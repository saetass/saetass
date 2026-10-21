"""Unit tests for the momentum-loss (LossSolver) operator.

Focus areas
-----------
1. Positivity: f must remain >= 0 at all grid points at every time step.
2. Step advection: a step-function initial condition should shift leftward in
   momentum space (toward lower energy) at the correct rate.
3. Mass conservation (with outflow): mass lost through the low-momentum
   boundary must equal the integral reduction of f.
"""

import astropy.units as u
import numpy as np
import pytest

from saetass import Grid, Particle, Solver, State, TimeFactor
from saetass import units as su

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loss_grid_and_solver(
    num_p: int,
    p_min: float,
    p_max: float,
    t_final: float,
    num_t: int,
    P_dot_const: float,
    f_init: np.ndarray,
    order: int = 2,
    cfl: float = 0.5,
):
    """Build a 1D (momentum-only) loss solver with a constant P_dot."""
    p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p) * su.MOMENTUM
    t_grid = np.linspace(0, t_final, num_t + 1) * su.TIME
    P_dot = np.full(num_p, P_dot_const) * su.MOMENTUM_LOSS_RATE
    init_f = f_init if isinstance(f_init, u.Quantity) else f_init * su.PSI_P

    loss_params = {
        "P_dot": P_dot,
        "limiter": "minmod",
        "cfl": cfl,
        "inflow_value_U": np.zeros(1) * su.MOMENTUM * su.PSI_P,
        "order": order,
        "adiabatic_losses": False,
    }

    grid = Grid(p_centers=p_centers, t_grid=t_grid)
    state = State(psi_p=init_f, grid=grid, particle=Particle.PROTON)
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="loss",
        operator_params={"loss": loss_params},
        substeps={"loss": 1},
        splitting_scheme="strang",
    )
    return solver, p_centers.to_value(su.MOMENTUM)


def _setup_loss_solver(
    P_dot,
    f_init,
    p_centers,
    t_grid,
    cfl=0.5,
    order=1,
    p_faces=None,
    is_p_log=True,
):
    """Convenience factory for LossSolver via the top-level Solver."""
    p_c = p_centers if isinstance(p_centers, u.Quantity) else p_centers * su.MOMENTUM
    t_g = t_grid if isinstance(t_grid, u.Quantity) else t_grid * su.TIME
    init_f = f_init if isinstance(f_init, u.Quantity) else f_init * su.PSI_P
    p_dot_in = (
        P_dot
        if (isinstance(P_dot, u.Quantity) or callable(P_dot))
        else P_dot * su.MOMENTUM_LOSS_RATE
    )

    loss_params = {
        "P_dot": p_dot_in,
        "limiter": "minmod",
        "cfl": cfl,
        "inflow_value_U": np.zeros(1) * su.MOMENTUM * su.PSI_P,
        "order": order,
        "adiabatic_losses": False,
    }

    grid = Grid(p_centers=p_c, t_grid=t_g)
    state = State(psi_p=init_f, grid=grid, particle=Particle.PROTON)
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="loss",
        operator_params={"loss": loss_params},
        substeps={"loss": 1},
        splitting_scheme="strang",
    )
    return solver, p_c.to_value(su.MOMENTUM)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLossSolverPositivity:
    """The loss solver must never produce negative values of f."""

    def test_step_function_stays_positive(self):
        """Step-function IC: sharp front is the worst case for overshoots."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        t_final = 0.05  # short run, few loss timescales
        num_t = 500

        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        # Step at mid-grid
        p_mid = np.sqrt(p_min * p_max)
        f_init = np.where(p_centers < p_mid, 1.0, 0.0).astype(float) * su.PSI_P

        # P_dot < 0: energy is lost (particles shift to lower p)
        # Rate chosen so ν = |V_gen|*dt/dp_log ~ 0.3 with cfl=0.5
        P_dot_const = -0.5 * p_mid  # units: [p] / [t]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.psi_p.to_value(su.PSI_P).flatten()

        assert np.all(f_final >= -1e-14), (
            f"Negative values in f after loss step: min={f_final.min():.3e}"
        )

    def test_gaussian_stays_positive(self):
        """Gaussian IC: a smooth initial condition should stay smooth and positive."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        t_final = 0.02
        num_t = 300

        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        log_p = np.log10(p_centers)
        log_p_mid = 0.5 * (np.log10(p_min) + np.log10(p_max))
        sigma = 0.3  # in log10 units
        f_init = np.exp(-((log_p - log_p_mid) ** 2) / (2 * sigma**2)) * su.PSI_P

        P_dot_const = -0.2 * p_centers[num_p // 2]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.psi_p.to_value(su.PSI_P).flatten()

        assert np.all(f_final >= -1e-14), (
            f"Negative values after Gaussian loss run: min={f_final.min():.3e}"
        )


class TestLossSolverPhysics:
    """Physical sanity checks for the loss solver."""

    def test_step_advects_leftward(self):
        """Step front should move toward lower momentum (losses drain energy)."""
        num_p = 300
        p_min, p_max = 1.0, 1e4
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        p_mid = np.sqrt(p_min * p_max)

        f_init = np.where(p_centers < p_mid, 1.0, 0.0).astype(float) * su.PSI_P

        # Short run: front should move, but only slightly
        t_final = 0.02
        num_t = 400
        P_dot_const = -0.3 * p_mid

        solver, pc = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.psi_p.to_value(su.PSI_P).flatten()

        # Locate the 50% crossing (step front position) before and after
        def _front_p(f, p):
            mid_val = 0.5
            # Find first index where f drops below 0.5
            for i in range(len(f) - 1):
                if f[i] >= mid_val and f[i + 1] < mid_val:
                    # Linear interpolation
                    return p[i] + (p[i + 1] - p[i]) * (f[i] - mid_val) / (
                        f[i] - f[i + 1]
                    )
            return None

        p_front_initial = p_mid
        p_front_final = _front_p(f_final, pc)

        assert p_front_final is not None, "Could not find step front in final f."
        assert p_front_final < p_front_initial, (
            f"Step front did not shift to lower p: initial={p_front_initial:.3f}, "
            f"final={p_front_final:.3f}"
        )

    def test_mass_non_increasing(self):
        """Total 'mass' integral cannot increase (losses only drain energy)."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        dp = np.diff(np.log10(p_centers))
        dp = np.append(dp, dp[-1])  # cell widths in log10 space

        f_init = np.ones(num_p) * su.PSI_P

        t_final = 0.05
        num_t = 500
        P_dot_const = -0.2 * p_centers[num_p // 2]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        mass_initial = float(np.sum(f_init.to_value(su.PSI_P) * dp))
        solver.step(num_t)
        f_final = solver.state.psi_p.to_value(su.PSI_P).flatten()
        mass_final = float(np.sum(f_final * dp))

        assert mass_final <= mass_initial + 1e-10, (
            f"Total mass increased: {mass_initial:.6g} → {mass_final:.6g}"
        )


class TestLossSolverOrder:
    """Check that order=2 is at least as accurate as order=1 for smooth data."""

    def test_order2_more_accurate_than_order1(self):
        """
        For the same number of time steps and CFL, the 2nd-order scheme should
        produce a lower L2 error than the 1st-order scheme when measured against
        a very fine 2nd-order reference run.
        """
        num_p = 150
        p_min, p_max = 1.0, 1e3
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        log_p = np.log10(p_centers)
        log_p_mid = 0.5 * (np.log10(p_min) + np.log10(p_max))
        sigma = 0.4

        f_init = np.exp(-((log_p - log_p_mid) ** 2) / (2 * sigma**2)) * su.PSI_P
        t_final = 0.01
        num_t = 60  # coarse — both schemes use same step count
        P_dot_const = -0.15 * p_centers[num_p // 2]

        # --- reference: very fine 2nd-order run (10× more steps, same cfl) ---
        solver_ref, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t * 10,
            P_dot_const,
            f_init.copy(),
            order=2,
            cfl=0.5,
        )
        solver_ref.step(num_t * 10)
        f_ref = solver_ref.state.psi_p.to_value(su.PSI_P).flatten()

        # --- order 1 (coarse) ---
        solver_1, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t,
            P_dot_const,
            f_init.copy(),
            order=1,
            cfl=0.5,
        )
        solver_1.step(num_t)
        f1 = solver_1.state.psi_p.to_value(su.PSI_P).flatten()

        # --- order 2 (coarse, same step count) ---
        solver_2, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t,
            P_dot_const,
            f_init.copy(),
            order=2,
            cfl=0.5,
        )
        solver_2.step(num_t)
        f2 = solver_2.state.psi_p.to_value(su.PSI_P).flatten()

        err1 = float(np.sqrt(np.mean((f1 - f_ref) ** 2)))
        err2 = float(np.sqrt(np.mean((f2 - f_ref) ** 2)))

        # 2nd-order scheme should be at least as accurate as 1st-order at the
        # same step count (allow 20% slack for CFL sub-cycling differences)
        assert err2 <= err1 * 1.2, (
            f"2nd-order scheme significantly worse than 1st-order: "
            f"L2(order1)={err1:.3e}, L2(order2)={err2:.3e}"
        )


class TestLossSolverExceptionsAndEdges:
    @pytest.mark.parametrize(
        "extra",
        [{}, {"v_centers_physical": np.ones((2, 2)) * su.VELOCITY}],
        ids=["without_velocity", "with_velocity"],
    )
    def test_adiabatic_losses_not_implemented(self, extra):
        grid = Grid(
            r_centers=np.array([1.0, 2.0]) * su.LENGTH,
            p_centers=np.array([50.0, 100.0]) * su.MOMENTUM,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        params = {
            "P_dot": np.full((2, 2), -0.1) * su.MOMENTUM_LOSS_RATE,
            "limiter": "minmod",
            "cfl": 0.5,
            "order": 1,
            "adiabatic_losses": True,
            **extra,
        }
        state = State(
            psi_p=np.ones((2, 2)) * su.PSI_P, grid=grid, particle=Particle.PROTON
        )
        with pytest.raises(NotImplementedError, match="Adiabatic losses"):
            Solver(
                grid=grid,
                state=state,
                problem_type="loss",
                operator_params={"loss": params},
            )

    def test_callable_P_dot(self):
        grid = Grid(
            p_centers=np.array([50.0, 100.0]) * su.MOMENTUM,
            is_p_log=True,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )

        def p_dot_func(t):
            t_val = t.value if hasattr(t, "value") else t
            return np.array([-0.1 * t_val, -0.1 * t_val]) * su.MOMENTUM_LOSS_RATE

        params = {
            "P_dot": p_dot_func,
            "adiabatic_losses": False,
            "limiter": "minmod",
            "order": 1,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
        }
        state = State(psi_p=np.ones(2) * su.PSI_P, grid=grid, particle=Particle.PROTON)
        solver = Solver(
            grid=grid,
            state=state,
            problem_type="loss",
            operator_params={"loss": params},
            substeps={"loss": 1},
        )
        assert solver.operator_subsolvers[0] is not None

    def test_inverse_generalized_errors(self):
        grid = Grid(
            p_centers=np.array([50.0, 100.0]) * su.MOMENTUM,
            is_p_log=True,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        params = {
            "P_dot": np.array([-1.0, -1.0]) * su.MOMENTUM_LOSS_RATE,
            "adiabatic_losses": False,
            "limiter": "minmod",
            "order": 1,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
        }
        state = State(psi_p=np.ones(2) * su.PSI_P, grid=grid, particle=Particle.PROTON)
        solver = Solver(
            grid=grid,
            state=state,
            problem_type="loss",
            operator_params={"loss": params},
            substeps={"loss": 1},
        )
        ls = solver.operator_subsolvers[0]

        # 1D mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            ls._inverse_generalized_variable(np.ones(3), grid)

        # 2D mismatch
        with pytest.raises(ValueError, match="Expected U.shape"):
            ls._inverse_generalized_variable(np.ones((2, 3)), grid)

        # 3D
        with pytest.raises(ValueError, match="only supports 1D or 2D"):
            ls._inverse_generalized_variable(np.ones((2, 2, 2)), grid)

    def test_non_positive_momentum_rejected_at_construction(self):
        # Faces [-2, 0, 2] give a legal Grid with centers [-1, 1]: the loss
        # transforms divide by p, so this must fail fast, not at every step.
        grid = Grid(
            p_faces=np.array([-2.0, 0.0, 2.0]) * su.MOMENTUM,
            is_p_log=False,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        params = {
            "P_dot": np.array([-1.0, -1.0]) * su.MOMENTUM_LOSS_RATE,
            "limiter": "minmod",
            "order": 1,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
        }
        state = State(psi_p=np.ones(2) * su.PSI_P, grid=grid, particle=Particle.PROTON)
        with pytest.raises(ValueError, match="strictly positive momentum"):
            Solver(
                grid=grid,
                state=state,
                problem_type="loss",
                operator_params={"loss": params},
                substeps={"loss": 1},
            )

    def test_inflow_psi_and_U_are_mutually_exclusive(self):
        grid = Grid(
            p_centers=np.logspace(0, 2, 10) * su.MOMENTUM,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        params = {
            "P_dot": np.full(10, -1.0) * su.MOMENTUM_LOSS_RATE,
            "limiter": "minmod",
            "order": 1,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
        }
        state = State(psi_p=np.ones(10) * su.PSI_P, grid=grid, particle=Particle.PROTON)
        with pytest.raises(
            ValueError, match="either inflow_value_psi or inflow_value_U"
        ):
            Solver(
                grid=grid,
                state=state,
                problem_type="loss",
                operator_params={"loss": params},
                substeps={"loss": 1},
            )

    def test_inflow_defaults_to_zero(self):
        """Gains without an inflow value let nothing enter through the last face."""
        grid = Grid(
            p_centers=np.logspace(0, 2, 10) * su.MOMENTUM,
            t_grid=np.linspace(0.0, 0.5, 6) * su.TIME,
        )
        params = {
            "P_dot": np.full(10, 1.0) * su.MOMENTUM_LOSS_RATE,
            "limiter": "minmod",
            "order": 2,
            "cfl": 0.5,
        }

        def run(extra):
            state = State(psi_p=np.ones(10) * su.PSI_P, grid=grid)
            solver = Solver(
                grid=grid,
                state=state,
                problem_type="loss",
                operator_params={"loss": {**params, **extra}},
            )
            return solver.run().psi_p.to_value(su.PSI_P)

        np.testing.assert_array_equal(
            run({}), run({"inflow_value_psi": 0.0 * su.PSI_P})
        )


class TestLossSolver2D:
    params = {
        "limiter": "minmod",
        "order": 1,
        "cfl": 0.5,
        "inflow_value_psi": 0.0 * su.PSI_P,
    }

    def test_requires_momentum_grid(self):
        grid = Grid(
            r_centers=np.linspace(0.0, 1.0, 5) * su.LENGTH,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        with pytest.raises(ValueError, match="requires a Grid with a momentum axis"):
            Solver(
                grid=grid,
                state=State(psi_p=np.ones(5) * su.PSI_P, grid=grid),
                problem_type="loss",
                operator_params={
                    "loss": {**self.params, "P_dot": np.ones(5) * su.MOMENTUM_LOSS_RATE}
                },
            )

    def test_2d_matches_1d_slices(self):
        """Each radial column of a 2D loss problem evolves as the 1D problem."""
        p = np.logspace(0, 2, 30) * su.MOMENTUM
        t_grid = np.linspace(0.0, 0.05, 11) * su.TIME
        log_p = np.log10(p.to_value(su.MOMENTUM))
        profile = np.exp(-((log_p - 1.0) ** 2) / (2 * 0.3**2))
        amplitudes = np.array([1.0, 2.0, 3.0])
        P_dot = -0.3 * p.to_value(su.MOMENTUM)

        grid_2d = Grid(
            r_centers=np.array([1.0, 2.0, 3.0]) * su.LENGTH, p_centers=p, t_grid=t_grid
        )
        solver_2d = Solver(
            grid=grid_2d,
            state=State(psi_p=np.outer(profile, amplitudes) * su.PSI_P, grid=grid_2d),
            problem_type="loss",
            operator_params={
                "loss": {
                    **self.params,
                    "P_dot": np.tile(P_dot[:, None], (1, 3)) * su.MOMENTUM_LOSS_RATE,
                }
            },
        )
        psi_2d = solver_2d.run().psi_p.to_value(su.PSI_P)

        for j, amplitude in enumerate(amplitudes):
            grid_1d = Grid(p_centers=p, t_grid=t_grid)
            solver_1d = Solver(
                grid=grid_1d,
                state=State(psi_p=amplitude * profile * su.PSI_P, grid=grid_1d),
                problem_type="loss",
                operator_params={
                    "loss": {**self.params, "P_dot": P_dot * su.MOMENTUM_LOSS_RATE}
                },
            )
            np.testing.assert_allclose(
                psi_2d[:, j], solver_1d.run().psi_p.to_value(su.PSI_P), rtol=1e-12
            )

    @pytest.mark.parametrize("kind", ["static", "callable", "separable"])
    def test_cfl_step_includes_the_last_momentum_of_every_radius(self, kind):
        """The last momentum of each radius bounds the CFL step, as the outer face."""
        p = np.logspace(0, 2, 20) * su.MOMENTUM
        t_grid = np.array([0.0, 0.2]) * su.TIME
        P_dot = np.tile(-0.3 * p.to_value(su.MOMENTUM)[:, None], (1, 3))
        P_dot[-1, 0] *= 20.0  # the last momentum loses fastest at the first radius
        rate = P_dot * su.MOMENTUM_LOSS_RATE
        if kind == "callable":
            rate_2d = lambda t: P_dot * su.MOMENTUM_LOSS_RATE  # noqa: E731
        elif kind == "separable":
            rate_2d = rate * TimeFactor(lambda t: 1.0)
        else:
            rate_2d = rate

        def subcycles(grid, P):
            solver = Solver(
                grid=grid,
                state=State(psi_p=np.ones(grid.shape) * su.PSI_P, grid=grid),
                problem_type="loss",
                operator_params={"loss": {**self.params, "P_dot": P}},
            )
            solver.run()
            return solver.operator_subsolvers[0]._cfl_subcycles

        grid_2d = Grid(
            r_centers=np.array([1.0, 2.0, 3.0]) * su.LENGTH, p_centers=p, t_grid=t_grid
        )
        grid_1d = Grid(p_centers=p, t_grid=t_grid)
        assert subcycles(grid_2d, rate_2d) == subcycles(grid_1d, rate[:, 0])
