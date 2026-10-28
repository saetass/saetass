"""Adiabatic losses: the divergence of the flow and the momentum change it causes."""

import numpy as np
import pytest

from saetass import Grid, Particle, Separable, Solver, State, TimeFactor
from saetass import units as su
from saetass.solvers.loss_solver import LossSolver, velocity_divergence


def _radial_grid(r_centers):
    return Grid(r_centers=r_centers * su.LENGTH)


def _faces(grid):
    return grid.r_faces.to_value(su.LENGTH)


class TestVelocityDivergence:
    @pytest.mark.parametrize(
        "r",
        [np.linspace(0.0, 10.0, 21), np.geomspace(0.1, 10.0, 30)],
        ids=["uniform_from_origin", "logarithmic"],
    )
    def test_homologous_flow_has_a_uniform_divergence(self, r):
        # v = alpha r has div v = 3 alpha everywhere, at the origin and boundaries too
        np.testing.assert_allclose(
            velocity_divergence(_radial_grid(r), 0.7 * r), 2.1, rtol=1e-13
        )

    def test_uniform_flow_gives_the_average_of_2v_over_r(self):
        r = np.linspace(1.0, 5.0, 9)
        grid = _radial_grid(r)
        faces = _faces(grid)
        divergence = velocity_divergence(grid, np.full(r.size, 3.0))
        average = 3.0 * 3.0 * np.diff(faces**2) / np.diff(faces**3)
        np.testing.assert_allclose(divergence, average, rtol=1e-13)

    def test_shock_concentrates_the_whole_jump_next_to_it(self):
        # A strong shock at r_s = 5: v = 4 upstream, v = (r_s / r)^2 downstream
        r = np.linspace(0.0, 10.0, 201)
        grid = _radial_grid(r)
        faces = _faces(grid)
        v = np.where(r < 5.0, 4.0, (5.0 / np.maximum(r, 5.0)) ** 2)
        divergence = velocity_divergence(grid, v)
        volumes = grid.volumes.to_value(su.VOLUME) / (4.0 * np.pi)
        shock = slice(99, 102)
        # Upstream expands as 2 v / r, downstream is incompressible, the shock compresses
        np.testing.assert_allclose(
            divergence[50:98],
            3 * 4 * np.diff(faces[50:99] ** 2) / np.diff(faces[50:99] ** 3),
        )
        np.testing.assert_allclose(divergence[103:-1], 0.0, atol=1e-4)
        assert np.all(divergence[shock] <= 0.0)
        # Gauss: the compression of the shock cells is the jump of r^2 v
        compression = np.sum(divergence[shock] * volumes[shock])
        # The telescoping sum leaves the fluxes through the outer faces of the shock cells
        expected = faces[102] ** 2 * 0.5 * (v[101] + v[102]) - faces[99] ** 2 * 4.0
        np.testing.assert_allclose(compression, expected, rtol=1e-12)

    def test_cell_at_the_origin_is_a_ball(self):
        # Grid places the first face at -dr / 2; the divergence uses the ball r < dr / 2
        r = np.linspace(0.0, 4.0, 5)
        grid = _radial_grid(r)
        assert _faces(grid)[0] < 0.0
        v = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        # Through the face at r = 1/2 flows 4 pi (1/2)^2 (1/2), out of a ball of 4 pi / 3 (1/2)^3
        np.testing.assert_allclose(velocity_divergence(grid, v)[0], 3.0, rtol=1e-13)

    def test_divergence_follows_the_leading_axes(self):
        r = np.linspace(0.0, 4.0, 5)
        v = np.stack([r, 2.0 * r])
        np.testing.assert_allclose(
            velocity_divergence(_radial_grid(r), v), [[3.0] * 5, [6.0] * 5]
        )

    def test_needs_two_cells(self):
        with pytest.raises(ValueError, match="at least 2 radial cells"):
            velocity_divergence(_radial_grid(np.array([1.0])), np.ones(1))


def _gaussian_in_log_p(p, center=1.5, width=0.25):
    return np.exp(-(((np.log10(p) - center) / width) ** 2))


def _homologous_problem(alpha, n_p=120, n_t=41, t_max=1.0, **loss):
    """Grid, initial state and parameters of adiabatic losses in the flow v = alpha r."""
    r = np.linspace(0.0, 2.0, 5)
    p = np.logspace(0.0, 3.0, n_p)
    grid = Grid(
        r_centers=r * su.LENGTH,
        p_centers=p * su.MOMENTUM,
        t_grid=np.linspace(0.0, t_max, n_t) * su.TIME,
    )
    psi = np.repeat(_gaussian_in_log_p(p)[:, None], r.size, axis=1)
    state = State(psi_p=psi * su.PSI_P, grid=grid, particle=Particle.PROTON)
    params = {
        "adiabatic_losses": True,
        "v_centers_physical": alpha * r * su.VELOCITY,
        **loss,
    }
    return state, params, r, p


def _run_losses(state, params):
    solver = Solver(state=state, problem_type="loss", operator_params={"loss": params})
    return solver.run().psi_p.to_value(su.PSI_P)


class TestAdiabaticLosses:
    @pytest.mark.parametrize("alpha", [0.8, -0.8], ids=["expansion", "compression"])
    def test_homologous_flow_shifts_the_spectrum(self, alpha):
        # psi_t = d/dp(alpha p psi): psi(p, t) = e^(alpha t) psi_0(p e^(alpha t))
        errors = []
        for n_p in (120, 240):
            state, params, r, p = _homologous_problem(alpha, n_p=n_p)
            psi = _run_losses(state, params)
            exact = np.exp(alpha) * _gaussian_in_log_p(p * np.exp(alpha))
            # The flow does not depend on the radius in this problem
            np.testing.assert_allclose(psi, psi[:, :1] * np.ones(r.size), rtol=1e-12)
            errors.append(np.linalg.norm(psi[:, 0] - exact) / np.linalg.norm(exact))
        assert errors[1] < 0.006, errors
        assert errors[0] / errors[1] > 3.0, errors  # second order in momentum

    def test_adiabatic_rate_adds_to_the_given_losses(self):
        alpha = 0.5
        P_dot = -0.2 * np.logspace(0.0, 3.0, 120) ** 1.5
        state, params, r, p = _homologous_problem(
            alpha, P_dot=P_dot * su.MOMENTUM_LOSS_RATE
        )
        with_adiabatic = _run_losses(state, params)

        state, _, _, _ = _homologous_problem(alpha)
        explicit = (
            P_dot[:, None] - alpha * p[:, None] * np.ones(r.size)
        ) * su.MOMENTUM_LOSS_RATE
        explicit_only = _run_losses(state, {"P_dot": explicit})
        np.testing.assert_allclose(
            with_adiabatic, explicit_only, rtol=1e-12, atol=1e-300
        )

    def test_time_dependent_flows(self):
        alpha = 0.6

        def growth(t):
            return 1.0 + t.to_value(su.TIME)

        def run(make_velocity):
            # Velocities are taken at the start of each step (#52): first order in time
            state, params, r, p = _homologous_problem(alpha, n_p=240, n_t=321)
            v = params["v_centers_physical"]
            return _run_losses(
                state, {**params, "v_centers_physical": make_velocity(v)}
            ), p

        callable_run, p = run(lambda v: lambda t: growth(t) * v)
        separable_run, _ = run(lambda v: v * TimeFactor(growth))
        np.testing.assert_allclose(separable_run, callable_run, rtol=1e-12, atol=1e-300)
        # The exact solution with div v = 3 alpha (1 + t): a shift by alpha (t + t^2 / 2)
        shift = alpha * 1.5
        exact = np.exp(shift) * _gaussian_in_log_p(p * np.exp(shift))
        error = np.linalg.norm(separable_run[:, 0] - exact) / np.linalg.norm(exact)
        assert error < 0.01, error

    @pytest.mark.parametrize(
        "separable_P_dot", [False, True], ids=["static_P_dot", "separable_P_dot"]
    )
    def test_separable_sums_stay_separable(self, separable_P_dot):
        state, params, r, p = _homologous_problem(0.4, n_t=3)
        P_dot = -0.1 * p * su.MOMENTUM_LOSS_RATE
        if separable_P_dot:
            P_dot = Separable([P_dot], [lambda t: 2.0])
        v = params["v_centers_physical"]
        solver = Solver(
            state=state,
            problem_type="loss",
            operator_params={
                "loss": {
                    **params,
                    "P_dot": P_dot,
                    "v_centers_physical": Separable([v], [0.5]),
                }
            },
        )
        loss = solver.operator_subsolvers[0]
        assert loss._separable is not None
        V = loss.velocities(0.0)
        expected = (2.0 if separable_P_dot else 1.0) * (
            -0.1 / np.log(10.0)
        ) - 0.5 * 0.4 / np.log(10.0)
        np.testing.assert_allclose(V, expected, rtol=1e-13)

    def test_callable_losses_with_a_static_flow(self):
        state, params, r, p = _homologous_problem(0.4, n_t=3)
        solver = Solver(
            state=state,
            problem_type="loss",
            operator_params={
                "loss": {**params, "P_dot": lambda t: -0.1 * p * su.MOMENTUM_LOSS_RATE}
            },
        )
        V = solver.operator_subsolvers[0].velocities(0.0)
        np.testing.assert_allclose(V, (-0.1 - 0.4) / np.log(10.0), rtol=1e-13)

    def test_invalid_combinations(self):
        state, params, r, p = _homologous_problem(0.4, n_t=3)
        v = params["v_centers_physical"]
        with pytest.raises(ValueError, match="only used with adiabatic_losses=True"):
            _run_losses(
                state, {"P_dot": -p * su.MOMENTUM_LOSS_RATE, "v_centers_physical": v}
            )
        with pytest.raises(
            ValueError, match="give P_dot, adiabatic_losses=True or both"
        ):
            _run_losses(state, {})
        # Built directly, without the conversion of Solver
        with pytest.raises(
            ValueError, match="needs P_dot, adiabatic_losses=True or both"
        ):
            LossSolver(state.grid, np.array([0.0, 1.0]), {"cfl": 0.5})
        with pytest.raises(ValueError, match="velocity of the flow"):
            _run_losses(state, {"adiabatic_losses": True})
        momentum_only = Grid(
            p_centers=p * su.MOMENTUM, t_grid=np.array([0.0, 1.0]) * su.TIME
        )
        state_1d = State(
            psi_p=np.ones(p.size) * su.PSI_P,
            grid=momentum_only,
            particle=Particle.PROTON,
        )
        with pytest.raises(ValueError, match="radial axis"):
            _run_losses(state_1d, params)
