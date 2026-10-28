try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import astropy.units as u
import numpy as np
import pytest

from saetass import Grid, Particle, Solver, State
from saetass import units as su


# Helper function to set up and run an advection problem
def run_advection_test(grid_params, solver_params, initial_f):
    """
    Helper function to initialize and run the solver for an advection problem.
    Handles both 1D and 2D cases.
    """
    r_g = (
        grid_params["r_grid"]
        if isinstance(grid_params["r_grid"], u.Quantity)
        else grid_params["r_grid"] * su.LENGTH
    )
    t_g = (
        grid_params["t_grid"]
        if isinstance(grid_params["t_grid"], u.Quantity)
        else grid_params["t_grid"] * su.TIME
    )
    p_g = grid_params.get("p_grid", None)
    if p_g is not None and not isinstance(p_g, u.Quantity):
        p_g = p_g * su.MOMENTUM

    init_f = initial_f if isinstance(initial_f, u.Quantity) else initial_f * su.PSI_P

    # Create grid and initial state
    grid = Grid(
        r_centers=r_g,
        t_grid=t_g,
        p_centers=p_g,
    )
    state = State(psi_p=init_f, grid=grid, particle=Particle.PROTON)

    # Create solver
    solver = Solver(
        state=state,
        problem_type="advection",
        operator_params={"advection": solver_params},
        substeps={"advection": 1},
        splitting_scheme="strang",
    )

    # Run simulation
    num_timesteps = len(t_g) - 1
    solver.step(num_timesteps)

    return solver.state.psi_p.to_value(su.PSI_P)


class Test1DRadialAdvection:
    """
    Tests for pure 1D radial advection problems.
    """

    def test_simple_translation(self, plot_results):
        """
        Validates that a Gaussian pulse advects correctly against the analytical
        solution, including spherical dilution.
        """
        num_r, r_end, t_final = 4000, 100.0, 10.0
        v_const = 5.0
        r_initial_peak = 20.0
        sigma = 2.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 7000) * su.TIME

        # Initial condition: Gaussian pulse
        r_raw = r_grid.to_value(su.LENGTH)
        f_initial = np.exp(-((r_raw - r_initial_peak) ** 2) / (2 * sigma**2)) * su.PSI_P

        # Advection parameters
        v_field = np.full(num_r, v_const) * su.VELOCITY
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        }

        f_final_numerical = run_advection_test(
            grid_params, solver_params, f_initial
        ).flatten()

        # Analytical solution at t_final for spherical advection
        r_shifted = r_raw - v_const * t_final
        f_analytical = np.zeros_like(r_raw)
        # Mask for valid regions (r > 0 and r_shifted > 0)
        mask = (r_raw > 0) & (r_shifted > 0)
        f_analytical[mask] = ((r_shifted[mask] / r_raw[mask]) ** 2) * np.exp(
            -((r_shifted[mask] - r_initial_peak) ** 2) / (2 * sigma**2)
        )

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                r_raw, f_initial.to_value(su.PSI_P), "k--", label="Initial Profile"
            )
            plt.plot(r_raw, f_final_numerical, label="Final (Numerical)", lw=2)
            plt.plot(
                r_raw,
                f_analytical,
                "r:",
                label="Final (Analytical)",
                lw=3,
                alpha=0.8,
            )
            plt.title("1D Advection: Numerical vs Analytical Solution")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        assert np.allclose(f_final_numerical, f_analytical, atol=1e-3)

    def test_spherical_dilution(self, plot_results):
        """
        Tests that the peak amplitude decreases as ~1/r^2 due to spherical expansion.
        """
        num_r, r_end, t_final = 4000, 100.0, 10.0
        v_const = 5.0
        r_initial = 20.0
        sigma = 2.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 5000) * su.TIME

        r_raw = r_grid.to_value(su.LENGTH)
        f_initial = np.exp(-((r_raw - r_initial) ** 2) / (2 * sigma**2)) * su.PSI_P
        peak_initial = np.max(f_initial.to_value(su.PSI_P))

        v_field = np.full(num_r, v_const) * su.VELOCITY
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        }

        f_final = run_advection_test(grid_params, solver_params, f_initial).flatten()

        peak_final_numerical = np.max(f_final)
        r_final_analytical = r_initial + v_const * t_final

        # Analytical peak height decrease due to spherical dilution (use analytical formula)
        peak_final_analytical = peak_initial * (r_initial / r_final_analytical) ** 2

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                r_raw,
                f_initial.to_value(su.PSI_P),
                "k--",
                label=f"Initial (Peak={peak_initial:.3f})",
            )
            plt.plot(r_raw, f_final, label=f"Final (Peak={peak_final_numerical:.3f})")
            plt.axhline(
                peak_final_analytical,
                color="r",
                ls="--",
                label=f"Analytical Peak Height ({peak_final_analytical:.3f})",
            )
            plt.title("1D Advection: Spherical Dilution Test")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # Check that the final peak height is close to the analytical prediction
        # Use a larger tolerance due to numerical diffusion affecting the peak
        assert np.isclose(peak_final_numerical, peak_final_analytical, rtol=1e-2)


class Test2DEnergyRadiusAdvection:
    """
    Tests for 2D advection problems (energy and radius).
    """

    def test_energy_independent_advection(self, plot_results):
        """
        Verifies that if velocity is the same for all energies, all energy slices
        advect identically.
        """
        num_r, num_E, r_end, t_final = 1000, 10, 50.0, 5.0
        v_const = 4.0
        r_initial = 10.0
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        p_grid = np.logspace(0, 2, num_E) * su.MOMENTUM  # Momentum grid
        t_grid = np.linspace(0, t_final, 1000) * su.TIME

        # Same velocity field for all energies
        v_field = np.full((num_E, num_r), v_const) * su.VELOCITY

        # Initial condition: Gaussian in radius, same for all energies
        r_raw = r_grid.to_value(su.LENGTH)
        f_initial_1d = np.exp(-((r_raw - r_initial) ** 2) / (2 * 1.0**2))
        f_initial_2d = np.tile(f_initial_1d, (num_E, 1)) * su.PSI_P

        grid_params = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        }

        f_final_2d = run_advection_test(grid_params, solver_params, f_initial_2d)

        # Get the final profiles for the lowest and highest energies
        f_final_low_E = f_final_2d[0, :]
        f_final_high_E = f_final_2d[-1, :]

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_initial_1d, "k--", label="Initial Profile")
            plt.plot(r_raw, f_final_low_E, label="Final (Low E)", lw=3, alpha=0.8)
            plt.plot(r_raw, f_final_high_E, "r:", label="Final (High E)", lw=3)
            plt.title("2D Energy-Independent Advection")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # The profiles for all energies should be identical
        assert np.allclose(f_final_low_E, f_final_high_E, atol=1e-7)


class TestAdvectionSolverExceptionsAndEdges:
    def test_invalid_parameters(self):
        grid = Grid(
            r_centers=np.array([1.0, 2.0]) * su.LENGTH,
            is_p_log=False,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        state = State(
            psi_p=np.ones((2,)) * su.PSI_P, grid=grid, particle=Particle.PROTON
        )
        # missing inflow value
        params = {"v_centers": np.array([[1.0, 1.0]]) * su.VELOCITY}
        with pytest.raises(
            ValueError,
            match=r"Missing required AdvectionSolver parameter\(s\) \['inflow_value_U'\]",
        ):
            Solver(
                state=state,
                problem_type="advection",
                operator_params={"advection": params},
                substeps={"advection": 1},
            )

        # a limiter for the first-order scheme, which does not limit slopes
        params["inflow_value_U"] = 0.0 * su.AREA * su.PSI_P
        params.update(order=1, limiter="minmod")
        with pytest.raises(ValueError, match="only applies to order 2"):
            Solver(
                state=state,
                problem_type="advection",
                operator_params={"advection": params},
                substeps={"advection": 1},
            )

        # invalid limiter
        params["order"] = 2
        params["limiter"] = "invalid_limiter"
        with pytest.raises(ValueError, match="limiter must be"):
            Solver(
                state=state,
                problem_type="advection",
                operator_params={"advection": params},
                substeps={"advection": 1},
            )

    @pytest.mark.parametrize(
        "given, expected",
        [
            ({}, {"order": 2, "limiter": "minmod", "cfl": 0.5}),
            ({"order": 1}, {"order": 1, "limiter": None, "cfl": 0.5}),
            ({"limiter": "mc", "cfl": 0.8}, {"order": 2, "limiter": "mc", "cfl": 0.8}),
        ],
        ids=["defaults", "first_order", "given"],
    )
    def test_scheme_defaults(self, given, expected):
        grid = Grid(
            r_centers=np.linspace(0.0, 1.0, 5) * su.LENGTH,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        params = {
            "v_centers": np.ones(5) * su.VELOCITY,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **given,
        }
        solver = Solver(
            state=State(psi_p=np.ones(5) * su.PSI_P, grid=grid),
            problem_type="advection",
            operator_params={"advection": params},
            quiet=True,
        )
        advection = solver.operator_subsolvers[0]
        assert {
            "order": advection.order,
            "limiter": advection.limiter,
            "cfl": advection.cfl,
        } == expected
        solver.run()

    def test_other_limiters(self):
        grid = Grid(
            r_centers=np.array([1.0, 3.0]) * su.LENGTH,
            p_centers=np.array([1.0]) * su.MOMENTUM,
            is_p_log=False,
            t_grid=np.array([0.0, 1.0]) * su.TIME,
        )
        f_init = np.array([[1.0, 2.0]]) * su.PSI_P
        state = State(psi_p=f_init, grid=grid, particle=Particle.PROTON)

        for limiter in ["vanleer", "mc"]:
            params = {
                "v_centers": np.array([[1.0, 1.0]]) * su.VELOCITY,
                "limiter": limiter,
                "order": 2,
                "cfl": 0.5,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            }
            solver = Solver(
                state=state.clone(),
                problem_type="advection",
                operator_params={"advection": params},
                substeps={"advection": 1},
            )
            solver.step(1)
            # Just test it executes successfully
            assert np.any(solver.state.psi_p.to_value(su.PSI_P))


@pytest.mark.parametrize("limiter", ["minmod", "vanleer", "mc"])
def test_smooth_pulse_converges_at_second_order(limiter):
    """A Gaussian pulse moved at constant speed converges at second order with every limiter."""
    v, t_final = 1.0, 0.4
    gaussian = lambda x: np.exp(-(((x - 0.9) / 0.15) ** 2))  # noqa: E731

    def error(num_r):
        r = np.linspace(0.0, 2.0, num_r)
        solver = Solver(
            state=State(
                psi_p=gaussian(r) * su.PSI_P,
                grid=Grid(
                    r_centers=r * su.LENGTH,
                    t_grid=np.linspace(0, t_final, 41) * su.TIME,
                ),
                particle=Particle.PROTON,
            ),
            problem_type="advection",
            operator_params={
                "advection": {
                    "v_centers": np.full(num_r, v) * su.VELOCITY,
                    "limiter": limiter,
                    "cfl": 0.5,
                    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
                }
            },
        )
        solver.step(40)
        # U = r^2 psi is shifted rigidly, so psi = (R / r)^2 psi_0(R) with R = r - v t
        R = r - v * t_final
        exact = (R[1:] / r[1:]) ** 2 * gaussian(R[1:])
        keep = r[1:] > 0.2
        diff = solver.state.psi_p.to_value(su.PSI_P)[1:] - exact
        return np.linalg.norm(diff[keep]) / np.linalg.norm(exact[keep])

    errors = [error(n) for n in (160, 320, 640)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.5), (errors, orders)
