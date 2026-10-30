try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import astropy.units as u
import numpy as np
import pytest

from saetass import DiffusionSolver, Grid, Particle, Solver, State
from saetass import units as su


# Helper function to set up and run a diffusion problem
def run_diffusion_test(grid_params, solver_params, initial_f):
    """
    Helper function to initialize and run the solver for a diffusion problem.
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

    # Convert solver params D_values to Quantity if not already
    solv_p = solver_params.copy()
    if (
        "D_values" in solv_p
        and not isinstance(solv_p["D_values"], u.Quantity)
        and not callable(solv_p["D_values"])
    ):
        solv_p["D_values"] = solv_p["D_values"] * su.DIFFUSION_COEFFICIENT

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
        problem_type="diffusion",
        operator_params={"diffusion": solv_p},
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )

    # Run simulation
    num_timesteps = len(t_g) - 1
    solver.step(num_timesteps)

    return solver.state.psi_p.to_value(su.PSI_P)


class Test1DRadialDiffusion:
    """
    Tests for pure 1D radial diffusion problems.
    A dummy energy dimension (num_E=1) is used to fit the solver's 2D structure.
    """

    @pytest.mark.parametrize("num_points", [200, 400, 800])
    def test_analytical_sinc(self, num_points, plot_results):
        """
        Validates against an analytical solution with a sinc-like initial profile.
        """
        r_0, r_end, t_final, D_const = 0.0, 1.0, 0.1, 1.0
        r_grid = np.linspace(r_0, r_end, num_points) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)
        f_initial = (np.pi / 2) * np.sinc(r_raw) * su.PSI_P

        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": np.full(num_points, D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }

        f_final_numerical = run_diffusion_test(
            grid_params, solver_params, f_initial
        ).flatten()
        f_final_analytical = f_initial.to_value(su.PSI_P) * np.exp(
            -(np.pi**2) * D_const * t_final
        )

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_final_numerical, label="Numerical", lw=2)
            plt.plot(r_raw, f_final_analytical, "r--", label="Analytical", lw=2)
            plt.title(f"1D Sinc Test - {num_points} points")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        assert np.allclose(f_final_numerical, f_final_analytical, atol=1e-4)

    def test_particle_conservation(self, plot_results):
        """
        Tests that the total number of particles is conserved in a closed system (where no enough time has passed for significant loss).
        """
        num_points, r_end, t_final, D_const = 800, 500.0, 1000.0, 0.1
        r_grid = np.linspace(0.0, r_end, num_points) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)
        dr = r_raw[1] - r_raw[0]
        f_initial = np.exp(-((r_raw - 50.0) ** 2) / (2 * 5**2)) * su.PSI_P

        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": np.full(num_points, D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }

        integrand_initial = 4 * np.pi * r_raw**2 * f_initial.to_value(su.PSI_P)
        n_particles_initial = np.sum(integrand_initial * dr)

        f_final_numerical = run_diffusion_test(
            grid_params, solver_params, f_initial
        ).flatten()

        integrand_final = 4 * np.pi * r_raw**2 * f_final_numerical
        n_particles_final = np.sum(integrand_final * dr)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                r_raw,
                f_initial.to_value(su.PSI_P),
                label=f"Initial, N={n_particles_initial:.4f}",
            )
            plt.plot(
                r_raw, f_final_numerical, label=f"Final, N={n_particles_final:.4f}"
            )
            plt.title("1D Particle Conservation Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        assert np.isclose(n_particles_initial, n_particles_final, rtol=1e-3)

    def test_discontinuous_diffusion_coefficient(self, plot_results):
        """
        Tests behavior with a discontinuous diffusion coefficient in 1D.
        A jump downwards in D should cause an accumulation of particles.
        """
        # Parameters
        num_points = 200
        r_end = 1.0
        t_final = 0.1

        # Grids
        r_grid = np.linspace(0.0, r_end, num_points) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition (Gaussian centered before the discontinuity)
        f_initial = np.exp(-((r_raw - 0.3) ** 2) / (2 * 0.05**2)) * su.PSI_P

        # Discontinuous diffusion coefficient
        D_values = np.ones(num_points)
        discontinuity_idx = int(num_points / 2)
        D_values[discontinuity_idx:] = 0.1  # D drops by a factor of 10
        D_qty = D_values * su.DIFFUSION_COEFFICIENT

        # Solver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": D_qty,
            "psi_end": 0.0 * su.PSI_P,
        }

        # Run simulation
        f_final = run_diffusion_test(grid_params, solver_params, f_initial).flatten()

        # Plotting for visual validation
        if plot_results:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            # Plot distribution
            ax1.plot(
                r_raw,
                f_initial.to_value(su.PSI_P),
                "k--",
                label="Initial Profile",
                alpha=0.5,
            )
            ax1.plot(r_raw, f_final, label="Final Profile", lw=2)
            ax1.axvline(
                r_raw[discontinuity_idx],
                color="r",
                linestyle="--",
                label="Discontinuity in D",
            )
            ax1.set_xlabel("Radius r")
            ax1.set_ylabel("f(r)", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Plot diffusion coefficient on a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(r_raw, D_values, "g-", label="Diffusion Coeff. D(r)", alpha=0.6)
            ax2.set_ylabel("D(r)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2.legend(loc="upper right")
            plt.title("1D Diffusion with Discontinuous Coefficient")
            fig.tight_layout()
            plt.show()

        # Check for accumulation: f should be higher just before the boundary
        f_before = f_final[discontinuity_idx - 1]
        f_after = f_final[discontinuity_idx]
        assert f_before > f_after

        # Check for gradient change: gradient should be steeper in the low-D region
        grad_high_D = np.mean(
            np.abs(np.diff(f_final[discontinuity_idx - 5 : discontinuity_idx - 1]))
        )
        grad_low_D = np.mean(
            np.abs(np.diff(f_final[discontinuity_idx : discontinuity_idx + 4]))
        )

        # Since D_high > D_low, we expect |grad_high| < |grad_low| for flux continuity
        assert grad_high_D < grad_low_D

    def test_boundary_conditions(self, plot_results):
        """
        Tests the three implemented boundary conditions (dirichlet, neumann, outflow)
        at the outer boundary r_end.
        """
        num_points = 200
        r_end = 5.0
        t_final = 1.0
        D_const = 1.0

        r_grid = np.linspace(0.0, r_end, num_points) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition: Gaussian pulse somewhat near the boundary
        f_initial = np.exp(-((r_raw - 3.5) ** 2) / (2 * 0.5**2)) * su.PSI_P

        grid_params = {"r_grid": r_grid, "t_grid": t_grid}

        # 1. Dirichlet: psi(r_end) = psi_end
        solver_params_dirichlet = {
            "boundary_condition": "dirichlet",
            "D_values": np.full(num_points, D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }
        f_dirichlet = run_diffusion_test(
            grid_params, solver_params_dirichlet, f_initial
        ).flatten()

        # 2. Neumann: zero flux (df/dr = 0)
        solver_params_neumann = {
            "boundary_condition": "neumann",
            "D_values": np.full(num_points, D_const) * su.DIFFUSION_COEFFICIENT,
        }
        f_neumann = run_diffusion_test(
            grid_params, solver_params_neumann, f_initial
        ).flatten()

        # 3. Outflow: f ~ 1/r (df/dr = -f/r)
        solver_params_outflow = {
            "boundary_condition": "outflow",
            "D_values": np.full(num_points, D_const) * su.DIFFUSION_COEFFICIENT,
        }
        f_outflow = run_diffusion_test(
            grid_params, solver_params_outflow, f_initial
        ).flatten()

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                r_raw,
                f_initial.to_value(su.PSI_P),
                "k--",
                label="Initial Profile",
                alpha=0.5,
            )
            plt.plot(r_raw, f_dirichlet, label="Dirichlet (f=0)")
            plt.plot(r_raw, f_neumann, label="Neumann (df/dr=0)")
            plt.plot(r_raw, f_outflow, label="Outflow (f ~ 1/r)")
            plt.title("Diffusion Boundary Conditions Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Assertions
        # 1. Dirichlet should exactly equal the psi_end value at the boundary
        assert np.isclose(f_dirichlet[-1], 0.0, atol=1e-12)

        # 2. Neumann should have zero gradient at the boundary: f_{N-1} ≈ f_N
        assert np.isclose(f_neumann[-2], f_neumann[-1], rtol=1e-3, atol=1e-4)

        # 3. Outflow should have dropping gradient f_{N-1} > f_N and f_N > 0 (it should let mass escape but not abruptly clamp to 0)
        assert f_outflow[-1] > 0.0
        assert f_outflow[-2] > f_outflow[-1]

        # 4. Outflow should drain mass faster than Neumann
        assert sum(f_outflow) < sum(f_neumann)


class Test2DEnergyRadiusDiffusion:
    """
    Tests for 2D problems involving both energy and radius dimensions.
    """

    def test_energy_dependent_diffusion(self, plot_results):
        """
        Verifies that diffusion is faster for energies with a higher diffusion coefficient.
        """
        num_r, num_E = 200, 10
        r_grid = np.linspace(0.0, 10.0, num_r) * su.LENGTH
        p_grid = np.logspace(0, 2, num_E) * su.MOMENTUM  # Momentum grid
        t_grid = np.linspace(0, 0.1, 100) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)
        p_raw = p_grid.to_value(su.MOMENTUM)

        # D(E) = D_0 * (p / p_0), i.e., diffusion is faster for higher "momentum"
        D_values = (
            (p_raw / p_raw[0])[:, np.newaxis]
            * np.ones((num_E, num_r))
            * su.DIFFUSION_COEFFICIENT
        )

        # Initial condition: Gaussian in radius, same for all energies
        f_initial = np.exp(-((r_raw - 5.0) ** 2) / (2 * 0.5**2))
        f_initial_2d = np.tile(f_initial, (num_E, 1)) * su.PSI_P

        grid_params = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params = {"D_values": D_values, "psi_end": 0.0 * su.PSI_P}

        f_final_2d = run_diffusion_test(grid_params, solver_params, f_initial_2d)

        # Calculate the standard deviation (width) of the profile for low and high energy
        def get_width(dist, r_coords):
            mean = np.sum(dist * r_coords) / np.sum(dist)
            variance = np.sum(dist * (r_coords - mean) ** 2) / np.sum(dist)
            return np.sqrt(variance)

        width_low_E = get_width(f_final_2d[0, :], r_raw)
        width_high_E = get_width(f_final_2d[-1, :], r_raw)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_final_2d[0, :], label=f"Low E (width={width_low_E:.2f})")
            plt.plot(
                r_raw, f_final_2d[-1, :], label=f"High E (width={width_high_E:.2f})"
            )
            plt.plot(r_raw, f_initial, "k--", label="Initial Profile")
            plt.title("2D Energy-Dependent Diffusion Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        # The profile for higher energy (and higher D) must be wider
        assert width_high_E > width_low_E

    def test_2d_decoupling_vs_1d(self, plot_results):
        """
        Verifies that a 2D run with D constant in energy matches a 1D run.
        """
        # 1. Run a standard 1D simulation
        num_r, D_const, t_final = 200, 1.0, 0.1
        r_grid = np.linspace(0.0, 1.0, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 200) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)
        f_initial_1d = (np.pi / 2) * np.sinc(r_raw) * su.PSI_P
        grid_params_1d = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params_1d = {
            "D_values": np.full(num_r, D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }
        f_final_1d = run_diffusion_test(
            grid_params_1d, solver_params_1d, f_initial_1d
        ).flatten()

        # 2. Run a 2D simulation with the same parameters
        num_E = 5
        p_grid = np.linspace(1, 5, num_E) * su.MOMENTUM
        f_initial_2d = np.tile(f_initial_1d.to_value(su.PSI_P), (num_E, 1)) * su.PSI_P
        grid_params_2d = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params_2d = {
            "D_values": np.full((num_E, num_r), D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }
        f_final_2d = run_diffusion_test(grid_params_2d, solver_params_2d, f_initial_2d)

        # 3. Compare the result of the 1D run with one slice of the 2D run
        f_slice_from_2d = f_final_2d[num_E // 2, :]

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_final_1d, "b-", label="1D Run Result", lw=4, alpha=0.7)
            plt.plot(
                r_raw,
                f_slice_from_2d,
                "r--",
                label="Slice from 2D Run",
                lw=2,
            )
            plt.title("2D Decoupling vs 1D Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        # The results should be identical
        assert np.allclose(f_final_1d, f_slice_from_2d, atol=1e-7)


def test_state_size_must_match_operator_grid():
    grid = Grid.uniform(
        r_min=0.0 * su.LENGTH,
        r_max=1.0 * su.LENGTH,
        num_r_cells=10,
        t_min=0.0 * su.TIME,
        t_max=1.0 * su.TIME,
        num_timesteps=2,
    )
    other_grid = Grid.uniform(
        r_min=0.0 * su.LENGTH, r_max=1.0 * su.LENGTH, num_r_cells=12
    )
    state = State(psi_p=np.ones(12) * su.PSI_P, grid=other_grid)
    # An operator used on its own checks the state it advances
    diffusion = DiffusionSolver(
        grid, grid.t_grid.to_value(su.TIME), {"D_values": np.ones(10)}
    )
    with pytest.raises(ValueError, match="1D state values must have size N"):
        diffusion.advance(1, state)
