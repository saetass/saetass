try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import astropy.units as u
import numpy as np

from saetass import Grid, Particle, Solver, State
from saetass import units as su


def run_solver_test(grid_params, operator_params, problem_types, initial_f):
    """
    Helper function to initialize and run the solver for multiple operators.
    """
    r_g = grid_params.get("r_grid", None)
    if r_g is not None and not isinstance(r_g, u.Quantity):
        r_g = r_g * su.LENGTH

    t_g = (
        grid_params["t_grid"]
        if isinstance(grid_params["t_grid"], u.Quantity)
        else grid_params["t_grid"] * su.TIME
    )

    p_g = grid_params.get("p_grid", None)
    if p_g is not None and not isinstance(p_g, u.Quantity):
        p_g = p_g * su.MOMENTUM

    init_f = initial_f if isinstance(initial_f, u.Quantity) else initial_f * su.PSI_P

    # Ensure operator parameters have proper Quantities
    op_params = {}
    for op_name, op_dict in operator_params.items():
        sub_p = op_dict.copy()
        if op_name == "advection":
            if (
                "v_centers" in sub_p
                and not isinstance(sub_p["v_centers"], u.Quantity)
                and not callable(sub_p["v_centers"])
            ):
                sub_p["v_centers"] = sub_p["v_centers"] * su.VELOCITY
        elif op_name == "diffusion":
            if (
                "D_values" in sub_p
                and not isinstance(sub_p["D_values"], u.Quantity)
                and not callable(sub_p["D_values"])
            ):
                sub_p["D_values"] = sub_p["D_values"] * su.DIFFUSION_COEFFICIENT
        elif op_name == "source":
            if (
                "source" in sub_p
                and not isinstance(sub_p["source"], u.Quantity)
                and not callable(sub_p["source"])
            ):
                sub_p["source"] = sub_p["source"] * su.SOURCE_PSI_P
        elif op_name == "loss":
            if (
                "P_dot" in sub_p
                and not isinstance(sub_p["P_dot"], u.Quantity)
                and not callable(sub_p["P_dot"])
            ):
                sub_p["P_dot"] = sub_p["P_dot"] * su.MOMENTUM_LOSS_RATE
        op_params[op_name] = sub_p

    grid = Grid(
        r_centers=r_g,
        t_grid=t_g,
        p_centers=p_g,
    )
    state = State(psi_p=init_f, grid=grid, particle=Particle.PROTON)

    solver = Solver(
        state=state,
        problem_type=problem_types,
        operator_params=op_params,
        splitting_scheme="strang",
    )

    num_timesteps = len(t_g) - 1
    solver.step(num_timesteps)

    return solver.state.psi_p.to_value(su.PSI_P)


class TestAdvectionDiffusion:
    """
    Tests for problems combining advection and diffusion operators.
    """

    def test_advection_diffusion_qualitative(self, plot_results):
        """
        Validates the combined effect of advection and diffusion qualitatively.
        The final profile should be advected and diffused compared to the initial state.
        """
        # Parameters
        num_r, r_end, t_final = 500, 100.0, 8.0
        v_const = 4.0  # Advection speed
        D_const = 0.5  # Diffusion coefficient
        r_initial_peak = 20.0
        sigma = 3.0

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 200) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition: Gaussian pulse
        f_initial = np.exp(-((r_raw - r_initial_peak) ** 2) / (2 * sigma**2)) * su.PSI_P

        # --- Run 1: Advection + Diffusion ---
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params_both = {
            "advection": {
                "v_centers": np.full(num_r, v_const) * su.VELOCITY,
                "limiter": "minmod",
                "order": 2,
                "cfl": 0.8,
                "inflow_value_U": 1.0 * su.AREA * su.PSI_P,
            },
            "diffusion": {
                "D_values": np.full(num_r, D_const) * su.DIFFUSION_COEFFICIENT,
                "psi_end": 0.0 * su.PSI_P,
            },
        }
        f_final_both = run_solver_test(
            grid_params, op_params_both, "advection-diffusion", f_initial
        ).flatten()

        # --- Run 2: Advection Only ---
        op_params_adv = {"advection": op_params_both["advection"]}
        f_final_adv_only = run_solver_test(
            grid_params, op_params_adv, "advection", f_initial
        ).flatten()

        # --- Run 3: Diffusion Only ---
        op_params_diff = {"diffusion": op_params_both["diffusion"]}
        f_final_diff_only = run_solver_test(
            grid_params, op_params_diff, "diffusion", f_initial
        ).flatten()

        # --- Analysis ---
        def get_width(dist):
            half_max = np.max(dist) / 2.0
            indices = np.where(dist > half_max)[0]
            if len(indices) < 2:
                return 0
            return r_raw[indices[-1]] - r_raw[indices[0]]

        width_initial = get_width(f_initial.to_value(su.PSI_P))
        width_adv_only = get_width(f_final_adv_only)
        width_both = get_width(f_final_both)

        peak_pos_adv_only = r_raw[np.argmax(f_final_adv_only)]
        peak_pos_both = r_raw[np.argmax(f_final_both)]

        # --- Plotting ---
        if plot_results:
            plt.figure(figsize=(12, 8))
            plt.plot(
                r_raw, f_initial.to_value(su.PSI_P), "k--", label="Initial Profile"
            )
            plt.plot(r_raw, f_final_diff_only, "g:", label="Final (Diffusion Only)")
            plt.plot(r_raw, f_final_adv_only, "b-.", label="Final (Advection Only)")
            plt.plot(
                r_raw, f_final_both, "r-", label="Final (Advection + Diffusion)", lw=2
            )
            plt.title("Combined Advection-Diffusion Test")
            plt.xlabel("Radius (pc)")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # --- Assertions ---
        # 1. The peak of the combined solution should be near the advected position.
        assert np.isclose(peak_pos_both, peak_pos_adv_only, rtol=0.1)

        # 2. The combined solution should be wider than the advection-only solution.
        assert width_both > width_adv_only

        # 3. The peak of the combined solution should be lower than the advection-only peak
        #    due to diffusion.
        assert np.max(f_final_both) < np.max(f_final_adv_only)

        # 4. The combined solution should be wider than the initial profile.
        assert width_both > width_initial


class TestAdvectionSource:
    """
    Tests for problems combining advection and a source term.
    """

    def test_constant_velocity_and_source(self, plot_results):
        """
        Validates that a source injects particles that are then carried away
        by a constant velocity field, creating a plume that goes like 1/r**2.
        """
        # Parameters
        num_r, r_end, t_final = 500, 40.0, 2.0
        v_const = 10.0  # pc/Myr

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 200) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r) * su.PSI_P

        # Source term: a spike at r=[0.9, 1.1]
        source_r_min, source_r_max = 0.9, 1.1
        Q_values = np.zeros(num_r)
        source_mask = (r_raw >= source_r_min) & (r_raw <= source_r_max)
        Q_values[source_mask] = 40.0
        Q_qty = Q_values * su.SOURCE_PSI_P

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "advection": {
                "v_centers": np.full(num_r, v_const) * su.VELOCITY,
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "source": {"source": Q_qty},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "advection-source", f_initial
        ).flatten()

        # Plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_final, label="Final Profile")
            plt.axvspan(
                source_r_min,
                source_r_max,
                color="red",
                alpha=0.3,
                label="Source Region",
            )
            plt.title("Advection-Source Test (Constant Velocity)")
            plt.xlabel("Radius (pc)")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # --- Assertions ---
        # 1. The distribution should be zero (or very close) upstream of the source.
        upstream_mask = r_raw < source_r_min
        assert np.allclose(f_final[upstream_mask], 0, atol=1e-9)

        # 2. The distribution should be non-zero downstream of the source.
        downstream_mask = r_raw > source_r_max
        assert np.any(f_final[downstream_mask] > 1e-9)

        # 3. Due to spherical geometry and constant velocity, f should decrease ~1/r^2.
        plume_mask = (r_raw > source_r_max) & (f_final > 0)
        plume_indices = np.where(plume_mask)[0]
        if len(plume_indices) > 10:  # Only check if the plume is well-developed
            mid_point = plume_indices[0] + len(plume_indices) // 2
            slope = (
                np.log(f_final[mid_point + 5]) - np.log(f_final[mid_point - 5])
            ) / (np.log(r_raw[mid_point + 5]) - np.log(r_raw[mid_point - 5]))
            expected_slope = -2
            assert np.isclose(slope, expected_slope, rtol=1e-3)

    def test_variable_velocity_and_source(self, plot_results):
        """
        Validates behavior with a source and a spatially varying velocity field (1/r^2).
        Particles should slow down and radial profile should be constant.
        """
        # Parameters
        num_r, r_end, t_final = 800, 6.0, 10.0

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 1000) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r) * su.PSI_P

        # Velocity field: v ~ 1/r^2, with a cap at small r (protect against division by zero at r=0)
        r_safe = np.maximum(r_raw, 0.1)
        v_field = 1.0 / (r_safe**2)
        v_field[0] = 0.0  # Ensure velocity is zero at the origin
        v_qty = v_field * su.VELOCITY

        # Source term: a spike at r=[0.9, 1.1]
        source_r_min, source_r_max = 0.9, 1.1
        Q_values = np.zeros(num_r)
        source_mask = (r_raw >= source_r_min) & (r_raw <= source_r_max)
        Q_values[source_mask] = 40.0
        Q_qty = Q_values * su.SOURCE_PSI_P

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "advection": {
                "v_centers": v_qty,
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "source": {"source": Q_qty},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "advection-source", f_initial
        ).flatten()

        # Plotting
        if plot_results:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(r_raw, f_final, label="Final Profile", color="C0")
            ax1.axvspan(
                source_r_min,
                source_r_max,
                color="red",
                alpha=0.3,
                label="Source Region",
            )
            ax1.set_xlabel("Radius (pc)")
            ax1.set_ylabel("f(r)", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(r_raw, v_field, "g--", label="Velocity Field")
            ax2.set_ylabel("Velocity (pc/Myr)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2.legend(loc="upper right")

            plt.title("Advection-Source Test (Variable Velocity)")
            fig.tight_layout()
            plt.show()

        # --- Assertions ---
        # 1. The distribution should be zero upstream of the source.
        upstream_mask = r_raw < source_r_min
        assert np.allclose(f_final[upstream_mask], 0, atol=1e-9)

        # 2. The distribution should be non-zero downstream of the source.
        downstream_mask = r_raw > source_r_max
        assert np.any(f_final[downstream_mask] > 1e-9)

        # 3. Due to v decreasing, f should be more or less constant (spherical geometry).
        # We check that the average slope in the plume region is close to zero.
        plume_mask = (r_raw > source_r_max) & (f_final > 0)
        plume_indices = np.where(plume_mask)[0]
        if len(plume_indices) > 10:  # Only check if the plume is well-developed
            mid_point = plume_indices[0] + len(plume_indices) // 2
            avg_slope = (f_final[mid_point + 5] - f_final[mid_point - 5]) / (
                r_raw[mid_point + 5] - r_raw[mid_point - 5]
            )
            assert np.isclose(avg_slope, 0, atol=1e-2)


class TestDiffusionSource:
    """
    Tests for problems combining diffusion and a source term.
    """

    def test_diffusion_source_steady_state(self, plot_results):
        """
        Validates that the solver reaches the correct analytical steady-state
        for a problem with spatially-dependent D(r) and Q(r).
        """
        # Parameters
        num_r, r_end, t_final = 1000, 1.0, 20
        D_0 = 1.0
        Q_0 = 4.0
        eps = 0.01

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r) * su.LENGTH
        t_grid = np.linspace(0, t_final, 100000) * su.TIME
        r_raw = r_grid.to_value(su.LENGTH)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r) * su.PSI_P

        # Spatially-dependent diffusion and source
        D_values = D_0 * (r_raw + eps) ** 2
        Q_values = Q_0 * r_raw
        D_qty = D_values * su.DIFFUSION_COEFFICIENT
        Q_qty = Q_values * su.SOURCE_PSI_P

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "diffusion": {"D_values": D_qty, "psi_end": 0.0 * su.PSI_P},
            "source": {"source": Q_qty},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "diffusion-source", f_initial
        ).flatten()

        # Analytical steady-state solution from DiffValidation4.py
        C1 = 1 - 2 * eps * np.log(eps + 1) - eps**2 / (eps + 1)
        analytical_steady_state = (Q_0 / (4 * D_0)) * (
            C1 - (r_raw - 2 * eps * np.log(eps + r_raw) - eps**2 / (eps + r_raw))
        )
        # Ensure boundary condition is met
        analytical_steady_state[-1] = 0.0

        # Plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_raw, f_final, label="Numerical Final State", lw=2)
            plt.plot(
                r_raw,
                analytical_steady_state,
                "r--",
                label="Analytical Steady State",
                lw=2,
            )
            plt.title("Diffusion-Source Steady State Test")
            plt.xlabel("Radius r")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # --- Assertions ---
        # Check that the final numerical solution is close to the analytical steady state.
        # A tolerance is needed as it's an approximation to a steady state.
        assert np.allclose(f_final, analytical_steady_state, atol=1e-3)
