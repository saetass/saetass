import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Apply unified plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import (
    add_time_colorbar,
    apply_plot_style,
    get_analytical_style,
    get_numerical_style,
    get_quantitative_style,
)

from saetass import Grid, Particle, Solver, State
from saetass import units as su

apply_plot_style()


def run_advection_simulation(
    r_grid, t_grid, psi_initial, solver_params, source_params=None, sample_count=0
):
    """
    Create and run an advection (+ optional source) Solver for the provided grids and params.
    Collect snapshots by advancing the solver in chunks to a set of sampled timesteps.

    Returns the final distribution (numpy array), a list of snapshots and their times.
    """
    grid = Grid(r_centers=r_grid * su.LENGTH, t_grid=t_grid * su.TIME)
    state = State(psi_p=psi_initial * su.PSI_P, grid=grid, particle=Particle.PROTON)

    operator_params = {"advection": solver_params}
    if source_params is not None:
        operator_params["source"] = source_params
        problem_type = "advection-source"
    else:
        problem_type = "advection"

    solver = Solver(
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={"advection": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

    # Prepare snapshots: always include initial state at index 0
    snapshots = [state.psi_p.to_value(su.PSI_P).flatten()]
    times = [t_grid[0]]

    # Determine which timestep indices to sample (include 0 and final)
    if sample_count > 0 and num_timesteps > 0:
        sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
        sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))
    else:
        sample_indices = np.array([0, num_timesteps], dtype=int)

    # Advance solver in chunks to reach each sampled timestep (efficient stepping)
    current_step = 0
    for next_step in sample_indices[1:]:
        steps_to_advance = int(next_step - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_step
        snapshots.append(solver.state.psi_p.to_value(su.PSI_P).flatten())
        times.append(t_grid[current_step])

    # Return final flattened array plus snapshots and times
    return solver.state.psi_p.to_value(su.PSI_P).flatten(), snapshots, times


def compute_plume_slope(r_grid, psi_values, source_r_max, min_points=10):
    """
    Compute slope of log(\psi) vs log(r) in the plume region (r > source_r_max).
    Returns slope and the mask used.
    """
    # Select only points between 1.5*source_r_max and r_grid = 6
    mask = (r_grid > 10 * source_r_max) & (r_grid < 20)
    x = np.log(r_grid[mask])
    y = np.log(psi_values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, mask


def validation_pointlike_source(
    resolutions,
    r_end=40.0,
    t_final=2.0,
    v_const=10.0,
    source_r_min=0.9,
    source_r_max=1.1,
    source_strength=40.0,
    plot_results=True,
):
    """
    Validation of point-like source advected with constant speed. The steady
    downstream profile should follow ~1/r^2. We run a sweep over spatial
    resolutions and measure the fitted slope of log(\psi) vs log(r) in the plume.
    """
    slopes = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running n_r={N}")
        r_grid = np.linspace(0.0, r_end, N)
        dr = r_grid[1] - r_grid[0]

        # Choose a sufficiently fine time sampling (many small steps)
        t_grid = np.linspace(0.0, t_final, 10000)

        # initial condition: zero everywhere
        psi_initial = np.zeros(N)

        # Source: spike between source_r_min and source_r_max
        Q_values = np.zeros(N)
        source_mask = (r_grid >= source_r_min) & (r_grid <= source_r_max)
        Q_values[source_mask] = source_strength

        # solver params
        v_field = np.full(N, v_const)
        solver_params = {
            "v_centers": v_field * su.VELOCITY,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        }

        # capture intermediate snapshots for time evolution (include initial and final)
        psi_num, snapshots, snap_times = run_advection_simulation(
            r_grid,
            t_grid,
            psi_initial,
            solver_params,
            source_params={"source": Q_values * su.SOURCE_PSI_P},
            sample_count=7,
        )

        # compute slope in plume region
        slope, mask = compute_plume_slope(r_grid, psi_num, source_r_max)
        slopes.append(slope)
        dxs.append(dr)

        print(f"  dx={dr:.4e}, slope={slope:.4e}")

        # store results and snapshots for plotting later so we can use same y-limits across runs
        all_results.append(
            {
                "N": N,
                "r_grid": r_grid,
                "psi_initial": psi_initial,
                "psi_num": psi_num,
                "slope": slope,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    # convergence plot and per-resolution plots with common y-limits
    if plot_results:
        slopes = np.array(slopes)
        dxs = np.array(dxs)
        res = np.array(resolutions)

        # plot each simulation with same y-limits
        last_fig = None
        last_N = None
        for rec in all_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            snapshots = rec["snapshots"]
            snap_times = rec["snap_times"]
            slope = rec["slope"]

            # Only produce log-log plots showing time evolution of the profile
            mask_pos_all = [(r_grid > 0) & (s > 0) for s in snapshots]
            fig_log = plt.figure(figsize=(6, 4))
            for idx, (s, t, mask_pos) in enumerate(
                zip(snapshots, snap_times, mask_pos_all)
            ):
                if np.any(mask_pos):
                    is_initial = idx == 0
                    is_final = idx == len(snapshots) - 1
                    style = get_numerical_style(
                        is_initial=is_initial,
                        is_final=is_final,
                        step_idx=idx,
                        total_steps=len(snapshots),
                    )
                    label = (
                        "Initial"
                        if is_initial
                        else ("Numerical (final)" if is_final else None)
                    )
                    plt.loglog(r_grid[mask_pos], s[mask_pos], label=label, **style)

            # add reference r^-2 scaled to the mid-plume of the final snapshot if available
            final = snapshots[-1]
            plume_idx = np.where((r_grid > source_r_max) & (final > 0))[0]
            if len(plume_idx) > 0:
                mid = plume_idx[len(plume_idx) // 3]
                scale = final[mid] * r_grid[mid] ** 2
                ref = scale * r_grid ** (-2)
                ana_style = get_analytical_style()
                plt.loglog(
                    r_grid[r_grid > 0],
                    ref[r_grid > 0],
                    label=r"$\propto r^{-2}$",
                    **ana_style,
                )

            add_time_colorbar(
                fig_log, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1]
            )

            plt.xlim(left=max(r_grid[1], 1e-6), right=r_end)
            # common y-limits computed earlier; use them in log space if positive
            # find positive y-bounds from all snapshots
            pos_vals = np.concatenate([s[s > 0] for s in snapshots])
            if pos_vals.size:
                y_min_log = pos_vals.min()
                y_max_log = pos_vals.max()
                plt.ylim(y_min_log * 0.8, y_max_log * 1.2)

            plt.xlabel(r"Radial coordinate: $r$ (a. u.)")
            plt.ylabel(r"Solution: $\psi(t,r)$ (a. u.)")
            plt.ylim([1e-3, 1e1])
            plt.xlim([1.0, r_end])
            plt.grid(False)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
            last_log_fig = fig_log
            last_fig = None
            last_N = N

        # convergence plot: |slope + 2| vs N using loglog
        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        slope_errors = np.abs(np.array(slopes) + 2.0)
        plt.loglog(res, slope_errors, label=r"$|\alpha - (-2)|$", **quant_style)
        plt.xlabel("Number of radial cells: $n_r$")
        plt.ylabel(r"Error: $|\alpha - \alpha_{\mathrm{theo}}|$")

        plt.grid(which="both")
        conv_fig = plt.gcf()
        plt.show()

        # save figures: last simulation and convergence (PDF)
        try:
            # Save figures to the parent `validation/figures/` directory
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir, f"advection_pointlike_source_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")

            # save last log-log figure if present
            if "last_log_fig" in locals():
                last_log_path = os.path.join(
                    out_dir, f"advection_pointlike_source_last_N{last_N}_loglog.pdf"
                )
                try:
                    last_log_fig.savefig(last_log_path, dpi=200, bbox_inches="tight")
                    print(f"Saved last log-log figure to: {last_log_path}")
                except Exception:
                    pass

            conv_path_pdf = os.path.join(
                out_dir, "advection_pointlike_source_convergence.pdf"
            )
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096]
    validation_pointlike_source(
        resolutions,
        r_end=40.0,
        t_final=3.0,
        v_const=10.0,
        source_r_min=0.9,
        source_r_max=1.1,
        source_strength=400.0,
        plot_results=True,
    )
