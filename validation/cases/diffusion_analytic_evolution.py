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


def run_diffusion_simulation(
    r_grid, t_grid, psi_initial, solver_params, sample_count=0
):
    """
    Create and run a diffusion Solver for the provided grids and params.
    Collect sampled snapshots by advancing the solver in chunks to sampled
    timestep indices (mirrors the approach used in other validation scripts).

    Returns the final distribution (numpy array), a list of snapshots and
    the corresponding snapshot times.
    """
    grid = Grid(r_centers=r_grid * su.LENGTH, t_grid=t_grid * su.TIME)
    state = State(psi_p=psi_initial * su.PSI_P, grid=grid, particle=Particle.PROTON)

    solver = Solver(
        state=state,
        problem_type="diffusion",
        operator_params={"diffusion": solver_params},
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

    # snapshots: include initial
    snapshots = [state.psi_p.to_value(su.PSI_P).flatten()]
    times = [t_grid[0]]

    if sample_count > 0 and num_timesteps > 0:
        sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
        sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))
    else:
        sample_indices = np.array([0, num_timesteps], dtype=int)

    current_step = 0
    for next_step in sample_indices[1:]:
        steps_to_advance = int(next_step - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_step
        snapshots.append(solver.state.psi_p.to_value(su.PSI_P).flatten())
        times.append(t_grid[current_step])

    return solver.state.psi_p.to_value(su.PSI_P).flatten(), snapshots, times


def compute_relative_L2(numerical, analytical, mask=None):
    if mask is None:
        mask = np.ones_like(numerical, dtype=bool)
    num = numerical[mask]
    theo = analytical[mask]

    sum_diff_sq = np.sum((num - theo) ** 2)
    sum_theo_sq = np.sum(theo**2)

    if sum_theo_sq == 0:
        return np.sqrt(sum_diff_sq)
    return np.sqrt(sum_diff_sq / sum_theo_sq)


def validation_diffusion_analytic(
    resolutions,
    r_end=1.0,
    t_final=0.1,
    D_const=1.0,
    plot_results=True,
):
    """
    Validation of 1D radial diffusion with a sinc-like initial profile.
    Analytical decay: \psi(r,t) = \psi_initial(r) * exp(-pi^2 * D_const * t).
    Sweep resolutions, measure relative L2 error and save figures to ../figures.
    """
    errors = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running n_r={N}")
        r_grid = np.linspace(0.0, r_end, N)
        dr = r_grid[1] - r_grid[0]

        t_grid = np.linspace(0.0, t_final, 10000)

        # sinc-like initial condition (as in tests)
        psi_initial = (np.pi / 2.0) * np.sinc(r_grid)

        solver_params = {
            "D_values": np.full(N, D_const) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        }

        psi_num, snapshots, snap_times = run_diffusion_simulation(
            r_grid, t_grid, psi_initial, solver_params, sample_count=6
        )

        # analytical solution at t_final
        psi_ana = psi_initial * np.exp(-(np.pi**2) * D_const * t_final)

        relL2 = compute_relative_L2(psi_num, psi_ana)
        errors.append(relL2)
        dxs.append(dr)

        print(f"  dx={dr:.4e}, steps={len(t_grid) - 1}, relL2={relL2:.4e}")

        all_results.append(
            {
                "N": N,
                "r_grid": r_grid,
                "psi_initial": psi_initial,
                "psi_num": psi_num,
                "psi_ana": psi_ana,
                "relL2": relL2,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        dxs = np.array(dxs)
        res = np.array(resolutions)
        errors = np.array(errors)

        # compute common y-limits across all results
        ymin = np.inf
        ymax = -np.inf
        for rec in all_results:
            ymin = min(ymin, np.min(rec["psi_initial"]))
            ymax = max(ymax, np.max(rec["psi_initial"]))
            ymin = min(ymin, np.min(rec["psi_num"]))
            ymax = max(ymax, np.max(rec["psi_num"]))

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 0.0, 1.0
        padding = 0.05 * (ymax - ymin) if (ymax - ymin) > 0 else 0.1
        ylims = (max(0.0, ymin - padding), ymax + padding)

        # convergence plot
        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        plt.loglog(
            res, errors, label=r"Relative error: $\mathcal{E}_{L_2}$", **quant_style
        )
        plt.xlabel("Number of radial cells: $n_r$")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(which="both")
        conv_fig = plt.gcf()
        plt.show()

        # per-resolution plots with consistent y-limits
        last_fig = None
        last_N = None
        for rec in all_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            psi_initial = rec["psi_initial"]
            psi_num = rec["psi_num"]
            psi_ana = rec["psi_ana"]
            snapshots = rec.get("snapshots", [psi_num])
            snap_times = rec.get("snap_times", [t_final])

            fig = plt.figure(figsize=(6, 4))
            # plot snapshots with varying opacity
            for i, (s, t) in enumerate(zip(snapshots, snap_times)):
                is_initial = i == 0
                is_final = i == len(snapshots) - 1
                style = get_numerical_style(
                    is_initial=is_initial,
                    is_final=is_final,
                    step_idx=i,
                    total_steps=len(snapshots),
                )

                label = (
                    "Initial"
                    if is_initial
                    else ("Numerical (final)" if is_final else None)
                )
                plt.plot(r_grid, s, label=label, **style)

            # analytical solution (final)
            ana_style = get_analytical_style()
            plt.plot(r_grid, psi_ana, label="Analytical (final)", **ana_style)

            add_time_colorbar(fig, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1])

            plt.xlim(0, r_end)
            plt.ylim(ylims)
            plt.xlabel(r"Radial coordinate: $r$ (a. u.)")
            plt.ylabel(r"Solution: $\psi(t,r)$ (a. u.)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            last_fig = fig
            last_N = N

        # save figures to ../figures
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir, f"diffusion_analytic_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")
            conv_path_pdf = os.path.join(out_dir, "diffusion_analytic_convergence.pdf")
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    validation_diffusion_analytic(
        resolutions,
        r_end=1.0,
        t_final=0.3,
        D_const=1.0,
        plot_results=True,
    )
