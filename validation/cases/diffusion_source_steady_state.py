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
    r_grid, t_grid, psi_initial, operator_params, sample_count=0
):
    grid = Grid(r_centers=r_grid * su.LENGTH, t_grid=t_grid * su.TIME)
    state = State(psi_p=psi_initial * su.PSI_P, grid=grid, particle=Particle.PROTON)

    # Decide problem_type depending on whether a source operator is present
    if "source" in operator_params:
        problem_type = "diffusion-source"
    else:
        problem_type = "diffusion"

    solver = Solver(
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

    # sampled snapshot collection (include initial and final)
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

    return solver.state.psi_p.to_value(su.PSI_P).flatten(), snapshots, times, solver


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


def analytical_steady_state(r_grid, D_0, Q_0, eps):
    # Analytical steady-state used in tests (see test_multiple_operator.py)
    C1 = 1 - 2 * eps * np.log(eps + 1) - eps**2 / (eps + 1)
    term = r_grid - 2 * eps * np.log(eps + r_grid) - eps**2 / (eps + r_grid)
    psi = (Q_0 / (4 * D_0)) * (C1 - term)
    # enforce boundary condition at last point
    psi[-1] = 0.0
    return psi


def validation_diffusion_source_steady_state(
    n_r=1024,
    r_end=1.0,
    t_min=0.1,
    t_max=3.0,
    n_times=10,
    D_0=1.0,
    Q_0=4.0,
    eps=0.01,
    t_steps=10000,
    sample_count=5,
    plot_results=True,
):
    """
    Validation for the diffusion+source steady-state problem used in tests.
    Runs several resolutions, computes relative L2 between numerical final
    state and analytical steady-state, and saves per-resolution and convergence
    plots to ../figures.
    """
    errors = []
    times_list = []
    all_results = []

    # build list of final times to iterate over
    t_finals = np.linspace(t_min, t_max, n_times)

    print(
        f"Running steady-state diffusion-source validation for n_r={n_r}, t in [{t_min}, {t_max}] ({n_times} points)"
    )

    r_grid = np.linspace(0.0, r_end, n_r)

    psi_initial = np.zeros(n_r)

    D_values = D_0 * (r_grid + eps) ** 2
    Q_values = Q_0 * r_grid

    solver_params = {
        "D_values": D_values * su.DIFFUSION_COEFFICIENT,
        "psi_end": 0.0 * su.PSI_P,
    }
    operator_params = {
        "diffusion": solver_params,
        "source": {"source": Q_values * su.SOURCE_PSI_P},
    }

    ana = analytical_steady_state(r_grid, D_0, Q_0, eps)

    for t_final in t_finals:
        print(f"Running t_final={t_final:.4g}")
        t_grid = np.linspace(0.0, t_final, t_steps)

        psi_num, snapshots, snap_times, solver = run_diffusion_simulation(
            r_grid, t_grid, psi_initial, operator_params, sample_count=sample_count
        )

        relL2 = compute_relative_L2(psi_num, ana)
        errors.append(relL2)
        times_list.append(t_final)

        print(f"  t_final={t_final:.4e}, relL2={relL2:.4e}")

        all_results.append(
            {
                "t_final": t_final,
                "r_grid": r_grid,
                "psi_num": psi_num,
                "ana": ana,
                "relL2": relL2,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        times_arr = np.array(times_list)
        errors = np.array(errors)

        # residuals vs time plot (how close to steady state as t increases)
        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        plt.semilogy(
            times_arr,
            errors,
            label=r"Relative error: $\mathcal{E}_{L_2}$",
            **quant_style,
        )
        plt.xlabel(r"Final simulation time: $t_\mathrm{end}$ (a. u.)")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(True, which="both")
        conv_fig = plt.gcf()
        plt.show()

        # consistent y-limits
        ymin = np.inf
        ymax = -np.inf
        for rec in all_results:
            ymin = min(ymin, np.min(rec["ana"]))
            ymax = max(ymax, np.max(rec["ana"]))
            ymin = min(ymin, np.min(rec["psi_num"]))
            ymax = max(ymax, np.max(rec["psi_num"]))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 0.0, 1.0
        padding = 0.05 * (ymax - ymin) if (ymax - ymin) > 0 else 0.1
        ylims = (max(0.0, ymin - padding), ymax + padding)

        last_fig = None
        for rec in all_results:
            t_final = rec["t_final"]
            r_grid = rec["r_grid"]
            snapshots = rec.get("snapshots", [])
            snap_times = rec.get("snap_times", [])
            ana = rec["ana"]

            fig = plt.figure(figsize=(6, 4))
            for idx, (s, t) in enumerate(zip(snapshots, snap_times)):
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
                plt.plot(r_grid, s, label=label, **style)

            # overlay analytical steady state (final)
            ana_style = get_analytical_style()
            plt.plot(r_grid, ana, label="Analytical SS", **ana_style)

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
            last_t = t_final

        # save figures to ../figures
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            # instead of saving the last-run figure, save the figure for t_end=1.3
            target_t = 1.3
            # find the recorded run closest to target_t
            rec_target = min(all_results, key=lambda r: abs(r["t_final"] - target_t))
            try:
                fig_target = plt.figure(figsize=(7.5, 5))
                r_grid = rec_target["r_grid"]
                snapshots = rec_target.get("snapshots", [])
                snap_times = rec_target.get("snap_times", [])
                ana = rec_target["ana"]
                for idx, (s, t) in enumerate(zip(snapshots, snap_times)):
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
                    plt.plot(r_grid, s, label=label, **style)

                ana_style = get_analytical_style()
                plt.plot(r_grid, ana, label="Analytical SS", **ana_style)

                add_time_colorbar(
                    fig_target, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1]
                )

                plt.xlim(0, r_end)
                plt.ylim(ylims)
                plt.xlabel(r"Radial coordinate: $r$ (a. u.)")
                plt.ylabel(r"Solution: $\psi(t,r)$ (a. u.)")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()
                target_path_pdf = os.path.join(
                    out_dir, f"diffusion_source_steady_t{target_t:.3g}.pdf"
                )
                fig_target.savefig(target_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved t_end={target_t} simulation figure to: {target_path_pdf}")
            except Exception as e:
                print(f"Error saving target plot: {e}")
                # fallback: save last_fig if target plot failed
                if last_fig is not None:
                    last_path_pdf = os.path.join(
                        out_dir, f"diffusion_source_steady_last_t{last_t:.3g}.pdf"
                    )
                    last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                    print(f"Saved last simulation figure to: {last_path_pdf}")
            conv_path_pdf = os.path.join(
                out_dir, "diffusion_source_steady_residuals.pdf"
            )
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved residuals vs time figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    # iterate final times from 0.1 to 2.0 (10 points)
    validation_diffusion_source_steady_state(
        n_r=8192,
        r_end=1.0,
        t_min=0.1,
        t_max=1.7,
        n_times=9,
        D_0=1.0,
        Q_0=4.0,
        eps=0.1,
        t_steps=10000,
        sample_count=7,
        plot_results=True,
    )
