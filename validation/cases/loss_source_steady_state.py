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


def run_loss_simulation(p_grid, t_grid, psi_initial, operator_params, sample_count=0):
    """
    Run a loss+source simulation on the momentum axis (p_grid) and collect
    sampled snapshots. Returns final psi, snapshots list, snapshot times, and solver.
    """
    grid = Grid(p_centers=p_grid * su.MOMENTUM, t_grid=t_grid * su.TIME)
    state = State(psi_p=psi_initial * su.PSI_P, grid=grid, particle=Particle.PROTON)

    # decide problem type
    if "source" in operator_params:
        problem_type = "loss-source"
    else:
        problem_type = "loss"

    solver = Solver(
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={"loss": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

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


def analytical_steady_state_loss(p_grid, Q0, b0, alpha, beta, p0, p_end):
    """
    Analytical steady-state for the loss+source model used in LossValidation1.py:
    \psi(p) = [Q0 * p0 / (1-alpha) / b0] * ( (p_end/p0)^(1-alpha) - (p/p0)^(1-alpha) ) * (p/p0)^(-beta)
    """
    pref = Q0 * p0 / (1.0 - alpha) / b0
    term = (p_end / p0) ** (1.0 - alpha) - (p_grid / p0) ** (1.0 - alpha)
    psi = pref * term * (p_grid / p0) ** (-beta)
    return psi


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


def validation_loss_source_steady_state(
    n_r=300,
    p_min=1.0,
    p_max=1000.0,
    p0=1.0,
    t_min=0.1,
    t_max=3.0,
    n_times=10,
    Q0=1.0,
    b0=1.0,
    alpha=4.0,
    beta=2.0,
    t_steps=2000,
    sample_count=6,
    plot_results=True,
):
    """
    Validation of a loss+source steady-state problem.
    Sweeps final times from t_min to t_max (n_times points), runs the solver, and
    compares the numerical final state with the analytical steady state.
    Produces temporal-evolution plots (sampled snapshots) and a residuals-vs-time plot.
    """
    p_grid = np.logspace(np.log10(p_min), np.log10(p_max), n_r)

    psi_initial = np.zeros(n_r)

    # define source and loss rates following LossValidation1.py
    Q_values = Q0 * (p_grid / p0) ** (-alpha)
    P_dot = -b0 * (p_grid / p0) ** beta  # P_dot is negative for losses

    # pack operator params
    loss_params = {
        "P_dot": P_dot * su.MOMENTUM_LOSS_RATE,
        "limiter": "minmod",
        "cfl": 0.2,
        "order": 2,
        "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
        "adiabatic_losses": False,
    }
    operator_params = {
        "loss": loss_params,
        "source": {"source": Q_values * su.SOURCE_PSI_P},
    }

    # analytical steady state (same for all times)
    ana = analytical_steady_state_loss(p_grid, Q0, b0, alpha, beta, p0, p_max)

    t_finals = np.linspace(t_min, t_max, n_times)

    residuals = []
    times_list = []
    all_results = []

    print(
        f"Running loss+source steady-state validation on p-grid n_r={n_r}, p in [{p_min},{p_max}]"
    )

    for t_final in t_finals:
        print(f"Running t_final={t_final:.4g}")
        t_grid = np.linspace(0.0, t_final, t_steps)

        psi_num, snapshots, snap_times, solver = run_loss_simulation(
            p_grid, t_grid, psi_initial, operator_params, sample_count=sample_count
        )

        relL2 = compute_relative_L2(psi_num, ana)
        residuals.append(relL2)
        times_list.append(t_final)

        print(f"  t_final={t_final:.4e}, relL2={relL2:.4e}")

        all_results.append(
            {
                "t_final": t_final,
                "p_grid": p_grid,
                "psi_num": psi_num,
                "ana": ana,
                "relL2": relL2,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        times_arr = np.array(times_list)
        residuals = np.array(residuals)

        # residuals vs time
        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        plt.semilogy(
            times_arr,
            residuals,
            label=r"Relative error: $\mathcal{E}_{L_2}$",
            **quant_style,
        )
        plt.xlabel(r"Final simulation time: $t_\mathrm{end}$ (a. u.)")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(True, which="both")
        conv_fig = plt.gcf()
        plt.show()
        # save residuals/convergence figure
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            conv_path = os.path.join(out_dir, "loss_source_residuals.pdf")
            conv_fig.savefig(conv_path, dpi=200, bbox_inches="tight")
            print(f"Saved residuals (convergence) figure to: {conv_path}")
        except Exception as e:
            print(f"Warning: could not save residuals figure: {e}")

        # compute y-limits for p^5 * \psi plotting
        ymin = np.inf
        ymax = -np.inf
        for rec in all_results:
            p = rec["p_grid"]
            psinum = rec["psi_num"]
            ana_ = rec["ana"]
            vals = (p**5) * np.maximum(psinum, 1e-300)
            vals_ana = (p**5) * np.maximum(ana_, 1e-300)
            ymin = min(ymin, np.min(vals))
            ymax = max(ymax, np.max(vals_ana))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 1e-20, 1e20

        for rec in all_results:
            t_final = rec["t_final"]
            p = rec["p_grid"]
            snapshots = rec.get("snapshots", [])
            snap_times = rec.get("snap_times", [])
            ana_ = rec["ana"]

            fig = plt.figure(figsize=(9, 6))
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
                plt.loglog(p, (p**5) * np.maximum(s, 1e-300), label=label, **style)

            ana_style = get_analytical_style()
            plt.loglog(
                p,
                (p**5) * np.maximum(ana_, 1e-300),
                label="Analytical SS",
                **ana_style,
            )

            add_time_colorbar(fig, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1])

            plt.xlim(p[0], p[-1])
            plt.ylim(1e-2, 1)
            plt.xlabel(r"Momentum coordinate: $p$ (a. u.)")
            plt.ylabel(r"Spectrum: $p^5 \psi(t,p)$ (a. u.)")
            plt.legend(loc="lower left")
            plt.grid(False)
            plt.tight_layout()
            # store the figure object so we can save exactly the same one later
            try:
                rec["fig"] = fig
            except Exception:
                pass
            plt.show()

        # save only the figure corresponding to t_end ~= 0.4
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)

            target_t = 0.4
            # pick recorded run closest to target_t
            rec_target = min(all_results, key=lambda r: abs(r["t_final"] - target_t))

            # save the exact figure that was shown for the chosen run, if available
            fig_target = rec_target.get("fig")
            if fig_target is not None:
                try:
                    target_path_pdf = os.path.join(
                        out_dir, "loss_source_steady_tend0p4.pdf"
                    )
                    fig_target.savefig(target_path_pdf, dpi=200, bbox_inches="tight")
                    print(f"Saved t_end=0.4 simulation figure to: {target_path_pdf}")
                except Exception as e:
                    print(f"Warning: could not save target figure from stored fig: {e}")
            else:
                try:
                    fig_target = plt.figure(figsize=(9, 6))
                    p = rec_target["p_grid"]
                    snapshots = rec_target.get("snapshots", [])
                    snap_times = rec_target.get("snap_times", [])
                    ana_ = rec_target["ana"]
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
                        plt.loglog(
                            p, (p**5) * np.maximum(s, 1e-300), label=label, **style
                        )

                    ana_style = get_analytical_style()
                    plt.loglog(
                        p,
                        (p**5) * np.maximum(ana_, 1e-300),
                        label="Analytical SS",
                        **ana_style,
                    )

                    add_time_colorbar(
                        fig_target, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1]
                    )

                    plt.xlim(p[0], p[-1])
                    plt.ylim(1e-2, 1)
                    plt.xlabel(r"Momentum coordinate: $p$ (a. u.)")
                    plt.ylabel(r"Spectrum: $p^5 \psi(t,p)$ (a. u.)")
                    plt.legend(loc="lower left")
                    plt.grid(False)
                    plt.tight_layout()
                    plt.show()
                    target_path_pdf = os.path.join(
                        out_dir, "loss_source_steady_tend0p4.pdf"
                    )
                    fig_target.savefig(target_path_pdf, dpi=200, bbox_inches="tight")
                    print(f"Saved t_end=0.4 simulation figure to: {target_path_pdf}")
                except Exception as e:
                    print(f"Error saving fallback target plot: {e}")
        except Exception as e:
            print(f"Warning: could not save loss-source figures: {e}")


if __name__ == "__main__":
    validation_loss_source_steady_state(
        n_r=1024,
        p_min=1.0,
        p_max=1000.0,
        p0=1.0,
        t_min=0.1,
        t_max=0.8,
        n_times=8,
        Q0=1.0,
        b0=1.0,
        alpha=4.0,
        beta=2.0,
        t_steps=10000,
        sample_count=8,
        plot_results=True,
    )
