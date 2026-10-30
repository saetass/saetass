import os
import sys

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# Apply standard validation plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.colors as mcolors
from plot_style import (
    apply_plot_style,
    get_quantitative_style,
)

from saetass import Grid, Particle, Solver, State
from saetass import units as su
from saetass.cli.palette import SAETASS_GREEN
from saetass.utils.bubble_profiles import BubbleProfileCalculator


def get_alpha_cmap(hex_color):
    """Create a colormap that builds opacity from 0.2 to 1.0 of the given color."""
    color_rgba = mcolors.to_rgba(hex_color)
    color_low_alpha = (color_rgba[0], color_rgba[1], color_rgba[2], 0.2)
    color_high_alpha = (color_rgba[0], color_rgba[1], color_rgba[2], 1.0)
    return mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [color_low_alpha, color_high_alpha]
    )


apply_plot_style()


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


def run_convergence_sim(factor, base_points=400, E_k=10 * u.GeV):
    """
    Run a simulation for a specific domain expansion factor.
    The spatial node density is kept constant, based on base_points inside 1 * R_b.
    """
    # 1. Dummy calculation to get reference R_b
    dummy_calc = BubbleProfileCalculator(
        r_grid=np.array([0, 1]) * u.pc,
        model="Morlino21",
        L_wind=2e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=20 * const.m_p / u.cm**3,
        t_b=3 * u.Myr,
    )
    R_b = dummy_calc.R_b
    R_TS = dummy_calc.R_TS

    # Calculate expanded domain and proportional number of points
    r_end = factor * R_b
    num_points = int(base_points * factor)

    print(
        f"Domain: r_end = {r_end.to('pc'):.1f} ({factor:.1f}xR_B), nodes = {num_points}"
    )

    r_grid = np.linspace(0.0, r_end.to("pc").value, num_points) * u.pc

    # 2. Instantiate actual physical setup
    calculator = BubbleProfileCalculator(
        r_grid=r_grid,
        model="Morlino21",
        L_wind=2e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=20 * const.m_p / u.cm**3,
        t_b=3 * u.Myr,
    )

    setup = calculator.get_all_profiles(
        E_k, eta_B=0.1, eta_inj=0.1, E_max=1 * u.PeV, diffusion_model="kraichnan"
    )

    r = setup["r_grid"].to("pc").value
    t_end = 3.0 * calculator.kwargs["t_b"].to("Myr").value

    # 3. Solver Setup
    num_timesteps = 50000
    t_grid = np.linspace(0, t_end, num_timesteps)
    psi_values = np.zeros(len(r))

    grid = Grid(r_centers=setup["r_grid"], t_grid=t_grid * u.Myr)

    op_params = {
        "advection": {
            "v_centers": setup["v_field"],
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        },
        "diffusion": {
            "D_values": setup["D_values"],
            "psi_end": 0.0 * su.PSI_P,
        },
        "source": {"source": setup["Q"]},
    }

    solver = Solver(
        state=State(psi_p=psi_values * su.PSI_P, grid=grid, particle=Particle.PROTON),
        problem_type="advection-source-diffusion",
        operator_params=op_params,
        substeps={"advection": 1, "diffusion": 1, "source": 1},
    )

    # 4. Execute simulation step
    solver.step(len(t_grid) - 1)

    psi_final = solver.state.psi_p.to_value(su.PSI_P)

    # Normalize with respect to forward shock interface (Termination shock)
    ts_idx = np.where(r >= R_TS.to("pc").value)[0][0] + 5
    ts_level = psi_final[ts_idx] if psi_final[ts_idx] > 0 else 1.0
    psi_normalized = psi_final / ts_level

    # 5. Theoretical Configuration
    psi_theoretical_raw = calculator.compute_analytical_CR_profile(
        D_values=setup["D_values"],
        f_gal=0.0,
        f_TS=1.0,
    )

    # Return everything needed for plotting
    return (
        r,
        psi_normalized,
        psi_theoretical_raw,
        R_TS.to("pc").value,
        R_b.to("pc").value,
    )


def run_validation():
    factors = [1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    results = []

    # 1. Run computations
    print("Testing Domain Size effect on Distribution Tail...")
    for factor in factors:
        r, psi_num, psi_theo, R_TS, R_b = run_convergence_sim(factor, base_points=500)

        # Calculate relative error in the ISM tail (r >= R_b)
        mask = r >= R_b
        relL2 = compute_relative_L2(psi_num, psi_theo, mask=mask)

        results.append(
            {
                "factor": factor,
                "r": r,
                "psi_num": psi_num,
                "psi_theo": psi_theo,
                "R_TS": R_TS,
                "R_b": R_b,
                "relL2": relL2,
            }
        )

    # 2. Generate and Save Solution Distributions plot
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    min_factor_res = results[0]
    min_domain = min_factor_res["r"][-1] * 1.05

    factors_array = np.array([res["factor"] for res in results])

    # Vertical lines
    last_res = results[-1]
    ax1.axvline(
        last_res["R_TS"],
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=r"$R_\mathrm{TS}$",
    )
    ax1.axvline(
        last_res["R_b"],
        color="gray",
        linestyle="-.",
        linewidth=1.5,
        label=r"$R_\mathrm{B}$",
    )

    cmap = get_alpha_cmap(SAETASS_GREEN)
    norm = plt.Normalize(vmin=factors_array.min(), vmax=factors_array.max())

    for i, res in enumerate(results):
        factor = res["factor"]
        is_final = i == len(results) - 1
        label = "Numerical" if is_final else None

        color = cmap(norm(factor))
        ls = "-" if is_final else "-"
        lw = 4.0 if is_final else 4.0

        ax1.semilogy(
            res["r"],
            res["psi_num"],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=label,
        )

    ax1.semilogy(
        results[-1]["r"], results[-1]["psi_theo"], "k--", lw=3, label="Steady state"
    )

    ax1.set_xlim(0, min_domain)
    ax1.set_ylim(1e-4, 2)
    ax1.set_xlabel(r"Radial coordinate: $r$ (pc)")
    ax1.set_ylabel(r"Norm. dens.: $\psi(t,r)/\psi_\mathrm{TS}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower center")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label(r"Domain factor: $r_\mathrm{end} / R_\mathrm{B}$")

    plt.tight_layout()

    # Save the distribution comparison plot
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    dist_path = os.path.join(out_dir, "domain_size_solutions.pdf")
    fig1.savefig(dist_path, dpi=200, bbox_inches="tight")
    print(f"Saved distribution profiles comparison to: {dist_path}")

    # 3. Generate Convergence Error plot
    factors_array = np.array([res["factor"] for res in results])
    relL2_array = np.array([res["relL2"] for res in results])

    fig2 = plt.figure(figsize=(6, 4))
    quant_style = get_quantitative_style()
    plt.semilogy(
        factors_array,
        relL2_array,
        label=r"Relative error: $\mathcal{E}_{L_2}$",
        **quant_style,
    )
    plt.xlabel(r"Domain factor: $r_\mathrm{end} / R_\mathrm{B}$")
    plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
    plt.grid(True, which="both")

    plt.tight_layout()

    # Save convergence figure
    conv_path = os.path.join(out_dir, "domain_size_convergence.pdf")
    fig2.savefig(conv_path, dpi=200, bbox_inches="tight")
    print(f"Saved error convergence figure to: {conv_path}")

    plt.show()


if __name__ == "__main__":
    run_validation()
