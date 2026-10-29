r"""
Validation of diffusive shock acceleration at the termination shock of a stellar wind.

Particles injected at a shock gain momentum each time they diffuse across it, where the flow is
compressed, and are advected away downstream. In steady state this produces the power law of
diffusive shock acceleration (:cite:ct:`Drury1983`), :math:`f \propto p^{-s}` with
:math:`s = 3 u_1 / (u_1 - u_2) = 3 r_c / (r_c - 1)` for the compression ratio
:math:`r_c = u_1 / u_2`: :math:`f \propto p^{-4}`, or :math:`\psi_p = 4 \pi p^2 f \propto p^{-2}`,
at a strong shock (:math:`r_c = 4`).

SAETASS has no shock operator: the acceleration comes out of the adiabatic term,
:math:`\dot{p} = -(p/3) \nabla \cdot v`, whose divergence concentrates the whole velocity jump in the
cells next to the shock, combined with advection and diffusion. This script uses the geometry of the
termination shock of :cite:ct:`Morlino2021`: a wind of constant speed :math:`u_1` inside the shock
radius :math:`R`, and :math:`u = u_2 (R / r)^2` outside, which keeps the shocked wind incompressible.
The diffusion coefficient :math:`D` is constant, with :math:`\varepsilon = D / (u_1 R) \ll 1`.

**Theory.** Upstream, particles reach a distance :math:`D / u_1` from the shock (the precursor),
:math:`f = f_s(p)\, e^{-u_1 (R - r) / D}`, as in :cite:ct:`Morlino2021`. Downstream, :math:`f = f_s`.
Integrating the steady transport equation times :math:`r^2` over the precursor and across the shock,
the spherical wind adds two corrections of order :math:`\varepsilon` to the planar result: the wind
expands upstream, :math:`\nabla \cdot v = 2 u_1 / r`, which cools the particles in the precursor, and
the precursor is a thin shell whose volume shrinks inward. To first order in :math:`\varepsilon`,

.. math::

    s = \frac{3 u_1 (1 - 2 \varepsilon)}{u_1 - u_2 - 2 \varepsilon u_1},

which tends to the planar :math:`3 r_c / (r_c - 1)` for :math:`\varepsilon \to 0`.

**Checks.** The script runs to steady state and compares, at the shock:

1. the slope of the spectrum for a strong shock at increasing resolution (cells per diffusion length
   :math:`D / u_1`, with time steps of a fixed Courant number at the shock), which converges to the
   theory at first order, as expected at a discontinuity;
2. the slope for several compression ratios, against the planar and spherical predictions. The
   first-order formula needs :math:`2 \varepsilon u_1 / (u_1 - u_2) \ll 1`, which is 0.05 for
   :math:`r_c = 4` but 0.08 for :math:`r_c = 2`, so its neglected terms grow for weaker shocks;
3. the radial profile around the shock against the precursor of :cite:ct:`Morlino2021`.

It prints the slopes, their differences from the theory and the order of convergence fitted to the
resolutions, which grow by factors of about :math:`\sqrt{2}`. The runs take about ten minutes.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import (
    SAETASS_BLUE,
    SAETASS_ORANGE,
    SAETASS_YELLOW,
    apply_plot_style,
    get_analytical_style,
    get_numerical_style,
    get_quantitative_style,
)

from saetass import Grid, Particle, Solver, State
from saetass import units as su
from saetass.cli import set_quiet

apply_plot_style()
set_quiet(True)

R_SHOCK = 1.0  # pc
U_WIND = 1.0  # pc/Myr, about 980 km/s
EPSILON = 0.02  # D / (u1 R)
P_INJECTION_INDEX = 4  # injection at the fifth momentum, 1.8 GeV/c
FIT_RANGE = (10.0, 300.0)  # GeV/c, where the slope is measured


def slope_planar(compression):
    """Slope of psi_p = 4 pi p^2 f at a planar shock, 2 - 3 r_c / (r_c - 1)."""
    return 2.0 - 3.0 * compression / (compression - 1.0)


def slope_spherical(compression, epsilon=EPSILON):
    """Slope of psi_p at the termination shock, to first order in epsilon = D / (u1 R)."""
    u1, u2 = 1.0, 1.0 / compression
    return 2.0 - 3.0 * u1 * (1.0 - 2.0 * epsilon) / (u1 - u2 - 2.0 * epsilon * u1)


def radial_grid(length, cells, growth=1.05, coarsest=0.02):
    """
    Radial cell centers from r = 0 to 2 R, with ``cells`` cells per diffusion length ``length``
    from 12 lengths upstream of the shock to 4 downstream, and cells growing by ``growth`` outside.
    """
    h = length / cells
    fine = np.arange(R_SHOCK - 12 * length, R_SHOCK + 4 * length + h / 2, h)
    inner, step, x = [], h, fine[0]
    while x > 0.0:
        step = min(step * growth, coarsest * R_SHOCK)
        x -= step
        inner.append(x)
    inner = np.array(inner[::-1])
    inner[0] = 0.0  # the diffusion operator needs a first center at the origin
    if inner[1] < 0.5 * (inner[2] - inner[1]):
        inner = np.delete(inner, 1)
    outer, step, x = [], h, fine[-1]
    while x < 2.0 * R_SHOCK:
        step = min(step * growth, coarsest * R_SHOCK)
        x += step
        outer.append(x)
    return np.concatenate([inner, fine, outer])


def run(compression=4.0, cells=20, epsilon=EPSILON, per_decade=16, p_max=1e3):
    """Steady state of a source at the shock, after 600 epsilon R / u1 (several acceleration times)."""
    D = epsilon * U_WIND * R_SHOCK
    length = D / U_WIND
    r = radial_grid(length, cells)
    p = np.logspace(0.0, np.log10(p_max), int(round(np.log10(p_max) * per_decade)) + 1)
    # Splitting steps of Courant number 2.5 / cells at the shock: the error of the
    # splitting at the shock decreases with the step, as that of the cells does
    dt = (2.5 / cells) * (length / cells) / U_WIND
    n_t = int(np.ceil(600.0 * epsilon * R_SHOCK / U_WIND / dt))
    grid = Grid(
        r_centers=r * su.LENGTH,
        p_centers=p * su.MOMENTUM,
        t_grid=np.arange(n_t + 1) * dt * su.TIME,
    )
    v = np.where(
        r < R_SHOCK,
        U_WIND,
        U_WIND / compression * (R_SHOCK / np.maximum(r, R_SHOCK)) ** 2,
    )
    shock = int(np.argmin(np.abs(r - R_SHOCK)))
    source = np.zeros((p.size, r.size))
    source[P_INJECTION_INDEX, shock] = 1.0
    solver = Solver(
        state=State(
            psi_p=np.zeros((p.size, r.size)) * su.PSI_P,
            grid=grid,
            particle=Particle.PROTON,
        ),
        problem_type="advection-diffusion-loss-source",
        operator_params={
            "advection": {
                "v_centers": v * su.VELOCITY,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "diffusion": {
                "D_values": np.full((p.size, r.size), D) * su.DIFFUSION_COEFFICIENT,
                "psi_end": 0.0 * su.PSI_P,
            },
            # The loss operator takes the velocity of the advection operator
            "loss": {"adiabatic_losses": True},
            "source": {"source": source * su.SOURCE_PSI_P},
        },
    )
    psi = solver.run().psi_p.to_value(su.PSI_P)
    return r, p, psi, shock


def fitted_slope(p, spectrum):
    selected = (p >= FIT_RANGE[0]) & (p <= FIT_RANGE[1])
    return np.polyfit(np.log(p[selected]), np.log(spectrum[selected]), 1)[0]


def validation_adiabatic_shock_acceleration(
    resolutions=(5, 7, 10, 14, 20, 28, 40),
    compressions=(4.0, 3.0, 2.5, 2.0),
    compression_cells=20,
    plot_results=True,
):
    print(
        f"Termination shock at R = {R_SHOCK:g} pc, u1 = {U_WIND:g} pc/Myr, "
        f"D / (u1 R) = {EPSILON:g}; slopes of psi_p fitted in {FIT_RANGE} GeV/c"
    )

    # 1. Convergence for a strong shock
    theory = slope_spherical(4.0)
    convergence = []
    for cells in resolutions:
        r, p, psi, shock = run(compression=4.0, cells=cells)
        slope = fitted_slope(p, psi[:, shock])
        convergence.append(slope)
        print(
            f"r_c = 4, {cells:2d} cells per diffusion length: slope {slope:.4f}, "
            f"spherical theory {theory:.4f} (difference {slope - theory:+.4f}), planar {slope_planar(4.0):.4f}"
        )
        if cells == compression_cells:
            strong = (r, p, psi, shock)
    # Slope converging as slope_inf + b cells^-order, fitted to every resolution
    (extrapolated, _, order), _ = curve_fit(
        lambda cells, limit, b, k: limit + b * cells**-k,
        np.asarray(resolutions, dtype=float),
        np.asarray(convergence),
        p0=(convergence[-1], -0.4, 1.0),
    )
    print(
        f"  fitted order of convergence {order:.2f}; extrapolated slope {extrapolated:.4f} "
        f"(difference {extrapolated - theory:+.4f}, of the order of the neglected "
        f"epsilon^2 = {EPSILON**2:g} terms)"
    )

    # 2. Compression ratios at a common resolution
    spectra = {4.0: strong}
    slopes = {4.0: convergence[resolutions.index(compression_cells)]}
    for compression in compressions:
        if compression in spectra:
            continue
        spectra[compression] = run(compression=compression, cells=compression_cells)
        r_c, p_c, psi_c, shock_c = spectra[compression]
        slopes[compression] = fitted_slope(p_c, psi_c[:, shock_c])
    for compression in compressions:
        print(
            f"r_c = {compression:g}: slope {slopes[compression]:.4f}, spherical theory "
            f"{slope_spherical(compression):.4f}, planar {slope_planar(compression):.4f}"
        )

    # 3. Radial profile around the shock: the precursor of Morlino et al. (2021)
    r, p, psi, shock = strong
    j = int(np.argmin(np.abs(p - 30.0)))
    x = (r - R_SHOCK) / (EPSILON * R_SHOCK)
    precursor = np.where(x < 0.0, np.exp(np.minimum(x, 0.0)), 1.0)
    upstream = (x > -6.0) & (x < -1.0)
    profile = psi[j] / psi[j, shock]
    precursor_error = np.max(np.abs(profile[upstream] - precursor[upstream]))
    print(
        f"Precursor at p = {p[j]:.1f} GeV/c: largest difference from e^(-u1 (R - r) / D) "
        f"between 1 and 6 diffusion lengths upstream, {precursor_error:.4f} of the value at the shock"
    )

    results = {
        "convergence": convergence,
        "order": order,
        "extrapolated": extrapolated,
        "slopes": slopes,
        "precursor_error": precursor_error,
    }
    if plot_results:
        _plot(
            spectra,
            slopes,
            resolutions,
            convergence,
            extrapolated,
            x,
            profile,
            precursor,
            p[j],
        )
    return results


def _plot(
    spectra,
    slopes,
    resolutions,
    convergence,
    extrapolated,
    x,
    profile,
    precursor,
    p_profile,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = [SAETASS_BLUE, SAETASS_ORANGE, SAETASS_YELLOW, "0.4"]

    # Spectra at the shock, multiplied by p^2 so that a strong shock is flat
    ax = axes[0, 0]
    for color, (compression, (r, p, psi, shock)) in zip(colors, spectra.items()):
        spectrum = psi[:, shock] * p**2
        ax.loglog(
            p,
            spectrum / spectrum[P_INJECTION_INDEX + 8],
            color=color,
            label=rf"$r_c = {compression:g}$",
        )
        anchor = p[P_INJECTION_INDEX + 8]
        ax.loglog(
            p,
            (p / anchor) ** (slope_spherical(compression) + 2.0),
            color=color,
            linestyle=":",
            alpha=0.8,
        )
    ax.axvspan(*FIT_RANGE, color="0.9", zorder=0)
    ax.set(
        xlabel="p [GeV/c]",
        ylabel=r"$p^2 \psi_p$ at the shock (normalized)",
        title="Steady spectra; dotted: theory",
        ylim=(1e-6, 10),
    )
    ax.legend()

    # Slope against the compression ratio
    ax = axes[0, 1]
    ratios = np.linspace(1.8, 4.2, 100)
    ax.plot(
        ratios,
        slope_planar(ratios),
        **get_analytical_style(),
        label="planar, $2 - 3 r_c / (r_c - 1)$",
    )
    ax.plot(
        ratios,
        slope_spherical(ratios),
        color=SAETASS_ORANGE,
        linestyle="-.",
        label=rf"spherical, $\varepsilon = {EPSILON:g}$",
    )
    ax.plot(
        list(slopes), list(slopes.values()), "o", color=SAETASS_BLUE, label="SAETASS"
    )
    ax.set(
        xlabel="compression ratio $r_c$",
        ylabel=r"slope of $\psi_p$ at the shock",
        title="Spectral slope",
    )
    ax.legend()

    # Convergence of the slope of a strong shock
    ax = axes[1, 0]
    error = np.abs(np.array(convergence) - extrapolated)
    ax.loglog(
        resolutions,
        np.abs(np.array(convergence) - slope_spherical(4.0)),
        **get_quantitative_style(),
        label="difference from the spherical theory",
    )
    ax.loglog(
        resolutions,
        error,
        "s--",
        color=SAETASS_BLUE,
        label="difference from the extrapolated slope",
    )
    reference = np.array(resolutions, dtype=float)
    ax.loglog(reference, error[0] * reference[0] / reference, "k:", label="first order")
    ax.set(
        xlabel="cells per diffusion length $D / u_1$",
        ylabel="slope error",
        title="Convergence, $r_c = 4$",
    )
    ax.legend()

    # Radial profile around the shock
    ax = axes[1, 1]
    ax.plot(x, profile, **get_numerical_style(is_final=True), label="SAETASS")
    ax.plot(
        x,
        precursor,
        **get_analytical_style(),
        label=r"$e^{-u_1 (R - r) / D}$ upstream, flat downstream",
    )
    ax.set(
        xlim=(-10, 4),
        xlabel=r"$(r - R) / (D / u_1)$",
        ylabel=r"$\psi_p / \psi_p(R)$",
        title=f"Profile at p = {p_profile:.0f} GeV/c, $r_c = 4$",
    )
    ax.legend()

    fig.tight_layout()
    path = os.path.join(
        os.path.dirname(__file__), "..", "figures", "adiabatic_shock_acceleration.pdf"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {path}")
    plt.show()


if __name__ == "__main__":
    validation_adiabatic_shock_acceleration()
