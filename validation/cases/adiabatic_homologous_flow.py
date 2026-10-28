r"""
Validation of adiabatic losses and gains in a homologous flow, against the exact solution.

In the flow :math:`v = \alpha r`, whose divergence :math:`\nabla \cdot v = 3 \alpha` is uniform, the
transport equation of :cite:ct:`Parker1965` without diffusion,

.. math::

    \frac{\partial f}{\partial t} + v\, \frac{\partial f}{\partial r} - \frac{\nabla \cdot v}{3}\, p\, \frac{\partial f}{\partial p} = 0,

keeps the phase-space density :math:`f` constant along the characteristics
:math:`r = r_0 e^{\alpha t}`, :math:`p = p_0 e^{-\alpha t}`: particles move with the flow and lose
momentum where it expands (:math:`\alpha > 0`) or gain it where it is compressed (:math:`\alpha < 0`).
For :math:`\psi = 4 \pi p^2 f`, the exact solution is

.. math::

    \psi(r, p, t) = e^{-2 \alpha t}\, \psi_0\!\left(r e^{-\alpha t},\, p e^{\alpha t}\right),

which conserves the number of particles. SAETASS solves it with the advection operator and the loss
operator with ``adiabatic_losses=True``, which takes the velocity of the advection. The script
compares both at several resolutions, for an expansion and a compression, and prints the relative
L2 errors and the observed order of convergence.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import (
    SAETASS_BLUE,
    SAETASS_ORANGE,
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

R_MAX = 4.0  # pc
P_MIN, P_MAX = 1.0, 1e3  # GeV/c
T_FINAL = 1.0  # Myr


def psi_initial(r, p):
    """A shell at r = 1.5 pc with a Gaussian spectrum in log10(p) around 30 GeV/c."""
    shell = np.exp(-(((r - 1.5) / 0.3) ** 2))
    spectrum = np.exp(-(((np.log10(p) - 1.5) / 0.2) ** 2))
    return spectrum[:, None] * shell[None, :]


def psi_exact(r, p, t, alpha):
    """Exact solution: psi_0 carried along r e^(-alpha t), p e^(alpha t), times e^(-2 alpha t)."""
    return np.exp(-2.0 * alpha * t) * psi_initial(
        r * np.exp(-alpha * t), p * np.exp(alpha * t)
    )


def run(alpha, n_r, per_decade):
    """Advection and adiabatic losses in v = alpha r up to T_FINAL."""
    r = np.linspace(0.0, R_MAX, n_r)
    p = np.logspace(np.log10(P_MIN), np.log10(P_MAX), int(3 * per_decade) + 1)
    # Splitting steps of a fixed Courant number, 0.5 at the outer boundary
    dt = 0.5 * (r[1] - r[0]) / (abs(alpha) * R_MAX)
    n_t = int(np.ceil(T_FINAL / dt))
    grid = Grid(
        r_centers=r * su.LENGTH,
        p_centers=p * su.MOMENTUM,
        t_grid=np.linspace(0.0, T_FINAL, n_t + 1) * su.TIME,
    )
    solver = Solver(
        state=State(
            psi_p=psi_initial(r, p) * su.PSI_P, grid=grid, particle=Particle.PROTON
        ),
        problem_type="advection-loss",
        operator_params={
            "advection": {
                "v_centers": alpha * r * su.VELOCITY,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "loss": {"adiabatic_losses": True},
        },
    )
    psi = solver.run().psi_p.to_value(su.PSI_P)
    return r, p, psi


def relative_L2(numerical, exact):
    return np.linalg.norm(numerical - exact) / np.linalg.norm(exact)


def validation_adiabatic_homologous_flow(
    alphas=(0.5, -0.5),
    levels=(
        (21, 5),
        (41, 10),
        (81, 20),
        (161, 40),
        (321, 80),
        (641, 160),
        (1281, 320),
    ),
    plot_results=True,
):
    results = {}
    for alpha in alphas:
        label = "expansion" if alpha > 0 else "compression"
        errors = []
        for n_r, per_decade in levels:
            r, p, psi = run(alpha, n_r, per_decade)
            exact = psi_exact(r, p, T_FINAL, alpha)
            errors.append(relative_L2(psi, exact))
            # Particles are conserved: compare the totals of r^2 psi over the grid cells
            weights = (
                r[None, :] ** 2
                * np.gradient(r)[None, :]
                * np.gradient(np.log(p))[:, None]
                * p[:, None]
            )
            number = np.sum(weights * psi) / np.sum(weights * psi_initial(r, p))
            print(
                f"alpha = {alpha:+.2f} /Myr ({label}), n_r = {n_r}, {per_decade} momenta per decade: "
                f"relative L2 error {errors[-1]:.4e}, particle number ratio {number:.6f}"
            )
        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        print(
            f"  observed orders of convergence: {np.array2string(orders, precision=2)}"
        )
        results[alpha] = {
            "errors": errors,
            "orders": orders,
            "r": r,
            "p": p,
            "psi": psi,
        }

    if plot_results:
        _plot(results, levels)
    return results


def _plot(results, levels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for alpha, result in results.items():
        r, p, psi = result["r"], result["p"], result["psi"]
        exact = psi_exact(r, p, T_FINAL, alpha)
        label = "expansion" if alpha > 0 else "compression"
        # Radial profile at the momentum of the peak, and spectrum at the radius of the peak
        j, i = np.unravel_index(np.argmax(exact), exact.shape)
        axes[0].plot(
            r,
            psi_initial(r, p)[np.argmax(psi_initial(r, p).max(axis=1))],
            **get_numerical_style(is_initial=True),
        )
        axes[0].plot(
            r,
            psi[j],
            **get_numerical_style(is_final=True),
            label="SAETASS" if alpha > 0 else None,
        )
        axes[0].plot(
            r, exact[j], **get_analytical_style(), label="Exact" if alpha > 0 else None
        )
        axes[0].annotate(
            label, (r[np.argmax(exact[j])], exact[j].max() * 1.04), ha="center"
        )
        axes[1].loglog(
            p,
            psi_initial(r, p)[:, np.argmax(psi_initial(r, p).max(axis=0))],
            **get_numerical_style(is_initial=True),
        )
        axes[1].loglog(p, psi[:, i], **get_numerical_style(is_final=True))
        axes[1].loglog(p, exact[:, i], **get_analytical_style())
        axes[1].annotate(
            label, (p[np.argmax(exact[:, i])], exact[:, i].max() * 1.3), ha="center"
        )
        n_r = [level[0] for level in levels]
        style = get_quantitative_style()
        style["color"] = SAETASS_ORANGE if alpha > 0 else SAETASS_BLUE
        axes[2].loglog(n_r, result["errors"], **style, label=label)
    reference = np.array([levels[0][0], levels[-1][0]], dtype=float)
    axes[2].loglog(
        reference,
        results[next(iter(results))]["errors"][0] * (reference / reference[0]) ** -2,
        "k:",
        label="second order",
    )
    axes[0].set(
        xlabel="r [pc]",
        ylabel=r"$\psi_p$ at the peak momentum",
        title=f"Radial profiles at t = {T_FINAL:g} Myr",
    )
    axes[1].set(
        xlabel="p [GeV/c]",
        ylabel=r"$\psi_p$ at the peak radius",
        title="Spectra",
        ylim=(1e-4, 5),
    )
    axes[2].set(
        xlabel="radial cells (momenta scale alike)",
        ylabel="relative L2 error",
        title="Convergence",
    )
    axes[0].legend()
    axes[2].legend()
    fig.tight_layout()
    path = os.path.join(
        os.path.dirname(__file__), "..", "figures", "adiabatic_homologous_flow.pdf"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {path}")
    plt.show()


if __name__ == "__main__":
    validation_adiabatic_homologous_flow()
