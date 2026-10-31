r"""
Recovery of D(r) of a wind bubble from SAETASS, with the batched solver and no assumed form.

The data are the radial profiles of psi of a SAETASS run at several energies, with noise.
The run needs 4800 cells: at 1200 cells its error at 3 pc and 100 GeV is 30 %, which it
loses as it converges to the batched solver, and an inversion would read it as a real
feature of D. The velocity, the source and the energy dependence of D are known, and D(r) at one energy
is free at uniformly spaced knots, which know nothing of the shock or of the bubble.

Usage: python pinn_diffusion_inversion.py output.npz [--device cuda] [--knots 160]
"""

import argparse
import json
import time

import astropy.units as u
import numpy as np
import torch
from pinn_parameter_search import AGE, ENERGIES, R_MAX, TRUTH, simulate

from saetass.cli import set_log_level
from saetass.pinn.inversion import DiffusionInversion
from saetass.pinn.radial_solver import RadialSolver
from saetass.pinn.wind_bubble import FastScales

#: Range of the radii of the data, in pc, and their relative noise.
FIRST_RADIUS, LAST_RADIUS = 3.0, 200.0
NOISE = 0.03
#: Cells of the SAETASS run of the data.
CELLS = 4800


def bubble_fields(solver: RadialSolver, device: str) -> dict:
    """D, v, source, times and the unit of psi of the bubble at ``TRUTH``, for each energy."""
    fast = FastScales(1 * u.PeV)
    energies = torch.as_tensor(ENERGIES, dtype=torch.float64, device=device)
    p = (energies**2 + 2.0 * energies * fast.mc2) ** 0.5
    L_wind, M_dot, rho, eta_B = (
        torch.as_tensor(10.0**value, dtype=torch.float64, device=device)
        for value in TRUTH
    )
    age = torch.full_like(L_wind, AGE)
    outer = torch.full_like(L_wind, R_MAX)
    scales = fast(p, L_wind, M_dot, rho, age, outer, eta_B=eta_B)

    # Lengths in units of the outer radius, and times in units of R_MAX / v_b
    v_b = scales.R_TS / scales.time_unit
    shock = (scales.R_TS / R_MAX).expand_as(p)
    edge = shock * scales.bubble_ratio
    diffusion = v_b * scales.R_TS / scales.peclet / (v_b * R_MAX)
    x = solver.centers[None]
    wind = 11.0 ** (fast.delta / 2.0) * (x / shock[:, None]) ** fast.delta
    zone = torch.where(x <= edge[:, None], 1.0, scales.diffusion_ratio[:, None])
    D = diffusion[:, None] * torch.where(x < shock[:, None], wind, zone)
    v = torch.where(
        x < shock[:, None],
        4.0,
        torch.where(x <= edge[:, None], (shock[:, None] / x) ** 2, 0.0),
    )
    return {
        "D": D,
        "v": v,
        "source": solver.shell(shock),
        "times": (AGE / (R_MAX / v_b)).expand_as(p),
        "log_unit": (scales.psi_unit * shock**2).log(),
        "edge": edge,
        "shock": shock,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nodes", type=int, default=1025)
    parser.add_argument("--contour", type=int, default=24)
    parser.add_argument("--knots", type=int, default=160)
    parser.add_argument("--radii", type=int, default=40)
    parser.add_argument("--realizations", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--regularization", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    set_log_level("ERROR")
    device = arguments.device
    nodes = torch.linspace(
        0.0, 1.0, arguments.nodes, dtype=torch.float64, device=device
    )
    solver = RadialSolver(nodes, arguments.contour)
    bubble = bubble_fields(solver, device)

    observed = np.geomspace(FIRST_RADIUS, LAST_RADIUS, arguments.radii)
    clean = np.array(
        [simulate(TRUTH, energy, observed, CELLS)[0] for energy in ENERGIES]
    )
    log_data = torch.as_tensor(np.log(clean), dtype=torch.float64, device=device)
    log_data = log_data - bubble["log_unit"][:, None]
    radii = torch.as_tensor(observed / R_MAX, dtype=torch.float64, device=device)

    factors = bubble["D"][:, -1] / bubble["D"][0, -1]
    with torch.no_grad():
        forward = solver.log_psi(
            bubble["D"], bubble["v"], bubble["source"], bubble["times"]
        )
        model = solver.interpolate(forward, radii.expand(ENERGIES.size, -1))
    print(
        "Forward model against SAETASS, rms of ln(psi):",
        float(torch.sqrt(torch.mean((model - log_data) ** 2))),
    )
    print("Factors", factors.tolist(), flush=True)

    inversion = DiffusionInversion(
        solver,
        bubble["v"],
        bubble["source"],
        bubble["times"],
        radii,
        log_data,
        NOISE,
        factors=factors,
        n_knots=arguments.knots,
        regularization=arguments.regularization,
    )
    start = float(bubble["D"][0].log().mean())
    begin = time.perf_counter()
    estimate = inversion.run(
        start, arguments.realizations, arguments.steps, seed=arguments.seed
    )
    elapsed = time.perf_counter() - begin

    truth = bubble["D"][0].log()
    recovered = inversion.log_diffusion(estimate.samples)
    median = recovered.quantile(0.5, dim=0)
    low, high = recovered.quantile(0.16, dim=0), recovered.quantile(0.84, dim=0)
    x = solver.centers
    shock, edge = bubble["shock"][0].item(), bubble["edge"][0].item()
    zones = {
        "wind": (0.01, 0.9 * shock),
        "bubble": (1.15 * shock, 0.9 * edge),
        "medium": (1.1 * edge, 0.66),
    }
    report = {
        "seconds": elapsed,
        "chi2_per_point": float(estimate.chi2.median()),
        "edge_in_pc": edge * R_MAX,
        "shock_in_pc": shock * R_MAX,
    }
    for name, (begin_x, end_x) in zones.items():
        mask = (x >= begin_x) & (x < end_x)
        error = (median - truth)[mask].abs()
        inside = ((truth >= low) & (truth <= high))[mask].float().mean()
        report[name] = {
            "median_abs_dln_D": float(error.median()),
            "max_abs_dln_D": float(error.max()),
            "inside_the_68_band": float(inside),
        }
    jump = (truth[1:] - truth[:-1]).abs().argmax()
    found = (median[1:] - median[:-1]).abs()
    near = (x[:-1] > 0.5 * edge) & (x[:-1] < 1.5 * edge)
    report["true_jump_in_pc"] = float(x[jump]) * R_MAX
    report["recovered_jump_in_pc"] = float(x[:-1][near][found[near].argmax()]) * R_MAX
    print(json.dumps(report, indent=2))
    np.savez(
        arguments.output,
        x=x.cpu().numpy() * R_MAX,
        truth=truth.cpu().numpy(),
        median=median.cpu().numpy(),
        low=low.cpu().numpy(),
        high=high.cpu().numpy(),
    )


if __name__ == "__main__":
    main()
