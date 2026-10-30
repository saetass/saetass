r"""
Acceptance of a wind bubble surrogate, with the criteria of issue #73.

Every point of this script was left out of the training and of the validation set that
monitors it. The script measures, against converged reference solutions:

1. The relative L2 error of psi over the radius, and the largest |Delta ln psi| above
   1e-6 of the largest psi of each time, on a Sobol test set of dimensionless bubbles.
2. The same error next to the error of SAETASS at its usual resolution, on random
   physical bubbles, in the units of a user.
3. The particle budget: the number of particles against the reference, and against the
   injected number at early times, when none has escaped yet, which needs no reference.
4. The cost of an evaluation against that of a simulation.

Usage: python pinn_wind_bubble.py surrogate.pt test.npz [--device cuda] [--bubbles 32]
"""

import argparse
import json
import time

import astropy.constants as const
import astropy.units as u
import numpy as np
import torch
from scipy.integrate import trapezoid

from saetass import Grid, Particle, Solver, State
from saetass import units as su
from saetass.cli import set_log_level
from saetass.pinn.dataset import bubble_from_log
from saetass.pinn.reference import solve
from saetass.pinn.surrogate import WindBubbleSurrogate
from saetass.pinn.training import Tensors, errors
from saetass.pinn.wind_bubble import DimensionlessBubble, physical_bubble
from saetass.utils.bubble_profiles import BubbleProfileCalculator


def summary(values: np.ndarray) -> dict:
    """Median, 95th percentile and worst case of an array of errors."""
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "worst": float(np.max(values)),
    }


def test_set_errors(surrogate: WindBubbleSurrogate, path: str) -> dict:
    """Errors on the test set, against the reference solutions stored in it."""
    tensors = Tensors(np.load(path), surrogate.delta, surrogate.device)
    result = errors(surrogate.model, tensors)
    l2 = result["l2"].cpu().numpy()
    log_max = result["log_max"].cpu().numpy()
    return {
        "bubbles": tensors.n,
        "l2_all_times": summary(l2),
        "l2_worst_time_of_each_bubble": summary(l2.max(axis=1)),
        "log_max": summary(log_max),
    }


def particle_budget(surrogate: WindBubbleSurrogate, n: int = 32, seed: int = 7) -> dict:
    """Number of particles of the surrogate, against the reference and against the injection."""
    rng = np.random.default_rng(seed)
    low, high = surrogate.bounds[:, 0], surrogate.bounds[:, 1]
    against_reference, against_injection = [], []
    times = np.geomspace(1.0, 3e3, 8)
    for _ in range(n):
        bubble = bubble_from_log(rng.uniform(low, high), surrogate.delta)
        solution = solve(bubble, times, resolution=4)
        x = solution.x[:-1]
        predicted = np.stack(
            [surrogate.dimensionless_psi(x, time, bubble) for time in times]
        )
        reference = trapezoid(solution.psi[:, :-1] * x**2, x, axis=1)
        number = trapezoid(predicted * x**2, x, axis=1)
        against_reference.append(np.abs(number / reference - 1.0))
        # Nothing has reached the outer boundary while the diffusion time across the
        # medium is much longer than t, nor has the flow carried it there
        outer = float(bubble.outer_radius)
        medium = float(bubble.diffusion_ratio / bubble.peclet)
        early = times < 0.02 * (outer - float(bubble.bubble_ratio)) ** 2 / medium
        if early.any():
            against_injection.append(np.abs(number[early] / times[early] - 1.0))
    injection = np.concatenate(against_injection) if against_injection else np.zeros(1)
    return {
        "bubbles": n,
        "against_reference": summary(np.concatenate(against_reference)),
        "against_injection_before_escaping": summary(injection),
    }


def random_physical(rng: np.random.Generator) -> dict:
    """Physical parameters of a bubble within the ranges of the study of the domain."""
    return dict(
        L_wind=10 ** rng.uniform(37.0, 39.0) * u.erg / u.s,
        M_dot=10 ** rng.uniform(-5.0, -3.7) * const.M_sun / u.yr,
        rho_0=10 ** rng.uniform(0.0, 2.0) * const.m_p / u.cm**3,
        t_b=rng.uniform(1.0, 8.0) * u.Myr,
        eta_B=10 ** rng.uniform(-2.0, -0.6),
    )


def simulate(
    r: np.ndarray, energy: u.Quantity, parameters: dict, n_steps: int, end: float
) -> tuple[np.ndarray, float]:
    """A run of :py:class:`~saetass.solver.Solver`, as in the example of the wind bubble."""
    physical = {k: v for k, v in parameters.items() if k != "eta_B"}
    calculator = BubbleProfileCalculator(r_grid=r * u.pc, model="Morlino21", **physical)
    setup = calculator.get_all_profiles(
        energy, eta_B=parameters["eta_B"], eta_inj=0.1, E_max=1 * u.PeV
    )
    solver = Solver(
        state=State(
            psi_p=np.zeros(r.size) * su.PSI_P,
            grid=Grid(
                r_centers=r * u.pc, t_grid=np.linspace(0.0, end, n_steps) * u.Myr
            ),
            particle=Particle.PROTON,
        ),
        problem_type="advection-source-diffusion",
        operator_params={
            "advection": {
                "v_centers": setup["v_field"],
                "cfl": 0.8,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "diffusion": {"D_values": setup["D_values"], "psi_end": 0.0 * su.PSI_P},
            "source": {"source": setup["Q"]},
        },
        quiet=True,
    )
    start = time.perf_counter()
    solver.run()
    return solver.state.psi_p.to_value(su.PSI_P), time.perf_counter() - start


def physical_comparison(surrogate: WindBubbleSurrogate, n: int, seed: int = 11) -> dict:
    """
    Surrogate and SAETASS at 600 cells against the converged solution of random bubbles.

    Returns the errors of both, the ratio of the costs, and the points that lie outside
    the domain of the surrogate, which are counted and left out.
    """
    rng = np.random.default_rng(seed)
    rows, outside = [], 0
    while len(rows) < n:
        parameters = random_physical(rng)
        energy = 10 ** rng.uniform(1.0, 5.5) * u.GeV
        probe = physical_bubble(energy, r_max=1e5 * u.pc, E_max=1 * u.PeV, **parameters)
        r_max = float(rng.uniform(2.0, 5.0) * probe.R_b)
        scales = physical_bubble(
            energy, r_max=r_max * u.pc, E_max=1 * u.PeV, **parameters
        )
        numbers = scales.dimensionless
        bubble = DimensionlessBubble(
            float(numbers.peclet[0]),
            float(numbers.diffusion_ratio[0]),
            float(numbers.bubble_ratio),
            float(numbers.domain_ratio),
            numbers.delta,
        )
        end = float(rng.uniform(0.1, 1.0)) * parameters["t_b"].to_value(su.TIME)
        target = end / scales.time_unit
        try:
            surrogate.log_ratio(np.array([1.5]), target, bubble)
        except ValueError:
            outside += 1
            continue

        r = np.linspace(0.0, r_max, 600)
        dr = r[1]
        # The rule of examples/01_multi_energy_diffusion.py for the number of steps
        n_steps = int((6000 + 9000 * np.log10(energy.to_value(u.GeV))) * end / 3.0) + 2
        psi, cost = simulate(r, energy, parameters, n_steps, end)

        reference = solve(bubble, np.array([target]), resolution=8)
        x = r / scales.R_TS
        unit = scales.psi_unit[0]
        exact = np.interp(x, reference.x, reference.psi[0]) * unit
        start = time.perf_counter()
        predicted = surrogate.dimensionless_psi(x, target, bubble) * unit
        if surrogate.device.type == "cuda":
            torch.cuda.synchronize()
        evaluation = time.perf_counter() - start

        # The cells of the origin and of the jump at R_b are artefacts of the
        # discretization, which do not converge, see the tests of the reference
        away = (np.abs(r - scales.R_b) > 3.0 * dr) & (r > 3.0 * dr)
        weight = np.linalg.norm(exact[away])
        relevant = away & (exact > 1e-6 * exact.max())
        rows.append(
            {
                "energy_GeV": float(energy.to_value(u.GeV)),
                "peclet": bubble.peclet,
                "saetass_l2": float(np.linalg.norm((psi - exact)[away]) / weight),
                "surrogate_l2": float(
                    np.linalg.norm((predicted - exact)[away]) / weight
                ),
                "surrogate_log_max": float(
                    np.max(np.abs(np.log(predicted[relevant] / exact[relevant])))
                ),
                "saetass_seconds": cost,
                "surrogate_seconds": evaluation,
            }
        )
    saetass = np.array([row["saetass_l2"] for row in rows])
    network = np.array([row["surrogate_l2"] for row in rows])
    return {
        "bubbles": n,
        "outside_the_domain": outside,
        "saetass_600_cells_l2": summary(saetass),
        "surrogate_l2": summary(network),
        "surrogate_log_max": summary(
            np.array([row["surrogate_log_max"] for row in rows])
        ),
        "surrogate_better_than_saetass": float(np.mean(network < saetass)),
        "seconds_saetass": summary(np.array([row["saetass_seconds"] for row in rows])),
        "seconds_surrogate": summary(
            np.array([row["surrogate_seconds"] for row in rows])
        ),
        "rows": rows,
    }


def throughput(surrogate: WindBubbleSurrogate, n_r: int = 600) -> dict:
    """Radial profiles evaluated per second, in batches of different bubbles."""
    rng = np.random.default_rng(3)
    size = 4096 if surrogate.device.type == "cuda" else 64
    theta = np.exp(
        rng.uniform(surrogate.bounds[:, 0], surrogate.bounds[:, 1], (size, 4))
    )
    bubble = DimensionlessBubble(
        theta[:, 0, None],
        theta[:, 1, None],
        theta[:, 2, None],
        theta[:, 3, None],
        surrogate.delta,
    )
    x = np.linspace(0.01, 1.0, n_r)[None, :] * (bubble.outer_radius * 0.99)
    surrogate.log_ratio(x, 10.0, bubble)
    start = time.perf_counter()
    for _ in range(5):
        surrogate.log_ratio(x, 10.0, bubble)
    if surrogate.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 5
    return {"profiles_per_second": size / elapsed, "batch": size, "cells": n_r}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("surrogate")
    parser.add_argument("test_set")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bubbles", type=int, default=32)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    set_log_level("ERROR")
    surrogate = WindBubbleSurrogate.load(arguments.surrogate, arguments.device)
    report = {
        "test_set": test_set_errors(surrogate, arguments.test_set),
        "throughput": throughput(surrogate),
        "particle_budget": particle_budget(surrogate),
        "physical_comparison": physical_comparison(surrogate, arguments.bubbles),
    }
    text = json.dumps(report, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if arguments.output:
        with open(arguments.output, "w") as file:
            file.write(text)


if __name__ == "__main__":
    main()
