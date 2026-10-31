r"""
Parameter search with a wind bubble surrogate or the batched solver, against SAETASS.

A mock observation is simulated with SAETASS: the radial profile of psi at several
energies, with noise. Four parameters of the bubble are then recovered with an ensemble
sampler that only uses the fast model, and the best fit is confirmed with a new SAETASS run.

With --confirm K, K samples of the posterior are run with SAETASS and weighted by the
ratio of its likelihood to that of the fast model, which turns the search into a sample of
the posterior of SAETASS itself, as long as the effective sample size stays high.

The script reports the time of the search, and what the same number of evaluations would
take with SAETASS, from the time of the runs it does.

Usage: python pinn_parameter_search.py surrogate.pt [--device cuda] [--walkers 2048]
       python pinn_parameter_search.py --nodes 512 --confirm 256
"""

import argparse
import json
import time

import astropy.constants as const
import astropy.units as u
import numpy as np
import torch

from saetass import units as su
from saetass.cli import set_log_level
from saetass.pinn.confirm import simulate as run_solver
from saetass.pinn.radial_solver import WindBubbleSolver
from saetass.pinn.surrogate import WindBubbleSurrogate
from saetass.pinn.wind_bubble import FastScales
from saetass.sweep import run_sweep

#: Parameters of the mock observation: log10 of L_wind, M_dot, rho_0 and eta_B.
TRUTH = np.array([38.3, -4.0, -22.5, -1.0])
#: Bounds of the flat priors of those parameters.
PRIOR = np.array([[37.3, 39.0], [-4.7, -3.7], [-23.5, -21.7], [-2.0, -0.6]])
#: Age of the bubble and outer radius, known, in Myr and pc.
AGE, R_MAX = 3.0, 300.0
#: Energies of the mock observation, in GeV, and relative noise of each point.
ENERGIES = np.array([1e2, 1e3, 1e4, 1e5])
NOISE = 0.05
N_RADII = 12


#: Contour of the batched solver in a search: more nodes do not change the posterior.
N_CONTOUR = 24


def simulate(
    theta: np.ndarray, energy: float, radii: np.ndarray, cells: int = 1200
) -> tuple[np.ndarray, float]:
    """A SAETASS run on ``cells`` cells, with psi at ``radii``, and its duration."""
    L_wind, M_dot, rho, eta_B = 10.0**theta
    r = np.linspace(0.0, R_MAX, cells)
    start = time.perf_counter()
    psi = run_solver(
        r * u.pc,
        energy * u.GeV,
        AGE * u.Myr,
        cells * int(6000 + 9000 * np.log10(energy)) // 600,
        L_wind=L_wind * u.erg / u.s,
        M_dot=M_dot * const.M_sun / u.yr,
        rho_0=rho * u.g / u.cm**3,
        t_b=AGE * u.Myr,
        E_max=1 * u.PeV,
        eta_B=eta_B,
    ).to_value(su.PSI_P)
    return np.exp(np.interp(radii, r, np.log(psi))), time.perf_counter() - start


def confirmation_run(task: tuple[np.ndarray, float, np.ndarray]) -> np.ndarray:
    """Logarithm of psi of one SAETASS run, for a worker of the sweep."""
    theta, energy, radii = task
    return np.log(simulate(theta, energy, radii)[0])


def confirm(
    samples: np.ndarray, log_fast: np.ndarray, radii: np.ndarray, data: np.ndarray
) -> dict:
    """Importance sampling of the posterior of SAETASS, from samples of the fast posterior."""
    tasks = [(theta, energy, radii) for theta in samples for energy in ENERGIES]
    begin = time.perf_counter()
    runs = run_sweep(confirmation_run, tasks, max_workers=16)
    seconds = time.perf_counter() - begin
    log_psi = np.array(runs).reshape(samples.shape[0], ENERGIES.size, -1)
    log_like = -0.5 * np.sum(((log_psi - np.log(data)) / NOISE) ** 2, axis=(1, 2))
    log_weights = log_like - log_fast
    weights = np.exp(log_weights - log_weights.max())
    effective = weights.sum() ** 2 / np.sum(weights**2)
    weights /= weights.sum()
    mean = np.sum(weights[:, None] * samples, axis=0)
    std = np.sqrt(np.sum(weights[:, None] * (samples - mean) ** 2, axis=0))
    return {
        "runs": len(tasks),
        "seconds": seconds,
        "samples": samples.shape[0],
        "effective_sample_size": float(effective),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pull": ((mean - TRUTH) / std).tolist(),
    }


class Likelihood:
    """Gaussian likelihood of ln(psi), with walkers in the rows of ``theta``."""

    def __init__(
        self,
        surrogate: WindBubbleSurrogate | WindBubbleSolver,
        radii: np.ndarray,
        data: np.ndarray,
    ) -> None:
        self.surrogate = surrogate
        device = surrogate.device
        self.fast = FastScales(1 * u.PeV)
        self.radii = torch.as_tensor(radii, dtype=torch.float64, device=device)
        self.energies = torch.as_tensor(ENERGIES, dtype=torch.float64, device=device)
        self.log_data = torch.as_tensor(
            np.log(data), dtype=torch.float64, device=device
        )
        mc2 = self.fast.mc2
        self.p = (self.energies**2 + 2.0 * self.energies * mc2) ** 0.5
        self.low = torch.as_tensor(PRIOR[:, 0], dtype=torch.float64, device=device)
        self.high = torch.as_tensor(PRIOR[:, 1], dtype=torch.float64, device=device)

    @torch.no_grad()
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        values = 10.0**theta
        L_wind, M_dot, rho, eta_B = (
            values[:, None, 0],
            values[:, None, 1],
            values[:, None, 2],
            values[:, None, 3],
        )
        age = torch.full_like(L_wind, AGE)
        outer = torch.full_like(L_wind, R_MAX)
        scales = self.fast(self.p[None, :], L_wind, M_dot, rho, age, outer, eta_B=eta_B)
        scales = type(scales)(
            **{
                name: getattr(scales, name).expand(theta.shape[0], self.p.numel())[
                    ..., None
                ]
                for name in scales.__dataclass_fields__
            }
        )
        times = AGE / scales.time_unit
        x = self.radii[None, None, :] / scales.R_TS
        log_psi = self.surrogate.log_psi(x, times, scales)
        chi2 = torch.sum(((log_psi - self.log_data) / NOISE) ** 2, dim=(1, 2))
        inside = self.surrogate.in_domain(scales, times).all(dim=(1, 2))
        inside &= torch.all((theta >= self.low) & (theta <= self.high), dim=1)
        # The radii must lie inside the domain of each bubble, which holds below r_max
        return torch.where(inside, -0.5 * chi2, torch.full_like(chi2, -torch.inf))


def stretch_sampler(
    log_prob, start: torch.Tensor, n_steps: int, scale: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine-invariant ensemble sampler of Goodman and Weare, with two half-ensembles."""
    walkers = start.clone()
    n, dimension = walkers.shape
    half = n // 2
    current = log_prob(walkers)
    chain, values = [], []
    for _ in range(n_steps):
        for first, second in ((0, half), (half, 0)):
            size = half if first == 0 else n - half
            active = slice(first, first + size)
            other = walkers[slice(second, second + (n - size))]
            partners = other[
                torch.randint(other.shape[0], (size,), device=start.device)
            ]
            z = (
                (scale - 1.0) * torch.rand(size, device=start.device, dtype=start.dtype)
                + 1.0
            ) ** 2 / scale
            proposal = partners + z[:, None] * (walkers[active] - partners)
            trial = log_prob(proposal)
            ratio = (dimension - 1) * torch.log(z) + trial - current[active]
            accept = (
                torch.log(torch.rand(size, device=start.device, dtype=start.dtype))
                < ratio
            )
            walkers[active] = torch.where(accept[:, None], proposal, walkers[active])
            current[active] = torch.where(accept, trial, current[active])
        chain.append(walkers.clone())
        values.append(current.clone())
    return torch.stack(chain), torch.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("surrogate", nargs="?")
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--confirm", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--walkers", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-radius", type=float, default=16.0)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    set_log_level("ERROR")
    rng = np.random.default_rng(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.nodes is not None:
        surrogate = WindBubbleSolver(
            arguments.nodes, N_CONTOUR, device=arguments.device
        )
    else:
        paths = arguments.surrogate.split(",")
        surrogate = (
            WindBubbleSurrogate.average(paths, arguments.device)
            if len(paths) > 1
            else WindBubbleSurrogate.load(paths[0], arguments.device)
        )

    radii = np.linspace(arguments.min_radius, 110.0, N_RADII)
    data, costs = [], []
    for energy in ENERGIES:
        psi, seconds = simulate(TRUTH, energy, radii)
        data.append(psi * np.exp(NOISE * rng.standard_normal(N_RADII)))
        costs.append(seconds)
    data = np.array(data)
    print(f"Mock observation: {sum(costs):.1f} s of SAETASS for one set of parameters")

    likelihood = Likelihood(surrogate, radii, data)
    device = surrogate.device
    low, high = likelihood.low, likelihood.high
    start = low + (high - low) * torch.rand(
        arguments.walkers, 4, dtype=torch.float64, device=device
    )
    begin = time.perf_counter()
    chain, values = stretch_sampler(likelihood, start, arguments.steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin

    settled = chain[arguments.steps // 2 :].reshape(-1, 4)
    good = values[arguments.steps // 2 :].reshape(-1)
    finite = torch.isfinite(good)
    best = settled[finite][good[finite].argmax()].cpu().numpy()
    mean = settled[finite].mean(0).cpu().numpy()
    std = settled[finite].std(0).cpu().numpy()
    evaluations = arguments.walkers * arguments.steps * ENERGIES.size

    fit, seconds = [], 0.0
    for index, energy in enumerate(ENERGIES):
        psi, cost = simulate(best, energy, radii)
        fit.append(psi)
        seconds += cost
    fit = np.array(fit)
    chi2_saetass = float(np.sum((np.log(fit / data) / NOISE) ** 2))
    truth, _ = zip(*[simulate(TRUTH, energy, radii) for energy in ENERGIES])
    chi2_truth = float(np.sum((np.log(np.array(truth) / data) / NOISE) ** 2))

    report = {
        "truth": TRUTH.tolist(),
        "best_fit": best.tolist(),
        "posterior_mean": mean.tolist(),
        "posterior_std": std.tolist(),
        "pull": ((mean - TRUTH) / std).tolist(),
        "n_points": int(data.size),
        "chi2_of_the_best_fit_with_saetass": chi2_saetass,
        "chi2_of_the_truth_with_saetass": chi2_truth,
        "search_seconds": elapsed,
        "profiles_evaluated": evaluations,
        "saetass_seconds_per_profile": float(sum(costs) / len(costs)),
        "saetass_hours_for_the_same_search_on_one_core": float(
            evaluations * sum(costs) / len(costs) / 3600.0
        ),
    }
    if arguments.confirm:
        picked = torch.randperm(settled[finite].shape[0], device=device)
        picked = picked[: arguments.confirm]
        report["saetass_posterior"] = confirm(
            settled[finite][picked].cpu().numpy(),
            good[finite][picked].cpu().numpy(),
            radii,
            data,
        )
    print(json.dumps(report, indent=2))
    if arguments.output:
        with open(arguments.output, "w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
