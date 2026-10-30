r"""
Datasets of reference solutions of the dimensionless wind bubble.

Each solution is stored as :math:`w = \ln(\psi / \psi_\mathrm{ss})`, the logarithm of its ratio to the exact steady state of :py:func:`~saetass.pinn.wind_bubble.steady_state`, on a tensor product of times and zone coordinates, which are the same for every bubble.
The zone coordinate :math:`\zeta` runs over :math:`[0, 1]` in the wind, :math:`[1, 2]` in the bubble and :math:`[2, 3]` in the interstellar medium, so that the shock and the edge of the bubble lie at fixed coordinates whatever the parameters.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from ..sweep import run_sweep
from .reference import reference_grid, solve
from .wind_bubble import DELTA, DimensionlessBubble

#: Number of parameters of a bubble: Pe, lambda, a and b.
N_PARAMETERS = 4

#: Default bounds of the logarithms of Pe, lambda, a and b.
DEFAULT_BOUNDS = np.log(
    np.array([[0.02, 300.0], [60.0, 700.0], [2.2, 23.0], [1.4, 7.0]], dtype=float)
)

#: Default bounds of the logarithm of the dimensionless time.
DEFAULT_TIME_BOUNDS = np.log(np.array([1e-3, 1e4]))

#: Lowest :math:`\psi` that the surrogates resolve, relative to the largest one at each time.
RELATIVE_FLOOR = 1e-9


@dataclass(frozen=True)
class Sampling:
    """
    Times and nodes of the snapshots of a dataset.

    Parameters
    ----------
    n_times : int, optional
        Number of snapshots. Default is ``64``.
    strides : tuple of int, optional
        Stride between the stored nodes of the reference grid in each zone. Default is ``(16, 16, 16)``.
    resolution : int, optional
        Resolution of the reference solutions. Default is ``4``.
    delta : float, optional
        Exponent of the diffusion coefficient in the wind. Default is ``1 / 3``.
    time_bounds : tuple of float, optional
        Bounds of the logarithm of the dimensionless time. Default is :py:data:`DEFAULT_TIME_BOUNDS`.
    """

    n_times: int = 64
    strides: tuple[int, int, int] = (16, 16, 16)
    resolution: int = 4
    delta: float = DELTA["kolmogorov"]
    time_bounds: tuple[float, float] = tuple(DEFAULT_TIME_BOUNDS)

    @property
    def times(self) -> np.ndarray:
        """The dimensionless times of the snapshots."""
        return np.exp(np.linspace(*self.time_bounds, self.n_times))


def sample_parameters(
    n: int, seed: int, bounds: np.ndarray = DEFAULT_BOUNDS
) -> np.ndarray:
    """
    Quasi-random logarithms of the parameters of ``n`` bubbles, from a scrambled Sobol sequence.

    Parameters
    ----------
    n : int
        Number of bubbles.
    seed : int
        Seed of the scrambling. Datasets with different seeds are independent.
    bounds : numpy.ndarray, optional
        Bounds of shape ``(N_PARAMETERS, 2)``. Default is :py:data:`DEFAULT_BOUNDS`.

    Returns
    -------
    numpy.ndarray
        Logarithms of the parameters, of shape ``(n, N_PARAMETERS)``.
    """
    sampler = qmc.Sobol(d=N_PARAMETERS, scramble=True, seed=seed)
    return qmc.scale(sampler.random(n), bounds[:, 0], bounds[:, 1])


def bubble_from_log(log_parameters: np.ndarray, delta: float) -> DimensionlessBubble:
    """The :py:class:`~saetass.pinn.wind_bubble.DimensionlessBubble` of a row of :py:func:`sample_parameters`."""
    Pe, ratio, a, b = np.exp(log_parameters)
    return DimensionlessBubble(float(Pe), float(ratio), float(a), float(b), delta)


def zone_indices(x: np.ndarray, a: float, strides: tuple[int, int, int]) -> np.ndarray:
    """Indices of the stored nodes of a reference grid: every stride-th node of each zone, without the outer Dirichlet node."""
    i_shock = int(np.searchsorted(x, 1.0))
    i_edge = int(np.searchsorted(x, a))
    return np.concatenate(
        [
            np.arange(0, i_shock, strides[0]),
            np.arange(i_shock, i_edge, strides[1]),
            np.arange(i_edge, x.size - 1, strides[2]),
        ]
    )


def zone_coordinate(x: np.ndarray, a: float, outer: float) -> np.ndarray:
    r"""
    Coordinate :math:`\zeta \in [0, 3]` of radii ``x``, linear in each zone.

    Parameters
    ----------
    x : numpy.ndarray
        Dimensionless radii.
    a : float or numpy.ndarray
        Radius of the bubble.
    outer : float or numpy.ndarray
        Outer radius.

    Returns
    -------
    numpy.ndarray
        Zone coordinate of ``x``.
    """
    return np.where(
        x < 1.0,
        x,
        np.where(x < a, 1.0 + (x - 1.0) / (a - 1.0), 2.0 + (x - a) / (outer - a)),
    )


def radius_from_zone(zeta: np.ndarray, a: float, outer: float) -> np.ndarray:
    """Inverse of :py:func:`zone_coordinate`."""
    return np.where(
        zeta < 1.0,
        zeta,
        np.where(
            zeta < 2.0, 1.0 + (zeta - 1.0) * (a - 1.0), a + (zeta - 2.0) * (outer - a)
        ),
    )


def simulate(task: tuple[np.ndarray, Sampling]) -> np.ndarray:
    r"""
    Reference solution of one bubble as :math:`w` on the stored nodes, of shape ``(n_times, n_zeta)``.

    Values below :py:data:`RELATIVE_FLOOR` of the largest :math:`\psi` of each time are stored as ``-inf``, which marks points where a surrogate only has to stay below the floor.

    Parameters
    ----------
    task : tuple
        Row of :py:func:`sample_parameters` and :py:class:`Sampling`.

    Returns
    -------
    numpy.ndarray
        Logarithm of the ratio of the solution to the steady state, in single precision.
    """
    log_parameters, sampling = task
    bubble = bubble_from_log(log_parameters, sampling.delta)
    solution = solve(bubble, sampling.times, resolution=sampling.resolution)
    psi = solution.psi
    keep = zone_indices(solution.x, bubble.bubble_ratio, sampling.strides)
    psi_max = np.max(psi, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = np.log(psi[:, keep]) - np.log(solution.steady[keep])
    w[psi[:, keep] < RELATIVE_FLOOR * psi_max] = -np.inf
    return w.astype(np.float32)


def zeta_nodes(sampling: Sampling) -> np.ndarray:
    """The zone coordinates of the stored nodes, the same for every bubble."""
    bubble = DimensionlessBubble(1.0, 10.0, 5.0, 2.0, sampling.delta)
    x = reference_grid(bubble, sampling.resolution)
    keep = zone_indices(x, 5.0, sampling.strides)
    return zone_coordinate(x[keep], 5.0, 10.0)


def build(
    n: int,
    seed: int,
    sampling: Sampling | None = None,
    bounds: np.ndarray = DEFAULT_BOUNDS,
    max_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Build a dataset of ``n`` bubbles.

    Parameters
    ----------
    n : int
        Number of bubbles.
    seed : int
        Seed of :py:func:`sample_parameters`.
    sampling : Sampling, optional
        Times and nodes of the snapshots. Default is ``None``, meaning ``Sampling()``.
    bounds : numpy.ndarray, optional
        Bounds of the parameters. Default is :py:data:`DEFAULT_BOUNDS`.
    max_workers : int, optional
        Number of processes, see :py:func:`~saetass.sweep.run_sweep`. Default is ``None``.

    Returns
    -------
    dict
        Arrays ``parameters`` of shape ``(n, N_PARAMETERS)``, ``log_times``, ``zeta``, and ``w`` of shape ``(n, n_times, n_zeta)``.
    """
    sampling = Sampling() if sampling is None else sampling
    parameters = sample_parameters(n, seed, bounds)
    results = run_sweep(
        simulate, [(row, sampling) for row in parameters], max_workers=max_workers
    )
    return {
        "parameters": parameters,
        "log_times": np.log(sampling.times),
        "zeta": zeta_nodes(sampling),
        "w": np.stack(results),
    }
