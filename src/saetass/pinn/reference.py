r"""
Reference solutions of the dimensionless wind bubble, for training and testing surrogates.

The surrogates need thousands of converged solutions, and the shock layers of a bubble need very small cells: :math:`D / v` is a fraction of a parsec at low energies.
This module solves the same equation as :py:class:`~saetass.solver.Solver` with two choices that make those solutions cheap:

- **Exponentially fitted fluxes** (Scharfetter-Gummel). Between two nodes the flux reproduces the exact steady state, so that layers of many decades need no special resolution, and the matrix is an M-matrix, positive in the semi-discrete sense. The trapezoidal stage of the time step can still leave negative values, about :math:`10^{-12}` of the largest one, far below the floor of the datasets.
- **Fully implicit time steps** (TR-BDF2, which is L-stable and second order), growing geometrically, with no CFL limit on the steps.

The nodes lie exactly on the shock and on the edge of the bubble, where :math:`\psi` is continuous, and the source enters at the node of the shock, so that it acts as a jump of the flux.

:py:func:`solve` is a tool to generate data and it is not a replacement for :py:class:`~saetass.solver.Solver`, which remains the reference that the surrogates are validated against.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .wind_bubble import DimensionlessBubble, steady_state, wind_exponent

#: Second-order, L-stable parameter of TR-BDF2.
_GAMMA = 2.0 - np.sqrt(2.0)


@dataclass(frozen=True)
class ReferenceSolution:
    r"""
    Dimensionless solution on the nodes of the reference grid.

    Parameters
    ----------
    x : numpy.ndarray
        Radii of the nodes, of shape ``(n_x,)``.
    times : numpy.ndarray
        Times of the snapshots, of shape ``(n_t,)``.
    psi : numpy.ndarray
        Dimensionless :math:`\psi`, of shape ``(n_t, n_x)``.
    steady : numpy.ndarray
        Exact steady state on the nodes, of shape ``(n_x,)``.
    i_shock : int
        Index of the node at the termination shock.
    i_edge : int
        Index of the node at the edge of the bubble.
    """

    x: np.ndarray
    times: np.ndarray
    psi: np.ndarray
    steady: np.ndarray
    i_shock: int
    i_edge: int


def _graded(start: float, end: float, n: int, left: float, right: float) -> np.ndarray:
    """``n + 1`` nodes in ``[start, end]``, with cells about ``left`` and ``right`` times smaller than uniform ones at each end."""

    def distance(strength: float, w: np.ndarray) -> np.ndarray:
        # Fraction of the half covered at a fraction w of its cells, from its end
        if strength <= 1.0:
            return w
        beta = np.arcsinh(strength)
        return np.sinh(beta * w) / strength

    u = np.linspace(0.0, 1.0, n + 1)
    fraction = np.where(
        u < 0.5,
        0.5 * distance(left, 2.0 * u),
        1.0 - 0.5 * distance(right, 2.0 * (1.0 - u)),
    )
    nodes = start + (end - start) * fraction
    nodes[0], nodes[-1] = start, end
    return nodes


def reference_grid(bubble: DimensionlessBubble, resolution: int = 1) -> np.ndarray:
    """
    Nodes of the reference grid, with nodes at the shock and at the edge of the bubble.

    Parameters
    ----------
    bubble : DimensionlessBubble
        Dimensionless bubble.
    resolution : int, optional
        Multiplier of the number of nodes. Default is ``1``.

    Returns
    -------
    numpy.ndarray
        The nodes, increasing from zero to the outer radius.
    """
    wind = _graded(0.0, 1.0, 300 * resolution, 1.0, 400.0)
    inner = _graded(1.0, float(bubble.bubble_ratio), 700 * resolution, 200.0, 400.0)
    outer = _graded(
        float(bubble.bubble_ratio),
        float(bubble.outer_radius),
        400 * resolution,
        100.0,
        1.0,
    )
    return np.concatenate([wind, inner[1:], outer[1:]])


def _bernoulli(z: np.ndarray) -> np.ndarray:
    r"""The Bernoulli function :math:`z / (e^z - 1)`, stable near zero and for large :math:`|z|`."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-4
    safe = np.where(small, 1.0, z)
    with np.errstate(over="ignore", invalid="ignore"):
        exact = safe / np.expm1(np.minimum(safe, 700.0))
    return np.where(small, 1.0 - 0.5 * z + z**2 / 12.0, exact)


def _operator(x: np.ndarray, bubble: DimensionlessBubble, i_shock: int, i_edge: int):
    r"""
    Tridiagonal system :math:`V\, d\psi / dt = A \psi + e_{TS}`, on the nodes before the Dirichlet one.

    Returns the lower, main and upper diagonals of :math:`V^{-1} A`, and :math:`V^{-1}` at the shock.
    """
    Pe, ratio = float(bubble.peclet), float(bubble.diffusion_ratio)
    left, right = x[:-1], x[1:]
    h = right - left
    mid = 0.5 * (left + right)

    # Integral of dx / (x^2 D) and exponent of the steady state over each element, exact
    # in the bubble and in the medium, so that the flux has their steady state as a null
    # vector; the wind uses the midpoint weight with the exact exponent, whose null vector
    # is its steady state, of zero flux
    resistance = np.empty_like(h)
    exponent = np.empty_like(h)
    wind = np.arange(h.size) < i_shock
    inner = (np.arange(h.size) >= i_shock) & (np.arange(h.size) < i_edge)
    outer = np.arange(h.size) >= i_edge
    wind_D = bubble.wind_diffusion * mid[wind] ** bubble.delta
    resistance[wind] = h[wind] / (mid[wind] ** 2 * wind_D)
    exponent[wind] = wind_exponent(left[wind], bubble) - wind_exponent(
        right[wind], bubble
    )
    inverse = 1.0 / left[inner] - 1.0 / right[inner]
    resistance[inner] = Pe * inverse
    exponent[inner] = Pe * inverse
    resistance[outer] = (Pe / ratio) * (1.0 / left[outer] - 1.0 / right[outer])
    exponent[outer] = 0.0

    # Flux from node i to node i + 1: (B(-z) psi_i - B(z) psi_i+1) / resistance
    forward = _bernoulli(-exponent) / resistance
    backward = _bernoulli(exponent) / resistance

    edges = np.concatenate([[0.0], mid, [x[-1]]])
    volumes = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    n = x.size - 1
    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    main[:-1] -= forward[:-1]
    upper[:-1] += backward[:-1]
    lower[1:] += forward[:-1]
    main[1:] -= backward[:-1]
    main[-1] -= forward[-1]
    return (
        lower / volumes[:n],
        main / volumes[:n],
        upper / volumes[:n],
        1.0 / volumes[i_shock],
    )


def _step(lower, main, upper, shift: float, rhs: np.ndarray) -> np.ndarray:
    """Solve ``(I - shift * A) y = rhs`` for the tridiagonal ``A``."""
    bands = np.zeros((3, main.size))
    bands[0, 1:] = -shift * upper[:-1]
    bands[1] = 1.0 - shift * main
    bands[2, :-1] = -shift * lower[1:]
    return solve_banded(
        (1, 1), bands, rhs, overwrite_ab=True, overwrite_b=False, check_finite=False
    )


def solve(
    bubble: DimensionlessBubble,
    times: np.ndarray,
    resolution: int = 1,
    steps_per_decade: int = 60,
    first_step: float = 1e-5,
) -> ReferenceSolution:
    """
    Solve the dimensionless wind bubble from an empty domain.

    Parameters
    ----------
    bubble : DimensionlessBubble
        Dimensionless bubble, with scalar fields.
    times : numpy.ndarray
        Increasing positive times of the snapshots.
    resolution : int, optional
        Multiplier of the number of nodes, and of the time steps per decade. Default is ``1``.
    steps_per_decade : int, optional
        Time steps per decade of time. Default is ``60``.
    first_step : float, optional
        Length of the first time step. Default is ``1e-5``.

    Returns
    -------
    ReferenceSolution
        Solution at ``times``.
    """
    x = reference_grid(bubble, resolution)
    i_shock = int(np.searchsorted(x, 1.0))
    i_edge = int(np.searchsorted(x, float(bubble.bubble_ratio)))
    lower, main, upper, source = _operator(x, bubble, i_shock, i_edge)

    times = np.asarray(times, dtype=float)
    growth = 10.0 ** (1.0 / (steps_per_decade * resolution))
    first = first_step / resolution**2
    psi = np.zeros(x.size - 1)
    forcing = np.zeros_like(psi)
    forcing[i_shock] = source
    snapshots = np.zeros((times.size, x.size))

    def apply(y: np.ndarray) -> np.ndarray:
        out = main * y + forcing
        out[1:] += lower[1:] * y[:-1]
        out[:-1] += upper[:-1] * y[1:]
        return out

    t, dt = 0.0, first
    for index, target in enumerate(times):
        while t < target * (1.0 - 1e-13):
            # Land exactly on the snapshot
            step = min(dt, target - t)
            shift = 0.5 * _GAMMA * step
            rhs = psi + shift * (apply(psi) + forcing)
            stage = _step(lower, main, upper, shift, rhs)
            weight = 1.0 / (_GAMMA * (2.0 - _GAMMA))
            rhs = (
                weight * stage
                - ((1.0 - _GAMMA) ** 2 * weight) * psi
                + (1.0 - _GAMMA) / (2.0 - _GAMMA) * step * forcing
            )
            psi = _step(lower, main, upper, (1.0 - _GAMMA) / (2.0 - _GAMMA) * step, rhs)
            t += step
            if step == dt:
                dt *= growth
        snapshots[index, :-1] = psi

    return ReferenceSolution(
        x=x,
        times=times,
        psi=snapshots,
        steady=steady_state(x, bubble),
        i_shock=i_shock,
        i_edge=i_edge,
    )
