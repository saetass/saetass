r"""
Batched and differentiable solver of the radial transport equation, on a CPU or a GPU.

Without energy losses each momentum obeys, with static coefficients and a constant source from an empty domain,

.. math::

    \frac{\partial \psi}{\partial t} = \frac{1}{r^2} \frac{\partial}{\partial r} \left[ r^2 \left( D \frac{\partial \psi}{\partial r} - v \psi \right) \right] + S(r),

with no flux at the centre and :math:`\psi = 0` at the outer radius. This module needs no time steps, no training and no assumed form for :math:`D`, :math:`v` or :math:`S`:

- **Space.** The fluxes are exponentially fitted (Scharfetter-Gummel) as in :py:mod:`saetass.pinn.reference`, which gives a tridiagonal system :math:`d\psi / dt = A \psi + F`, and they define the steady state.
- **Scale.** The unknown is :math:`y = \psi / \psi_\mathrm{ss}`, with the exact steady state of the discrete system. It is computed in logarithms, so that :math:`\psi` keeps its relative accuracy over any number of decades, and its matrix :math:`B = \Psi^{-1} A \Psi` has a diagonal that dominates its rows, since :math:`B \mathbb{1} = -\Psi^{-1} F \le 0`.
- **Time.** The Laplace transform gives :math:`y(t)` as a contour integral (Weideman's Talbot contour), a few tridiagonal solves at complex :math:`s`. At each :math:`s` the fluxes are fitted again to the exponential solutions of the transformed equation inside each cell, so that the fit also holds for the transient, and a front needs no resolved cells: where a fit to the steady state alone becomes first-order upwind, this one does not. The solves are parallel cyclic reduction, which only uses batched tensor operations, so autograd differentiates through them.
- **Control.** The solution satisfies :math:`0 \le y \le 1`, which is checked at every node of every problem where :math:`\psi_\mathrm{ss}` is within thirteen decades of its largest value. The problems that break it are recomputed with implicit time steps (TR-BDF2).

The contour keeps the absolute accuracy of :math:`\psi`, not its relative accuracy, in two regions: ahead of an early front, and where the steady state lies many decades below its maximum. There :math:`y` is only known to lie in :math:`[0, 1]`, which is what the results are clamped to. A front that advection carries much further than diffusion spreads it cannot be represented by a contour at all (:math:`\hat{\psi} \propto e^{-s\tau}` grows on the left of the contour for a delay :math:`\tau > t`), so the problem must have some diffusion across the cells the front has not reached.

The solver is a discretization of the same equation as :py:class:`~saetass.solver.Solver`, which remains the reference: results that matter should be confirmed with it.
"""

import math
from dataclasses import dataclass

import torch

from .reference import _GAMMA, _graded
from .wind_bubble import DELTA, BatchedScales, torch_log_steady_state

#: Tolerance of the check :math:`0 \le y \le 1`.
BOUND_TOLERANCE = 1e-3
#: Logarithm of the steady state over its largest value down to which the check applies.
LOG_RELEVANT = -30.0
#: Smallest :math:`y` that the contour resolves: the front of an early solution lies below it.
SMALLEST_RATIO = 1e-13
#: Logarithm that stands for zero in the fluxes of cells without a source below them.
_LOG_ZERO = -1e5


def log_bernoulli(z: torch.Tensor) -> torch.Tensor:
    r"""Logarithm of the Bernoulli function :math:`z / (e^z - 1)`, for any :math:`z`."""
    size = z.abs()
    small = size < 1e-4
    safe = size.where(~small, torch.ones_like(size))
    exact = -0.5 * (z + safe) + safe.log() - torch.log(-torch.expm1(-safe))
    return exact.where(~small, -0.5 * z - z**2 / 24.0)


def solve_tridiagonal(
    lower: torch.Tensor, main: torch.Tensor, upper: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """
    Solve batched tridiagonal systems by parallel cyclic reduction, in logarithmic depth.

    Parameters
    ----------
    lower, main, upper : torch.Tensor
        Diagonals, of shape ``(..., n)``. ``lower[..., 0]`` and ``upper[..., -1]`` are ignored.
    rhs : torch.Tensor
        Right-hand sides, of shape ``(..., n)``.

    Returns
    -------
    torch.Tensor
        The solutions, of shape ``(..., n)``.
    """
    n = main.shape[-1]
    index = torch.arange(n, device=main.device)
    a = torch.cat([torch.zeros_like(lower[..., :1]), lower[..., 1:]], dim=-1)
    c = torch.cat([upper[..., :-1], torch.zeros_like(upper[..., :1])], dim=-1)
    b, d = main, rhs
    for level in range(math.ceil(math.log2(n))):
        stride = 1 << level
        down = (index - stride).clamp(min=0)
        up = (index + stride).clamp(max=n - 1)
        # Out of range, the neighbour of a row is the row itself, and its weight is zero
        alpha = (-a / b[..., down]).where(index >= stride, torch.zeros_like(a))
        gamma = (-c / b[..., up]).where(index + stride < n, torch.zeros_like(c))
        b = b + alpha * c[..., down] + gamma * a[..., up]
        d = d + alpha * d[..., down] + gamma * d[..., up]
        a = alpha * a[..., down]
        c = gamma * c[..., up]
    return d / b


def talbot_contour(n_nodes: int, device: torch.device | str | None = None):
    r"""
    Nodes of the upper half of Weideman's optimal Talbot contour, for a time of one.

    Parameters
    ----------
    n_nodes : int
        Number of nodes of the whole contour, which needs half as many solves.
    device : torch.device or str, optional
        Device of the nodes.

    Returns
    -------
    tuple of torch.Tensor
        The scaled points :math:`s` and the weights of :math:`\mathrm{Re}\, \hat{y}(s)`.
    """
    theta = 2 * torch.arange(n_nodes // 2, dtype=torch.float64, device=device) + 1
    theta = theta * math.pi / n_nodes
    cotangent = 1.0 / torch.tan(0.6407 * theta)
    z = -0.6122 + 0.5017 * theta * cotangent + 0.2645j * theta
    slope = 0.5017 * (cotangent - 0.6407 * theta / torch.sin(0.6407 * theta) ** 2)
    weight = 2.0 * torch.exp(n_nodes * z) * (slope + 0.2645j) / 1j
    return n_nodes * z, weight


@dataclass(frozen=True)
class ScaledSystem:
    r"""
    Tridiagonal system :math:`dy / dt = B y + f`, with :math:`y = \psi / \psi_\mathrm{ss}`, on the nodes before the outer one.

    Every tensor has the shape ``(..., n)``.

    Parameters
    ----------
    lower, main, upper : torch.Tensor
        Diagonals of :math:`B`.
    forcing : torch.Tensor
        The source :math:`f`.
    log_steady : torch.Tensor
        :math:`\ln \psi_\mathrm{ss}`.
    """

    lower: torch.Tensor
    main: torch.Tensor
    upper: torch.Tensor
    forcing: torch.Tensor
    log_steady: torch.Tensor
    log_resistance: torch.Tensor
    exponent: torch.Tensor
    volumes: torch.Tensor
    capacities: torch.Tensor

    @classmethod
    def from_fluxes(
        cls,
        log_resistance: torch.Tensor,
        exponent: torch.Tensor,
        volumes: torch.Tensor,
        capacities: torch.Tensor,
        forcing: torch.Tensor,
    ) -> "ScaledSystem":
        r"""
        Assemble the system from the fluxes across the ``n`` cells between the nodes.

        The flux from node :math:`i` to node :math:`i + 1` is :math:`[B(-z_i) \psi_i - B(z_i) \psi_{i+1}] / R_i`.

        Parameters
        ----------
        log_resistance : torch.Tensor
            :math:`\ln R_i`, with :math:`R_i = \int dr / (r^2 D)` over the cell.
        exponent : torch.Tensor
            :math:`z_i = \int v / D\, dr` over the cell.
        volumes : torch.Tensor
            Volumes of the control volumes of the nodes, over :math:`4 \pi`.
        capacities : torch.Tensor
            Volumes of the cells, over :math:`4 \pi`.
        forcing : torch.Tensor
            Source :math:`F \ge 0` at the nodes.
        """
        log_forward = log_bernoulli(-exponent) - log_resistance
        log_backward = log_bernoulli(exponent) - log_resistance
        # With cumulative source C_i, forward_i psi_i - backward_i psi_i+1 = C_i, and
        # backward_i / forward_i = exp(-z_i). Then psi_i exp(-E_i), with E the sum of z,
        # is a sum over the outer cells, which logcumsumexp evaluates without overflow
        cumulative = torch.cumsum(forcing * volumes, dim=-1)
        positive = cumulative > 0.0
        log_cumulative = cumulative.where(positive, torch.ones_like(cumulative)).log()
        log_cumulative = log_cumulative.where(
            positive, torch.full_like(cumulative, _LOG_ZERO)
        )
        energy = torch.cumsum(exponent, dim=-1)
        energy = torch.cat(
            [torch.zeros_like(energy[..., :1]), energy[..., :-1]], dim=-1
        )
        terms = log_cumulative - energy - log_forward
        log_steady = torch.logcumsumexp(terms.flip(-1), dim=-1).flip(-1) + energy

        difference = log_steady[..., :-1] - log_steady[..., 1:]
        zero = torch.zeros_like(volumes[..., :1])
        # Where there is no source the steady state can be too small for its inverse
        inverse = (-log_steady).where(forcing > 0.0, torch.zeros_like(forcing)).exp()
        lower = torch.cat([zero, (log_forward[..., :-1] + difference).exp()], dim=-1)
        upper = torch.cat([(log_backward[..., :-1] - difference).exp(), zero], dim=-1)
        inflow = torch.cat([zero, log_backward[..., :-1].exp()], dim=-1)
        return cls(
            lower=lower / volumes,
            main=-(log_forward.exp() + inflow) / volumes,
            upper=upper / volumes,
            forcing=forcing * inverse,
            log_steady=log_steady,
            log_resistance=log_resistance,
            exponent=exponent,
            volumes=volumes,
            capacities=capacities,
        )

    def select(self, mask: torch.Tensor) -> "ScaledSystem":
        """The problems of a boolean mask of the batch."""
        return ScaledSystem(
            **{name: getattr(self, name)[mask] for name in self.__dataclass_fields__}
        )

    def _fitted(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Diagonals of the transformed problem at ``s``, in the variable :math:`y`, with the cell problem solved exactly.

        Inside a cell with constant coefficients, :math:`D \hat{\psi}'' - v \hat{\psi}' - s \hat{\psi} = 0` has exponential solutions. The flux of the cell in terms of its two nodes follows from them, with :math:`q^2 = z^2 / 4 + s V R`, and reduces to the fluxes of :py:meth:`from_fluxes` at :math:`s = 0`. Unlike those it is also fitted to the transient, so it does not add numerical diffusion to a front when the cells do not resolve it.
        """
        z = self.exponent.to(s.dtype)
        half = 0.5 * z
        stiffness = s * self.capacities * self.log_resistance.exp()
        q = (half**2 + stiffness).sqrt()

        small = q.abs() < 1e-3
        safe = q.where(~small, torch.ones_like(q))
        decay = torch.exp(-2.0 * safe)
        # q coth q = q + excess, and log(q / sinh q), with their series near zero
        excess = (2.0 * safe * decay / (1.0 - decay)).where(
            ~small, 1.0 - q + q**2 / 3.0
        )
        log_ratio = ((2.0 * safe).log() - safe - (1.0 - decay).log()).where(
            ~small, -(q**2) / 6.0
        )
        # q + z / 2 and q - z / 2 without the loss of digits of a difference of large terms
        positive = self.exponent >= 0.0
        plus = (q + half).where(positive, stiffness / (q - half))
        minus = (q - half).where(~positive, stiffness / (q + half))

        conductance = (-self.log_resistance).exp()
        leaving = (plus + excess) * conductance
        entering = (minus + excess) * conductance
        difference = self.log_steady[..., :-1] - self.log_steady[..., 1:]
        log_out = log_ratio + half - self.log_resistance
        log_in = log_ratio - half - self.log_resistance
        zero = torch.zeros_like(z[..., :1])
        lower = (log_out[..., :-1] + difference).exp()
        upper = (log_in[..., :-1] - difference).exp()
        main = leaving + torch.cat([zero, entering[..., :-1]], dim=-1)
        return (
            torch.cat([zero, -lower], dim=-1),
            main,
            torch.cat([-upper, zero], dim=-1),
        )

    def contour(self, t: torch.Tensor, n_contour: int) -> torch.Tensor:
        r"""
        :math:`y(t)` from the Laplace inversion on a contour of ``n_contour`` nodes.

        Parameters
        ----------
        t : torch.Tensor
            Positive times, of the shape of the batch.
        n_contour : int
            Number of nodes of the contour.
        """
        points, weights = talbot_contour(n_contour, self.main.device)
        rate = 1.0 / t[..., None]
        total = self.forcing * self.volumes
        y = torch.zeros_like(self.main)
        for point, weight in zip(points, weights):
            s = point * rate
            solution = solve_tridiagonal(*self._fitted(s), total.to(s.dtype) / s)
            y = y + (weight * rate * solution).real
        return y

    def stepped(self, t: torch.Tensor, n_steps: int = 400) -> torch.Tensor:
        r"""
        :math:`y(t)` from TR-BDF2 steps that grow geometrically from :math:`t = 0`, L-stable and without oscillations.

        Parameters
        ----------
        t : torch.Tensor
            Positive times, of the shape of the batch.
        n_steps : int, optional
            Number of steps. Default is ``400``.
        """

        def solve(shift: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return solve_tridiagonal(
                -shift * self.lower, 1.0 - shift * self.main, -shift * self.upper, rhs
            )

        def apply(y: torch.Tensor) -> torch.Tensor:
            out = self.main * y + self.forcing
            shifted = y.roll(1, dims=-1)
            out = out + self.lower * shifted
            return out + self.upper * y.roll(-1, dims=-1)

        grid = torch.expm1(
            14.0 * torch.linspace(0.0, 1.0, n_steps + 1, device=t.device, dtype=t.dtype)
        ) / math.expm1(14.0)
        y = torch.zeros_like(self.main)
        weight = 1.0 / (_GAMMA * (2.0 - _GAMMA))
        for begin, end in zip(grid[:-1], grid[1:]):
            step = (end - begin) * t[..., None]
            shift = 0.5 * _GAMMA * step
            stage = solve(shift, y + shift * (apply(y) + self.forcing))
            shift = (1.0 - _GAMMA) / (2.0 - _GAMMA) * step
            rhs = (
                weight * stage - (1.0 - _GAMMA) ** 2 * weight * y + shift * self.forcing
            )
            y = solve(shift, rhs)
        return y

    def ratio(
        self, t: torch.Tensor, n_contour: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        :math:`y(t)` from the contour, with the problems that break :math:`0 \le y \le 1` recomputed with time steps.

        Parameters
        ----------
        t : torch.Tensor
            Positive times, of the shape of the batch.
        n_contour : int
            Number of nodes of the contour.

        Returns
        -------
        tuple of torch.Tensor
            :math:`y`, and which problems were recomputed.
        """
        y = self.contour(t, n_contour)
        with torch.no_grad():
            # Far below the largest steady state, the contour keeps the absolute accuracy
            # of psi, not the relative one, which is no sign of a failure
            top = self.log_steady.amax(dim=-1, keepdim=True)
            relevant = self.log_steady > top + LOG_RELEVANT
            excess = ((y - 1.0).abs() - 1.0).where(relevant, torch.zeros_like(y))
            broken = ~torch.isfinite(excess).all(dim=-1)
            broken |= excess.amax(dim=-1) > BOUND_TOLERANCE
        if broken.any():
            y = y.clone()
            y[broken] = self.select(broken).stepped(t.expand(broken.shape)[broken])
        return y, broken


def control_volumes(nodes: torch.Tensor) -> torch.Tensor:
    """Control volumes over 4 pi of the nodes before the outer one, between the midpoints of the cells."""
    edges = 0.5 * (nodes[..., :-1] + nodes[..., 1:])
    edges = torch.cat([torch.zeros_like(nodes[..., :1]), edges], dim=-1)
    edges = torch.cat([edges, nodes[..., -1:]], dim=-1)
    return ((edges[..., 1:] ** 3 - edges[..., :-1] ** 3) / 3.0)[..., :-1]


class RadialSolver:
    r"""
    Solver for any :math:`D(r)`, :math:`v(r)` and :math:`S(r)` on a fixed grid, batched and differentiable.

    The coefficients enter by cell, so that a jump on a node needs no averaging, and they take any shape: the solver has no family of problems.

    Parameters
    ----------
    nodes : torch.Tensor
        Radii of the nodes, in double precision, of shape ``(n_nodes,)``. They increase from zero, and the last one is the outer radius, where :math:`\psi` vanishes.
    n_contour : int, optional
        Number of nodes of the contour. Default is ``48``.
    """

    def __init__(self, nodes: torch.Tensor, n_contour: int = 48) -> None:
        if nodes.dtype != torch.float64:
            raise ValueError("The solver needs nodes in double precision.")
        if nodes[0] != 0.0 or torch.any(nodes[1:] <= nodes[:-1]):
            raise ValueError("The nodes must increase from zero.")
        self.nodes = nodes
        self.n_contour = n_contour
        self.widths = nodes[1:] - nodes[:-1]
        self.centers = 0.5 * (nodes[:-1] + nodes[1:])
        self.volumes = control_volumes(nodes)
        self.capacities = (nodes[1:] ** 3 - nodes[:-1] ** 3) / 3.0

    def shell(self, radius: torch.Tensor) -> torch.Tensor:
        r"""
        Source of one particle per unit time and per :math:`4 \pi` at a shell, shared by its two nodes.

        Parameters
        ----------
        radius : torch.Tensor
            Radii of the shells, of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            The source at the nodes before the outer one, of shape ``(batch, n_nodes - 1)``.
        """
        cell = torch.searchsorted(self.nodes, radius).clamp(1, self.nodes.numel() - 2)
        cell = cell - 1
        weight = (radius - self.nodes[cell]) / self.widths[cell]
        source = torch.zeros(radius.numel(), self.volumes.numel()).to(radius)
        source = source.scatter(-1, cell[:, None], (1.0 - weight)[:, None])
        source = source.scatter_add(-1, cell[:, None] + 1, weight[:, None])
        return source / self.volumes

    def system(
        self, D: torch.Tensor, v: torch.Tensor, source: torch.Tensor
    ) -> ScaledSystem:
        """
        The discrete problem of some coefficients.

        Parameters
        ----------
        D, v : torch.Tensor
            Diffusion coefficient and velocity in each cell, of shape ``(..., n_nodes - 1)``, in units of the nodes and of the time.
        source : torch.Tensor
            Non-negative source at the nodes before the outer one, of shape ``(..., n_nodes - 1)``, with some particles.

        Returns
        -------
        ScaledSystem
            The system.
        """
        log_resistance = (self.widths / (self.centers**2 * D)).log()
        return ScaledSystem.from_fluxes(
            log_resistance,
            v * self.widths / D,
            self.volumes.expand_as(source),
            self.capacities.expand_as(source),
            source,
        )

    def log_psi(
        self,
        D: torch.Tensor,
        v: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        :math:`\ln \psi` at the nodes before the outer one, from an empty domain.

        Parameters
        ----------
        D, v, source : torch.Tensor
            See :py:meth:`system`.
        t : torch.Tensor
            Positive times, of the shape of the batch.

        Returns
        -------
        torch.Tensor
            :math:`\ln \psi`, of shape ``(..., n_nodes - 1)``.
        """
        system = self.system(D, v, source)
        y, _ = system.ratio(t, self.n_contour)
        return y.clamp(SMALLEST_RATIO, 1.0).log() + system.log_steady

    def interpolate(self, log_psi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r"""
        Linear interpolation of :math:`\ln \psi` to any radii inside the last node.

        Parameters
        ----------
        log_psi : torch.Tensor
            Values at the nodes before the outer one, of shape ``(..., n_nodes - 1)``.
        r : torch.Tensor
            Radii, of shape ``(..., m)``.

        Returns
        -------
        torch.Tensor
            Values at ``r``, of shape ``(..., m)``.
        """
        return _interpolate(self.nodes[:-1], log_psi, r)


def _interpolate(
    nodes: torch.Tensor, values: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation over the last axis, with increasing ``nodes`` of shape ``(..., n)`` or ``(n,)``."""
    nodes = nodes.expand(*values.shape[:-1], -1).contiguous()
    upper = torch.searchsorted(nodes, x.contiguous()).clamp(1, nodes.shape[-1] - 1)
    left = nodes.gather(-1, upper - 1)
    right = nodes.gather(-1, upper)
    start = values.gather(-1, upper - 1)
    return start + (x - left) / (right - left) * (values.gather(-1, upper) - start)


class WindBubbleSolver:
    r"""
    Solver of the wind bubble, with the same interface as :py:class:`~saetass.pinn.surrogate.WindBubbleSurrogate` for parameter searches.

    Each bubble has its own grid, with nodes on the shock and on the edge of the bubble and the integrals of the fluxes in closed form, so that the steady state of the grid is the exact one at any resolution. Radii between the nodes are evaluated from the closed form of :math:`\psi_\mathrm{ss}` and the interpolation of :math:`\ln(\psi / \psi_\mathrm{ss})`, which is smooth. It needs no training, and its domain is not limited: it does not refuse any bubble.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes of each bubble. Default is ``512``.
    n_contour : int, optional
        Number of nodes of the contour. Default is ``48``, and ``24`` keeps a parameter search accurate in half the time, but it leaves errors of order ``1e-6`` of the largest :math:`\psi`.
    delta : float, optional
        Exponent of the diffusion coefficient in the wind. Default is that of Kolmogorov turbulence.
    device : torch.device or str, optional
        Device of the computation.
    """

    def __init__(
        self,
        n_nodes: int = 512,
        n_contour: int = 48,
        delta: float = DELTA["kolmogorov"],
        device: torch.device | str = "cpu",
    ) -> None:
        if delta >= 1.0:
            raise ValueError("The wind needs an exponent below one.")
        self.n_contour = n_contour
        self.delta = delta
        self.device = torch.device(device)
        # The shares and the grading of the grid of the reference solutions
        n_wind, n_inner = round(3 * n_nodes / 14), round(7 * n_nodes / 14)
        n_outer = n_nodes - 1 - n_wind - n_inner

        def fractions(n: int, left: float, right: float) -> torch.Tensor:
            grid = _graded(0.0, 1.0, n, left, right)
            return torch.as_tensor(grid, dtype=torch.float64, device=self.device)

        self._wind = fractions(n_wind, 1.0, 400.0)
        self._inner = fractions(n_inner, 200.0, 400.0)
        self._outer = fractions(n_outer, 100.0, 1.0)

    def nodes(self, bubble_ratio: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
        """Nodes of the bubbles of shape ``(batch,)``, of shape ``(batch, n_nodes)``."""
        a, outer = bubble_ratio[..., None], outer[..., None]
        inner = 1.0 + (a - 1.0) * self._inner[1:]
        beyond = a + (outer - a) * self._outer[1:]
        wind = self._wind.expand(*a.shape[:-1], -1)
        return torch.cat([wind, inner, beyond], dim=-1)

    def system(
        self,
        peclet: torch.Tensor,
        ratio: torch.Tensor,
        bubble_ratio: torch.Tensor,
        outer: torch.Tensor,
    ) -> tuple[ScaledSystem, torch.Tensor]:
        """
        Discrete problems of the bubbles and their nodes, for fields of shape ``(batch,)``.
        """
        x = self.nodes(bubble_ratio, outer)
        left, right = x[..., :-1], x[..., 1:]
        n_wind = self._wind.numel() - 1
        n_inner = self._inner.numel() - 1
        Pe, lam = peclet[..., None], ratio[..., None]

        # The fields of the reference grid, with the integrals of the fluxes in closed form
        mid = 0.5 * (left + right)[..., :n_wind]
        edges = x[..., : n_wind + 1]
        scale = 4.0 * Pe / 11.0 ** (self.delta / 2.0) / (1.0 - self.delta)
        drops = scale * (
            edges[..., 1:] ** (1.0 - self.delta) - edges[..., :-1] ** (1.0 - self.delta)
        )
        wind_resistance = (
            (right - left)[..., :n_wind]
            * Pe
            / (11.0 ** (self.delta / 2.0) * mid ** (2.0 + self.delta))
        )
        inverse = 1.0 / left[..., n_wind:] - 1.0 / right[..., n_wind:]
        inner = Pe * inverse[..., :n_inner]
        beyond = (Pe / lam) * inverse[..., n_inner:]
        resistance = torch.cat([wind_resistance, inner, beyond], dim=-1)
        exponent = torch.cat([drops, inner, torch.zeros_like(beyond)], dim=-1)

        volumes = control_volumes(x)
        source = torch.zeros_like(volumes)
        source[..., n_wind] = 1.0 / volumes[..., n_wind]
        capacities = (right**3 - left**3) / 3.0
        system = ScaledSystem.from_fluxes(
            resistance.log(), exponent, volumes, capacities, source
        )
        return system, x

    def log_ratio(
        self, x: torch.Tensor, times: torch.Tensor, scales: BatchedScales
    ) -> torch.Tensor:
        r"""
        :math:`\ln(\psi / \psi_\mathrm{ss})`, for bubbles of shape ``(...,)`` and radii in the last axis of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Dimensionless radii, of shape ``(..., m)``.
        times : torch.Tensor
            Dimensionless times, of shape ``(...,)``.
        scales : BatchedScales
            Scales of the bubbles, as tensors of shape ``(...,)``.

        Returns
        -------
        torch.Tensor
            The ratio at ``x``, of shape ``(..., m)``.
        """
        batch = x.shape[:-1]
        fields = (
            scales.peclet,
            scales.diffusion_ratio,
            scales.bubble_ratio,
            scales.bubble_ratio * scales.domain_ratio,
        )
        Pe, ratio, a, outer = (field.expand(batch).reshape(-1) for field in fields)
        system, nodes = self.system(Pe, ratio, a, outer)
        y, _ = system.ratio(times.expand(batch).reshape(-1), self.n_contour)
        log_y = y.clamp(SMALLEST_RATIO, 1.0).log()
        # The ratio is finite at the outer boundary, where both functions vanish
        log_y = torch.cat([log_y, log_y[..., -1:]], dim=-1)
        flat = x.reshape(-1, x.shape[-1])
        return _interpolate(nodes, log_y, flat).reshape(x.shape)

    def log_psi(
        self, x: torch.Tensor, times: torch.Tensor, scales: BatchedScales
    ) -> torch.Tensor:
        r"""
        Logarithm of the physical :math:`\psi`, differentiable and batched, for parameter searches.

        The bubbles are the leading axes of the fields of ``scales`` and of ``times``, which end in an axis of length one, as they do in the calls to :py:meth:`~saetass.pinn.surrogate.WindBubbleSurrogate.log_psi`, and the last axis of ``x`` holds the radii of each bubble.

        Parameters
        ----------
        x : torch.Tensor
            Dimensionless radii, of shape ``(..., m)``.
        times : torch.Tensor
            Dimensionless times, of shape ``(..., 1)``.
        scales : BatchedScales
            Scales of the bubbles, as tensors of shape ``(..., 1)``.

        Returns
        -------
        torch.Tensor
            :math:`\ln \psi` in canonical :py:data:`~saetass.units.PSI_P` units, of shape ``(..., m)``.
        """
        batch = torch.broadcast_shapes(x.shape, scales.peclet.shape)
        x = x.expand(batch)
        bubble = BatchedScales(
            **{
                name: getattr(scales, name).expand(*batch[:-1], 1)[..., 0]
                for name in scales.__dataclass_fields__
            }
        )
        w = self.log_ratio(x, times.expand(*batch[:-1], 1)[..., 0], bubble)
        a = scales.bubble_ratio
        return (
            w
            + torch_log_steady_state(
                x,
                scales.peclet,
                scales.diffusion_ratio,
                a,
                a * scales.domain_ratio,
                self.delta,
            )
            + scales.psi_unit.log()
        )

    def in_domain(self, scales: BatchedScales, times: torch.Tensor) -> torch.Tensor:
        """Every bubble is inside the domain of the solver: the method exists for the interface of the surrogates."""
        shape = torch.broadcast_shapes(scales.peclet.shape, times.shape)
        return torch.ones(shape, dtype=torch.bool, device=times.device)
