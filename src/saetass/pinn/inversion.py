r"""
Recovery of a diffusion coefficient :math:`D(r)` from :math:`\psi`, with no assumed form.

The unknown is :math:`\ln D` at uniformly spaced knots, placed with no knowledge of the interfaces of the problem, and piecewise linear between them. The gradient of the misfit comes from autograd through :py:class:`~saetass.pinn.radial_solver.RadialSolver`. Each estimate minimizes

.. math::

    \chi^2(u) + \lambda \sum_k \sqrt{(u_{k+1} - u_k)^2 + \epsilon},

a smoothed total variation, which allows jumps without saying where they lie. The uncertainty comes from several estimates, each with new noise added to the data (randomized maximum a posteriori).

The bands are not a calibrated posterior: the regularization biases the estimates in a way that the spread does not include, above all where the data say little about :math:`D`. They should be checked with mock data before reading them as confidence intervals.
"""

from dataclasses import dataclass

import torch

from .radial_solver import RadialSolver


@dataclass(frozen=True)
class DiffusionEstimate:
    r"""
    Result of :py:meth:`DiffusionInversion.run`.

    Parameters
    ----------
    knots : torch.Tensor
        Radii of the knots, of shape ``(n_knots,)``.
    samples : torch.Tensor
        Estimates of :math:`\ln D` at the knots, of shape ``(n_realizations, n_knots)``.
    chi2 : torch.Tensor
        :math:`\chi^2` per data point of each estimate against its noisy data.
    """

    knots: torch.Tensor
    samples: torch.Tensor
    chi2: torch.Tensor

    @property
    def median(self) -> torch.Tensor:
        r"""Median of :math:`\ln D` at the knots."""
        return self.samples.quantile(0.5, dim=0)

    def band(self, level: float = 0.68) -> tuple[torch.Tensor, torch.Tensor]:
        """Central interval of the estimates that holds a fraction ``level`` of them."""
        return (
            self.samples.quantile(0.5 * (1.0 - level), dim=0),
            self.samples.quantile(0.5 * (1.0 + level), dim=0),
        )


class DiffusionInversion:
    r"""
    Inverse problem for :math:`D(r)` from :math:`\ln \psi` at several radii and channels.

    A channel is an energy bin or a time: it has its own source and time, and its coefficient is :math:`D(r)` times a known factor.

    Parameters
    ----------
    solver : RadialSolver
        Forward model.
    velocity : torch.Tensor
        Velocity in each cell, known, of shape ``(n_channels, n_nodes - 1)``.
    source : torch.Tensor
        Source at the nodes, known, of shape ``(n_channels, n_nodes - 1)``.
    times : torch.Tensor
        Time of each channel, of shape ``(n_channels,)``.
    radii : torch.Tensor
        Radii of the data, of shape ``(n_radii,)``.
    log_data : torch.Tensor
        Observed :math:`\ln \psi`, of shape ``(n_channels, n_radii)``.
    noise : float
        Standard deviation of ``log_data``.
    factors : torch.Tensor, optional
        Known factor of the coefficient of each channel, of shape ``(n_channels,)``. Default is one.
    n_knots : int, optional
        Number of knots. Default is ``160``.
    regularization : float, optional
        The weight :math:`\lambda`. Default is ``3.0``.
    """

    def __init__(
        self,
        solver: RadialSolver,
        velocity: torch.Tensor,
        source: torch.Tensor,
        times: torch.Tensor,
        radii: torch.Tensor,
        log_data: torch.Tensor,
        noise: float,
        factors: torch.Tensor | None = None,
        n_knots: int = 160,
        regularization: float = 3.0,
    ) -> None:
        self.solver = solver
        self.velocity = velocity
        self.source = source
        self.times = times
        self.radii = radii
        self.log_data = log_data
        self.noise = noise
        self.factors = torch.ones_like(times) if factors is None else factors
        self.regularization = regularization
        centers = solver.centers
        self.knots = torch.linspace(0.0, 1.0, n_knots).to(centers) * solver.nodes[-1]
        place = centers / solver.nodes[-1] * (n_knots - 1)
        self._cell = place.floor().long().clamp(0, n_knots - 2)
        self._weight = place - self._cell

    def log_diffusion(self, u: torch.Tensor) -> torch.Tensor:
        r""":math:`\ln D` in the cells of the solver, from its values ``u`` at the knots, of shape ``(..., n_knots)``."""
        return (
            u[..., self._cell] * (1.0 - self._weight)
            + u[..., self._cell + 1] * self._weight
        )

    def log_psi(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        :math:`\ln \psi` at the radii of the data.

        Parameters
        ----------
        u : torch.Tensor
            :math:`\ln D` at the knots, of shape ``(batch, n_knots)``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(batch, n_channels, n_radii)``.
        """
        batch, channels = u.shape[0], self.times.numel()
        cells = self.log_diffusion(u)[:, None] + self.factors.log()[None, :, None]
        D = cells.exp().reshape(batch * channels, -1)
        values = self.solver.log_psi(
            D,
            self.velocity.repeat(batch, 1),
            self.source.repeat(batch, 1),
            self.times.repeat(batch),
        )
        radii = self.radii.expand(batch * channels, -1)
        return self.solver.interpolate(values, radii).reshape(batch, channels, -1)

    def objective(
        self, u: torch.Tensor, log_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        :math:`\chi^2` and the total variation of each estimate.

        Parameters
        ----------
        u : torch.Tensor
            :math:`\ln D` at the knots, of shape ``(batch, n_knots)``.
        log_data : torch.Tensor
            Data of each estimate, of shape ``(batch, n_channels, n_radii)``.
        """
        chi2 = (((self.log_psi(u) - log_data) / self.noise) ** 2).sum(dim=(1, 2))
        steps = u[:, 1:] - u[:, :-1]
        return chi2, (steps**2 + 1e-3).sqrt().sum(dim=1)

    def run(
        self,
        start: float,
        n_realizations: int = 32,
        n_steps: int = 2400,
        learning_rate: float = 0.08,
        seed: int = 0,
    ) -> DiffusionEstimate:
        r"""
        Compute the estimates with Adam.

        Parameters
        ----------
        start : float
            Flat initial value of :math:`\ln D`.
        n_realizations : int, optional
            Number of estimates, each with new noise. Default is ``32``.
        n_steps : int, optional
            Number of iterations. Default is ``2400``.
        learning_rate : float, optional
            Initial learning rate, which follows a cosine. Default is ``0.08``.
        seed : int, optional
            Seed of the noise. Default is ``0``.

        Returns
        -------
        DiffusionEstimate
            The estimates.
        """
        generator = torch.Generator(device=self.log_data.device).manual_seed(seed)
        shape = (n_realizations, *self.log_data.shape)
        log_data = self.log_data + self.noise * torch.randn(
            shape,
            generator=generator,
            device=generator.device,
            dtype=self.log_data.dtype,
        )
        u = self.knots.new_full((n_realizations, self.knots.numel()), start)
        u.requires_grad_(True)
        optimizer = torch.optim.Adam([u], learning_rate)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, 1e-3)
        for _ in range(n_steps):
            chi2, variation = self.objective(u, log_data)
            optimizer.zero_grad()
            (chi2 + self.regularization * variation).sum().backward()
            optimizer.step()
            schedule.step()
        with torch.no_grad():
            chi2, _ = self.objective(u, log_data)
        return DiffusionEstimate(
            knots=self.knots, samples=u.detach(), chi2=chi2 / self.log_data.numel()
        )
