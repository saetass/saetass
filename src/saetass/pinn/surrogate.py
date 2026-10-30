r"""
Public interface of the wind bubble surrogate.

:py:class:`WindBubbleSurrogate` evaluates a trained network at any radii, momenta and times, without a grid:

.. math::

    \psi(t, r, p) = \psi_\mathrm{unit}(p)\; \psi_\mathrm{ss}(x; \theta)\; e^{w(t, \zeta; \theta)},

where the steady state is the closed form of :py:func:`~saetass.pinn.wind_bubble.steady_state` and the network only provides :math:`w`.
Inputs outside the domain the network was trained on raise a :py:class:`ValueError` instead of extrapolating, and :py:meth:`WindBubbleSurrogate.residual` reports how well a prediction satisfies the transport equation, which needs no simulation.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import torch

from .. import units as su
from .dataset import zone_coordinate
from .network import BubbleNetwork
from .training import physics_residual
from .wind_bubble import (
    BatchedScales,
    DimensionlessBubble,
    PhysicalBubble,
    log_steady_state,
    physical_bubble,
    torch_log_steady_state,
)


class WindBubbleSurrogate:
    r"""
    A trained surrogate of the wind bubble without energy losses.

    Parameters
    ----------
    model : BubbleNetwork
        Trained network.
    delta : float
        Exponent of the diffusion coefficient in the wind that the network was trained for.
    device : torch.device or str, optional
        Device of the evaluations. Default is ``'cpu'``.
    """

    def __init__(
        self,
        model: BubbleNetwork,
        delta: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.delta = delta
        self.bounds = model.bounds.detach().cpu().numpy().astype(float)
        self.time_bounds = model.time_bounds.detach().cpu().numpy().astype(float)

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str = "cpu"
    ) -> "WindBubbleSurrogate":
        """
        Load a surrogate saved by :py:meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            File of the surrogate.
        device : torch.device or str, optional
            Device of the evaluations. Default is ``'cpu'``.

        Returns
        -------
        WindBubbleSurrogate
            The surrogate.
        """
        stored = torch.load(path, map_location="cpu", weights_only=True)
        model = BubbleNetwork(
            stored["bounds"],
            stored["time_bounds"],
            width=stored["width"],
            n_blocks=stored["n_blocks"],
        )
        model.load_state_dict(stored["state"])
        return cls(model, stored["delta"], device)

    def save(self, path: str | Path) -> None:
        """
        Save the surrogate.

        Parameters
        ----------
        path : str or pathlib.Path
            File of the surrogate.
        """
        torch.save(
            {
                "state": self.model.state_dict(),
                "bounds": self.model.bounds,
                "time_bounds": self.model.time_bounds,
                "width": self.model.lift.out_features,
                "n_blocks": len(self.model.blocks),
                "delta": self.delta,
            },
            path,
        )

    def _check_domain(self, log_parameters: np.ndarray, log_t: np.ndarray) -> None:
        names = ("Pe", "lambda", "a", "b")
        for index, name in enumerate(names):
            values = log_parameters[..., index]
            low, high = self.bounds[index]
            if np.any(values < low) or np.any(values > high):
                raise ValueError(
                    f"{name} = [{np.exp(values.min()):.4g}, {np.exp(values.max()):.4g}] "
                    f"lies outside the domain of the surrogate, "
                    f"[{np.exp(low):.4g}, {np.exp(high):.4g}]."
                )
        low, high = self.time_bounds
        if np.any(log_t < low) or np.any(log_t > high):
            raise ValueError(
                f"The dimensionless time [{np.exp(log_t.min()):.4g}, {np.exp(log_t.max()):.4g}] "
                f"lies outside the domain of the surrogate, [{np.exp(low):.4g}, {np.exp(high):.4g}]."
            )

    def _inputs(
        self, x: np.ndarray, times: np.ndarray, bubble: DimensionlessBubble
    ) -> list[torch.Tensor]:
        """Validated inputs of the network, broadcast against each other."""
        fields = (
            bubble.peclet,
            bubble.diffusion_ratio,
            bubble.bubble_ratio,
            bubble.domain_ratio,
        )
        shape = np.broadcast_shapes(
            np.shape(x), np.shape(times), *map(np.shape, fields)
        )
        log_parameters = np.stack(
            [np.broadcast_to(np.log(field), shape) for field in fields], axis=-1
        )
        log_t = np.broadcast_to(np.log(times), shape)
        self._check_domain(log_parameters, log_t)
        zeta = zone_coordinate(
            np.broadcast_to(x, shape), bubble.bubble_ratio, bubble.outer_radius
        )
        return [
            torch.tensor(array, dtype=torch.float32, device=self.device)
            for array in (zeta, log_t, log_parameters)
        ]

    @torch.no_grad()
    def log_ratio(
        self, x: np.ndarray, times: np.ndarray, bubble: DimensionlessBubble
    ) -> np.ndarray:
        r"""
        Predicted :math:`w = \ln(\psi / \psi_\mathrm{ss})`, all inputs broadcast against each other.

        Parameters
        ----------
        x : numpy.ndarray
            Dimensionless radii.
        times : numpy.ndarray
            Dimensionless times.
        bubble : DimensionlessBubble
            Dimensionless bubble.

        Returns
        -------
        numpy.ndarray
            :math:`w`, in double precision.

        Raises
        ------
        ValueError
            If a time or a parameter lies outside the domain of the surrogate.
        """
        return self.model(*self._inputs(x, times, bubble)).double().cpu().numpy()

    def dimensionless_psi(
        self, x: np.ndarray, times: np.ndarray, bubble: DimensionlessBubble
    ) -> np.ndarray:
        r"""
        Dimensionless :math:`\psi`, the quantity of :py:func:`saetass.pinn.reference.solve`.

        Parameters
        ----------
        x : numpy.ndarray
            Dimensionless radii.
        times : numpy.ndarray
            Dimensionless times.
        bubble : DimensionlessBubble
            Dimensionless bubble.

        Returns
        -------
        numpy.ndarray
            Dimensionless :math:`\psi`, zero below :py:data:`~saetass.pinn.dataset.RELATIVE_FLOOR` where the network has no information.
        """
        w = self.log_ratio(x, times, bubble)
        return np.exp(w + log_steady_state(np.broadcast_to(x, w.shape), bubble))

    def in_domain(self, scales: BatchedScales, times: torch.Tensor) -> torch.Tensor:
        """
        Whether a batch of bubbles and times lies inside the domain of the surrogate.

        Parameters
        ----------
        scales : BatchedScales
            Scales of the bubbles, as tensors.
        times : torch.Tensor
            Dimensionless times.

        Returns
        -------
        torch.Tensor
            Boolean mask, broadcast from the inputs.
        """
        bounds = self.model.bounds
        time_bounds = self.model.time_bounds
        fields = (
            scales.peclet,
            scales.diffusion_ratio,
            scales.bubble_ratio,
            scales.domain_ratio,
        )
        inside = (torch.log(times) >= time_bounds[0]) & (
            torch.log(times) <= time_bounds[1]
        )
        for index, field in enumerate(fields):
            logarithm = torch.log(field)
            inside = inside & (logarithm >= bounds[index, 0])
            inside = inside & (logarithm <= bounds[index, 1])
        return inside

    def log_psi(
        self, x: torch.Tensor, times: torch.Tensor, scales: BatchedScales
    ) -> torch.Tensor:
        r"""
        Logarithm of the physical :math:`\psi`, differentiable and batched, for parameter searches.

        This method does not check the domain, see :py:meth:`in_domain`.

        Parameters
        ----------
        x : torch.Tensor
            Dimensionless radii.
        times : torch.Tensor
            Dimensionless times.
        scales : BatchedScales
            Scales of the bubbles, as tensors, for instance from :py:class:`~saetass.pinn.wind_bubble.FastScales`.

        Returns
        -------
        torch.Tensor
            :math:`\ln \psi` in canonical :py:data:`~saetass.units.PSI_P` units, broadcast from the inputs.
        """
        fields = (
            scales.peclet,
            scales.diffusion_ratio,
            scales.bubble_ratio,
            scales.domain_ratio,
        )
        x, times, *fields = torch.broadcast_tensors(x, times, *fields)
        Pe, ratio, a, b = fields
        outer = a * b
        zeta = torch.where(
            x < 1.0,
            x,
            torch.where(
                x < a, 1.0 + (x - 1.0) / (a - 1.0), 2.0 + (x - a) / (outer - a)
            ),
        )
        # The scales are in double precision, since physical inputs overflow single
        # precision, and the network in the precision of its weights
        dtype = self.model.bounds.dtype
        theta = torch.stack([torch.log(field) for field in fields], dim=-1).to(dtype)
        w = self.model(zeta.to(dtype), torch.log(times).to(dtype), theta).to(x.dtype)
        return (
            w
            + torch_log_steady_state(x, Pe, ratio, a, outer, self.delta)
            + (torch.log(scales.psi_unit))
        )

    def psi_p(
        self,
        r: u.Quantity,
        momenta: u.Quantity,
        t: u.Quantity,
        scales: PhysicalBubble | None = None,
        **parameters,
    ) -> u.Quantity:
        r"""
        Differential density :math:`\psi` at any radii, momenta and times.

        Parameters
        ----------
        r : astropy.units.Quantity
            Radii, of shape ``(n_r,)``. Units compatible with :py:data:`~saetass.units.LENGTH`.
        momenta : astropy.units.Quantity
            Particle momenta or kinetic energies, scalar or of shape ``(n_p,)``. Units compatible with :py:data:`~saetass.units.MOMENTUM` or :py:data:`~saetass.units.ENERGY`.
        t : astropy.units.Quantity
            Time, a scalar. Units compatible with :py:data:`~saetass.units.TIME`.
        scales : PhysicalBubble, optional
            Scales of the bubble at these momenta, to reuse them between calls. Default is ``None``, meaning :py:func:`~saetass.pinn.wind_bubble.physical_bubble` with ``parameters``.
        **parameters
            Arguments of :py:func:`~saetass.pinn.wind_bubble.physical_bubble`.

        Returns
        -------
        astropy.units.Quantity
            :math:`\psi` of shape ``(n_p, n_r)``. Units compatible with :py:data:`~saetass.units.PSI_P`.

        Raises
        ------
        ValueError
            If the inputs lie outside the domain of the surrogate or of the bubble.
        """
        if scales is None:
            scales = physical_bubble(momenta, **parameters)
        if abs(scales.delta - self.delta) > 1e-12:
            raise ValueError(
                f"The surrogate was trained for delta = {self.delta:.4g}, not {scales.delta:.4g}."
            )
        radii = np.atleast_1d(r.to_value(su.LENGTH))
        if np.any(radii < 0.0) or np.any(radii >= scales.r_max):
            raise ValueError("The radii must lie in [0, r_max).")
        numbers = scales.dimensionless
        peclet = np.atleast_1d(numbers.peclet)[:, None]
        ratio = np.atleast_1d(numbers.diffusion_ratio)[:, None]
        bubble = DimensionlessBubble(
            peclet, ratio, numbers.bubble_ratio, numbers.domain_ratio, self.delta
        )
        x = radii / scales.R_TS
        times = t.to_value(su.TIME) / scales.time_unit
        psi = self.dimensionless_psi(x, times, bubble)
        return psi * np.atleast_1d(scales.psi_unit)[:, None] * su.PSI_P

    def residual(
        self, x: np.ndarray, times: np.ndarray, bubble: DimensionlessBubble
    ) -> np.ndarray:
        r"""
        Normalized residual of the transport equation at the given points, see :py:func:`~saetass.pinn.training.physics_residual`.

        Parameters
        ----------
        x : numpy.ndarray
            Dimensionless radii, away from the interfaces.
        times : numpy.ndarray
            Dimensionless times.
        bubble : DimensionlessBubble
            Dimensionless bubble.

        Returns
        -------
        numpy.ndarray
            Relative imbalance of the equation at each point.
        """
        tensors = self._inputs(x, times, bubble)
        with torch.enable_grad():
            result = physics_residual(self.model, *tensors, self.delta)
        return result.detach().double().cpu().numpy()
