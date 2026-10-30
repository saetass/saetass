r"""
Network of the wind bubble surrogate.

The network predicts :math:`w(t, \zeta; \theta) = \ln(\psi / \psi_\mathrm{ss})` in the variables of :py:mod:`saetass.pinn.dataset`.
Its inputs are continuous functions of the radius, among them :math:`|\zeta - 1|` and :math:`|\zeta - 2|`, so that the output is continuous at the shock and at the edge of the bubble by construction, as :math:`\psi` is, while its radial derivative can jump there, as the jumps of :math:`D` and :math:`v` require.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Encoder:
    r"""
    Map of the logarithmic time and parameters and of the zone coordinate to the inputs of the network.

    Parameters
    ----------
    bounds : torch.Tensor
        Bounds of the logarithms of the parameters, of shape ``(n_parameters, 2)``.
    time_bounds : torch.Tensor
        Bounds of the logarithm of the time, of shape ``(2,)``.
    """

    def __init__(self, bounds: torch.Tensor, time_bounds: torch.Tensor) -> None:
        self.low = bounds[:, 0]
        self.width = bounds[:, 1] - bounds[:, 0]
        self.t_low = time_bounds[0]
        self.t_width = time_bounds[1] - time_bounds[0]

    def __call__(
        self, zeta: torch.Tensor, log_t: torch.Tensor, log_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Inputs of shape ``(..., n_inputs)``, from ``zeta`` and ``log_t`` of shape ``(...,)`` and ``log_parameters`` of shape ``(..., n_parameters)``."""
        scaled = 2.0 * (log_parameters - self.low.to(zeta)) / self.width.to(zeta) - 1.0
        tau = 2.0 * (log_t - self.t_low.to(zeta)) / self.t_width.to(zeta) - 1.0
        to_shock = torch.abs(zeta - 1.0)
        to_edge = torch.abs(zeta - 2.0)
        # Distances at several scales: the layers next to each interface are very thin
        features = [zeta - 1.5, tau, to_shock, to_edge]
        for scale in (0.3, 0.1, 0.03, 0.01, 0.003):
            features.append(torch.exp(-to_shock / scale))
            features.append(torch.exp(-to_edge / scale))
        return torch.cat([torch.stack(features, dim=-1), scaled], dim=-1)

    @staticmethod
    def n_inputs(n_parameters: int) -> int:
        """Number of inputs of the network."""
        return 14 + n_parameters


class ResidualBlock(nn.Module):
    """Pre-normalized residual block, with a smooth activation so that the physics residual is defined."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.second(F.silu(self.first(self.norm(z))))


class BubbleNetwork(nn.Module):
    r"""
    Residual network for :math:`w(t, \zeta; \theta)`.

    The output is multiplied by ``scale``, the typical size of :math:`|w|`.

    Parameters
    ----------
    bounds : torch.Tensor
        Bounds of the logarithms of the parameters, of shape ``(n_parameters, 2)``.
    time_bounds : torch.Tensor
        Bounds of the logarithm of the time, of shape ``(2,)``.
    width : int, optional
        Width of the hidden layers. Default is ``512``.
    n_blocks : int, optional
        Number of residual blocks. Default is ``6``.
    n_fourier : int, optional
        Number of random Fourier features of the inputs. Default is ``128``.
    scale : float, optional
        Scale of the output. Default is ``5.0``.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        time_bounds: torch.Tensor,
        width: int = 512,
        n_blocks: int = 6,
        n_fourier: int = 128,
        scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.register_buffer("bounds", bounds.clone())
        self.register_buffer("time_bounds", time_bounds.clone())
        self.scale = scale
        n_inputs = Encoder.n_inputs(bounds.shape[0])
        # A fixed generator: the features must be the same when a surrogate is loaded
        generator = torch.Generator().manual_seed(0)
        frequencies = torch.randn(n_fourier, n_inputs, generator=generator)
        self.register_buffer("frequencies", frequencies)
        self.lift = nn.Linear(n_inputs + 2 * n_fourier, width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, zeta: torch.Tensor, log_t: torch.Tensor, log_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Predicted :math:`w`, with the shape of ``zeta``."""
        encoder = Encoder(self.bounds, self.time_bounds)
        inputs = encoder(zeta, log_t, log_parameters)
        phase = inputs @ self.frequencies.T
        z = self.lift(torch.cat([inputs, torch.sin(phase), torch.cos(phase)], dim=-1))
        for block in self.blocks:
            z = block(z)
        return self.scale * self.head(self.norm(z)).squeeze(-1)
