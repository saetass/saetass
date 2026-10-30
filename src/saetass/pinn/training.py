r"""
Training and evaluation of the wind bubble surrogate.

The loss combines three terms:

- **Data**: squared error of :math:`w = \ln(\psi / \psi_\mathrm{ss})` against reference solutions.
- **Floor**: where :math:`\psi` is negligible at that time, a one-sided penalty that only keeps the prediction below :py:data:`~saetass.pinn.dataset.RELATIVE_FLOOR`.
- **Physics**: the transport equation at random points, which do not need a reference solution. For :math:`\psi = \psi_\mathrm{ss}\, e^w` it reads

  .. math::

      \frac{\partial w}{\partial t} = D \left( \frac{\partial^2 w}{\partial x^2} + \left(\frac{\partial w}{\partial x}\right)^2 \right) + \beta(x)\, \frac{\partial w}{\partial x},
      \qquad
      \beta = 2 D\, \frac{\partial \ln \psi_\mathrm{ss}}{\partial x} + \frac{\partial D}{\partial x} + \frac{2 D}{x} - v,

  whose coefficients are closed forms, see :py:func:`residual_coefficients`.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .dataset import RELATIVE_FLOOR, radius_from_zone
from .network import BubbleNetwork
from .wind_bubble import DimensionlessBubble, log_steady_state


def _log_steady_grid(
    parameters: np.ndarray, zeta: np.ndarray, delta: float
) -> np.ndarray:
    """Logarithm of the steady state of each bubble on the zone coordinates, of shape ``(n, n_zeta)``."""
    out = np.empty((parameters.shape[0], zeta.size))
    for row, values in enumerate(np.exp(parameters)):
        bubble = DimensionlessBubble(*values, delta)
        out[row] = log_steady_state(
            radius_from_zone(zeta, bubble.bubble_ratio, bubble.outer_radius), bubble
        )
    return out


class Tensors:
    r"""
    A dataset of :py:func:`saetass.pinn.dataset.build` on a device, ready to draw batches from.

    Parameters
    ----------
    data : dict
        Dataset.
    delta : float
        Exponent of the diffusion coefficient in the wind.
    device : torch.device or str
        Device of the tensors.
    """

    def __init__(self, data: dict, delta: float, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.delta = delta
        self.parameters = torch.as_tensor(
            data["parameters"], device=self.device, dtype=torch.float32
        )
        self.log_times = torch.as_tensor(
            data["log_times"], device=self.device, dtype=torch.float32
        )
        self.zeta = torch.as_tensor(
            data["zeta"], device=self.device, dtype=torch.float32
        )
        w = torch.as_tensor(
            np.asarray(data["w"]), device=self.device, dtype=torch.float32
        )
        self.valid = torch.isfinite(w)
        self.w = torch.where(self.valid, w, torch.zeros_like(w))
        log_steady = _log_steady_grid(
            data["parameters"], np.asarray(data["zeta"]), delta
        )
        self.log_steady = torch.as_tensor(
            log_steady, device=self.device, dtype=torch.float32
        )
        # Largest ln(psi) of each time, over the nodes above the floor
        total = self.w + self.log_steady[:, None, :]
        total = torch.where(self.valid, total, torch.full_like(total, -torch.inf))
        self.log_floor = torch.amax(total, dim=2) + math.log(RELATIVE_FLOOR)
        self.n, self.n_times, self.n_zeta = w.shape

    def batch(self, size: int) -> dict[str, torch.Tensor]:
        """A random batch of ``size`` points of the tensor."""
        i = torch.randint(self.n, (size,), device=self.device)
        j = torch.randint(self.n_times, (size,), device=self.device)
        k = torch.randint(self.n_zeta, (size,), device=self.device)
        return {
            "zeta": self.zeta[k],
            "log_t": self.log_times[j],
            "theta": self.parameters[i],
            "w": self.w[i, j, k],
            "valid": self.valid[i, j, k],
            "log_steady": self.log_steady[i, k],
            "log_floor": self.log_floor[i, j],
        }


def data_loss(
    model: BubbleNetwork, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error of :math:`w` above the floor, and the one-sided penalty below it."""
    pred = model(batch["zeta"], batch["log_t"], batch["theta"])
    valid = batch["valid"]
    fit = (pred[valid] - batch["w"][valid]) ** 2
    excess = torch.relu(
        pred[~valid] + batch["log_steady"][~valid] - batch["log_floor"][~valid]
    )
    return fit.mean(), (excess**2).mean() if excess.numel() else pred.sum() * 0.0


def residual_coefficients(
    x: torch.Tensor, theta: torch.Tensor, delta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Coefficients :math:`D` and :math:`\beta` of the equation for :math:`w`, from the closed form of the steady state.

    In the wind :math:`\beta = 4 + (2 + \delta) D / x`, in the bubble :math:`\beta = (1 - 2 / \psi_\mathrm{ss}) / x^2 + 2 D / x` and in the interstellar medium :math:`\beta = 2 D (\partial_x \ln \psi_\mathrm{ss} + 1 / x)`.

    Parameters
    ----------
    x : torch.Tensor
        Dimensionless radii.
    theta : torch.Tensor
        Logarithms of the parameters, of shape ``(..., 4)``.
    delta : float
        Exponent of the diffusion coefficient in the wind.

    Returns
    -------
    tuple of torch.Tensor
        :math:`D` and :math:`\beta` at ``x``.
    """
    Pe, ratio, a, b = torch.exp(theta).unbind(-1)
    outer = a * b
    edge = (Pe / ratio) * (1.0 / a - 1.0 / outer)
    safe_x = torch.clamp(x, min=1e-6)

    wind_D = 11.0 ** (delta / 2.0) / Pe * safe_x**delta
    wind_beta = 4.0 + (2.0 + delta) * wind_D / safe_x

    inner_x = torch.clamp(x, min=1.0)
    inner_D = 1.0 / Pe
    steady = 1.0 + (edge - 1.0) * torch.exp(Pe * (1.0 / a - 1.0 / inner_x))
    inner_beta = (1.0 - 2.0 / steady) / inner_x**2 + 2.0 * inner_D / inner_x

    outer_D = ratio / Pe
    distance = torch.clamp(1.0 / safe_x - 1.0 / outer, min=1e-6)
    outer_beta = 2.0 * outer_D * (1.0 / safe_x - 1.0 / (safe_x**2 * distance))

    D = torch.where(x < 1.0, wind_D, torch.where(x < a, inner_D, outer_D))
    beta = torch.where(x < 1.0, wind_beta, torch.where(x < a, inner_beta, outer_beta))
    return D, beta


def physics_residual(
    model: BubbleNetwork,
    zeta: torch.Tensor,
    log_t: torch.Tensor,
    theta: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    r"""
    Normalized residual of the equation for :math:`w` at the given points.

    It is computed with respect to :math:`\ln t`, and divided by the size of the coefficients of the equation over the zone, in the coordinate :math:`\zeta`.

    Parameters
    ----------
    model : BubbleNetwork
        Network.
    zeta : torch.Tensor
        Zone coordinates, inside a zone.
    log_t : torch.Tensor
        Logarithms of the dimensionless times.
    theta : torch.Tensor
        Logarithms of the parameters, of shape ``(..., 4)``.
    delta : float
        Exponent of the diffusion coefficient in the wind.

    Returns
    -------
    torch.Tensor
        Normalized residual at each point.
    """
    zeta = zeta.detach().requires_grad_(True)
    log_t = log_t.detach().requires_grad_(True)
    w = model(zeta, log_t, theta)
    w_s, w_z = torch.autograd.grad(w.sum(), (log_t, zeta), create_graph=True)
    w_zz = torch.autograd.grad(w_z.sum(), zeta, create_graph=True)[0]

    Pe, ratio, a, b = torch.exp(theta).unbind(-1)
    outer = a * b
    x = torch.where(
        zeta < 1.0,
        zeta,
        torch.where(
            zeta < 2.0, 1.0 + (zeta - 1.0) * (a - 1.0), a + (zeta - 2.0) * (outer - a)
        ),
    ).detach()
    slope = torch.where(
        zeta < 1.0, torch.ones_like(zeta), torch.where(zeta < 2.0, a - 1.0, outer - a)
    ).detach()
    D, beta = residual_coefficients(x, theta, delta)
    t = torch.exp(log_t).detach()
    w_x = w_z / slope
    w_xx = w_zz / slope**2
    diffusion = t * D * (w_xx + w_x**2)
    drift = t * beta * w_x
    # The scale depends on the coefficients only: dividing by the terms of the residual
    # would saturate it at one, without a gradient
    scale = 1.0 + t * (D / slope**2 + torch.abs(beta) / slope)
    residual = (w_s - diffusion - drift) / scale
    # Far below the floor the network is only bounded from above: nothing constrains w
    return torch.where(w.detach() > -20.0, residual, torch.zeros_like(residual))


@dataclass
class TrainingOptions:
    """
    Options of :py:func:`train`.

    Parameters
    ----------
    steps : int
        Number of optimization steps.
    batch_size : int
        Number of data points of each step.
    physics_batch : int
        Number of collocation points of each step, ``0`` to train on data only.
    physics_weight : float
        Weight of the physics residual in the loss.
    floor_weight : float
        Weight of the one-sided penalty.
    learning_rate : float
        Initial learning rate, which decays with a cosine schedule.
    log_every : int
        Steps between evaluations on the validation set.
    seed : int
        Seed of the initialization and of the batches.
    checkpoint : str or None
        File where the weights are saved at each evaluation, ``None`` not to save them.
    """

    steps: int = 200_000
    batch_size: int = 32768
    physics_batch: int = 0
    physics_weight: float = 1.0
    floor_weight: float = 0.1
    learning_rate: float = 5e-4
    log_every: int = 10_000
    seed: int = 0
    checkpoint: str | None = None
    history: list = field(default_factory=list)


@torch.no_grad()
def errors(
    model: BubbleNetwork, tensors: Tensors, batch_size: int = 2**20
) -> dict[str, torch.Tensor]:
    r"""
    Errors of the model on a whole dataset, against its stored solutions.

    Parameters
    ----------
    model : BubbleNetwork
        Network.
    tensors : Tensors
        Dataset.
    batch_size : int, optional
        Number of points evaluated at once. Default is ``2**20``.

    Returns
    -------
    dict
        ``l2``, of shape ``(n, n_times)``: relative :math:`L_2` error of :math:`\psi` over the radius; ``log_max``, of the same shape: largest :math:`|\Delta \ln \psi|` above :math:`10^{-6}` of the largest :math:`\psi` of that time.
    """
    n, n_t, n_z = tensors.n, tensors.n_times, tensors.n_zeta
    zeta = tensors.zeta.expand(n_t, n_z).reshape(-1)
    log_t = tensors.log_times[:, None].expand(n_t, n_z).reshape(-1)
    pred = torch.empty(n, n_t, n_z, device=tensors.device)
    for i in range(n):
        theta = tensors.parameters[i].expand(zeta.numel(), -1)
        pred[i] = torch.cat(
            [
                model(z, t, p)
                for z, t, p in zip(
                    zeta.split(batch_size),
                    log_t.split(batch_size),
                    theta.split(batch_size),
                )
            ]
        ).reshape(n_t, n_z)

    a, outer = torch.exp(tensors.parameters[:, 2:3]), None
    outer = a * torch.exp(tensors.parameters[:, 3:4])
    z = tensors.zeta[None, :]
    x = torch.where(
        z < 1.0,
        z,
        torch.where(z < 2.0, 1.0 + (z - 1.0) * (a - 1.0), a + (z - 2.0) * (outer - a)),
    )
    widths = torch.zeros_like(x)
    widths[:, 1:] += 0.5 * (x[:, 1:] - x[:, :-1])
    widths[:, :-1] += 0.5 * (x[:, 1:] - x[:, :-1])

    log_true = tensors.w + tensors.log_steady[:, None, :]
    log_pred = pred + tensors.log_steady[:, None, :]
    # Double precision: ln(psi) ranges over hundreds, its exponential is compared
    reference = tensors.log_floor[..., None] - math.log(RELATIVE_FLOOR)
    true = torch.where(
        tensors.valid,
        torch.exp((log_true - reference).double()),
        torch.zeros_like(log_true, dtype=torch.float64),
    )
    guess = torch.exp((log_pred - reference).double().clamp(max=50.0))
    weights = widths[:, None, :].double()
    l2 = torch.sqrt(
        torch.sum(weights * (guess - true) ** 2, dim=2)
        / torch.sum(weights * true**2, dim=2)
    )
    relevant = tensors.valid & (log_true > reference - math.log(1e6))
    gap = torch.where(
        relevant, torch.abs(log_pred - log_true), torch.zeros_like(log_true)
    )
    return {"l2": l2, "log_max": torch.amax(gap, dim=2)}


def train(
    model: BubbleNetwork,
    data: Tensors,
    validation: Tensors,
    options: TrainingOptions,
) -> BubbleNetwork:
    """
    Train ``model`` in place and return it.

    Parameters
    ----------
    model : BubbleNetwork
        Network, on the device of ``data``.
    data : Tensors
        Training set.
    validation : Tensors
        Validation set, used to report the errors and never for the gradients.
    options : TrainingOptions
        Options.

    Returns
    -------
    BubbleNetwork
        The trained network.
    """
    torch.manual_seed(options.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=0.0
    )
    warmup = max(options.steps // 50, 1)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 1e-3, 1.0, warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, options.steps - warmup, eta_min=options.learning_rate * 1e-3
            ),
        ],
        milestones=[warmup],
    )
    device = data.device
    low, high = data.parameters.amin(0), data.parameters.amax(0)
    start = time.perf_counter()
    for step in range(1, options.steps + 1):
        batch = data.batch(options.batch_size)
        fit, floor = data_loss(model, batch)
        loss = fit + options.floor_weight * floor
        physics = torch.zeros((), device=device)
        if options.physics_batch:
            size = options.physics_batch
            i = torch.randint(data.n, (size,), device=device)
            theta = low + (high - low) * torch.rand(size, 4, device=device)
            log_t = data.log_times[0] + (
                data.log_times[-1] - data.log_times[0]
            ) * torch.rand(size, device=device)
            k = torch.randint(data.n_zeta - 1, (size,), device=device)
            zeta = data.zeta[k] + (data.zeta[k + 1] - data.zeta[k]) * torch.rand(
                size, device=device
            )
            # Only inside the zones, and away from the outer boundary, where the ratio to
            # the steady state is singular
            inside = (
                (torch.abs(zeta - 1.0) > 1e-5)
                & (torch.abs(zeta - 2.0) > 1e-5)
                & (zeta < 2.98)
            )
            residual = physics_residual(
                model, zeta[inside], log_t[inside], theta[inside], data.delta
            )
            del i
            physics = torch.mean(residual**2)
            loss = loss + options.physics_weight * physics
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % options.log_every == 0 or step == options.steps:
            result = errors(model, validation)
            worst = result["l2"].amax(1)
            record = {
                "step": step,
                "seconds": time.perf_counter() - start,
                "fit": fit.item(),
                "floor": floor.item(),
                "physics": physics.item(),
                "l2_median": result["l2"].nanmedian().item(),
                "l2_p95": torch.nanquantile(
                    result["l2"].flatten().float(), 0.95
                ).item(),
                "l2_worst": worst.max().item(),
                "log_max": result["log_max"].max().item(),
            }
            options.history.append(record)
            if options.checkpoint is not None:
                torch.save(model.state_dict(), options.checkpoint)
            print(
                " ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in record.items()
                ),
                flush=True,
            )
    return model
