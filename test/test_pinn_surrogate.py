"""Tests of the network, the physics residual and the public interface of the surrogates."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saetass import units as su  # noqa: E402
from saetass.pinn import dataset  # noqa: E402
from saetass.pinn.network import BubbleNetwork  # noqa: E402
from saetass.pinn.reference import solve  # noqa: E402
from saetass.pinn.surrogate import WindBubbleSurrogate  # noqa: E402
from saetass.pinn.training import (  # noqa: E402
    Tensors,
    TrainingOptions,
    errors,
    physics_residual,
    residual_coefficients,
    train,
)
from saetass.pinn.wind_bubble import (  # noqa: E402
    DELTA,
    DimensionlessBubble,
    FastScales,
    log_steady_state,
    physical_bubble,
)

DELTA_K = DELTA["kolmogorov"]
BOUNDS = torch.as_tensor(dataset.DEFAULT_BOUNDS, dtype=torch.float32)
TIME_BOUNDS = torch.as_tensor(
    np.array(dataset.DEFAULT_TIME_BOUNDS), dtype=torch.float32
)
PHYSICAL = dict(
    L_wind=2e38 * u.erg / u.s,
    M_dot=1e-4 * const.M_sun / u.yr,
    rho_0=20 * const.m_p / u.cm**3,
    t_b=3 * u.Myr,
    r_max=300 * u.pc,
    E_max=1 * u.PeV,
)


@pytest.fixture
def network():
    torch.manual_seed(0)
    model = BubbleNetwork(BOUNDS, TIME_BOUNDS, width=32, n_blocks=2)
    torch.nn.init.normal_(model.head.weight, std=0.1)
    return model


def _points(zeta, n_parameters=4):
    zeta = torch.as_tensor(zeta, dtype=torch.float32)
    log_t = torch.zeros_like(zeta)
    theta = BOUNDS.mean(dim=1).expand(zeta.numel(), n_parameters)
    return zeta, log_t, theta


@pytest.mark.parametrize("interface", [1.0, 2.0])
def test_network_is_continuous_with_a_kink_at_the_interfaces(network, interface):
    for block in network.blocks:
        torch.nn.init.normal_(block.second.weight, std=0.1)
    h = 1e-5
    zeta = torch.tensor([interface - h, interface + h], requires_grad=True)
    _, log_t, theta = _points(np.zeros(2))
    w = network(zeta, log_t, theta)
    left, right = torch.autograd.grad(w.sum(), zeta)[0]
    assert torch.abs(w[1] - w[0]) < 2 * h * max(abs(left), abs(right))
    assert torch.abs(left - right) > 1e-2 * max(abs(left), abs(right))


@pytest.mark.parametrize(
    "bubble",
    [
        DimensionlessBubble(1.0, 200.0, 5.0, 3.0),
        DimensionlessBubble(20.0, 200.0, 5.0, 3.0),
    ],
)
def test_coefficients_of_the_residual_describe_the_solutions(bubble):
    """The equation for w holds on a reference solution, by finite differences."""
    middle = 200.0
    ratio = 1.01
    solution = solve(
        bubble, np.array([middle / ratio, middle, middle * ratio]), resolution=4
    )
    x = solution.x
    log_steady = log_steady_state(x, bubble)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(solution.psi) - log_steady
    w_t = (w[2] - w[0]) / (middle * (ratio - 1.0 / ratio))

    theta = torch.as_tensor(
        np.log(
            [
                bubble.peclet,
                bubble.diffusion_ratio,
                bubble.bubble_ratio,
                bubble.domain_ratio,
            ]
        )
    ).expand(x.size - 2, 4)
    D, beta = residual_coefficients(torch.as_tensor(x[1:-1]), theta, DELTA_K)
    D, beta = D.numpy(), beta.numpy()
    left, right = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    w1 = w[1]
    w_x = (w1[2:] * left**2 - w1[:-2] * right**2 + w1[1:-1] * (right**2 - left**2)) / (
        left * right * (left + right)
    )
    w_xx = (
        2.0
        * (w1[2:] * left + w1[:-2] * right - w1[1:-1] * (left + right))
        / (left * right * (left + right))
    )
    balance = D * (w_xx + w_x**2) + beta * w_x
    # Well inside the bubble, and in the medium before the outer boundary
    a = bubble.bubble_ratio
    inner = (x[1:-1] > 1.3) & (x[1:-1] < 0.8 * a)
    medium = (x[1:-1] > 1.2 * a) & (x[1:-1] < 0.8 * bubble.outer_radius)
    significant = np.abs(w_t[1:-1]) > 1e-8
    for zone in (inner & significant, medium & significant):
        if zone.any():
            assert np.allclose(w_t[1:-1][zone], balance[zone], rtol=0.05, atol=1e-7)


def test_residual_vanishes_for_a_constant_ratio():
    """A network returning w = 0 is the steady state, which solves the equation exactly."""
    model = BubbleNetwork(BOUNDS, TIME_BOUNDS, width=16, n_blocks=1)
    zeta, log_t, theta = _points(np.linspace(0.1, 2.9, 40))
    assert torch.allclose(
        physics_residual(model, zeta, log_t, theta, DELTA_K), torch.zeros(40), atol=1e-6
    )


def _small_dataset(n=4):
    sampling = dataset.Sampling(n_times=8, resolution=1, strides=(30, 70, 40))
    parameters = dataset.sample_parameters(n, seed=0)
    w = np.stack([dataset.simulate((row, sampling)) for row in parameters])
    return {
        "parameters": parameters,
        "log_times": np.log(sampling.times),
        "zeta": dataset.zeta_nodes(sampling),
        "w": w,
    }


def test_dataset_has_fixed_zone_coordinates():
    data = _small_dataset(2)
    assert data["w"].shape == (2, 8, data["zeta"].size)
    assert np.all(np.diff(data["zeta"]) > 0.0)
    assert 1.0 in data["zeta"] and 2.0 in data["zeta"]
    assert np.all(data["zeta"] < 3.0)
    x = dataset.radius_from_zone(data["zeta"], 5.0, 10.0)
    assert np.allclose(dataset.zone_coordinate(x, 5.0, 10.0), data["zeta"])


def test_training_reduces_the_errors():
    data = Tensors(_small_dataset(), DELTA_K, "cpu")
    torch.manual_seed(0)
    model = BubbleNetwork(BOUNDS, TIME_BOUNDS, width=48, n_blocks=2)
    before = errors(model, data)["log_max"].max()
    options = TrainingOptions(
        steps=120, batch_size=2048, physics_batch=128, log_every=120
    )
    train(model, data, data, options)
    assert errors(model, data)["log_max"].max() < before
    assert options.history[-1]["physics"] > 0.0


@pytest.fixture
def surrogate():
    torch.manual_seed(0)
    return WindBubbleSurrogate(
        BubbleNetwork(BOUNDS, TIME_BOUNDS, width=32, n_blocks=2), DELTA_K
    )


def test_surrogate_returns_the_steady_state_for_a_null_network(surrogate):
    bubble = DimensionlessBubble(5.0, 200.0, 6.0, 3.0)
    x = np.linspace(0.8, 17.0, 50)
    psi = surrogate.dimensionless_psi(x, 10.0, bubble)
    assert np.allclose(psi, np.exp(log_steady_state(x, bubble)))


def test_surrogate_is_mesh_free_in_physical_units(surrogate):
    r = np.array([1.0, 17.3, 41.0, 250.0]) * u.pc
    momenta = [10.0, 1e3] * u.GeV
    psi = surrogate.psi_p(r, momenta, 1.5 * u.Myr, **PHYSICAL)
    assert psi.shape == (2, 4)
    assert psi.unit.is_equivalent(su.PSI_P)
    assert np.all(psi.value > 0.0)


def test_surrogate_rejects_parameters_outside_its_domain(surrogate):
    with pytest.raises(ValueError, match="Pe"):
        surrogate.dimensionless_psi(
            np.array([1.5]), 10.0, DimensionlessBubble(1e4, 200.0, 6.0, 3.0)
        )
    with pytest.raises(ValueError, match="time"):
        surrogate.dimensionless_psi(
            np.array([1.5]), 1e9, DimensionlessBubble(5.0, 200.0, 6.0, 3.0)
        )


def test_surrogate_rejects_radii_outside_the_domain(surrogate):
    with pytest.raises(ValueError, match="radii"):
        surrogate.psi_p(np.array([301.0]) * u.pc, 1 * u.TeV, 1 * u.Myr, **PHYSICAL)


def test_surrogate_rejects_another_turbulence(surrogate):
    scales = physical_bubble(1 * u.TeV, diffusion_model="kraichnan", **PHYSICAL)
    with pytest.raises(ValueError, match="delta"):
        surrogate.psi_p(np.array([20.0]) * u.pc, 1 * u.TeV, 1 * u.Myr, scales=scales)


def test_surrogate_reports_the_residual(surrogate):
    bubble = DimensionlessBubble(5.0, 200.0, 6.0, 3.0)
    residual = surrogate.residual(np.linspace(0.2, 15.0, 30), 10.0, bubble)
    assert residual.shape == (30,)
    assert np.all(np.isfinite(residual))


def _double(values):
    return torch.as_tensor(values, dtype=torch.float64)


def test_batched_evaluation_matches_the_physical_interface(surrogate):
    fast = FastScales(1 * u.PeV)
    p = _double([[10.0], [1e3]])
    L_wind, M_dot = _double(2e38), _double(1e-4)
    rho = _double((20 * const.m_p / u.cm**3).to_value(u.g / u.cm**3))
    scales = fast(p, L_wind, M_dot, rho, _double(3.0), _double(300.0))
    r = np.array([5.0, 17.3, 41.0, 250.0])
    times = 1.5 / scales.time_unit
    x = _double(r) / scales.R_TS
    assert surrogate.in_domain(scales, times).all()
    log_psi = surrogate.log_psi(x, times, scales)
    exact = surrogate.psi_p(
        r * u.pc, p.numpy()[:, 0] * su.MOMENTUM, 1.5 * u.Myr, **PHYSICAL
    )
    assert np.allclose(log_psi.detach().numpy(), np.log(exact.value), atol=1e-3)


def test_batched_evaluation_is_differentiable(surrogate):
    fast = FastScales(1 * u.PeV)
    L_wind = _double(2e38).requires_grad_()
    rho = _double((20 * const.m_p / u.cm**3).to_value(u.g / u.cm**3))
    scales = fast(
        _double(100.0),
        L_wind,
        _double(1e-4),
        rho,
        _double(3.0),
        _double(300.0),
    )
    x = _double([0.5, 2.0, 30.0])
    surrogate.log_psi(x, 2.0 / scales.time_unit, scales).sum().backward()
    assert torch.isfinite(L_wind.grad) and L_wind.grad != 0.0


def test_domain_mask(surrogate):
    fast = FastScales(1 * u.PeV)
    rho = _double((20 * const.m_p / u.cm**3).to_value(u.g / u.cm**3))
    scales = fast(
        _double([1e3, 1e-3]),
        _double(2e38),
        _double(1e-4),
        rho,
        _double(3.0),
        _double(300.0),
    )
    mask = surrogate.in_domain(scales, 2.0 / scales.time_unit)
    assert mask.tolist() == [True, False]


def test_surrogate_is_saved_and_loaded(surrogate, tmp_path):
    bubble = DimensionlessBubble(5.0, 200.0, 6.0, 3.0)
    x = np.linspace(0.5, 15.0, 10)
    path = tmp_path / "surrogate.pt"
    surrogate.save(path)
    loaded = WindBubbleSurrogate.load(path)
    assert np.array_equal(
        surrogate.log_ratio(x, 10.0, bubble), loaded.log_ratio(x, 10.0, bubble)
    )
