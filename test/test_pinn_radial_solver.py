"""Tests of the batched solver of the radial transport equation and of the inversion of D(r)."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saetass import units as su  # noqa: E402
from saetass.pinn.confirm import simulate  # noqa: E402
from saetass.pinn.inversion import DiffusionInversion  # noqa: E402
from saetass.pinn.radial_solver import (  # noqa: E402
    RadialSolver,
    ScaledSystem,
    WindBubbleSolver,
    log_bernoulli,
    solve_tridiagonal,
    talbot_contour,
)
from saetass.pinn.reference import solve  # noqa: E402
from saetass.pinn.wind_bubble import (  # noqa: E402
    BatchedScales,
    DimensionlessBubble,
    FastScales,
    log_steady_state,
    physical_bubble,
)

BUBBLES = [
    DimensionlessBubble(22.0, 250.0, 6.4, 3.5),
    DimensionlessBubble(1.0, 270.0, 6.4, 3.5),
    DimensionlessBubble(100.0, 500.0, 15.0, 2.0),
    DimensionlessBubble(0.05, 100.0, 3.0, 6.0),
    DimensionlessBubble(300.0, 60.0, 23.0, 1.4),
]


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def fields(bubble, n=1):
    return tuple(
        tensor(float(value)).expand(n)
        for value in (
            bubble.peclet,
            bubble.diffusion_ratio,
            bubble.bubble_ratio,
            bubble.outer_radius,
        )
    )


def scales(bubble, n=1, unit=1.0):
    Pe, ratio, a, outer = fields(bubble, n)
    one = torch.ones_like(Pe)
    return BatchedScales(Pe, ratio, a, outer / a, one, one, unit * one)


@pytest.mark.parametrize("z", [1e-9, 1e-5, 0.3, 40.0, 900.0, 1e5])
def test_log_bernoulli(z):
    z = tensor([z, -z])
    exact = torch.log(z / torch.expm1(z.clamp(max=700.0)))
    exact[1] = torch.log(z[1] / torch.expm1(z[1]))
    value = log_bernoulli(z)
    assert torch.all(torch.isfinite(value))
    if z[0] < 700.0:
        assert torch.allclose(value, exact, rtol=1e-9, atol=1e-9)
    assert value[1] - value[0] == pytest.approx(z[0].item(), rel=1e-8)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 64, 129])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_solve_tridiagonal_against_a_dense_solve(n, dtype):
    generator = torch.Generator().manual_seed(n)
    lower, upper, rhs = torch.rand((3, 4, n), generator=generator, dtype=torch.float64)
    main = 3.0 + torch.rand((4, n), generator=generator, dtype=torch.float64)
    shift = 1.0 + 1.0j if dtype == torch.complex128 else 0.0
    lower, main, upper, rhs = (
        lower.to(dtype),
        (main + shift).to(dtype),
        upper.to(dtype),
        rhs.to(dtype),
    )
    x = solve_tridiagonal(lower, main, upper, rhs)
    for k in range(4):
        matrix = torch.diag(main[k])
        if n > 1:
            matrix = (
                matrix + torch.diag(lower[k, 1:], -1) + torch.diag(upper[k, :-1], 1)
            )
        assert torch.allclose(matrix @ x[k], rhs[k], rtol=1e-12, atol=1e-12)


def test_contour_inverts_a_known_transform():
    # The transform of 1 - exp(-t) is 1 / (s (s + 1))
    points, weights = talbot_contour(32)
    for t in (0.01, 1.0, 100.0):
        s = points / t
        value = (weights / t * (1.0 / (s * (s + 1.0)))).real.sum()
        assert value.item() == pytest.approx(-np.expm1(-t), rel=1e-10)


@pytest.mark.parametrize("bubble", BUBBLES)
def test_steady_state_of_the_grid_is_exact(bubble):
    solver = WindBubbleSolver(128)
    system, x = solver.system(*fields(bubble))
    assert torch.all(x[:, 1:] > x[:, :-1])
    exact = log_steady_state(x[0, :-1].numpy(), bubble)
    assert np.allclose(system.log_steady[0].numpy(), exact, atol=1e-10)
    # The steady state is a null vector of B + f, whatever its range of decades
    rows = system.main + system.lower + system.upper + system.forcing
    assert torch.all(rows.abs() < 1e-7 * system.main.abs())


@pytest.mark.parametrize("bubble", BUBBLES)
def test_long_times_reach_the_steady_state(bubble):
    system, _ = WindBubbleSolver(128).system(*fields(bubble))
    top = system.log_steady.amax(dim=-1, keepdim=True)
    relevant = system.log_steady > top - 20.0
    y, broken = system.ratio(tensor([1e9]), 48)
    assert not broken.any()
    assert torch.allclose(y[relevant], torch.ones_like(y[relevant]), rtol=1e-6)


@pytest.mark.parametrize("bubble", BUBBLES)
def test_against_the_reference(bubble):
    times = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    reference = solve(bubble, times)
    n = times.size
    solver = WindBubbleSolver(512)
    x = tensor(reference.x[:-1]).expand(n, -1)
    w = solver.log_ratio(x, tensor(times), scales(bubble, n)).numpy()
    psi = np.exp(w + log_steady_state(reference.x[:-1], bubble))
    for k in range(n):
        truth = reference.psi[k, :-1]
        error = np.linalg.norm(psi[k] - truth) / np.linalg.norm(truth)
        # At this resolution most of the difference is the error of the reference
        assert error < 3e-3


def test_fitted_fluxes_have_no_numerical_diffusion_at_a_front():
    # A front inside cells of Peclet number far above one, at a resolved reference
    bubble = DimensionlessBubble(200.0, 250.0, 10.0, 3.0)
    t = tensor([1.0])
    fine, x = WindBubbleSolver(4096).system(*fields(bubble))
    coarse, nodes = WindBubbleSolver(128).system(*fields(bubble))
    truth = np.interp(nodes[0, :-1].numpy(), x[0, :-1].numpy(), fine.contour(t, 48)[0])
    fitted = coarse.contour(t, 48)[0].numpy()
    plain = coarse.stepped(t, 800)[0].numpy()
    weight = np.exp(coarse.log_steady[0].numpy())
    assert np.linalg.norm(weight * (fitted - truth)) < 0.25 * np.linalg.norm(
        weight * (plain - truth)
    )


def test_uniform_diffusion_from_a_point_source():
    # psi = erfc(r / sqrt(4 D t)) / (D r), per 4 pi, in an infinite medium
    D, t = 2.0, 0.5
    nodes = torch.linspace(0.0, 30.0, 3001, dtype=torch.float64)
    solver = RadialSolver(nodes)
    source = torch.zeros(1, 3000, dtype=torch.float64)
    source[0, 0] = 1.0 / solver.volumes[0]
    cells = torch.full((1, 3000), D, dtype=torch.float64)
    log_psi = solver.log_psi(cells, 0.0 * cells, source, tensor([t]))
    r = tensor([[0.5, 1.0, 2.0, 4.0]])
    exact = torch.erfc(r / (4.0 * D * t) ** 0.5) / (D * r)
    assert torch.allclose(solver.interpolate(log_psi, r), exact.log(), rtol=2e-3)


def test_shell_holds_one_particle():
    nodes = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    solver = RadialSolver(nodes)
    source = solver.shell(tensor([0.0, 0.23, 0.5, 0.97]))
    assert torch.allclose(
        (source * solver.volumes).sum(dim=1), torch.ones(4, dtype=torch.float64)
    )
    assert source.shape == (4, 10)
    with pytest.raises(ValueError):
        RadialSolver(nodes.float())
    with pytest.raises(ValueError):
        RadialSolver(nodes + 1.0)


def test_contour_failures_are_recomputed_with_time_steps():
    system, _ = WindBubbleSolver(256).system(*fields(BUBBLES[2]))
    t = tensor([0.05])
    assert not system.ratio(t, 64)[1].any()
    y, broken = system.ratio(t, 4)
    assert broken.all()
    assert torch.allclose(y, system.stepped(t))


def test_select_keeps_every_field():
    a, _ = WindBubbleSolver(64).system(*fields(BUBBLES[0], 3))
    picked = a.select(torch.tensor([True, False, True]))
    assert isinstance(picked, ScaledSystem)
    assert picked.main.shape[0] == 2
    assert picked.capacities.shape[0] == 2


def test_wind_bubble_gradient():
    bubble = BUBBLES[0]
    solver = WindBubbleSolver(128, n_contour=24)
    x = tensor([[2.0, 7.0, 15.0]])

    def value(Pe):
        one = torch.ones_like(Pe)
        problem = BatchedScales(
            Pe,
            tensor([250.0]),
            tensor([6.4]),
            tensor([3.5]),
            one,
            one,
            one,
        )
        return solver.log_psi(x, tensor([[30.0]]), _with_axis(problem)).sum()

    Pe = tensor([bubble.peclet]).requires_grad_(True)
    gradient = torch.autograd.grad(value(Pe), Pe)[0]
    h = 1e-5
    numeric = (value(Pe.detach() + h) - value(Pe.detach() - h)) / (2 * h)
    assert gradient.item() == pytest.approx(numeric.item(), rel=1e-5)


def _with_axis(problem):
    return BatchedScales(
        **{
            name: getattr(problem, name)[..., None]
            for name in problem.__dataclass_fields__
        }
    )


def test_physical_results_against_solver():
    physical = dict(
        L_wind=2e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=20 * const.m_p / u.cm**3,
        t_b=3 * u.Myr,
        E_max=1 * u.PeV,
    )
    energy = 1e4 * u.GeV
    r = np.linspace(0.0, 300.0, 2400) * u.pc
    t = 3 * u.Myr
    reference = simulate(r, energy, t, 84000, **physical).to_value(su.PSI_P)

    bubble = physical_bubble(energy, r_max=300 * u.pc, **physical)
    fast = FastScales(1 * u.PeV)
    mc2 = fast.mc2
    p = tensor([(1e4**2 + 2 * 1e4 * mc2) ** 0.5])
    values = dict(
        L_wind=tensor([2e38]),
        M_dot=tensor([1e-4]),
        rho_0=tensor([20 * const.m_p.to_value(u.g)]),
        t_b=tensor([3.0]),
        r_max=tensor([300.0]),
    )
    problem = fast(p, **values)
    assert problem.peclet.item() == pytest.approx(
        np.ravel(bubble.dimensionless.peclet)[0], rel=1e-6
    )
    problem = _with_axis(problem)
    x = tensor(r.value[1:-1]) / problem.R_TS
    solver = WindBubbleSolver(1024)
    log_psi = solver.log_psi(x, 3.0 / problem.time_unit, problem)[0].numpy()
    # The cell of the origin and those next to the jump of D are local artefacts of the
    # discretization of SAETASS, which do not converge
    radii = r.value[1:-1]
    away = (np.abs(radii - bubble.R_b) > 3.0) & (radii > 3.0)
    error = (np.exp(log_psi) - reference[1:-1])[away]
    assert np.linalg.norm(error) / np.linalg.norm(reference[1:-1][away]) < 5e-3
    assert solver.in_domain(problem, 3.0 / problem.time_unit).all()


@pytest.fixture(scope="module")
def inversion_problem():
    solver = RadialSolver(torch.linspace(0.0, 1.0, 129, dtype=torch.float64), 24)
    true_D = torch.where(solver.centers < 0.4, tensor(0.01), tensor(0.2))
    velocity = torch.zeros(2, 128, dtype=torch.float64)
    source = solver.shell(tensor([0.2, 0.2]))
    times = tensor([0.5, 3.0])
    radii = torch.linspace(0.03, 0.9, 30, dtype=torch.float64)
    with torch.no_grad():
        values = solver.log_psi(true_D.expand(2, -1), velocity, source, times)
        log_data = solver.interpolate(values, radii.expand(2, -1))
    problem = DiffusionInversion(
        solver, velocity, source, times, radii, log_data, 0.02, n_knots=24
    )
    return problem, true_D


def test_inversion_gradient(inversion_problem):
    problem, _ = inversion_problem
    start = torch.full((1, 24), -3.0, dtype=torch.float64).requires_grad_(True)
    data = problem.log_data[None]
    chi2 = problem.objective(start, data)[0].sum()
    gradient = torch.autograd.grad(chi2, start)[0]
    for k in (2, 11, 20):
        step = torch.zeros_like(start.detach())
        step[0, k] = 1e-5
        high = problem.objective(start.detach() + step, data)[0].sum()
        low = problem.objective(start.detach() - step, data)[0].sum()
        numeric = (high - low) / 2e-5
        assert gradient[0, k].item() == pytest.approx(numeric.item(), rel=1e-5)


def test_inversion_recovers_a_jump(inversion_problem):
    problem, true_D = inversion_problem
    estimate = problem.run(-3.0, n_realizations=4, n_steps=500)
    assert estimate.chi2.median() < 2.0
    recovered = problem.log_diffusion(estimate.median)
    centers = problem.solver.centers
    inner = (centers > 0.05) & (centers < 0.3)
    outer = (centers > 0.5) & (centers < 0.85)
    assert (recovered - true_D.log())[inner].abs().median() < 0.2
    assert (recovered - true_D.log())[outer].abs().median() < 0.2
    low, high = estimate.band()
    assert torch.all(low <= high)
