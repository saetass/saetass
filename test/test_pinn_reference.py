"""Tests of the dimensionless wind bubble and of the reference solutions of the surrogates."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import units as su
from saetass.pinn.confirm import simulate
from saetass.pinn.reference import reference_grid, solve
from saetass.pinn.wind_bubble import (
    DimensionlessBubble,
    FastScales,
    log_steady_state,
    physical_bubble,
    steady_state,
)
from saetass.utils.bubble_profiles import BubbleProfileCalculator

BUBBLES = [
    DimensionlessBubble(22.0, 250.0, 6.4, 3.5),
    DimensionlessBubble(1.0, 270.0, 6.4, 3.5),
    DimensionlessBubble(100.0, 500.0, 15.0, 2.0),
    DimensionlessBubble(0.05, 100.0, 3.0, 6.0),
    DimensionlessBubble(5.0, 900.0, 8.0, 3.0, delta=0.5),
]
PHYSICAL = dict(
    L_wind=2e38 * u.erg / u.s,
    M_dot=1e-4 * const.M_sun / u.yr,
    rho_0=20 * const.m_p / u.cm**3,
    t_b=3 * u.Myr,
)


def _flux(x, bubble):
    """Total flux x^2 (v psi - D dpsi/dx) of the steady state, by centered differences."""
    h = 1e-6 * x
    gradient = (steady_state(x + h, bubble) - steady_state(x - h, bubble)) / (2 * h)
    psi = steady_state(x, bubble)
    a = bubble.bubble_ratio
    v = np.where(x < 1.0, 4.0, np.where(x < a, x**-2.0, 0.0))
    D = np.where(
        x < 1.0,
        bubble.wind_diffusion * x**bubble.delta,
        np.where(x < a, 1.0 / bubble.peclet, bubble.diffusion_ratio / bubble.peclet),
    )
    return x**2 * (v * psi - D * gradient)


@pytest.mark.parametrize("bubble", BUBBLES)
def test_steady_state_has_the_flux_of_the_source(bubble):
    a, outer = bubble.bubble_ratio, bubble.outer_radius
    # In the wind the flux vanishes: use the logarithmic form where psi underflows
    wind = np.linspace(0.9, 0.999, 20)
    assert np.allclose(_flux(wind, bubble), 0.0, atol=1e-6 * steady_state(1.0, bubble))
    beyond = np.concatenate(
        [np.linspace(1.001, a * 0.999, 40), np.linspace(a * 1.001, outer * 0.99, 40)]
    )
    assert np.allclose(_flux(beyond, bubble), 1.0, rtol=1e-6)


@pytest.mark.parametrize("bubble", BUBBLES)
def test_steady_state_is_continuous(bubble):
    a = bubble.bubble_ratio
    for point in (1.0, a):
        left, right = steady_state(np.array([point * (1 - 1e-10), point]), bubble)
        assert left == pytest.approx(right, rel=1e-6)
    assert steady_state(np.array([bubble.outer_radius]), bubble)[0] == pytest.approx(
        0.0, abs=1e-14
    )


@pytest.mark.parametrize("bubble", BUBBLES)
def test_log_steady_state(bubble):
    x = np.linspace(0.5, bubble.outer_radius * 0.99, 200)
    psi = steady_state(x, bubble)
    finite = psi > 1e-250
    assert np.allclose(np.exp(log_steady_state(x, bubble))[finite], psi[finite])


def test_steady_state_matches_the_analytic_profile_of_the_calculator():
    # That profile assumes an infinite medium, without the Dirichlet boundary of a run
    r = np.linspace(0.0, 300.0, 3000) * u.pc
    calculator = BubbleProfileCalculator(r_grid=r, model="Morlino21", **PHYSICAL)
    setup = calculator.get_all_profiles(
        1e3 * u.GeV, eta_B=0.1, eta_inj=0.1, E_max=1 * u.PeV
    )
    profile = calculator.compute_analytical_CR_profile(
        D_values=setup["D_values"], f_gal=0.0, f_TS=1.0
    )
    scales = physical_bubble(1e3 * u.GeV, r_max=300 * u.pc, E_max=1 * u.PeV, **PHYSICAL)
    x = r.to_value(su.LENGTH) / scales.R_TS
    bubble = scales.dimensionless
    bubble = DimensionlessBubble(
        float(bubble.peclet[0]),
        float(bubble.diffusion_ratio[0]),
        float(bubble.bubble_ratio),
        1e12,
        bubble.delta,
    )
    inside = (x > 1.0) & (x < bubble.bubble_ratio)
    psi = steady_state(x, bubble) / steady_state(1.0, bubble)
    assert np.allclose(psi[inside], profile[inside], rtol=1e-3)


def test_physical_bubble_uses_the_profiles_of_the_calculator():
    r = np.array([5.0, 40.0, 200.0]) * u.pc
    calculator = BubbleProfileCalculator(r_grid=r, model="Morlino21", **PHYSICAL)
    D = calculator.compute_diffusion_profile(100 * u.GeV).to_value(
        su.DIFFUSION_COEFFICIENT
    )
    scales = physical_bubble(100 * u.GeV, r_max=300 * u.pc, E_max=1 * u.PeV, **PHYSICAL)
    assert scales.R_TS == pytest.approx(calculator.R_TS.to_value(su.LENGTH))
    assert scales.R_b == pytest.approx(calculator.R_b.to_value(su.LENGTH))
    assert scales.D_b[0] == pytest.approx(D[1])
    assert scales.D_ISM[0] == pytest.approx(D[2])
    wind = scales.dimensionless.wind_diffusion[0] * (5.0 / scales.R_TS) ** (1.0 / 3.0)
    assert wind == pytest.approx(D[0] / (scales.v_b * scales.R_TS))


@pytest.mark.parametrize("model", ["kolmogorov", "kraichnan", "bohm"])
@pytest.mark.parametrize(
    "L_wind, M_dot, density, age, eta_B",
    [
        (2e38, 1e-4, 20.0, 3.0, 0.1),
        (3e37, 8e-6, 1.5, 7.0, 0.02),
        (1e39, 3e-4, 90.0, 1.2, 0.3),
    ],
)
def test_fast_scales_match_the_calculator(model, L_wind, M_dot, density, age, eta_B):
    p = np.array([3.0, 1e2, 1e5])
    rho = (density * const.m_p / u.cm**3).to_value(u.g / u.cm**3)
    exact = physical_bubble(
        p * su.MOMENTUM,
        L_wind=L_wind * u.erg / u.s,
        M_dot=M_dot * const.M_sun / u.yr,
        rho_0=rho * u.g / u.cm**3,
        t_b=age * u.Myr,
        r_max=2000 * u.pc,
        E_max=1 * u.PeV,
        eta_B=eta_B,
        diffusion_model=model,
    )
    fast = FastScales(1 * u.PeV, diffusion_model=model)(
        p, L_wind, M_dot, rho, age, 2000.0, eta_B=eta_B
    )
    numbers = exact.dimensionless
    assert np.allclose(fast.peclet, numbers.peclet, rtol=1e-9)
    assert np.allclose(fast.diffusion_ratio, numbers.diffusion_ratio, rtol=1e-9)
    assert np.isclose(fast.bubble_ratio, numbers.bubble_ratio, rtol=1e-9)
    assert np.isclose(fast.domain_ratio, numbers.domain_ratio, rtol=1e-9)
    assert np.isclose(fast.R_TS, exact.R_TS, rtol=1e-9)
    assert np.isclose(fast.time_unit, exact.time_unit, rtol=1e-9)
    assert np.allclose(fast.psi_unit, exact.psi_unit, rtol=1e-6)


def test_physical_bubble_rejects_a_domain_inside_the_bubble():
    with pytest.raises(ValueError, match="r_max"):
        physical_bubble(1 * u.GeV, r_max=10 * u.pc, E_max=1 * u.PeV, **PHYSICAL)


def test_physical_bubble_rejects_a_core():
    with pytest.raises(ValueError, match="R_c"):
        physical_bubble(
            1 * u.GeV, r_max=300 * u.pc, E_max=1 * u.PeV, R_c=1 * u.pc, **PHYSICAL
        )


def test_injection_does_not_depend_on_the_momenta_evaluated():
    one = physical_bubble(1e3 * u.GeV, r_max=300 * u.pc, E_max=1 * u.PeV, **PHYSICAL)
    two = physical_bubble(
        [1.0, 1e3] * u.GeV, r_max=300 * u.pc, E_max=1 * u.PeV, **PHYSICAL
    )
    assert one.rate[0] == pytest.approx(two.rate[1])


@pytest.mark.parametrize("bubble", BUBBLES)
def test_grid_has_nodes_on_the_interfaces(bubble):
    x = reference_grid(bubble)
    assert x[0] == 0.0
    assert np.all(np.diff(x) > 0.0)
    assert 1.0 in x
    assert bubble.bubble_ratio in x
    assert x[-1] == bubble.outer_radius


@pytest.mark.parametrize("bubble", BUBBLES)
def test_solution_reaches_the_exact_steady_state(bubble):
    solution = solve(bubble, np.array([1e6]))
    significant = solution.steady > 1e-6 * solution.steady.max()
    assert np.allclose(
        solution.psi[0, significant], solution.steady[significant], rtol=1e-3
    )


@pytest.mark.parametrize("bubble", BUBBLES)
def test_solution_conserves_particles_before_they_escape(bubble):
    time = 1e-3
    solution = solve(bubble, np.array([time]), resolution=2)
    x = solution.x
    mid = 0.5 * (x[1:] + x[:-1])
    edges = np.concatenate([[0.0], mid, [x[-1]]])
    volumes = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    assert np.sum(volumes * solution.psi[0]) == pytest.approx(time, rel=1e-9)


@pytest.mark.parametrize("bubble", BUBBLES)
def test_solution_stays_above_the_floor_of_the_datasets_and_grows(bubble):
    solution = solve(bubble, np.array([1.0, 10.0, 100.0]))
    assert np.all(solution.psi >= -1e-11 * solution.psi.max())
    assert np.all(np.diff(solution.psi, axis=0) >= -1e-11 * solution.psi.max())


def test_solution_converges_at_second_order():
    bubble = DimensionlessBubble(1.0, 200.0, 5.0, 3.0)
    times = np.array([30.0])
    fine = solve(bubble, times, resolution=8)

    def error(resolution):
        coarse = solve(bubble, times, resolution=resolution)
        psi = np.interp(fine.x, coarse.x, coarse.psi[0])
        return np.linalg.norm(psi - fine.psi[0]) / np.linalg.norm(fine.psi[0])

    assert error(2) / error(4) > 2.5


def test_solution_agrees_with_the_solver():
    energy = 1e5 * u.GeV
    r = np.linspace(0.0, 300.0, 1200)
    end = 0.3
    psi = simulate(
        r * u.pc, energy, end * u.Myr, 5000, E_max=1 * u.PeV, **PHYSICAL
    ).to_value(su.PSI_P)

    scales = physical_bubble(energy, r_max=300 * u.pc, E_max=1 * u.PeV, **PHYSICAL)
    numbers = scales.dimensionless
    bubble = DimensionlessBubble(
        float(numbers.peclet[0]),
        float(numbers.diffusion_ratio[0]),
        float(numbers.bubble_ratio),
        float(numbers.domain_ratio),
        numbers.delta,
    )
    solution = solve(bubble, np.array([end / scales.time_unit]), resolution=4)
    reference = np.interp(
        r / scales.R_TS, solution.x, solution.psi[0] * scales.psi_unit[0]
    )
    # The cell of the origin and those next to the jump of D are local artefacts of the
    # discretization, which do not converge: see the exact steady state
    away = (np.abs(r - scales.R_b) > 3.0) & (r > 3.0)
    assert (
        np.linalg.norm((psi - reference)[away]) / np.linalg.norm(reference[away]) < 5e-3
    )
