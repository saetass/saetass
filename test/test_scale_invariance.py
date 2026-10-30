"""
Scale invariance of the numerical kernels.

Expressing a problem in units 2^k times larger or smaller must not change its solution:
multiplying by powers of two is exact in floating point, so the solution must be bitwise
identical once the scaling is undone. These tests guarantee that no kernel depends on the
absolute magnitude of its inputs (through absolute tolerances, or through products that
under- or overflow), whatever units or normalization the inputs are expressed in.

Each problem is defined in dimensionless variables and handed to SAETASS with lengths,
times and the distribution expressed in scales (L, T, Psi). Momentum scales are kept at 1,
since the logarithmic momentum coordinate log10(p) is not exactly shifted by powers of two.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pytest

from saetass import Grid, Solver, State
from saetass import units as su

REFERENCE = (1.0, 1.0, 1.0)
SCALES = [
    pytest.param((2.0**5, 2.0**-3, 2.0**40), id="moderate"),
    # Distributions far beyond 1e+-154, where products of slopes over- or underflow
    pytest.param((2.0**-4, 2.0**6, 2.0**560), id="psi~1e+168"),
    pytest.param((2.0**3, 2.0**-2, 2.0**-600), id="psi~1e-181"),
]


def _run(grid, psi_initial, problem_type, operator_params, psi_scale):
    solver = Solver(
        state=State(grid=grid, psi_p=psi_initial * psi_scale * su.PSI_P),
        problem_type=problem_type,
        operator_params=operator_params,
    )
    return solver.run().psi_p.to_value(su.PSI_P) / psi_scale


def _plot(x, reference, scaled, scales, title, xlabel, log=False):
    """Overlay the reference solution and the rescaled one, after undoing the scaling."""
    L, T, Psi = scales
    plt.figure(figsize=(10, 6))
    plot = plt.loglog if log else plt.plot
    for i, (ref_row, scaled_row) in enumerate(
        zip(np.atleast_2d(reference), np.atleast_2d(scaled))
    ):
        plot(x, ref_row, lw=3, alpha=0.5, label="Reference" if i == 0 else None)
        plot(
            x,
            scaled_row,
            "k:",
            lw=2,
            label=f"Scaled (L={L:g}, T={T:g}, Psi={Psi:.3g})" if i == 0 else None,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()


def _step_and_bump(x, x0, width):
    """Smooth bump plus a discontinuity, to exercise the slope limiters."""
    return np.exp(-(((x - x0) / width) ** 2)) + 0.5 * (x > x0 + 2 * width)


def _advection(scales, limiter):
    L, T, Psi = scales
    r = np.linspace(0.0, 10.0, 48)
    t = np.linspace(0.0, 1.0, 17)

    def velocity(time):
        t_hat = time.to_value(su.TIME) / T
        return r / (1.0 + t_hat) * (L / T) * su.VELOCITY

    def source(r_phys, p_phys, time):
        r_hat, t_hat = r_phys.to_value(su.LENGTH) / L, time.to_value(su.TIME) / T
        return np.exp(-r_hat) * (1.0 + t_hat) * (Psi / T) * su.SOURCE_PSI_P

    grid = Grid(r_centers=r * L * su.LENGTH, t_grid=t * T * su.TIME)
    params = {
        "advection": {
            "v_centers": velocity,
            "limiter": limiter,
            "order": 2,
            "cfl": 0.5,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        },
        "source": {"source": source},
    }
    return _run(grid, _step_and_bump(r, 3.0, 0.8), "advection-source", params, Psi)


def _diffusion(scales):
    L, T, Psi = scales
    r = np.linspace(0.0, 5.0, 48)
    t = np.linspace(0.0, 1.0, 17)

    def diffusion_coefficient(time):
        t_hat = time.to_value(su.TIME) / T
        return (
            np.full(r.size, 0.05 * (1.0 + t_hat))
            * (L**2 / T)
            * su.DIFFUSION_COEFFICIENT
        )

    def psi_end(time):
        return 0.1 * (1.0 + time.to_value(su.TIME) / T) * Psi * su.PSI_P

    def source(r_phys, p_phys, time):
        return (
            np.exp(-((r_phys.to_value(su.LENGTH) / L) ** 2))
            * (Psi / T)
            * su.SOURCE_PSI_P
        )

    grid = Grid(r_centers=r * L * su.LENGTH, t_grid=t * T * su.TIME)
    params = {
        "diffusion": {"D_values": diffusion_coefficient, "psi_end": psi_end},
        "source": {"source": source},
    }
    return _run(grid, _step_and_bump(r, 2.0, 0.5), "diffusion-source", params, Psi)


def _loss(scales, limiter):
    _, T, Psi = scales
    p = np.logspace(0, 3, 48)
    t = np.linspace(0.0, 0.05, 11)
    grid = Grid(p_centers=p * su.MOMENTUM, t_grid=t * T * su.TIME)
    params = {
        "loss": {
            "P_dot": -(p**2) / T * su.MOMENTUM_LOSS_RATE,
            "limiter": limiter,
            "order": 2,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
        },
        "source": {"source": p**-2.0 * (Psi / T) * su.SOURCE_PSI_P},
    }
    return _run(grid, _step_and_bump(np.log10(p), 1.5, 0.3), "loss-source", params, Psi)


def _all_operators_2d(scales):
    L, T, Psi = scales
    r = np.linspace(0.0, 10.0, 24)
    p = np.logspace(0, 2, 12)
    t = np.linspace(0.0, 0.5, 9)
    shape = (p.size, r.size)
    grid = Grid(
        r_centers=r * L * su.LENGTH, p_centers=p * su.MOMENTUM, t_grid=t * T * su.TIME
    )
    params = {
        "advection": {
            "v_centers": np.broadcast_to(1.0 + 0.1 * r, shape) * (L / T) * su.VELOCITY,
            "limiter": "vanleer",
            "order": 2,
            "cfl": 0.5,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        },
        "diffusion": {
            "D_values": np.broadcast_to(0.05 * p[:, None] ** 0.5, shape)
            * (L**2 / T)
            * su.DIFFUSION_COEFFICIENT
        },
        "loss": {
            "P_dot": np.broadcast_to(-0.5 * p[:, None] ** 2, shape)
            / T
            * su.MOMENTUM_LOSS_RATE,
            "limiter": "vanleer",
            "order": 2,
            "cfl": 0.5,
            "inflow_value_psi": 0.0 * su.PSI_P,
        },
        "source": {
            "source": np.outer(p**-2.0, np.exp(-r)) * (Psi / T) * su.SOURCE_PSI_P
        },
    }
    psi_initial = np.outer(p**-2.0, _step_and_bump(r, 4.0, 1.0))
    return _run(grid, psi_initial, "advection-diffusion-loss-source", params, Psi)


def _shock_acceleration(scales):
    """Advection, diffusion and adiabatic losses across a strong shock at r = 5."""
    L, T, Psi = scales
    r = np.linspace(0.0, 10.0, 40)
    p = np.logspace(0, 2, 16)
    t = np.linspace(0.0, 2.0, 9)
    grid = Grid(
        r_centers=r * L * su.LENGTH, p_centers=p * su.MOMENTUM, t_grid=t * T * su.TIME
    )
    v = np.where(r < 5.0, 2.0, 0.5 * (5.0 / np.maximum(r, 5.0)) ** 2)
    params = {
        "advection": {
            "v_centers": v * (L / T) * su.VELOCITY,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
        },
        "diffusion": {
            "D_values": np.full((p.size, r.size), 0.5)
            * (L**2 / T)
            * su.DIFFUSION_COEFFICIENT
        },
        "loss": {"adiabatic_losses": True},
        "source": {
            "source": np.outer(p == p[2], np.exp(-((r - 5.0) ** 2)))
            * (Psi / T)
            * su.SOURCE_PSI_P
        },
    }
    psi_initial = np.zeros((p.size, r.size))
    return _run(grid, psi_initial, "advection-diffusion-loss-source", params, Psi)


@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("limiter", ["minmod", "vanleer", "mc"])
def test_advection_is_scale_invariant(limiter, scales, plot_results):
    scaled, reference = _advection(scales, limiter), _advection(REFERENCE, limiter)
    np.testing.assert_array_equal(scaled, reference)

    if plot_results:
        _plot(
            np.linspace(0.0, 10.0, 48),
            reference,
            scaled,
            scales,
            f"Advection-source, {limiter} limiter",
            "r / L",
        )


@pytest.mark.parametrize("scales", SCALES)
def test_diffusion_is_scale_invariant(scales, plot_results):
    scaled, reference = _diffusion(scales), _diffusion(REFERENCE)
    np.testing.assert_array_equal(scaled, reference)

    if plot_results:
        _plot(
            np.linspace(0.0, 5.0, 48),
            reference,
            scaled,
            scales,
            "Diffusion-source",
            "r / L",
        )


@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("limiter", ["minmod", "vanleer", "mc"])
def test_loss_is_scale_invariant(limiter, scales, plot_results):
    scaled, reference = _loss(scales, limiter), _loss(REFERENCE, limiter)
    np.testing.assert_array_equal(scaled, reference)

    if plot_results:
        _plot(
            np.logspace(0, 3, 48),
            reference,
            scaled,
            scales,
            f"Loss-source, {limiter} limiter",
            "p [GeV/c]",
            log=True,
        )


@pytest.mark.parametrize("scales", SCALES)
def test_coupled_2d_problem_is_scale_invariant(scales, plot_results):
    reference = _all_operators_2d(REFERENCE)
    assert np.all(np.isfinite(reference)) and np.any(reference > 0)
    scaled = _all_operators_2d(scales)
    np.testing.assert_array_equal(scaled, reference)

    if plot_results:
        rows = slice(None, None, 3)  # every third momentum
        _plot(
            np.linspace(0.0, 10.0, 24),
            reference[rows],
            scaled[rows],
            scales,
            "Advection-diffusion-loss-source, every third momentum",
            "r / L",
        )


@pytest.mark.parametrize("scales", SCALES)
def test_adiabatic_losses_at_a_shock_are_scale_invariant(scales, plot_results):
    reference = _shock_acceleration(REFERENCE)
    # Particles injected at the third momentum are accelerated to the higher ones
    assert np.all(np.isfinite(reference)) and np.any(reference[5:] > 0)
    scaled = _shock_acceleration(scales)
    np.testing.assert_array_equal(scaled, reference)

    if plot_results:
        rows = slice(None, None, 3)  # every third momentum
        _plot(
            np.linspace(0.0, 10.0, 40),
            reference[rows],
            scaled[rows],
            scales,
            "Shock acceleration, every third momentum",
            "r / L",
        )
