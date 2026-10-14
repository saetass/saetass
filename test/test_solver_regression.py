"""
Regression tests that freeze the numerical output of every operator.

The golden data in ``test/data/golden_solver_outputs.npz`` was generated with the NumPy
implementation of the operators of SAETASS 0.5: a new implementation must reproduce it
to rounding error.
The cases are small and short, but together they reach every branch of the kernels
(orders, limiters, boundary conditions, flow directions, 1D and 2D grids, static and
time-dependent parameters, and both splitting schemes). Regenerate the data with
``--update-golden`` only for an intended, explained change of the numerics.
"""

from pathlib import Path

import numpy as np
import pytest

from saetass import Grid, Particle, Solver, State
from saetass import units as su

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_solver_outputs.npz"

#: Tolerance relative to the largest value of each case, a few hundred roundings.
RTOL = 1e-12

N_R, N_P, N_STEPS = 40, 32, 30
R = np.linspace(0.0, 1.0, N_R)
P = np.logspace(0.0, 4.0, N_P)
R_2D, P_2D = np.linspace(0.0, 1.0, 24), np.logspace(0.0, 4.0, 12)


def _hyperbolic(order=2, limiter="minmod"):
    return {"order": order, "limiter": limiter, "cfl": 0.8}


def _wave(t):
    """Smooth time modulation of the dynamic parameters."""
    return 1.0 + 0.3 * np.sin(20.0 * t.to_value(su.TIME))


def _run(problem_type, params, psi, r=None, p=None, dt=1e-3, scheme="strang"):
    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=dt * np.arange(N_STEPS + 1) * su.TIME,
    )
    state = State(grid=grid, psi_p=psi * su.PSI_P, particle=Particle.PROTON)
    solver = Solver(
        grid=grid,
        state=state,
        problem_type=problem_type,
        operator_params=params,
        splitting_scheme=scheme,
    )
    solver.step(N_STEPS)
    return solver.state.psi_p.to_value(su.PSI_P)


# ---------------------------------------------------------------------------
# Advection in r: the velocity changes sign, so the outer face is an inflow
# ---------------------------------------------------------------------------

V_R = 1.0 - 1.5 * R
PSI_R = np.exp(-(((R - 0.4) / 0.15) ** 2)) + 0.1 * (R > 0.7)


def _advection(order=2, limiter="minmod", dynamic=False, v=V_R):
    velocity = (lambda t: _wave(t) * v * su.VELOCITY) if dynamic else v * su.VELOCITY
    params = {
        "advection": {
            "v_centers": velocity,
            "inflow_value_U": 0.05 * su.AREA * su.PSI_P,
            **_hyperbolic(order, limiter),
        }
    }
    return _run("advection", params, PSI_R, r=R, dt=0.02)


# ---------------------------------------------------------------------------
# Losses in p: losses everywhere, or gains with an inflow at the highest momentum
# ---------------------------------------------------------------------------

PSI_P = P**-2.0 * np.exp(-P / 3e3) + 1e-12 * (P > 1e3)


def _loss(order=2, limiter="minmod", dynamic=False, gain=False):
    P_dot = (1.0 if gain else -1.0) * P**1.5
    rate = (
        (lambda t: _wave(t) * P_dot * su.MOMENTUM_LOSS_RATE)
        if dynamic
        else P_dot * su.MOMENTUM_LOSS_RATE
    )
    inflow = (
        {"inflow_value_psi": 1e-9 * su.PSI_P}
        if gain
        else {"inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P}
    )
    params = {"loss": {"P_dot": rate, **inflow, **_hyperbolic(order, limiter)}}
    return _run("loss", params, PSI_P, p=P, dt=2e-3)


# ---------------------------------------------------------------------------
# Diffusion in r
# ---------------------------------------------------------------------------


def _diffusion(boundary_condition="dirichlet", dynamic=False):
    D = 0.05 * (1.0 + R**2)
    if dynamic:
        params = {
            "D_values": lambda t: _wave(t) * D * su.DIFFUSION_COEFFICIENT,
            "psi_end": lambda t: 0.01 * _wave(t) * su.PSI_P,
        }
    else:
        params = {"D_values": D * su.DIFFUSION_COEFFICIENT, "psi_end": 0.02 * su.PSI_P}
    params["boundary_condition"] = boundary_condition
    return _run("diffusion", {"diffusion": params}, PSI_R, r=R, dt=0.01)


def _source(dynamic=False):
    Q = np.exp(-((R / 0.3) ** 2))
    source = (
        (lambda r, p, t: _wave(t) * Q * su.SOURCE_PSI_P)
        if dynamic
        else Q * su.SOURCE_PSI_P
    )
    return _run("source", {"source": {"source": source}}, PSI_R, r=R, dt=0.01)


# ---------------------------------------------------------------------------
# Full transport equation on a 2D grid
# ---------------------------------------------------------------------------


def _full_2d(dynamic=False, scheme="strang", limiter="minmod"):
    shape = (len(P_2D), len(R_2D))
    v = np.broadcast_to(1.0 - 0.5 * R_2D, shape)
    D = 0.02 * np.outer(P_2D ** (1.0 / 3.0), 1.0 + R_2D)
    P_dot = np.broadcast_to((-0.5 * P_2D**1.5)[:, None], shape)
    Q = np.outer(P_2D**-2.0, np.exp(-((R_2D / 0.2) ** 2)))
    psi = np.outer(P_2D**-2.0, np.exp(-(((R_2D - 0.3) / 0.2) ** 2)))

    def maybe_dynamic(values, unit, coords=False):
        if not dynamic:
            return values * unit
        if coords:
            return lambda r, p, t: _wave(t) * values * unit
        return lambda t: _wave(t) * values * unit

    params = {
        "advection": {
            "v_centers": maybe_dynamic(v, su.VELOCITY),
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic(2, limiter),
        },
        "loss": {
            "P_dot": maybe_dynamic(P_dot, su.MOMENTUM_LOSS_RATE),
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic(2, limiter),
        },
        "source": {"source": maybe_dynamic(Q, su.SOURCE_PSI_P, coords=True)},
        "diffusion": {
            "D_values": maybe_dynamic(D, su.DIFFUSION_COEFFICIENT),
            "psi_end": 0.0 * su.PSI_P,
        },
    }
    return _run(
        "advection-loss-source-diffusion",
        params,
        psi,
        r=R_2D,
        p=P_2D,
        dt=0.01,
        scheme=scheme,
    )


CASES = {
    "advection_order1": lambda: _advection(order=1),
    "advection_minmod": lambda: _advection(limiter="minmod"),
    "advection_vanleer": lambda: _advection(limiter="vanleer"),
    "advection_mc": lambda: _advection(limiter="mc"),
    "advection_dynamic": lambda: _advection(limiter="vanleer", dynamic=True),
    "advection_order1_dynamic": lambda: _advection(order=1, dynamic=True),
    "loss_order1": lambda: _loss(order=1),
    "loss_minmod": lambda: _loss(limiter="minmod"),
    "loss_vanleer": lambda: _loss(limiter="vanleer"),
    "loss_mc": lambda: _loss(limiter="mc"),
    "loss_dynamic": lambda: _loss(limiter="mc", dynamic=True),
    "loss_gain_inflow": lambda: _loss(gain=True),
    "loss_gain_inflow_order1": lambda: _loss(order=1, gain=True),
    "diffusion_dirichlet": lambda: _diffusion("dirichlet"),
    "diffusion_neumann": lambda: _diffusion("neumann"),
    "diffusion_outflow": lambda: _diffusion("outflow"),
    "diffusion_dynamic": lambda: _diffusion("dirichlet", dynamic=True),
    "source_static": lambda: _source(),
    "source_dynamic": lambda: _source(dynamic=True),
    "full_2d_strang": lambda: _full_2d(),
    "full_2d_vanleer": lambda: _full_2d(limiter="vanleer"),
    "full_2d_dynamic_lie": lambda: _full_2d(dynamic=True, scheme="lie"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_golden_output(case, update_golden):
    result = CASES[case]()
    assert np.all(np.isfinite(result))

    if update_golden:
        golden = dict(np.load(GOLDEN_PATH)) if GOLDEN_PATH.exists() else {}
        golden[case] = result
        np.savez_compressed(GOLDEN_PATH, **golden)
        pytest.skip(f"Golden output of {case!r} updated")

    with np.load(GOLDEN_PATH) as golden:
        expected = golden[case]
    np.testing.assert_allclose(
        result, expected, rtol=RTOL, atol=RTOL * np.max(np.abs(expected))
    )
