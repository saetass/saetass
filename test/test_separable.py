"""Tests of time-dependent parameters with a separable time dependence (``saetass.parameters``)."""

import astropy.units as u
import numba
import numpy as np
import pytest
from test_kernels import _use_python_kernels
from test_solver_regression import RTOL

from saetass import Grid, Particle, Separable, Solver, State, TimeFactor, parallel
from saetass import units as su
from saetass.parameters import CanonicalSeparable

N_STEPS = 6
R = np.linspace(0.0, 1.0, 30)
P = np.linspace(50.0, 120.0, 20)


def _wave(t):
    return 1.0 + 0.5 * np.sin(30.0 * t.to_value(su.TIME))


def _ramp(t):
    return (1.0 + 2.0 * t.to_value(su.TIME)) * u.dimensionless_unscaled


def _parameter(operator, two_d):
    """Profiles of a parameter of ``operator`` and their unit, on the 1D or 2D grid."""
    shape = (
        (P.size, R.size) if two_d else ((P.size,) if operator == "loss" else (R.size,))
    )
    rng = np.random.default_rng(11)
    if operator == "advection":
        r = np.broadcast_to(R, shape)
        return [1.0 - 1.5 * r, 0.3 * np.ones(shape)], su.VELOCITY
    if operator == "loss":
        p = P[:, None] if two_d else P
        return [-0.5 * np.broadcast_to(p**1.5, shape), -(rng.random(shape) + 0.5)], (
            su.MOMENTUM_LOSS_RATE
        )
    if operator == "diffusion":
        return [0.05 * np.ones(shape), 0.02 * rng.random(shape)], (
            su.DIFFUSION_COEFFICIENT
        )
    return [np.ones(shape), rng.random(shape)], su.SOURCE_PSI_P


def _run(operator, two_d, value, scheme="strang", order=2):
    p = P if (two_d or operator == "loss") else None
    r = None if (operator == "loss" and not two_d) else R
    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=np.linspace(0.0, 0.05, N_STEPS + 1) * su.TIME,
    )
    shape = grid.shape
    psi = np.exp(-np.linspace(0.0, 3.0, int(np.prod(shape)))).reshape(shape)
    params = {"source": {"source": value}}
    if operator == "advection":
        params = {
            "advection": {
                "v_centers": value,
                "inflow_value_U": 0.01 * su.AREA * su.PSI_P,
                "order": order,
                "limiter": "minmod",
                "cfl": 0.8,
            }
        }
    elif operator == "loss":
        params = {
            "loss": {
                "P_dot": value,
                "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
                "order": order,
                "limiter": "vanleer",
                "cfl": 0.8,
            }
        }
    elif operator == "diffusion":
        params = {"diffusion": {"D_values": value, "psi_end": 0.0 * su.PSI_P}}
    state = State(grid=grid, psi_p=psi * su.PSI_P, particle=Particle.PROTON)
    solver = Solver(state, operator, params, splitting_scheme=scheme)
    solver.run()
    return solver.state.psi_p.to_value(su.PSI_P), solver.operator_subsolvers[0]


def _separable_and_callable(operator, two_d):
    profiles, unit = _parameter(operator, two_d)
    separable = Separable([profiles[0] * unit, profiles[1] * unit], [_wave, _ramp])

    def as_callable(*args):
        t = args[-1]
        return _wave(t) * profiles[0] * unit + _ramp(t).value * profiles[1] * unit

    return separable, as_callable


@pytest.mark.parametrize("two_d", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize("operator", ["advection", "loss", "diffusion", "source"])
def test_separable_matches_the_equivalent_callable(operator, two_d):
    separable, as_callable = _separable_and_callable(operator, two_d)
    result, subsolver = _run(operator, two_d, separable)
    expected, reference = _run(operator, two_d, as_callable)
    np.testing.assert_allclose(
        result, expected, rtol=RTOL, atol=RTOL * np.max(np.abs(expected))
    )
    if operator in ("advection", "loss"):
        assert subsolver._separable is not None
        assert subsolver._cfl_subcycles == reference._cfl_subcycles


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("operator", ["advection", "loss"])
def test_separable_kernels_match_their_python_source(operator, order, monkeypatch):
    separable, _ = _separable_and_callable(operator, True)
    compiled, _ = _run(operator, True, separable, order=order)
    _use_python_kernels(monkeypatch)
    python, _ = _run(operator, True, separable, order=order)
    np.testing.assert_array_equal(python, compiled)


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs NUMBA_NUM_THREADS >= 2"
)
@pytest.mark.parametrize("operator", ["advection", "loss"])
def test_separable_parallel_matches_serial(operator, monkeypatch):
    separable, _ = _separable_and_callable(operator, True)
    previous = parallel.get_max_threads()
    try:
        parallel.set_max_threads(1)
        serial, _ = _run(operator, True, separable)
        parallel.set_max_threads(numba.config.NUMBA_NUM_THREADS)
        monkeypatch.setattr(parallel, "MIN_PARALLEL_WORK_NS", 0.0)
        threaded, _ = _run(operator, True, separable)
    finally:
        parallel.set_max_threads(previous)
    np.testing.assert_array_equal(threaded, serial)


def test_zero_steps_keep_the_distribution():
    """An advance of zero steps only converts the state to U and back."""
    separable, _ = _separable_and_callable("loss", True)
    _, subsolver = _run("loss", True, separable)
    state = subsolver.grid  # the grid of the subsolver, to build a state on it
    psi = np.full(state.shape, 2.0)
    zero = State(grid=state, psi_p=psi * su.PSI_P, particle=Particle.PROTON)
    subsolver.advance(0, zero)
    np.testing.assert_allclose(zero.psi_p.to_value(su.PSI_P), psi, rtol=1e-15)
    assert subsolver._cfl_subcycles == 0


def test_calling_a_separable():
    separable = Separable(
        [np.ones(3) * su.VELOCITY, 2.0 * np.ones(3) * su.VELOCITY], [_ramp, 0.5]
    )
    t = 0.5 * su.TIME
    expected = (_ramp(t).value + 1.0) * np.ones(3)
    np.testing.assert_allclose(separable(t).to_value(su.VELOCITY), expected)
    # Sources call their parameters with the coordinates first
    np.testing.assert_allclose(separable(None, None, t).to_value(su.VELOCITY), expected)

    # A factor may also return a plain float
    plain = Separable([np.ones(3) * su.VELOCITY], [_wave])
    np.testing.assert_allclose(plain(t).to_value(su.VELOCITY), _wave(t) * np.ones(3))

    canonical = separable.canonical(su.VELOCITY, "test")
    assert isinstance(canonical, CanonicalSeparable)
    np.testing.assert_allclose(canonical(0.5), expected)
    np.testing.assert_allclose(canonical.factor_values(0.5), [2.0, 0.5])


@pytest.mark.parametrize(
    "profiles, factors, error, match",
    [
        ([], [], ValueError, "at least one profile"),
        ([np.ones(3) * su.VELOCITY], [1.0, 2.0], ValueError, "one factor per profile"),
        ([np.ones(3)], [1.0], TypeError, "astropy Quantities"),
        (
            [np.ones(3) * su.VELOCITY, np.ones(4) * su.VELOCITY],
            [1.0, 1.0],
            ValueError,
            "broadcast to a common shape",
        ),
        (
            [np.ones(3) * su.VELOCITY, np.ones(3) * su.LENGTH],
            [1.0, 1.0],
            u.UnitsError,
            "compatible units",
        ),
        ([np.ones(3) * su.VELOCITY], ["one"], TypeError, "callables or numbers"),
    ],
)
def test_invalid_separable(profiles, factors, error, match):
    with pytest.raises(error, match=match):
        Separable(profiles, factors)


def test_separable_units_are_checked():
    grid = Grid(r_centers=R * su.LENGTH)
    from saetass.solvers import AdvectionSolver, LossSolver

    wrong_unit = Separable([np.ones(R.size) * su.MOMENTUM], [1.0])
    with pytest.raises(u.UnitsError):
        AdvectionSolver.convert_params({"v_centers": wrong_unit}, grid)

    dimensionful_factor = Separable(
        [np.ones(R.size) * su.VELOCITY], [lambda t: 2.0 * su.LENGTH]
    )
    required = {
        "limiter": "minmod",
        "cfl": 0.5,
        "order": 1,
        "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
    }
    canonical = AdvectionSolver.convert_params(
        {**required, "v_centers": dimensionful_factor}, grid
    )["v_centers"]
    with pytest.raises(u.UnitsError):
        canonical(0.0)

    not_dynamic = Separable([np.ones(R.size) * su.VELOCITY], [1.0])
    with pytest.raises(TypeError, match="does not accept a Separable"):
        LossSolver.convert_params({"v_centers_physical": not_dynamic}, grid)


def test_profiles_broadcast_to_a_common_shape():
    separable = Separable(
        [np.ones((2, 1)) * su.VELOCITY, np.arange(3.0) * su.VELOCITY], [_ramp, 1.0]
    )
    assert [profile.shape for profile in separable.profiles] == [(2, 3), (2, 3)]
    t = 0.5 * su.TIME
    expected = _ramp(t).value + np.arange(3.0) * np.ones((2, 1))
    np.testing.assert_allclose(separable(t).to_value(su.VELOCITY), expected)


def test_the_formula_builds_a_separable():
    profiles, unit = _parameter("loss", True)
    first, second = profiles[0] * unit, profiles[1] * unit
    wave, ramp = TimeFactor(_wave), TimeFactor(_ramp)

    separable = first * wave + ramp * second
    assert isinstance(separable, Separable)
    assert separable.factors == [_wave, _ramp]

    t = 0.02 * su.TIME
    assert wave(t) == _wave(t)
    terms = {
        "sum with a constant": (first * wave + second, _wave(t) * first + second),
        "constant first": (second + first * wave, second + _wave(t) * first),
        "difference": (first * wave - second, _wave(t) * first - second),
        "from a constant": (second - first * wave, second - _wave(t) * first),
        "of separables": (first * wave - second * wave, _wave(t) * (first - second)),
        "negation": (-(first * wave), -_wave(t) * first),
        "scaled": (2.0 * (first * wave) * 3, 6.0 * _wave(t) * first),
    }
    for name, (term, expected) in terms.items():
        assert isinstance(term, Separable), name
        np.testing.assert_allclose(
            term(t).to_value(unit), expected.to_value(unit), rtol=1e-15, err_msg=name
        )

    # The formula and the explicit constructor give the same simulation
    explicit, _ = _separable_and_callable("loss", True)
    by_formula = first * TimeFactor(_wave) + second * TimeFactor(_ramp)
    np.testing.assert_array_equal(
        _run("loss", True, by_formula)[0], _run("loss", True, explicit)[0]
    )


def test_invalid_formulas():
    profile = np.ones(3) * su.VELOCITY
    separable = profile * TimeFactor(_wave)
    with pytest.raises(TypeError, match="needs a callable of time"):
        TimeFactor(2.0)
    for invalid in (
        lambda: TimeFactor(_wave) * 2.0,
        lambda: TimeFactor(_wave) * np.ones(3),
        lambda: separable + 1.0,
        lambda: 1.0 + separable,
        lambda: separable - "one",
        lambda: "one" - separable,
        lambda: separable * profile,
        lambda: separable * TimeFactor(_wave),
    ):
        with pytest.raises(TypeError):
            invalid()
