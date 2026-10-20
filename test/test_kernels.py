"""
Tests of the Numba kernels of the hyperbolic and diffusion operators, and of the inputs
they reject.

Coverage cannot trace code compiled by Numba, so the kernels are also run with every
compiled function replaced by its Python source, and both must give bitwise the same
results. They do because Numba compiles every floating-point operation as written, in the
same order and with the same double-precision rounding as Python: the kernels are compiled
without ``fastmath``, which would let the compiler reorder or fuse operations.
"""

import numpy as np
import pytest
from numba.core.registry import CPUDispatcher
from test_solver_regression import CASES, RTOL

import saetass.solvers.diffusion_kernels as diffusion_kernels
import saetass.solvers.hyperbolic_kernels as hyperbolic_kernels
from saetass import (
    AdvectionSolver,
    DiffusionSolver,
    Grid,
    LossSolver,
    Particle,
    Solver,
    State,
)
from saetass import units as su
from saetass.parallel import TwinKernel
from saetass.solvers.hyperbolic_solver import HyperbolicSolver

#: Golden cases reaching every branch of the kernels.
PYTHON_CASES = [
    "advection_order1",
    "advection_minmod",
    "advection_vanleer",
    "advection_mc",
    "loss_minmod",
    "loss_gain_inflow",
    "loss_gain_inflow_order1",
    "diffusion_dirichlet",
    "diffusion_neumann",
    "diffusion_outflow",
    "diffusion_dynamic",
    "full_2d_dynamic_lie",
]


def _use_python_kernels(monkeypatch):
    """Replace every compiled kernel by its Python source."""
    for module in (hyperbolic_kernels, diffusion_kernels):
        for name, value in list(vars(module).items()):
            if isinstance(value, CPUDispatcher):
                monkeypatch.setattr(module, name, value.py_func)
            elif isinstance(value, TwinKernel):
                py_func = value.py_func
                monkeypatch.setattr(
                    module, name, lambda n_threads, *args, f=py_func: f(*args)
                )


@pytest.mark.parametrize("case", PYTHON_CASES)
def test_python_source_matches_compiled_kernels(case):
    compiled = CASES[case]()
    with pytest.MonkeyPatch.context() as monkeypatch:
        _use_python_kernels(monkeypatch)
        python = CASES[case]()
    np.testing.assert_array_equal(python, compiled)


def _hyperbolic_solver(problem_type, params, r=None, p=None, psi=None):
    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=np.linspace(0.0, 1e-3, 3) * su.TIME,
    )
    state = State(grid=grid, psi_p=psi * su.PSI_P, particle=Particle.PROTON)
    return Solver(grid, state, problem_type, {problem_type: params})


def test_two_cell_grid_has_zero_slopes():
    """With two cells, MUSCL-Hancock has no limited slopes but still runs."""
    params = {
        "P_dot": -np.ones(2) * su.MOMENTUM_LOSS_RATE,
        "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
        "order": 2,
        "limiter": "minmod",
        "cfl": 0.5,
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        _use_python_kernels(monkeypatch)
        solver = _hyperbolic_solver(
            "loss", params, p=np.array([50.0, 100.0]), psi=np.ones(2)
        )
        result = solver.run().psi_p.to_value(su.PSI_P)
    assert np.all(np.isfinite(result)) and np.all(result >= 0.0)


def _advection_params(order, inflow):
    return {
        "v_centers": np.ones(5) * su.VELOCITY,
        "inflow_value_U": inflow * su.AREA * su.PSI_P,
        "order": order,
        "limiter": "minmod",
        "cfl": 0.5,
    }


def test_advection_away_from_the_origin_needs_first_order():
    """Second-order advection only supports grids whose first face is r = 0."""
    r = np.linspace(1.0, 2.0, 5)
    first = _hyperbolic_solver(
        "advection", _advection_params(1, 0.0), r=r, psi=np.ones(5)
    )
    assert np.all(np.isfinite(first.run().psi_p.to_value(su.PSI_P)))

    second = _hyperbolic_solver(
        "advection", _advection_params(2, 0.0), r=r, psi=np.ones(5)
    )
    with pytest.raises(ValueError, match="Invalid left face"):
        second.run()


def test_inflow_must_match_the_rows():
    r, p = np.linspace(0.0, 1.0, 5), np.array([50.0, 75.0, 100.0])
    params = _advection_params(2, np.zeros(2))
    with pytest.raises(ValueError, match="inflow_value_U must have 1 or 3 values"):
        _hyperbolic_solver("advection", params, r=r, p=p, psi=np.ones((3, 5)))


class _PythonTransformLoss(LossSolver):
    _KERNEL_TRANSFORM = hyperbolic_kernels.TRANSFORM_NONE


class _PythonTransformAdvection(AdvectionSolver):
    _KERNEL_TRANSFORM = hyperbolic_kernels.TRANSFORM_NONE


def _subsolver(solver_class, two_d, dynamic):
    """A hyperbolic subsolver on a small grid, with its state."""
    r = np.linspace(0.0, 1.0, 12)
    p = np.array([50.0, 60.0, 75.0, 90.0, 100.0]) if two_d else None
    is_loss = issubclass(solver_class, LossSolver)
    if is_loss and not two_d:
        r, p = None, np.linspace(50.0, 100.0, 12)
    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
    )
    shape = grid.shape
    rng = np.random.default_rng(5)
    if is_loss:
        rate = -(rng.random(shape) + 0.5)
        values, unit, key = rate, su.MOMENTUM_LOSS_RATE, "P_dot"
        inflow = {"inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P}
    else:
        velocity = 1.0 - 1.5 * np.broadcast_to(r, shape)
        values, unit, key = velocity, su.VELOCITY, "v_centers"
        inflow = {"inflow_value_U": 0.01 * su.AREA * su.PSI_P}
    param = (
        (lambda t: (1.0 + t.to_value(su.TIME)) * values * unit)
        if dynamic
        else values * unit
    )
    params = solver_class.convert_params(
        {key: param, **inflow, "order": 2, "limiter": "mc", "cfl": 0.8}, grid
    )
    subsolver = solver_class(grid, np.array([0.0, 0.05]), params)
    state = State(
        grid=grid, psi_p=rng.random(shape) * su.PSI_P, particle=Particle.PROTON
    )
    return subsolver, state


@pytest.mark.parametrize("n_steps", [0, 2])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("two_d", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize(
    "kernel_class, python_class",
    [
        (AdvectionSolver, _PythonTransformAdvection),
        (LossSolver, _PythonTransformLoss),
    ],
    ids=["advection", "loss"],
)
def test_kernel_and_python_transforms_agree(
    kernel_class, python_class, two_d, dynamic, n_steps
):
    """
    Converting to U in the kernels or in Python gives the same results.

    Losses agree bitwise. The kernels multiply by :math:`1/r^2` where Python divides by
    :math:`r^2`, so advection agrees to rounding.
    """
    results = []
    for solver_class in (kernel_class, python_class):
        subsolver, state = _subsolver(solver_class, two_d, dynamic)
        for _ in range(3):
            subsolver.advance(n_steps, state)
        results.append(state._values.copy())
    if kernel_class is LossSolver:
        np.testing.assert_array_equal(results[0], results[1])
    else:
        scale = np.max(np.abs(results[1]))
        np.testing.assert_allclose(results[0], results[1], rtol=RTOL, atol=RTOL * scale)


def test_advection_from_the_origin_needs_three_cells():
    with pytest.raises(ValueError, match="at least 3 radial cells"):
        _hyperbolic_solver(
            "advection",
            {**_advection_params(2, 0.0), "v_centers": np.ones(2) * su.VELOCITY},
            r=np.array([0.0, 1.0]),
            psi=np.ones(2),
        )


def test_base_class_has_no_kernel_scales():
    subsolver, _ = _subsolver(LossSolver, two_d=False, dynamic=False)
    scale, inv_scale = HyperbolicSolver._kernel_scales(subsolver)
    assert scale.size == inv_scale.size == 0


def test_zero_pivots_are_replaced():
    """An exactly zero pivot is replaced by machine epsilon instead of dividing by zero."""
    lower = np.array([0.0, 0.0, 1.0])
    diag = np.array([0.0, 0.0, 2.0])
    upper = np.array([1.0, 1.0, 0.0])
    inv_pivots, normalized = np.empty((1, 3)), np.empty((1, 3))
    diffusion_kernels._factorize_block.py_func(
        lower[None], diag[None], upper[None], inv_pivots, normalized, 0, 1
    )
    eps = np.finfo(np.float64).eps
    assert inv_pivots[0, 0] == 1.0 / eps
    assert inv_pivots[0, 1] == 1.0 / eps
    assert np.all(np.isfinite(normalized))


def test_psi_end_must_match_the_momenta():
    r, p = np.linspace(0.0, 1.0, 5), np.array([50.0, 75.0, 100.0])
    grid = Grid(
        r_centers=r * su.LENGTH,
        p_centers=p * su.MOMENTUM,
        t_grid=np.linspace(0.0, 1e-3, 3) * su.TIME,
    )
    state = State(grid=grid, psi_p=np.ones((3, 5)) * su.PSI_P, particle=Particle.PROTON)
    params = {
        "D_values": np.ones((3, 5)) * su.DIFFUSION_COEFFICIENT,
        "psi_end": lambda t: np.zeros(2) * su.PSI_P,
    }
    solver = Solver(grid, state, "diffusion", {"diffusion": params})
    with pytest.raises(ValueError, match="psi_end must have 1 or 3 values"):
        solver.run()


def _numpy_matrices(sub, D_values, dt):
    """The Crank-Nicolson matrices built with NumPy array operations, the reference for the compiled assembly."""
    D_face, G = np.empty_like(sub._D_face), np.empty_like(sub._G)
    sub._compute_conductances(D_values, D_face, G)
    a_left, a_right = G[:, :-1], G[:, 1:]
    A_diag = 2.0 * sub.V_b / dt + (a_left + a_right)
    B_diag = 2.0 * sub.V_b / dt - (a_left + a_right)
    A_lower, B_lower = np.zeros_like(A_diag), np.zeros_like(A_diag)
    A_upper, B_upper = np.zeros_like(A_diag), np.zeros_like(A_diag)
    A_lower[:, 1:], B_lower[:, 1:] = -a_left[:, 1:], a_left[:, 1:]
    A_upper[:, :-1], B_upper[:, :-1] = -a_right[:, :-1], a_right[:, :-1]
    if sub.boundary_condition == "dirichlet":
        for matrix in (A_lower, A_upper, B_diag, B_lower, B_upper):
            matrix[:, -1] = 0.0
        A_diag[:, -1] = 1.0
    return D_face, G, A_lower, A_diag, A_upper, B_lower, B_diag, B_upper


@pytest.mark.parametrize("python", [False, True], ids=["compiled", "python"])
@pytest.mark.parametrize("boundary_condition", ["dirichlet", "neumann", "outflow"])
def test_compiled_assembly_matches_numpy(boundary_condition, python, monkeypatch):
    """The compiled assembly (and its Python source) reproduce the NumPy matrices bitwise."""
    if python:
        _use_python_kernels(monkeypatch)
    r = np.array([0.0, 0.1, 0.25, 0.3, 0.5, 0.8, 1.0])
    p = np.array([50.0, 60.0, 75.0])
    grid = Grid(r_centers=r * su.LENGTH, p_centers=p * su.MOMENTUM)
    rng = np.random.default_rng(7)
    D = rng.random((3, 7)) + 0.1
    # Infinite coefficients on both sides make a harmonic denominator vanish
    D[1, 3:5] = np.inf
    params = DiffusionSolver.convert_params(
        {
            "D_values": D * su.DIFFUSION_COEFFICIENT,
            "boundary_condition": boundary_condition,
        },
        grid,
    )
    sub = DiffusionSolver(grid, np.array([0.0, 0.01]), params)
    with np.errstate(divide="ignore"):
        sub._build_matrices(0.01)
    compiled = (
        sub._D_face,
        sub._G,
        sub._A_lower,
        sub._A_diag,
        sub._A_upper,
        sub._B_lower,
        sub._B_diag,
        sub._B_upper,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = _numpy_matrices(sub, D, 0.01)
    for got, want in zip(compiled, expected):
        np.testing.assert_array_equal(got, want)


def test_unknown_boundary_condition():
    grid = Grid(r_centers=np.linspace(0.0, 1.0, 5) * su.LENGTH)
    params = DiffusionSolver.convert_params(
        {
            "D_values": np.ones(5) * su.DIFFUSION_COEFFICIENT,
            "boundary_condition": "robin",
        },
        grid,
    )
    with pytest.raises(ValueError, match="Unknown boundary_condition: robin"):
        DiffusionSolver(grid, np.array([0.0, 0.01]), params)


def test_dynamic_coefficients_must_match_the_grid():
    r, p = np.linspace(0.0, 1.0, 5), np.array([50.0, 75.0, 100.0])
    grid = Grid(
        r_centers=r * su.LENGTH,
        p_centers=p * su.MOMENTUM,
        t_grid=np.linspace(0.0, 1e-3, 3) * su.TIME,
    )
    state = State(grid=grid, psi_p=np.ones((3, 5)) * su.PSI_P, particle=Particle.PROTON)
    params = {"D_values": lambda t: np.ones((2, 5)) * su.DIFFUSION_COEFFICIENT}
    with pytest.raises(ValueError, match="2D D_values must have shape"):
        Solver(grid, state, "diffusion", {"diffusion": params})
