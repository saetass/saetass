"""
Cost of one call to each operator, with a single CFL sub-cycle per call.

The 1D grids have 1000 cells and the 2D grid has 71 momenta and 400 radii, a typical
size. Divide the time by the number of cells to compare with the cost per cell of a
kernel.
"""

import numpy as np
import pytest
from problems import operator_case

SHAPE_1D = (1000,)
SHAPE_2D = (71, 400)

#: Rounds per benchmark; each starts from the same initial distribution.
ROUNDS = 200

HYPERBOLIC_VARIANTS = {
    "order1": {"order": 1},
    "minmod": {"limiter": "minmod"},
    "vanleer": {"limiter": "vanleer"},
    "mc": {"limiter": "mc"},
    "minmod-dynamic": {"limiter": "minmod", "dynamic": True},
}


def _run(bench, case):
    case.advance()  # compile the Numba kernels outside the measurement
    bench.extra_info["cells"] = case.initial.size
    bench.pedantic(case.advance, setup=case.reset, rounds=ROUNDS, warmup_rounds=5)
    values = case.state._values
    assert np.all(np.isfinite(values)) and np.any(values > 0.0)


@pytest.mark.parametrize("shape", [SHAPE_1D, SHAPE_2D], ids=["1d", "2d"])
@pytest.mark.parametrize("variant", sorted(HYPERBOLIC_VARIANTS))
@pytest.mark.parametrize("operator", ["advection", "loss"])
def test_hyperbolic(bench, operator, variant, shape):
    bench.group = f"{operator} {'x'.join(map(str, shape))}"
    _run(bench, operator_case(operator, shape, **HYPERBOLIC_VARIANTS[variant]))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("shape", [SHAPE_1D, SHAPE_2D], ids=["1d", "2d"])
@pytest.mark.parametrize("operator", ["diffusion", "source"])
def test_diffusion_and_source(bench, operator, shape, dynamic):
    bench.group = f"{operator} {'x'.join(map(str, shape))}"
    _run(bench, operator_case(operator, shape, dynamic=dynamic))
