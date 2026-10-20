"""
Stress cases: the solver at the restrictive end of the physical parameters.

Fast winds, strong magnetic fields, dense gas and momentum ranges of ten decades force
many CFL sub-cycles per step, so these problems are slower than the ones of
``test_bench_problems.py``. They only use the API of SAETASS 0.5, so the same file also
measures that release. Building the solver is not
measured; the loss rates are cached, so only the first build pays for them.
"""

import numpy as np
import problems
import pytest

#: Rounds per benchmark; each builds a fresh solver from the same initial state.
ROUNDS = 3

PROBLEMS = {
    "electrons_2d": problems.electrons_2d,
    "electrons_2d_dynamic": lambda **kw: problems.electrons_2d(dynamic=True, **kw),
    "protons_2d": problems.protons_2d,
    "electron_cooling_1d": problems.electron_cooling_1d,
    "wind_bubble_1d_stress": problems.wind_bubble_1d_stress,
    "wind_bubble_1d_stress_dynamic": lambda **kw: problems.wind_bubble_1d_stress(
        dynamic=True, **kw
    ),
    "full_2d_large": problems.full_2d_large,
    "full_2d_huge": problems.full_2d_huge,
}


def _check(values):
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    assert np.any(values > 0.0)


def cfl_subcycles(problem) -> dict[str, float]:
    """CFL sub-cycles per call of each hyperbolic operator, from the diagnostics."""
    diagnostics = problem.solver.diagnostics(warn=False)
    return {
        op.name: op.numbers["cfl_subcycles"]
        for op in diagnostics.operators
        if "cfl_subcycles" in op.numbers
    }


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_stress(bench, name):
    build = PROBLEMS[name]
    build(n_steps=1).run()  # compile the Numba kernels outside the measurement
    problem = build()
    bench.group = "stress"
    bench.extra_info["steps"] = problem.n_steps
    bench.extra_info["cells"] = int(problem.solver.state._values.size)
    for operator, subcycles in cfl_subcycles(problem).items():
        bench.extra_info[f"{operator}_cfl_subcycles"] = subcycles
    result = bench.pedantic(
        lambda problem: problem.run(),
        setup=lambda: ((build(),), {}),
        rounds=ROUNDS,
        warmup_rounds=0,
    )
    _check(result)
