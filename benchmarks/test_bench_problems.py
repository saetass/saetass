"""
End-to-end cost of the representative problems, through the public :py:class:`Solver`.

These include everything a user pays for: operator splitting, the progress bar, the
Python overhead of every step and the kernels. Building the solver is not measured.
"""

import numpy as np
import problems
import pytest

#: Rounds per benchmark; each builds a fresh solver from the same initial state.
ROUNDS = 5

PROBLEMS = {
    # Small grid and cheap steps: the Python overhead of every step
    "diffusion_source_1d": problems.diffusion_source_1d,
    # One energy of the multi-energy example
    "wind_bubble_1d": problems.wind_bubble_1d,
    "wind_bubble_1d_order1": lambda: problems.wind_bubble_1d(order=1),
    # The full equation on a grid of typical size
    "full_2d": problems.full_2d,
    "full_2d_dynamic": lambda: problems.full_2d(dynamic=True),
}


def _check(values):
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    assert np.any(values > 0.0)


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_problem(bench, name):
    build = PROBLEMS[name]
    build().run()  # compile the Numba kernels outside the measurement
    bench.group = "problems"
    bench.extra_info["steps"] = build().n_steps
    result = bench.pedantic(
        lambda problem: problem.run(),
        setup=lambda: ((build(),), {}),
        rounds=ROUNDS,
        warmup_rounds=1,
    )
    _check(result)


def test_energy_sweep(bench):
    """Independent runs of a parameter sweep, one after the other."""

    def run_all(runs):
        return [run.run().copy() for run in runs]

    problems.energy_sweep(n_runs=1, n_steps=2)[0].run()  # compile
    bench.group = "sweep"
    sweep = problems.energy_sweep()
    bench.extra_info["runs"] = len(sweep)
    results = bench.pedantic(
        run_all,
        setup=lambda: ((problems.energy_sweep(),), {}),
        rounds=3,
        warmup_rounds=0,
    )
    for values in results:
        _check(values)
