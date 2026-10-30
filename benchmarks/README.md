# Performance benchmarks

These benchmarks measure the cost of the numerical core with
[pytest-benchmark](https://pytest-benchmark.readthedocs.io), installed by the `test` extra.
They are not part of the test suite (`pytest` only collects `test/`): timings need
several rounds, and a machine without other heavy work (see below). Correctness is
checked by the test suite: `test/test_solver_regression.py` compares every operator with
golden outputs, so the benchmarks only check that their results are finite and
non-negative.

| File | What it measures |
|---|---|
| `test_bench_operators.py` | One call to each operator (one CFL sub-cycle), on a 1D grid of 1000 cells and a 2D grid of 71 x 400 cells, for every order, limiter and static or time-dependent parameters. |
| `test_bench_problems.py` | Representative problems through `Solver`, including splitting and per-step overhead: a small 1D diffusion-source problem with cheap steps, one energy of the wind bubble example (1D, 600 cells), the full equation on a 71 x 400 grid, a sweep of independent 1D runs, and a larger sweep run serially and with `saetass.sweep.run_sweep`. |
| `test_bench_stress.py` | Stress cases at the restrictive end of the physical parameters (3000 km/s wind, 100 uG field, dense gas, ten decades of momentum, up to tens of thousands of CFL sub-cycles per step): electrons and protons on the full equation, electron cooling, a fast 1D wind bubble, static and time-dependent, and two large grids for thread scaling. Slower than the other files. |
| `test_bench_emissions.py` | Every emission mechanism end to end with its cross-section model, and the two integrations over the distribution on large grids. |
| `problems.py` | The problems, shared with `profile_problem.py`. |

## Running them

From the root of the repository, in the development environment (`pip install -e ".[dev]"`):

```bash
# Every benchmark, saving the results as a reference
pytest benchmarks --benchmark-save=baseline

# The same with every kernel serial
SAETASS_MAX_THREADS=1 pytest benchmarks --benchmark-save=baseline-serial

# Compare against saved run 0001, failing if a median gets 10% slower
pytest benchmarks --benchmark-compare=0001 --benchmark-compare-fail=median:10%

# Only check that they run, as CI does
pytest benchmarks --benchmark-disable
```

Results are saved in `.benchmarks/<machine>/` (ignored by git), numbered in order; the
saved runs can be listed and compared with `pytest-benchmark list` and
`pytest-benchmark compare 0001 0002 --group-by=group`, or with
`python benchmarks/compare.py 0001 0002`, which prints the speedup of each benchmark.
Every saved run records the Numba version, threads and threading layer, and the path of
the measured package, in `machine_info`.

Useful options: `-k advection` selects benchmarks by name and
`--benchmark-columns=min,median,iqr,rounds` shortens the table. Compare medians rather
than means: the mean is sensitive to the occasional slow round.

### Threads and the load of the machine

The kernels choose their threads from the work of the simulation and the cores of the
machine (see `saetass.parallel`). Two variables fix them for a comparison:
`SAETASS_MAX_THREADS` limits the threads of every kernel call (`1` runs every kernel
serially), and `NUMBA_NUM_THREADS`, read when Numba starts, sets the threads available
to Numba (CI uses `1`).

Other processes change the timings in two ways: they compete for the cores, and on Linux
SAETASS leaves the cores they keep busy free, so a loaded machine also runs with fewer
threads. Benchmark on a machine without other heavy work, or set `SAETASS_MAX_THREADS`
below its idle cores. Even then, whole runs on a laptop drift by up to 10% from one to
the next (temperature, background processes), more than the spread within a run
suggests: re-run the affected benchmarks, alternating the two versions, before trusting
a small difference.

### Other versions

The benchmarks only use the API of SAETASS 0.5, except the variants named `separable`
and the sweep with `run_sweep`, which are skipped where it is missing, so they also
measure that release. To measure another version, put its `src` first with `PYTHONPATH`
and keep pytest from reading `pyproject.toml`, whose `pythonpath = ["src"]` would
otherwise put the `src` of the repository first:

```bash
cd benchmarks
printf "[pytest]\n" > /tmp/empty.ini
PYTHONPATH=/path/to/other/src pytest -c /tmp/empty.ini --rootdir . \
    --benchmark-storage=../.benchmarks --benchmark-save=other
```

Check the measured package in `machine_info["saetass_path"]` of the saved run.

## Profiling

`profile_problem.py` runs one of the problems under cProfile, after a first run that
compiles the Numba kernels:

```bash
cd benchmarks
SAETASS_MAX_THREADS=1 python profile_problem.py full_2d --limit 30
```
