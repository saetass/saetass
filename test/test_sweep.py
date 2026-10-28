"""Tests of the parallel runs of independent simulations in ``saetass.sweep``."""

import os

import numpy as np
import pytest

from saetass import Grid, Particle, Solver, State
from saetass import units as su
from saetass.sweep import WORKER_ENVIRONMENT, _init_worker, default_workers, run_sweep


def _diffuse(D):
    """A small diffusion-source run, defined at the top level so workers can import it."""
    r = np.linspace(0.0, 1.0, 50)
    grid = Grid(r_centers=r * su.LENGTH, t_grid=np.linspace(0.0, 0.1, 21) * su.TIME)
    state = State(
        grid=grid, psi_p=np.zeros(r.size) * su.PSI_P, particle=Particle.PROTON
    )
    params = {
        "diffusion": {"D_values": np.full(r.size, D) * su.DIFFUSION_COEFFICIENT},
        "source": {"source": np.exp(-((r / 0.2) ** 2)) * su.SOURCE_PSI_P},
    }
    solver = Solver(state, "diffusion-source", params)
    return solver.run().psi_p.to_value(su.PSI_P)


def _environment(name):
    return os.environ.get(name)


def _fail(task):
    raise RuntimeError(f"task {task} failed")


def test_results_match_serial_runs_in_order():
    tasks = [0.01, 0.1, 1.0]
    results = run_sweep(_diffuse, tasks, max_workers=2)
    assert len(results) == len(tasks)
    for D, result in zip(tasks, results):
        np.testing.assert_array_equal(result, _diffuse(D))


def test_empty_sweep():
    assert run_sweep(_diffuse, []) == []


def test_errors_of_a_task_reach_the_caller():
    with pytest.raises(RuntimeError, match="task 1 failed"):
        run_sweep(_fail, [1], max_workers=1, quiet=False)


@pytest.mark.parametrize("max_workers", [0, -2, 1.5, True, "4"])
def test_invalid_max_workers(max_workers):
    with pytest.raises(ValueError, match="max_workers"):
        run_sweep(_diffuse, [0.1], max_workers=max_workers)


def test_default_workers():
    from saetass import parallel

    assert default_workers() == parallel.usable_cores() >= 1


def test_worker_initializer_makes_kernels_serial(monkeypatch):
    from saetass import parallel

    monkeypatch.setattr(parallel, "_max_threads", parallel.get_max_threads())
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)
    _init_worker(quiet=True)
    assert parallel.get_max_threads() == 1
    import sys

    sys.stdout.write("discarded")


def test_workers_run_single_threaded_and_the_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    names = sorted(WORKER_ENVIRONMENT)
    assert run_sweep(_environment, names, max_workers=2) == ["1"] * len(names)
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ
