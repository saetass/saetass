r"""
Parallel runs of independent simulations.

A parameter sweep, such as the same problem at several energies or for several diffusion models, is a set of independent simulations.
They scale much better as separate processes than with threads inside each simulation: every process advances its own grid serially, with no synchronization between them.
:py:func:`run_sweep` runs them in a pool of worker processes, each with a single thread (see :py:mod:`saetass.parallel` and :py:data:`WORKER_ENVIRONMENT`), and returns their results in order.

Workers are started with the ``spawn`` method by default: forking a process whose OpenMP threads are already running is unsafe.
The function must therefore be importable by the workers, that is, defined at the top level of a module, or of a script whose code runs under ``if __name__ == "__main__":``.
Functions defined in a notebook can be run with ``mp_context="fork"`` on Linux, provided that the notebook has not run any simulation with threads before.

The compiled kernels are cached on disk, so workers do not compile them again after the first run on a machine.
"""

import contextlib
import logging
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .cli.output import plural

logger = logging.getLogger(__name__)

__all__ = ["WORKER_ENVIRONMENT", "default_workers", "run_sweep"]

#: Environment of the workers of :py:func:`run_sweep`: one thread for Numba, OpenMP and the
#: linear algebra libraries of NumPy. Each library would otherwise start a thread per core
#: in every worker, which on a machine with 120 cores made sweeps up to 2x slower.
WORKER_ENVIRONMENT = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def default_workers() -> int:
    """
    Default number of workers of :py:func:`run_sweep`: the physical cores this process can use.

    Returns
    -------
    int
        :py:func:`saetass.parallel.usable_cores`, which accounts for the CPU affinity of the process, the CPU quota of its container and the logical CPUs that share a physical core.
    """
    from . import parallel

    return parallel.usable_cores()


@contextlib.contextmanager
def _worker_environment():
    """Set :py:data:`WORKER_ENVIRONMENT` while workers start, then restore the environment."""
    previous = {name: os.environ.get(name) for name in WORKER_ENVIRONMENT}
    os.environ.update(WORKER_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def _init_worker(quiet: bool) -> None:
    """Make the kernels of a worker serial and, if ``quiet``, silence its output."""
    from . import parallel

    parallel.set_max_threads(1)
    if quiet:
        # For the lifetime of the worker, which ends with the pool
        sys.stdout = sys.stderr = open(os.devnull, "w")


def run_sweep(
    function: Callable[[Any], Any],
    tasks: Iterable[Any],
    max_workers: int | None = None,
    quiet: bool = True,
    mp_context: str | None = "spawn",
) -> list[Any]:
    """
    Run ``function(task)`` for every task in parallel worker processes.

    Parameters
    ----------
    function : callable
        Function of one argument, importable by the workers (see :py:mod:`saetass.sweep`). It typically builds a :py:class:`~saetass.solver.Solver`, runs it and returns the quantities of interest. Its arguments and results must be picklable.
    tasks : iterable
        Arguments of the successive calls, for instance energies, or tuples or dictionaries of parameters.
    max_workers : int, optional
        Number of worker processes, at most the number of tasks. Default is ``None``, meaning :py:func:`default_workers`.
    quiet : bool, optional
        Whether to discard what the workers print and log. Their :py:class:`~saetass.solver.Solver` banners and progress bars are hidden either way, unless :py:func:`~saetass.cli.output.set_quiet` or ``SAETASS_QUIET`` turn them on (see :py:mod:`saetass.cli.output`). Default is ``True``.
    mp_context : str or None, optional
        Start method of the workers (``'spawn'``, ``'forkserver'`` or ``'fork'``), or ``None`` for the default of the platform. Default is ``'spawn'``.

    Returns
    -------
    list
        The results of the calls, in the order of ``tasks``.

    Raises
    ------
    ValueError
        If ``max_workers`` is not a positive integer.
    Exception
        The first exception raised by a call, after the remaining calls finish.
    """
    tasks = list(tasks)
    if max_workers is None:
        max_workers = default_workers()
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError(
            f"max_workers must be a positive integer, got {max_workers!r}."
        )
    if not tasks:
        return []

    context = None if mp_context is None else multiprocessing.get_context(mp_context)
    workers = min(max_workers, len(tasks))
    logger.info(
        "Sweep of %s on %s, started with %s",
        plural(len(tasks), "task"),
        plural(workers, "worker process", "worker processes"),
        mp_context or "the default method",
    )
    start = time.perf_counter()
    # Workers start on demand while the tasks are submitted, inheriting the environment
    with (
        _worker_environment(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(quiet,),
        ) as pool,
    ):
        results = list(pool.map(function, tasks))
    logger.info("Sweep finished in %.3g s", time.perf_counter() - start)
    return results
