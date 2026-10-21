r"""
Threads used by the numerical kernels.

SAETASS parallelizes a kernel only over independent rows of the grid (for instance the momenta of a spatial operator), so its result never depends on the number of threads.
Every such kernel is compiled twice from the same Python function and with the same flags: a serial version and a parallel one that distributes the rows with ``numba.prange``.

The threads of a call are chosen in two stages:

* Before each call, the kernel estimates its work and runs serially unless it exceeds :py:data:`MIN_PARALLEL_WORK_NS`: starting a parallel region costs microseconds, more than the whole work of a small call.
* Larger calls use a team of threads, :py:func:`team_size`: one thread per :py:data:`MIN_WORK_PER_THREAD_NS` of the largest call of the simulation so far, at most the physical cores this process can use (:py:func:`usable_cores`) that other processes do not keep busy, and at most the limit set with :py:func:`set_max_threads` or ``SAETASS_MAX_THREADS``.
  Every parallel call uses the whole team, and the team only grows during a simulation (each operator with parallel kernels starts a new one with :py:func:`reset_team` when it is created), because OpenMP rebuilds its team, at a cost of about 150 microseconds, whenever two consecutive parallel regions ask for different numbers of threads.
  A simulation with heavy calls, such as the losses of electrons over many sub-cycles, thus uses every idle core, and one with light calls a few threads, whose start costs less than their gain.

The OpenMP threads of Numba busy-wait for the slowest of them, so a team larger than the idle cores of a shared machine makes every call up to a hundred times slower.
On Linux, the cores kept busy by other processes are therefore measured (from ``/proc/stat``, every minute) and left free.
A limit (``SAETASS_MAX_THREADS``) replaces that measurement, for instance to share a machine with other simulations of one's own.
Independent simulations, such as the energies or parameters of a sweep, scale much better as separate processes with one thread each: see :py:func:`saetass.sweep.run_sweep`.

The threads available to Numba, which bound every call, are set with ``NUMBA_NUM_THREADS`` before Numba starts.
"""

import functools
import logging
import math
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numba
from numba import njit

from .cli.output import plural

__all__ = [
    "MIN_PARALLEL_WORK_NS",
    "MIN_WORK_PER_THREAD_NS",
    "get_max_threads",
    "reset_team",
    "set_max_threads",
    "team_size",
    "threads_for",
    "twin_kernel",
    "TwinKernel",
    "usable_cores",
]

logger = logging.getLogger(__name__)

#: Minimum estimated serial work of a kernel call, in nanoseconds, for it to run in parallel.
MIN_PARALLEL_WORK_NS = 100_000.0

#: Estimated serial work of the largest call per thread of the team, in nanoseconds.
MIN_WORK_PER_THREAD_NS = 20_000.0

#: Seconds over which the busy cores of other processes are measured.
_LOAD_INTERVAL = 0.05

#: Seconds for which a measurement of the busy cores is reused.
_LOAD_LIFETIME = 60.0


def _parse_max_threads(value: Any, name: str) -> int:
    # Through str, so that floats and booleans are rejected instead of truncated
    try:
        n = int(str(value).strip())
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return n


_max_threads = (
    _parse_max_threads(os.environ["SAETASS_MAX_THREADS"], "SAETASS_MAX_THREADS")
    if os.environ.get("SAETASS_MAX_THREADS")
    else None
)


def get_max_threads() -> int | None:
    """
    Limit of the threads of a kernel call.

    Returns
    -------
    int or None
        The limit set by :py:func:`set_max_threads` or the environment variable ``SAETASS_MAX_THREADS``, or ``None`` if there is none.
    """
    return _max_threads


def set_max_threads(n: int | None) -> None:
    """
    Set the limit of the threads of a kernel call.

    Calls never use more threads than this process can use (:py:func:`usable_cores`) or than Numba has (``numba.get_num_threads()``).
    With a limit, the cores kept busy by other processes are not measured.

    Parameters
    ----------
    n : int or None
        Maximum number of threads, at least 1. ``1`` runs every kernel serially and ``None`` removes the limit.
    """
    global _max_threads
    _max_threads = (
        None if n is None else _parse_max_threads(n, "The maximum number of threads")
    )


#: Largest estimated work of a parallel call since the last :py:func:`reset_team`, in nanoseconds.
_team_work_ns = 0.0

#: Threads of the last team reported to the debug log.
_logged_team = 1


def reset_team() -> None:
    """
    Start a new team of threads, sized by the calls that follow.

    Operators with parallel kernels call it when they are created, so that a simulation with light calls does not keep the large team of a previous one.
    """
    global _team_work_ns
    _team_work_ns = 0.0


def threads_for(rows: int, work_ns: float) -> int:
    """
    Number of threads a kernel call should use.

    Parameters
    ----------
    rows : int
        Number of independent rows (or groups of rows) the kernel can distribute.
    work_ns : float
        Estimated serial cost of the call, in nanoseconds.

    Returns
    -------
    int
        ``1`` for a serial call, otherwise the team of the largest call so far, :py:func:`team_size`, whatever the rows of this call (threads without rows idle briefly, which costs less than resizing the team).
    """
    global _team_work_ns, _logged_team
    if rows < 2 or work_ns < MIN_PARALLEL_WORK_NS:
        return 1
    _team_work_ns = max(_team_work_ns, work_ns)
    n = team_size(_team_work_ns)
    if n != _logged_team:
        _logged_team = n
        logger.debug(
            "Kernel thread team: %s for calls of about %.0f us "
            "(usable cores %d, limit %s, Numba threads %d)",
            plural(n, "thread"),
            _team_work_ns / 1e3,
            usable_cores(),
            _max_threads,
            numba.get_num_threads(),
        )
    return n


def team_size(work_ns: float | None = None) -> int:
    """
    Number of threads of the parallel kernel calls.

    Parameters
    ----------
    work_ns : float, optional
        Estimated serial work of the largest call, in nanoseconds, which asks for one thread per :py:data:`MIN_WORK_PER_THREAD_NS` (and at least two). If ``None``, as many threads as possible. Default is ``None``.

    Returns
    -------
    int
        The threads asked for by ``work_ns``, at most the physical cores this process can use (:py:func:`usable_cores`) minus those that other processes keep busy (measured on Linux), or at most the limit of :py:func:`set_max_threads` instead; never more than the threads of Numba.
    """
    n = numba.get_num_threads()
    if work_ns is not None:
        n = min(n, max(2, int(work_ns // MIN_WORK_PER_THREAD_NS)))
    if n < 2:
        return 1
    if _max_threads is not None:
        return max(1, min(n, _max_threads, usable_cores()))
    return max(1, min(n, _idle_cores()))


@functools.cache
def usable_cores() -> int:
    """
    Physical cores this process can use.

    Returns
    -------
    int
        The smallest of the logical CPUs of the process (its CPU affinity, or ``PYTHON_CPU_COUNT`` on Python 3.13 and later), the CPU quota of its container (Linux control groups) and the physical cores among its CPUs (hyperthreads of a core share its arithmetic units). Physical cores are not counted on Windows.
    """
    cpus = _affinity()
    counts = [_logical_cpus(cpus), _cgroup_quota(), _physical_cores(cpus)]
    return max(1, min(count for count in counts if count is not None))


def _affinity() -> set[int] | None:
    """The CPUs this process may run on, where the system tells (Linux)."""
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _logical_cpus(cpus: set[int] | None) -> int | None:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count()
    return len(cpus) if cpus is not None else os.cpu_count()


def _cgroup_quota(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """CPUs allowed by the quota of the control group of a container, if any."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota == "max":
            return None
    except (OSError, ValueError):
        try:
            quota = (root / "cpu" / "cpu.cfs_quota_us").read_text()
            period = (root / "cpu" / "cpu.cfs_period_us").read_text()
        except OSError:
            return None
    quota, period = int(quota), int(period)
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def _physical_cores(
    cpus: set[int] | None, root: Path = Path("/sys/devices/system/cpu")
) -> int | None:
    """Physical cores among ``cpus`` (Linux) or of the machine (macOS)."""
    if sys.platform == "darwin":
        return _sysctl_int("hw.physicalcpu")
    if cpus is None:
        return None
    cores = set()
    for cpu in cpus:
        topology = root / f"cpu{cpu}" / "topology"
        try:
            cores.add((topology / "core_cpus_list").read_text().strip())
        except OSError:
            try:
                cores.add((topology / "thread_siblings_list").read_text().strip())
            except OSError:
                return None
    return len(cores)


def _sysctl_int(name: str) -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
        return int(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


_load_sample: tuple[float, float] | None = None  # (time, busy cores)


def _idle_cores() -> int:
    """Usable cores minus those that other processes keep busy, measured at most every :py:data:`_LOAD_LIFETIME` seconds."""
    global _load_sample
    now = time.monotonic()
    if _load_sample is None or now - _load_sample[0] > _LOAD_LIFETIME:
        _load_sample = (now, _busy_cpus(_affinity()))
        logger.debug(
            "Cores kept busy by other processes: %.1f of %d usable",
            _load_sample[1],
            usable_cores(),
        )
    return max(1, usable_cores() - math.ceil(_load_sample[1]))


def _busy_cpus(cpus: Iterable[int] | None, stat: Path = Path("/proc/stat")) -> float:
    """
    CPUs among ``cpus`` that other processes keep busy, over :py:data:`_LOAD_INTERVAL` seconds.

    The busy time of every CPU comes from ``/proc/stat``, where time stolen by the hypervisor of a virtual machine counts as busy; the CPU time of this process (for instance its OpenMP threads spinning after a parallel region) is subtracted.
    Without ``/proc/stat`` (other systems than Linux), no CPU counts as busy.
    """
    before = _cpu_times(stat)
    if before is None or cpus is None:
        return 0.0
    own_before, wall_before = time.process_time(), time.perf_counter()
    time.sleep(_LOAD_INTERVAL)
    after = _cpu_times(stat)
    own = (time.process_time() - own_before) / (time.perf_counter() - wall_before)
    busy = 0.0
    for cpu in cpus:
        if cpu not in before or cpu not in after:
            continue
        # user, nice, system, idle, iowait, irq, softirq, steal (guest time is in user)
        delta = [b - a for a, b in zip(before[cpu][:8], after[cpu][:8])]
        total = sum(delta)
        if total > 0:
            busy += 1.0 - (delta[3] + delta[4]) / total
    return max(0.0, busy - own)


def _cpu_times(stat: Path) -> dict[int, list[int]] | None:
    try:
        lines = stat.read_text().splitlines()
    except OSError:
        return None
    times = {}
    for line in lines:
        name, *values = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            times[int(name[3:])] = [int(value) for value in values]
    return times


class TwinKernel:
    """
    A kernel compiled serially and in parallel from the same Python function.

    Calling it with a number of threads runs the serial version for one thread, and otherwise the parallel version with that many threads.
    Both versions are cached on disk, so that new processes do not compile them again.

    Parameters
    ----------
    py_func : callable
        Kernel distributing independent rows with ``numba.prange``, which runs as ``range`` in the serial version.
    **flags
        Numba compilation flags shared by both versions, such as ``error_model``.
    """

    def __init__(self, py_func: Callable, **flags: Any) -> None:
        self.py_func = py_func
        self.serial = njit(cache=True, **flags)(py_func)
        # A distinct qualified name gives the parallel version its own cache entry
        parallel_func = types.FunctionType(
            py_func.__code__,
            py_func.__globals__,
            f"{py_func.__name__}_parallel",
            py_func.__defaults__,
            py_func.__closure__,
        )
        parallel_func.__qualname__ = f"{py_func.__qualname__}_parallel"
        parallel_func.__module__ = py_func.__module__
        self.parallel = njit(parallel=True, cache=True, **flags)(parallel_func)
        # Versions not called yet: their first call compiles them or loads them from the cache
        self._first_calls = {"serial", "parallel"}

    def __call__(self, n_threads: int, *args: Any) -> Any:
        version = "serial" if n_threads <= 1 else "parallel"
        if version in self._first_calls:
            return self._first_call(version, n_threads, args)
        return self._call(n_threads, args)

    def _call(self, n_threads: int, args: tuple) -> Any:
        if n_threads <= 1:
            return self.serial(*args)
        previous = numba.get_num_threads()
        numba.set_num_threads(min(n_threads, previous))
        try:
            return self.parallel(*args)
        finally:
            numba.set_num_threads(previous)

    def _first_call(self, version: str, n_threads: int, args: tuple) -> Any:
        """Call a version for the first time, reporting how long compiling or loading it took."""
        start = time.perf_counter()
        result = self._call(n_threads, args)
        self._first_calls.discard(version)
        logger.debug(
            "Kernel %s (%s): first call, including its compilation or cache load, "
            "took %.3g s",
            self.py_func.__name__,
            version,
            time.perf_counter() - start,
        )
        return result


def twin_kernel(**flags: Any) -> Callable[[Callable], TwinKernel]:
    """
    Decorator compiling a kernel into a :py:class:`TwinKernel`.

    Parameters
    ----------
    **flags
        Numba compilation flags shared by the serial and parallel versions.
    """

    def decorate(py_func: Callable) -> TwinKernel:
        return TwinKernel(py_func, **flags)

    return decorate
