"""Tests of the thread management of the kernels in ``saetass.parallel``."""

import pytest

from saetass import parallel


@pytest.fixture
def max_threads():
    """Restore the thread limit after the test."""
    previous = parallel.get_max_threads()
    yield
    parallel.set_max_threads(previous)


@pytest.fixture
def cores(monkeypatch):
    """Pretend 32 usable cores and 32 Numba threads, with no load from other processes."""
    monkeypatch.setattr(parallel, "usable_cores", lambda: 32)
    monkeypatch.setattr(parallel.numba, "get_num_threads", lambda: 32)
    monkeypatch.setattr(parallel, "_busy_cpus", lambda cpus: 0.0)
    monkeypatch.setattr(parallel, "_load_sample", None)


@pytest.mark.parametrize("value", [1, 4, "8", " 2 ", None])
def test_set_max_threads(value, max_threads):
    parallel.set_max_threads(value)
    assert parallel.get_max_threads() == (None if value is None else int(str(value)))


@pytest.mark.parametrize("value", [0, -1, 2.5, "two", True])
def test_invalid_max_threads(value, max_threads):
    with pytest.raises(ValueError, match="positive integer"):
        parallel.set_max_threads(value)


def test_threads_for_small_work_is_serial(cores, max_threads):
    parallel.set_max_threads(None)
    small = 0.99 * parallel.MIN_PARALLEL_WORK_NS
    assert parallel.threads_for(rows=1000, work_ns=small) == 1
    assert parallel.threads_for(rows=1, work_ns=1e12) == 1
    parallel.set_max_threads(1)
    assert parallel.threads_for(rows=1000, work_ns=1e12) == 1


def test_threads_for_uses_the_same_team(cores, monkeypatch, max_threads):
    """Parallel calls use the team of the largest call so far, which only grows."""
    parallel.set_max_threads(None)
    monkeypatch.setattr(parallel, "_team_work_ns", 0.0)
    per_thread = parallel.MIN_WORK_PER_THREAD_NS
    smallest = parallel.MIN_PARALLEL_WORK_NS
    assert parallel.threads_for(rows=1000, work_ns=smallest) == int(
        smallest // per_thread
    )
    assert parallel.threads_for(rows=1000, work_ns=10 * per_thread) == 10
    # A lighter call keeps the team, whatever its rows, and a heavier one grows it
    assert parallel.threads_for(rows=1000, work_ns=smallest) == 10
    assert parallel.threads_for(rows=3, work_ns=1e5 * per_thread) == 32
    assert parallel.threads_for(rows=1000, work_ns=smallest) == 32
    parallel.reset_team()
    assert parallel.threads_for(rows=1000, work_ns=smallest) == int(
        smallest // per_thread
    )

    parallel.set_max_threads(8)
    assert parallel.threads_for(rows=1000, work_ns=1e9) == 8
    parallel.set_max_threads(64)  # never more than the usable cores
    assert parallel.threads_for(rows=1000, work_ns=1e9) == 32


def test_team_size(cores, max_threads):
    parallel.set_max_threads(None)
    assert parallel.team_size() == 32
    assert parallel.team_size(0.0) == 2  # at least two threads
    assert parallel.team_size(4 * parallel.MIN_WORK_PER_THREAD_NS) == 4


def test_team_leaves_busy_cores_free(cores, monkeypatch, max_threads):
    parallel.set_max_threads(None)
    measured = []

    def busy(cpus):
        measured.append(cpus)
        return 4.2

    monkeypatch.setattr(parallel, "_busy_cpus", busy)
    assert parallel.team_size() == 27  # 32 cores minus 5 partly busy
    assert parallel.team_size() == 27
    assert len(measured) == 1  # measured once per _LOAD_LIFETIME
    monkeypatch.setattr(parallel, "_LOAD_LIFETIME", -1.0)
    parallel.team_size()
    assert len(measured) == 2

    monkeypatch.setattr(parallel, "_busy_cpus", lambda cpus: 100.0)
    assert parallel.team_size() == 1

    # A limit replaces the measurement
    parallel.set_max_threads(16)
    assert parallel.team_size() == 16

    monkeypatch.setattr(parallel.numba, "get_num_threads", lambda: 1)
    assert parallel.team_size() == 1


def test_usable_cores_is_the_smallest_count(monkeypatch):
    monkeypatch.setattr(parallel, "_affinity", lambda: {0, 1, 2, 3})
    monkeypatch.setattr(parallel, "_logical_cpus", lambda cpus: 4)
    monkeypatch.setattr(parallel, "_cgroup_quota", lambda: None)
    monkeypatch.setattr(parallel, "_physical_cores", lambda cpus: 2)
    assert parallel.usable_cores.__wrapped__() == 2
    monkeypatch.setattr(parallel, "_physical_cores", lambda cpus: None)
    monkeypatch.setattr(parallel, "_cgroup_quota", lambda: 3)
    assert parallel.usable_cores.__wrapped__() == 3
    assert parallel.usable_cores() >= 1


def test_affinity_and_logical_cpus(monkeypatch):
    cpus = parallel._affinity()
    assert cpus is None or len(cpus) >= 1
    monkeypatch.delattr(parallel.os, "sched_getaffinity", raising=False)
    assert parallel._affinity() is None

    monkeypatch.setattr(parallel.os, "process_cpu_count", lambda: 5, raising=False)
    assert parallel._logical_cpus({0, 1}) == 5
    monkeypatch.delattr(parallel.os, "process_cpu_count")
    assert parallel._logical_cpus({0, 1}) == 2
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 7)
    assert parallel._logical_cpus(None) == 7


def test_cgroup_quota(tmp_path):
    assert parallel._cgroup_quota(tmp_path) is None
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert parallel._cgroup_quota(tmp_path) is None
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert parallel._cgroup_quota(tmp_path) == 3

    # Version 1 of control groups
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert parallel._cgroup_quota(tmp_path) is None
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("50000\n")
    assert parallel._cgroup_quota(tmp_path) == 1


def _topology(root, siblings, name="core_cpus_list"):
    for cpu, core in siblings.items():
        directory = root / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        (directory / name).write_text(core + "\n")


def test_physical_cores_on_linux(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel.sys, "platform", "linux")
    # Two cores with two hyperthreads each
    _topology(tmp_path, {0: "0,2", 1: "1,3", 2: "0,2", 3: "1,3"})
    assert parallel._physical_cores({0, 1, 2, 3}, tmp_path) == 2
    assert parallel._physical_cores({0, 2}, tmp_path) == 1
    assert parallel._physical_cores(None, tmp_path) is None
    assert parallel._physical_cores({7}, tmp_path) is None

    older = tmp_path / "older"
    _topology(older, {0: "0-1", 1: "0-1"}, name="thread_siblings_list")
    assert parallel._physical_cores({0, 1}, older) == 1


def test_physical_cores_on_macos(monkeypatch):
    monkeypatch.setattr(parallel.sys, "platform", "darwin")
    monkeypatch.setattr(parallel, "_sysctl_int", {"hw.physicalcpu": 8}.get)
    assert parallel._physical_cores(None) == 8


def test_sysctl_int(monkeypatch):
    class Result:
        stdout = "10\n"

    monkeypatch.setattr(parallel.subprocess, "run", lambda *args, **kwargs: Result())
    assert parallel._sysctl_int("hw.physicalcpu") == 10
    Result.stdout = "unknown oid"
    assert parallel._sysctl_int("hw.physicalcpu") is None

    def missing(*args, **kwargs):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(parallel.subprocess, "run", missing)
    assert parallel._sysctl_int("hw.physicalcpu") is None


def _stat(path, rows):
    lines = ["cpu  1 2 3 4 5 6 7 8 0 0", "intr 1 2 3"]
    lines += [f"cpu{cpu} " + " ".join(map(str, values)) for cpu, values in rows.items()]
    path.write_text("\n".join(lines) + "\n")


def test_busy_cpus(tmp_path, monkeypatch):
    stat = tmp_path / "stat"
    assert parallel._busy_cpus({0}, stat) == 0.0  # no /proc/stat
    _stat(stat, {0: [0] * 10})
    assert parallel._busy_cpus(None, stat) == 0.0

    # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    before = {0: [100, 0, 0, 100, 0, 0, 0, 0, 0, 0], 1: [0] * 10, 2: [0] * 10}
    after = {
        0: [200, 0, 0, 100, 0, 0, 0, 0, 50, 0],  # busy, with guest time inside user
        1: [0, 0, 0, 50, 50, 0, 0, 0, 0, 0],  # idle or waiting for input/output
        2: [0] * 10,  # no time elapsed
    }
    snapshots = iter([before, after])
    monkeypatch.setattr(parallel, "_cpu_times", lambda path: next(snapshots))
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: None)
    clock = iter([0.0, 0.0])  # this process used no CPU time
    monkeypatch.setattr(parallel.time, "process_time", lambda: next(clock))
    assert parallel._busy_cpus({0, 1, 2, 5}, stat) == pytest.approx(1.0)


def test_busy_cpus_subtracts_this_process(tmp_path, monkeypatch):
    rows = {0: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    snapshots = iter([{0: [0] * 10}, rows])
    monkeypatch.setattr(parallel, "_cpu_times", lambda path: next(snapshots))
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: None)
    clock = iter([0.0, 1.0])
    walls = iter([0.0, 2.0])
    monkeypatch.setattr(parallel.time, "process_time", lambda: next(clock))
    monkeypatch.setattr(parallel.time, "perf_counter", lambda: next(walls))
    assert parallel._busy_cpus({0}, tmp_path) == pytest.approx(0.5)


def test_cpu_times(tmp_path):
    stat = tmp_path / "stat"
    _stat(stat, {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3: [0] * 10})
    times = parallel._cpu_times(stat)
    assert set(times) == {0, 3}
    assert times[0][:4] == [1, 2, 3, 4]
    assert parallel._cpu_times(tmp_path / "missing") is None


def test_measured_cores_are_consistent():
    """On the machine running the tests, the detections agree with each other."""
    usable = parallel.usable_cores()
    assert 1 <= usable <= (parallel.os.cpu_count() or 1)
    assert parallel._busy_cpus(parallel._affinity()) >= 0.0
