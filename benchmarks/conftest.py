import numba
import pytest

import saetass


@pytest.fixture
def bench(benchmark):
    """The ``benchmark`` fixture, recording the threads available to the kernels."""
    benchmark.extra_info["numba_threads"] = numba.config.NUMBA_NUM_THREADS
    return benchmark


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info["numba_version"] = numba.__version__
    machine_info["numba_threads"] = numba.config.NUMBA_NUM_THREADS
    machine_info["numba_threading_layer"] = numba.config.THREADING_LAYER
    # The measured code, to check runs of other versions (see README.md)
    machine_info["saetass_path"] = saetass.__file__
