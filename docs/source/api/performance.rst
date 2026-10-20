Performance
===========

The numerical kernels of SAETASS are compiled with Numba and cached on disk, so a new process only loads them.
Each operator advances every row of the grid (for instance every momentum of a spatial operator) in a single compiled pass per sub-cycle, and rows are independent: the results do not depend on the number of threads.

Threads inside one simulation are managed by :py:mod:`saetass.parallel`. A kernel call runs serially unless its work is large enough, and then on a team of threads sized by the largest calls of the simulation: a few threads for light calls, and every idle physical core for heavy ones, such as the losses of electrons over many sub-cycles. 1D grids run serially. On Linux, cores kept busy by other processes are left free, because a team larger than the idle cores makes every call up to a hundred times slower. Set a limit with :py:func:`~saetass.parallel.set_max_threads` or the environment variable ``SAETASS_MAX_THREADS``, for instance to run several simulations side by side, and the threads available to Numba with ``NUMBA_NUM_THREADS``.

The ``benchmarks`` directory of the repository measures the cost of every operator and of representative problems.

------------------

Threads
-------

.. automodule:: saetass.parallel
    :members:
