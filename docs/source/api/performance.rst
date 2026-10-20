Performance
===========

The numerical kernels of SAETASS are compiled with Numba and cached on disk, so a new process only loads them.
Each operator advances every row of the grid (for instance every momentum of a spatial operator) in a single compiled pass per sub-cycle, and rows are independent: the results do not depend on the number of threads.

Two levels of parallelism are available:

* **Threads inside one simulation**, managed by :py:mod:`saetass.parallel`. A kernel call runs serially unless its work is large enough, and then on a team of threads sized by the largest calls of the simulation: a few threads for light calls, and every idle physical core for heavy ones, such as the losses of electrons over many sub-cycles. 1D grids run serially. On Linux, cores kept busy by other processes are left free, because a team larger than the idle cores makes every call up to a hundred times slower. Set a limit with :py:func:`~saetass.parallel.set_max_threads` or the environment variable ``SAETASS_MAX_THREADS``, for instance to run several simulations side by side, and the threads available to Numba with ``NUMBA_NUM_THREADS``.
* **Processes for independent simulations**, such as the energies or parameters of a sweep, with :py:func:`saetass.sweep.run_sweep`. This scales much better than threads: each process advances its own grid with no synchronization.

Optionally, a time-dependent parameter can be given as a **separable time dependence**, a sum of fixed profiles scaled by functions of time (:py:class:`~saetass.parameters.Separable`). The advection and loss operators then advance all the CFL sub-cycles of a step in one compiled call, as for static parameters, instead of evaluating a Python callable on the whole grid at every sub-cycle. It gives the same results as the equivalent callable and only pays off when an operator needs many sub-cycles per step, as the losses of high-energy electrons do.

The ``benchmarks`` directory of the repository measures the cost of every operator and of representative problems.

------------------

Threads
-------

.. automodule:: saetass.parallel
    :members:

Separable time dependences
--------------------------

.. automodule:: saetass.parameters
    :members: Separable, CanonicalSeparable

Parameter sweeps
----------------

.. automodule:: saetass.sweep
    :members:
