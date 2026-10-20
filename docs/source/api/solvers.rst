Solvers
=======

SAETASS is designed to be modular and extensible, allowing users to implement and integrate their own
solvers for any physical process in the spherical astroparticle transport equation. The solvers are
responsible for advancing the state of the system in time according to the governing equations and numerical methods.
SAETASS includes a set of built-in solvers for common astrophysical processes, such as a :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`
for advection and loss terms, a :py:class:`~saetass.solvers.diffusion_solver.DiffusionSolver` for diffusion term and a :py:class:`~saetass.solvers.source_solver.SourceSolver`
for source terms.

Operator parameters are given to the :py:class:`~saetass.solver.Solver` as physical :py:class:`~astropy.units.Quantity` objects, or as callables returning them for time-dependent parameters.
Each :py:class:`~saetass.solver.SubSolver` declares the parameters it accepts, together with their canonical units, in :py:attr:`~saetass.solver.SubSolver.PARAM_SPECS` through :py:class:`~saetass.solver.ParamSpec` entries.
The :py:class:`~saetass.solver.Solver` converts them with :py:meth:`~saetass.solver.SubSolver.convert_params` into bare floats in the canonical units of :py:mod:`saetass.units`, so the numerical kernels never handle units, and rejects any parameter that is not declared or whose units are incompatible.

Users can also create custom operators by providing their own implementations inheriting from :py:class:`~saetass.solver.SubSolver` and declaring their accepted parameters in :py:attr:`~saetass.solver.SubSolver.PARAM_SPECS`.
Operators can also report their characteristic timescale and dimensionless numbers through :py:meth:`~saetass.solver.SubSolver.characteristic_scales`, which :py:meth:`~saetass.solver.Solver.diagnostics` uses to detect time steps that compromise the accuracy of a simulation (see :py:mod:`saetass.diagnostics`).

------------------

The solver module
------------------

.. automodule:: saetass.solver
    :members:

Physical solvers
----------------

Hyperbolic solver
~~~~~~~~~~~~~~~~~

.. automodule:: saetass.solvers.hyperbolic_solver
    :members:
    :show-inheritance:

Hyperbolic kernels
~~~~~~~~~~~~~~~~~~

.. automodule:: saetass.solvers.hyperbolic_kernels
    :members: advance_rows, reciprocal_spacings, LIMITERS, LEFT_ZERO_FLUX, LEFT_EXTRAPOLATED, NS_PER_CELL

Advection Solver
^^^^^^^^^^^^^^^^
.. automodule:: saetass.solvers.advection_solver
    :members:
    :undoc-members:
    :show-inheritance:

Loss Solver
^^^^^^^^^^^
.. automodule:: saetass.solvers.loss_solver
    :members:
    :undoc-members:
    :show-inheritance:

Diffusion solver
~~~~~~~~~~~~~~~~

.. automodule:: saetass.solvers.diffusion_solver
    :members:
    :undoc-members:
    :show-inheritance:

Source solver
~~~~~~~~~~~~~

.. automodule:: saetass.solvers.source_solver
    :members:
    :undoc-members:
    :show-inheritance:
