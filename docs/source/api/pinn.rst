PINN surrogates
===============

The :py:mod:`saetass.pinn` subpackage holds two fast models of the transport equation, for what needs very many evaluations, such as parameter searches:

- A **batched solver** (:py:mod:`saetass.pinn.radial_solver`), differentiable, which runs on a GPU and needs no training and no time steps. It accepts any :math:`D(r)`, :math:`v(r)` and :math:`S(r)`, and it is the most accurate of the two.
- **Surrogates** (:py:mod:`saetass.pinn.surrogate`): networks that are trained once over a space of physical parameters and then evaluate :math:`\psi(t, r, p)` at any point, more than ten times faster than the solver, but only for the family they were trained on.

Solutions that a result depends on should be confirmed with :py:class:`~saetass.solver.Solver`, which remains the reference.

Both need PyTorch, an optional dependency:

.. code-block:: bash

    python -m pip install "saetass[pinn]"

Batched solver
--------------

:py:class:`~saetass.pinn.radial_solver.RadialSolver` solves the equation for each momentum without losses, with static coefficients and a constant source from an empty domain, for a whole batch of problems at once, and :py:class:`~saetass.pinn.radial_solver.WindBubbleSolver` does it for wind bubbles with the interface of the surrogates:

.. code-block:: python

    import torch

    from saetass.pinn.radial_solver import RadialSolver

    nodes = torch.linspace(0.0, 1.0, 513, dtype=torch.float64, device="cuda")
    solver = RadialSolver(nodes)
    x = solver.centers
    D = 0.01 * torch.where(x < 0.3, 1.0, 30.0)[None]
    v = (0.3 / x).clamp(max=1.0)[None]
    source = solver.shell(torch.tensor([0.1]).to(x))
    log_psi = solver.log_psi(D, v, source, t=torch.tensor([2.0]).to(x))

Everything in the call is differentiable, with respect to each value of ``D`` and ``v`` if needed, and :py:class:`~saetass.pinn.inversion.DiffusionInversion` uses that to recover :math:`D(r)` from :math:`\psi` with no assumed form.
There is no training set: a new problem only costs its solve.
With ``validation/cases/pinn_diffusion_inversion.py`` and data from SAETASS with 3 % of noise, 160 free values of :math:`D` reach a :math:`\chi^2` per point of 1.0 and a median error of 1 % in each zone, and they place a jump of 265 times one knot away from its true radius with 160 radii of data, but seven pc away with 40. Next to the shock the estimate oscillates by up to a factor of 2.2, outside its band. The coverage of the 68 % band ranged from 75 % to 95 %, so it is not calibrated.

Results on a shared NVIDIA L40S, with 48 nodes of the contour, on the 128 bubbles of the test set of the surrogate:

.. list-table::
   :header-rows: 1

   * - Nodes per bubble
     - :math:`L_2` error of :math:`\psi`: median
     - 95th percentile
     - Worst
     - Time per profile
   * - 128
     - 0.076 %
     - 0.97 %
     - 4.0 %
     - 0.14 ms
   * - 512
     - 0.0047 %
     - 0.062 %
     - 0.25 %
     - 0.14 ms
   * - 1024
     - 0.0012 %
     - 0.014 %
     - 0.057 %
     - 0.17 ms

Above :math:`10^{-6}` of the largest :math:`\psi`, the largest :math:`|\Delta \ln \psi|` against the same solver on 4096 nodes, over 256 random bubbles and times, is 0.019 % in the median, 1.1 % in the 95th percentile and 3.2 % in the worst case with 1024 nodes, and 0.062 %, 3.6 % and 10 % with 512.
That comparison is not independent, but the reference solutions are not accurate enough to measure it: at the worst bubble of the test set they converge to the solver as they are refined, with differences of 325 %, 93 %, 25 % and 6.5 % at resolutions 1, 2, 4 and 8.
The same happens with :py:class:`~saetass.solver.Solver`: at 3 pc and 100 GeV in the wind of ``validation/cases/pinn_parameter_search.py``, the difference is 33 % at 1200 cells, 8 % at 2400, 1.8 % at 4800 and 0.15 % at 9600.

Limits of the solver:

- No energy losses, no time-dependent coefficients or sources, and a source that does not change sign.
- Results are accurate relative to :math:`\psi` where it exceeds about :math:`10^{-6}` of its maximum. Below that the check of the module only guarantees :math:`0 \le \psi \le \psi_\mathrm{ss}`.
- A front carried by advection with almost no diffusion, such as the one of ``electrons_2d`` in the benchmarks, cannot be represented on the contour.
- The wind bubble needs :math:`\delta < 1`, so it does not cover Bohm diffusion.

.. warning::

    A surrogate only knows the family of problems it was trained on, and it refuses inputs outside that family. It does not replace :py:class:`~saetass.solver.Solver` for other profiles, operators or parameters.

What the surrogates cover
-------------------------

The current surrogate describes the wind bubble of :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator` with the Morlino21 profiles, for advection, diffusion and a source at the termination shock:

- Kolmogorov turbulence, and no energy or adiabatic losses, so that each momentum evolves independently.
- Static profiles, and a constant injection rate from an empty domain.
- A core radius ``R_c`` of zero, and the default ``D_ISM``.

Validity and accuracy
---------------------

The domain is :math:`\mathrm{Pe} \in [0.02, 300]`, :math:`\lambda \in [60, 700]`, :math:`a \in [2.2, 23]`, :math:`b \in [1.4, 7]` and a dimensionless time :math:`t\, v_b / R_{TS} \in [0.01, 10^4]`, which covers Kolmogorov bubbles from 1 GeV to 1 PeV.
Results of ``validation/cases/pinn_wind_bubble.py`` with the mean of two networks, on bubbles that took no part in the training:

.. list-table::
   :header-rows: 1

   * - Measure
     - Result
   * - Relative :math:`L_2` error of :math:`\psi`, test set of 128 bubbles, all times
     - median 0.08 %, 95th percentile 0.41 %, worst 2.2 %
   * - Largest :math:`|\Delta \ln \psi|` where :math:`\psi` exceeds :math:`10^{-6}` of its maximum, same set
     - median 0.44 %, 95th percentile 1.9 %, worst 14 %
   * - Particle budget against the reference, 32 bubbles
     - median 0.013 %, worst 0.29 %
   * - :math:`L_2` against SAETASS at 4800 cells, 8 physical bubbles above 100 GeV
     - surrogate: median 0.12 %, worst 0.47 %; SAETASS at 600 cells: median 1.4 %, worst 2.1 %

The largest errors of the test set lie at the earliest times of the domain and in the extremes of the parameters.
Two limits should be kept in mind:

- The network is trained on data only, and the physics enters through the variables, the coordinates and the exact steady state. A physics residual in the loss did not improve the accuracy: the residual reduced its interior tails, but the error against the reference grew slightly.
- For that reason :py:meth:`~saetass.pinn.surrogate.WindBubbleSurrogate.residual` is not a certificate of accuracy: it has heavy tails next to the interfaces even where the prediction is accurate. :py:meth:`~saetass.pinn.surrogate.WindBubbleSurrogate.spread` is an indication, which flagged 64 % of the points with an error above 3 %, with 0.5 % of false alarms, so it can miss errors.

How it works
------------

Without losses, the problem depends on four dimensionless numbers (see :py:mod:`saetass.pinn.wind_bubble`), and its steady state has a closed form.
The network only learns the logarithm of the ratio of :math:`\psi` to that steady state, in coordinates where the shock and the edge of the bubble are fixed, and from reference solutions of :py:mod:`saetass.pinn.reference`.

.. code-block:: python

    import astropy.constants as const
    import astropy.units as u
    import numpy as np

    from saetass.pinn.surrogate import WindBubbleSurrogate

    surrogate = WindBubbleSurrogate.average(
        ["wind_bubble_0.pt", "wind_bubble_1.pt"], device="cuda"
    )
    psi = surrogate.psi_p(
        r=np.array([5.0, 30.0, 120.0]) * u.pc,
        momenta=[1e2, 1e4] * u.GeV,
        t=3 * u.Myr,
        L_wind=2e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=20 * const.m_p / u.cm**3,
        t_b=3 * u.Myr,
        r_max=300 * u.pc,
        E_max=1 * u.PeV,
    )

For parameter searches, :py:meth:`~saetass.pinn.surrogate.WindBubbleSurrogate.log_psi` evaluates batches of tensors, and it is differentiable, with the scales of :py:class:`~saetass.pinn.wind_bubble.FastScales`.

Dimensionless problem
---------------------

.. automodule:: saetass.pinn.wind_bubble
    :members:

Reference solutions
-------------------

.. automodule:: saetass.pinn.reference
    :members:

Datasets
--------

.. automodule:: saetass.pinn.dataset
    :members:

Batched solver of the radial equation
-------------------------------------

.. automodule:: saetass.pinn.radial_solver
    :members:

Inversion of the diffusion coefficient
--------------------------------------

.. automodule:: saetass.pinn.inversion
    :members:

Network
-------

.. automodule:: saetass.pinn.network
    :members:

Training
--------

.. automodule:: saetass.pinn.training
    :members:

Surrogate
---------

.. automodule:: saetass.pinn.surrogate
    :members:
