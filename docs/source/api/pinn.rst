PINN surrogates
===============

The :py:mod:`saetass.pinn` subpackage provides mesh-free surrogates of the transport equation: networks that are trained once over a space of physical parameters and then evaluate :math:`\psi(t, r, p)` at any point, on a CPU or a GPU, in a time much shorter than a simulation.
They are meant for what needs very many evaluations, such as parameter searches.
Solutions that a result depends on should be confirmed with :py:class:`~saetass.solver.Solver`, which remains the reference.

The surrogates need PyTorch, an optional dependency:

.. code-block:: bash

    python -m pip install "saetass[pinn]"

.. warning::

    A surrogate only knows the family of problems it was trained on, and it refuses inputs outside that family. It does not replace :py:class:`~saetass.solver.Solver` for other profiles, operators or parameters.

What is covered
---------------

The current surrogate describes the wind bubble of :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator` with the Morlino21 profiles, for advection, diffusion and a source at the termination shock:

- Kolmogorov turbulence, and no energy or adiabatic losses, so that each momentum evolves independently.
- Static profiles, and a constant injection rate from an empty domain.
- A core radius ``R_c`` of zero, and the default ``D_ISM``.

How it works
------------

Without losses, the problem depends on four dimensionless numbers (see :py:mod:`saetass.pinn.wind_bubble`), and its steady state has a closed form.
The network only learns the logarithm of the ratio of :math:`\psi` to that steady state, in coordinates where the shock and the edge of the bubble are fixed, and from reference solutions of :py:mod:`saetass.pinn.reference`.

.. code-block:: python

    import astropy.constants as const
    import astropy.units as u
    import numpy as np

    from saetass.pinn.surrogate import WindBubbleSurrogate

    surrogate = WindBubbleSurrogate.load("wind_bubble.pt", device="cuda")
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
