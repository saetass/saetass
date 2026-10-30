Utils
=====

The utility module in SAETASS provides supplementary physical calculators to easily compute complex backgrounds and processes required by the transport solvers. These tools are designed to be independent, robust and extensible for a wide range of astrophysical environments.

SAETASS currently includes built-in utilities to compute particle loss timescales through the :py:class:`~saetass.utils.energy_losses.EnergyLossCalculator`, determine the spatial profiles of stellar wind bubbles via the :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`, evaluate high-energy interaction cross-sections through modular models in :py:mod:`~saetass.utils.cross_sections`, and calculate multi-messenger non-thermal radiative signatures (gamma-rays and neutrinos) through the :py:class:`~saetass.utils.emissions.EmissionCalculator`.

Users can make use of these utilities to build physical setups, pass the generated data directly to the SAETASS solvers, and compute synthetic observable signatures from cosmic ray distributions.

------------------

Energy losses module
--------------------

.. automodule:: saetass.utils.energy_losses
    :members:

Bubble profiles module
----------------------

.. automodule:: saetass.utils.bubble_profiles
    :members:

Emissions module
----------------

.. automodule:: saetass.utils.emissions
    :members:

Cross-sections module
---------------------

.. automodule:: saetass.utils.cross_sections
    :members:
