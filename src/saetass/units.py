r"""
The units module defines the canonical physical units of SAETASS, which act as the single internal standard for every physical dimension handled by the package.

All physical inputs of the public API, i.e. :py:class:`~saetass.grid.Grid`, :py:class:`~saetass.state.State`, the operator parameters passed to :py:class:`~saetass.solver.Solver` and the calculators in ``saetass.utils``, must be given as :py:class:`astropy.units.Quantity` objects.
Any unit with the correct physical dimensions is accepted: values are converted once, at the boundary of the package, into the canonical units defined here.
Bare numbers or incompatible units raise an error instead of being silently interpreted in some implicit unit.
Function arguments are validated with :py:func:`astropy.units.quantity_input`, while the physical inputs it cannot see, such as entries of the operator parameter dictionaries, keyword arguments or values returned by user callables, are validated with :py:func:`validate_quantity`, which applies the same rules and raises the same errors.

Past this boundary, the numerical kernels only operate on bare ``float64`` arrays expressed in canonical units, so that no unit bookkeeping takes place inside the time loop.
Results are returned to the user as Quantities again, e.g. through :py:attr:`~saetass.state.State.psi_p`, and can be converted to any other compatible unit with :py:meth:`~astropy.units.Quantity.to`.

The base canonical units are:

.. list-table::
   :header-rows: 1

   * - Dimension
     - Constant
     - Canonical unit
   * - Length
     - :py:data:`LENGTH`
     - :math:`\mathrm{pc}`
   * - Time
     - :py:data:`TIME`
     - :math:`\mathrm{Myr}`
   * - Mass
     - :py:data:`MASS`
     - :math:`M_\odot`
   * - Energy
     - :py:data:`ENERGY`
     - :math:`\mathrm{GeV}`
   * - Momentum
     - :py:data:`MOMENTUM`
     - :math:`\mathrm{GeV}/c`

The remaining constants are derived from them, e.g. :py:data:`PSI_P` is :math:`\mathrm{pc^{-3}\,(GeV/c)^{-1}}`.
They are meant both to build input Quantities and to document the dimensions each argument expects, which is how the rest of the API reference refers to them.

Momentum is expressed in the named unit :py:data:`GEV_C`, registered in Astropy as ``GeV_c``.
Quantities built as ``p * u.GeV / const.c`` are converted to it automatically.

.. code-block:: python

    import astropy.units as u
    import numpy as np

    from saetass import Grid, State
    from saetass import units as su

    grid = Grid.log_spaced(
        r_min=0.01 * u.pc, r_max=50 * u.pc, num_r_cells=200,
        p_min=1 * su.MOMENTUM, p_max=1e5 * su.MOMENTUM, num_p_cells=60,
        t_min=0 * u.yr, t_max=1 * u.Myr, num_timesteps=100,
    )
    state = State(grid=grid, psi_p=np.zeros(grid.shape) * u.cm**-3 / su.MOMENTUM)

    state.psi_p.to(u.cm**-3 / su.MOMENTUM)  # results are Quantities again

--------------
"""

from typing import Any, Final

import astropy.constants as const
import astropy.units as u

# =============================================================================
# 1. Base Canonical Physical Units (Astrophysical Standard)
# =============================================================================
#: Canonical length unit, :math:`\mathrm{pc}`.
LENGTH: Final[u.Unit] = u.pc
#: Canonical time unit, :math:`\mathrm{Myr}`.
TIME: Final[u.Unit] = u.Myr
#: Canonical mass unit, :math:`M_\odot`.
MASS: Final[u.Unit] = u.M_sun
#: Canonical energy unit, :math:`\mathrm{GeV}`.
ENERGY: Final[u.Unit] = u.GeV

# =============================================================================
# 2. Kinematic and Dynamic Units
# =============================================================================
#: Canonical velocity unit, :math:`\mathrm{pc\,Myr^{-1}}`.
VELOCITY: Final[u.Unit] = LENGTH / TIME  # pc / Myr

#: Named momentum unit :math:`\mathrm{GeV}/c`, registered in Astropy as ``GeV_c``.
#:
#: It cannot be written as ``(u.GeV / const.c).unit``, which keeps only the unit of
#: :math:`c` and silently drops its value, yielding :math:`\mathrm{GeV\,s\,m^{-1}}`
#: (:math:`1\,\mathrm{GeV\,s\,m^{-1}} \approx 3 \times 10^8\,\mathrm{GeV}/c`).
#: Registering it allows unit strings containing it, e.g. ``"1 / (GeV_c pc3)"``, to be
#: parsed back by Astropy.
GEV_C: Final[u.Unit] = u.def_unit(
    ["GeV_c"],
    u.GeV / const.c,
    format={"latex": r"GeV/c"},
    doc="Momentum unit GeV/c",
)
u.add_enabled_units([GEV_C])

#: Canonical momentum unit, :py:data:`GEV_C`.
MOMENTUM: Final[u.Unit] = GEV_C  # GeV / c
#: Canonical spatial diffusion coefficient unit, :math:`\mathrm{pc^2\,Myr^{-1}}`.
DIFFUSION_COEFFICIENT: Final[u.Unit] = LENGTH**2 / TIME  # pc^2 / Myr
#: Alias of :py:data:`DIFFUSION_COEFFICIENT`.
DIFFUSION: Final[u.Unit] = DIFFUSION_COEFFICIENT  # Alias

# Loss and Evolution Rates
#: Canonical energy loss rate unit, :math:`\mathrm{GeV\,Myr^{-1}}`.
ENERGY_LOSS_RATE: Final[u.Unit] = ENERGY / TIME  # GeV / Myr
#: Canonical momentum loss rate unit, :math:`(\mathrm{GeV}/c)\,\mathrm{Myr^{-1}}`.
MOMENTUM_LOSS_RATE: Final[u.Unit] = MOMENTUM / TIME  # (GeV / c) / Myr

# =============================================================================
# 3. Geometric and Environmental Units
# =============================================================================
#: Canonical area unit, :math:`\mathrm{pc^2}`.
AREA: Final[u.Unit] = LENGTH**2  # pc^2
#: Canonical volume unit, :math:`\mathrm{pc^3}`.
VOLUME: Final[u.Unit] = LENGTH**3  # pc^3
#: Canonical number density unit, :math:`\mathrm{pc^{-3}}`.
NUMBER_DENSITY: Final[u.Unit] = LENGTH**-3  # pc^-3
#: Canonical mass density unit, :math:`M_\odot\,\mathrm{pc^{-3}}`.
MASS_DENSITY: Final[u.Unit] = MASS / LENGTH**3  # M_sun / pc^3

#: Canonical magnetic field unit, :math:`\mu\mathrm{G}`.
MAGNETIC_FIELD: Final[u.Unit] = u.uG  # microGauss
#: Canonical luminosity unit, :math:`\mathrm{GeV\,Myr^{-1}}`.
LUMINOSITY: Final[u.Unit] = ENERGY / TIME  # GeV / Myr (convertible to erg / s)
#: Canonical mass loss rate unit, :math:`M_\odot\,\mathrm{Myr^{-1}}`.
MASS_LOSS_RATE: Final[u.Unit] = MASS / TIME  # M_sun / Myr (convertible to M_sun / yr)

#: Canonical cross-section unit, :math:`\mathrm{pc^2}`.
CROSS_SECTION: Final[u.Unit] = LENGTH**2  # pc^2 (convertible to cm^2 / mbarn)
#: Canonical differential cross-section unit, :math:`\mathrm{pc^2\,GeV^{-1}}`.
DIFFERENTIAL_CROSS_SECTION: Final[u.Unit] = LENGTH**2 / ENERGY  # pc^2 / GeV

# =============================================================================
# 4. Particle Distribution Representations (State)
# =============================================================================
#: Canonical unit of the phase-space distribution
#: :math:`f_\mathrm{ps} = \frac{dn}{d^3r\,d^3p}`, :math:`\mathrm{pc^{-3}\,(GeV/c)^{-3}}`.
F_PS: Final[u.Unit] = NUMBER_DENSITY / (MOMENTUM**3)

#: Canonical unit of the momentum differential density
#: :math:`\psi_p = \frac{dn}{dp} = 4\pi p^2 f_\mathrm{ps}`, :math:`\mathrm{pc^{-3}\,(GeV/c)^{-1}}`.
PSI_P: Final[u.Unit] = NUMBER_DENSITY / MOMENTUM

#: Canonical unit of the energy differential density
#: :math:`\psi_E = \frac{dn}{dE}`, :math:`\mathrm{pc^{-3}\,GeV^{-1}}`.
PSI_E: Final[u.Unit] = NUMBER_DENSITY / ENERGY

#: Canonical unit of a source term
#: :math:`Q = \frac{\partial \psi_p}{\partial t}`, :math:`\mathrm{pc^{-3}\,(GeV/c)^{-1}\,Myr^{-1}}`.
SOURCE_PSI_P: Final[u.Unit] = PSI_P / TIME
#: Canonical unit of a source term :math:`\frac{\partial \psi_E}{\partial t}`, :math:`\mathrm{pc^{-3}\,GeV^{-1}\,Myr^{-1}}`.
SOURCE_PSI_E: Final[u.Unit] = PSI_E / TIME


# =============================================================================
# 5. Validation of Physical Inputs
# =============================================================================
def validate_quantity(value: Any, unit: u.UnitBase, name: str) -> u.Quantity:
    """
    Validate a physical input and return it expressed in ``unit``.

    Function arguments are validated with :py:func:`astropy.units.quantity_input`.
    This function applies the same rules, and raises the same errors, to the physical
    inputs that the decorator cannot see, such as entries of parameter dictionaries,
    ``**kwargs``, values returned by user callables or property assignments.

    Parameters
    ----------
    value : astropy.units.Quantity
        Physical input to validate.
    unit : astropy.units.UnitBase
        Unit in which the input is returned, typically one of the canonical units of this module.
    name : str
        Description of the input, used in error messages.

    Returns
    -------
    astropy.units.Quantity
        The input expressed in ``unit``, without copying it when it already is.

    Raises
    ------
    TypeError
        If ``value`` is not a Quantity.
    astropy.units.UnitsError
        If ``value`` has units incompatible with ``unit``.
    """
    if not hasattr(value, "unit"):
        raise TypeError(
            f"{name} has no 'unit' attribute. You should pass in an astropy Quantity instead."
        )
    try:
        return u.Quantity(value, copy=False).to(unit, copy=False)
    except u.UnitConversionError:
        raise u.UnitsError(
            f"{name} must be in units convertible to '{unit}'."
        ) from None


def require_bare(value: Any, name: str) -> None:
    """
    Reject a Quantity where a bare number or array is expected.

    Used at the internal boundary of SAETASS, where numerical kernels receive bare floats
    in canonical units, and for non-physical options such as CFL numbers.

    Parameters
    ----------
    value : Any
        Input to validate.
    name : str
        Description of the input, used in error messages.

    Raises
    ------
    TypeError
        If ``value`` is a Quantity.
    """
    if hasattr(value, "unit"):
        raise TypeError(
            f"{name} must be a bare number or array, not an astropy Quantity ({value!r})."
        )


__all__ = [
    "LENGTH",
    "TIME",
    "MASS",
    "ENERGY",
    "VELOCITY",
    "GEV_C",
    "MOMENTUM",
    "DIFFUSION_COEFFICIENT",
    "DIFFUSION",
    "ENERGY_LOSS_RATE",
    "MOMENTUM_LOSS_RATE",
    "AREA",
    "VOLUME",
    "NUMBER_DENSITY",
    "MASS_DENSITY",
    "MAGNETIC_FIELD",
    "LUMINOSITY",
    "MASS_LOSS_RATE",
    "CROSS_SECTION",
    "DIFFERENTIAL_CROSS_SECTION",
    "F_PS",
    "PSI_P",
    "PSI_E",
    "SOURCE_PSI_P",
    "SOURCE_PSI_E",
    "validate_quantity",
    "require_bare",
]
