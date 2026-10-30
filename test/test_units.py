import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import units as su


def test_canonical_base_units():
    """Verify base canonical units adhere to astrophysical standard."""
    assert su.LENGTH == u.pc
    assert su.TIME == u.Myr
    assert su.MASS == u.M_sun
    assert su.ENERGY == u.GeV


def test_canonical_kinematic_and_dynamic_units():
    """Verify derived kinematic and dynamic units."""
    assert su.VELOCITY.is_equivalent(u.pc / u.Myr)
    assert su.VELOCITY.is_equivalent(u.km / u.s)
    assert su.MOMENTUM.is_equivalent(u.GeV / const.c)
    assert su.MOMENTUM.is_equivalent(u.g * u.cm / u.s)
    assert su.DIFFUSION_COEFFICIENT.is_equivalent(u.pc**2 / u.Myr)
    assert su.DIFFUSION_COEFFICIENT.is_equivalent(u.cm**2 / u.s)
    assert su.DIFFUSION == su.DIFFUSION_COEFFICIENT
    assert su.ENERGY_LOSS_RATE.is_equivalent(u.GeV / u.Myr)
    assert su.ENERGY_LOSS_RATE.is_equivalent(u.GeV / u.s)
    assert su.MOMENTUM_LOSS_RATE.is_equivalent((u.GeV / const.c) / u.Myr)


def test_canonical_geometric_and_environmental_units():
    """Verify geometric and medium units."""
    assert su.AREA.is_equivalent(u.pc**2)
    assert su.VOLUME.is_equivalent(u.pc**3)
    assert su.NUMBER_DENSITY.is_equivalent(u.pc**-3)
    assert su.NUMBER_DENSITY.is_equivalent(u.cm**-3)
    assert su.MASS_DENSITY.is_equivalent(u.M_sun / u.pc**3)
    assert su.MAGNETIC_FIELD.is_equivalent(u.uG)
    assert su.MAGNETIC_FIELD.is_equivalent(u.G)
    assert su.CROSS_SECTION.is_equivalent(u.pc**2)
    assert su.CROSS_SECTION.is_equivalent(u.cm**2)
    assert su.CROSS_SECTION.is_equivalent(u.mbarn)


def test_canonical_state_units():
    """Verify particle distribution units."""
    assert su.F_PS.is_equivalent(u.pc**-3 / (u.GeV / const.c) ** 3)
    assert su.F_PS.is_equivalent(u.cm**-3 / (u.GeV / const.c) ** 3)
    assert su.PSI_P.is_equivalent(u.pc**-3 / (u.GeV / const.c))
    assert su.PSI_P.is_equivalent(u.cm**-3 / (u.GeV / const.c))
    assert su.PSI_E.is_equivalent(u.pc**-3 / u.GeV)
    assert su.PSI_E.is_equivalent(u.cm**-3 / u.GeV)
    assert su.SOURCE_PSI_P.is_equivalent(su.PSI_P / u.Myr)
    assert su.SOURCE_PSI_E.is_equivalent(su.PSI_E / u.Myr)


def test_canonical_units_have_unit_scale():
    """
    Canonical units must match their physical definitions in scale, not only in
    dimension: ``is_equivalent`` cannot detect e.g. GeV s/m masquerading as GeV/c.
    """
    gev_c = u.GeV / const.c
    assert (1.0 * su.MOMENTUM).to_value(gev_c.unit) == pytest.approx(gev_c.value)
    assert (1.0 * su.MOMENTUM * const.c).to_value(u.GeV) == pytest.approx(1.0)
    assert (1.0 * su.MOMENTUM_LOSS_RATE).to_value(gev_c.unit / u.Myr) == pytest.approx(
        gev_c.value
    )
    assert (1.0 * su.PSI_P).to_value(u.pc**-3 / gev_c.unit) == pytest.approx(
        1.0 / gev_c.value
    )
    assert (1.0 * su.F_PS).to_value(u.pc**-3 / gev_c.unit**3) == pytest.approx(
        1.0 / gev_c.value**3
    )


def test_momentum_unit_string_roundtrip():
    """Unit strings of composite canonical units must be parseable back."""
    for unit in (su.MOMENTUM, su.PSI_P, su.F_PS, su.SOURCE_PSI_P):
        assert u.Unit(unit.to_string()) == unit


def test_validate_quantity_converts_without_copying():
    """validate_quantity returns the input in the requested unit, as a view if possible."""
    r = np.linspace(1.0, 2.0, 5) * u.kpc
    out = su.validate_quantity(r, su.LENGTH, "r")
    assert out.unit == su.LENGTH
    np.testing.assert_allclose(out.value, r.to_value(u.pc))

    r_pc = np.linspace(1.0, 2.0, 5) * su.LENGTH
    assert np.shares_memory(su.validate_quantity(r_pc, su.LENGTH, "r"), r_pc)


def test_validate_quantity_errors_match_quantity_input():
    """validate_quantity raises the same error types as astropy's quantity_input."""
    with pytest.raises(TypeError, match="'r' has no 'unit' attribute"):
        su.validate_quantity(np.ones(3), su.LENGTH, "'r'")
    with pytest.raises(u.UnitsError, match="'r' must be in units convertible to 'pc'"):
        su.validate_quantity(np.ones(3) * u.s, su.LENGTH, "'r'")


def test_require_bare():
    """require_bare accepts bare values and rejects Quantities."""
    su.require_bare(0.5, "cfl")
    su.require_bare(np.ones(3), "values")
    with pytest.raises(TypeError, match="cfl must be a bare number or array"):
        su.require_bare(0.5 * u.s, "cfl")
