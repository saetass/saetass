import astropy.units as u
import numpy as np
import pytest

from saetass import Separable
from saetass import units as su
from saetass.utils.energy_losses import EnergyLossCalculator, Particle


@pytest.fixture
def base_args():
    return {
        "E_grid": np.logspace(-1, 3, 50) * u.GeV,
        "r_grid": np.linspace(0.1, 10, 20) * u.pc,
        "n_gas": np.ones(20) * u.cm**-3,
    }


class TestEnergyLossCalculator:
    def test_initialization(self, base_args):
        calc_p = EnergyLossCalculator(**base_args, particle="proton")
        assert calc_p.particle == Particle.PROTON

        # Test raw ndarray inputs without units fail fail-fast boundary
        with pytest.raises(TypeError):
            EnergyLossCalculator(
                E_grid=np.logspace(-1, 3, 50),
                r_grid=np.linspace(0.1, 10, 20),
                n_gas=np.ones(20),
                particle="electron",
            )

        with pytest.raises(ValueError):
            EnergyLossCalculator(**base_args, particle="invalid")

    def test_invalid_units(self, base_args):
        # Pass wrong units
        wrong_args = base_args.copy()
        wrong_args["E_grid"] = np.ones(50) * u.K
        with pytest.raises((u.UnitsError, TypeError, ValueError)):
            EnergyLossCalculator(**wrong_args, particle="proton")

        wrong_args2 = base_args.copy()
        wrong_args2["r_grid"] = np.ones(50) * u.K
        with pytest.raises((u.UnitsError, TypeError, ValueError)):
            EnergyLossCalculator(**wrong_args2, particle="proton")

        wrong_args3 = base_args.copy()
        wrong_args3["n_gas"] = np.ones(50) * u.K
        with pytest.raises((u.UnitsError, TypeError, ValueError)):
            EnergyLossCalculator(**wrong_args3, particle="proton")

    def test_ionization_losses(self, base_args):
        calc_p = EnergyLossCalculator(**base_args, particle="proton")
        dE = calc_p.compute_ionization_losses()
        assert dE.unit.is_equivalent(u.GeV / u.s)

        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        dE_e = calc_e.compute_ionization_losses()
        assert dE_e.unit.is_equivalent(u.GeV / u.s)

    def test_pion_losses(self, base_args):
        calc_p = EnergyLossCalculator(**base_args, particle="proton")
        dE = calc_p.compute_pion_production_losses()
        assert dE.unit.is_equivalent(u.GeV / u.s)

    def test_synchrotron_losses(self, base_args):
        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        B_field = np.ones(len(base_args["r_grid"])) * u.G
        dE = calc_e.compute_synchrotron_losses(B_field=B_field)
        assert dE.unit.is_equivalent(u.GeV / u.s)

        U_B = np.ones(len(base_args["r_grid"])) * u.eV / u.cm**3
        dE_UB = calc_e.compute_synchrotron_losses(U_B=U_B)
        assert dE_UB.unit.is_equivalent(u.GeV / u.s)

        # Test U_B as scalar
        dE_UB_scalar = calc_e.compute_synchrotron_losses(U_B=1.0 * u.eV / u.cm**3)
        assert dE_UB_scalar.unit.is_equivalent(u.GeV / u.s)

        with pytest.raises(ValueError):
            calc_e.compute_synchrotron_losses()  # Neither provided

    def test_bremsstrahlung_losses(self, base_args):
        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        mask = np.ones(len(base_args["r_grid"]), dtype=bool)  # Ionised
        dE_ws = calc_e.compute_bremsstrahlung_losses(mask)
        assert dE_ws.unit.is_equivalent(u.GeV / u.s)

        mask_false = np.zeros(len(base_args["r_grid"]), dtype=bool)  # Neutral
        dE_ss = calc_e.compute_bremsstrahlung_losses(mask_false)
        assert dE_ss.unit.is_equivalent(u.GeV / u.s)

    def test_coulomb_losses(self, base_args):
        calc_p = EnergyLossCalculator(**base_args, particle="proton")
        T_gas = np.ones(len(base_args["r_grid"])) * u.K
        dE = calc_p.compute_coulomb_losses(T_gas)
        assert dE.unit.is_equivalent(u.GeV / u.s)

        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        dE_e = calc_e.compute_coulomb_losses(T_gas)
        assert dE_e.unit.is_equivalent(u.GeV / u.s)

    def test_inverse_compton_losses(self, base_args):
        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        eps_grid = np.logspace(-3, 0, 10) * u.eV
        dn_deps = np.ones((10, len(base_args["r_grid"]))) * u.cm**-3 / u.eV
        dE = calc_e.compute_inverse_compton_losses(eps_grid, dn_deps)
        assert dE.unit.is_equivalent(u.GeV / u.s)

        # Invalid eps_grid shape
        with pytest.raises(ValueError):
            calc_e.compute_inverse_compton_losses(np.ones((2, 2)) * u.eV, dn_deps)

        # Invalid dn_deps shape
        with pytest.raises(ValueError):
            calc_e.compute_inverse_compton_losses(
                eps_grid, np.ones(2) * u.cm**-3 / u.eV
            )

        # Mismatch
        with pytest.raises(ValueError):
            calc_e.compute_inverse_compton_losses(
                eps_grid, np.ones((11, len(base_args["r_grid"]))) * u.cm**-3 / u.eV
            )

    def test_total_losses_and_timescales(self, base_args):
        calc_e = EnergyLossCalculator(**base_args, particle="electron")
        with pytest.raises(RuntimeError):
            calc_e.compute_total_losses()

        calc_e.compute_ionization_losses()
        E_dot_total = calc_e.compute_total_losses()
        assert E_dot_total.unit.is_equivalent(u.GeV / u.s)

        # Repeat should use cached total correctly
        P_dot = calc_e.get_momentum_loss_rate()
        assert P_dot.shape == (len(base_args["E_grid"]), len(base_args["r_grid"]))

        timescales = calc_e.get_loss_timescales()
        assert "ionization" in timescales
        assert "total" in timescales

        timescales_r = calc_e.get_loss_timescales(r_index=0)
        assert timescales_r["total"].shape == (len(base_args["E_grid"]),)


class TestTimeDependentLosses:
    @pytest.fixture
    def calculator(self, base_args):
        calc = EnergyLossCalculator(**base_args, particle="electron")
        calc.compute_synchrotron_losses(B_field=10 * u.uG)
        calc.compute_bremsstrahlung_losses(ionised_mask=np.zeros(20, dtype=bool))
        calc.compute_ionization_losses()
        calc.compute_coulomb_losses(T_gas=1e4 * u.K)
        return calc

    def test_without_time_factors_the_rate_is_static(self, calculator):
        static = calculator.get_momentum_loss_rate()
        separable = calculator.get_momentum_loss_rate(time_factors={})
        assert isinstance(separable, Separable)
        assert separable.factors == [1.0]
        np.testing.assert_array_equal(
            separable(0 * u.Myr).to_value(static.unit), static.value
        )

    def test_time_factors_scale_each_mechanism(self, calculator):
        def B_squared(t):
            return (1.0 + t.to_value(u.Myr)) ** 2

        def gas(t):
            return 2.0 - t.to_value(u.Myr)

        P_dot = calculator.get_momentum_loss_rate(
            time_factors={
                "synchrotron": B_squared,
                "bremsstrahlung": gas,
                "ionization": gas,
                "coulomb": 0.5,
            }
        )
        # Mechanisms that share a function share a profile, and constant ones fold into one
        assert P_dot.factors == [B_squared, gas, 1.0]

        t = 0.3 * u.Myr
        components = calculator._E_dot_components
        E_dot = (
            B_squared(t) * components["synchrotron"]
            + gas(t) * (components["bremsstrahlung"] + components["ionization"])
            + 0.5 * components["coulomb"]
        )
        expected = E_dot * calculator.dp_dE[:, np.newaxis]
        np.testing.assert_allclose(
            P_dot(t).to_value(su.MOMENTUM_LOSS_RATE),
            expected.to_value(su.MOMENTUM_LOSS_RATE),
            rtol=1e-14,
        )

        only_dynamic = calculator.get_momentum_loss_rate(
            time_factors=dict.fromkeys(components, gas)
        )
        assert only_dynamic.factors == [gas]

    def test_invalid_time_factors(self, base_args, calculator):
        with pytest.raises(ValueError, match="have not been computed: \\['pion'\\]"):
            calculator.get_momentum_loss_rate(time_factors={"pion": 2.0})
        with pytest.raises(TypeError, match="'synchrotron' must be a callable"):
            calculator.get_momentum_loss_rate(time_factors={"synchrotron": "B"})
        empty = EnergyLossCalculator(**base_args, particle="electron")
        with pytest.raises(RuntimeError, match="No energy loss mechanisms"):
            empty.get_momentum_loss_rate(time_factors={})
