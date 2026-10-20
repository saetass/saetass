from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import units as su
from saetass.grid import Grid
from saetass.state import State
from saetass.utils.cross_sections import AnalyticalBremsstrahlung
from saetass.utils.emissions import EmissionCalculator
from saetass.utils.energy_losses import Particle

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import naima
except ImportError:
    naima = None


DEBUG_DIR = Path(__file__).parent / "debug"


def _handle_plot(fig, filename: str, output_dir: Path = DEBUG_DIR) -> None:
    """Save matplotlib figure to debug directory without blocking GUI windows."""
    if fig is None or plt is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_spectra_comparison(
    E_out: u.Quantity,
    flux_1: u.Quantity,
    flux_2: u.Quantity,
    label_1: str,
    label_2: str,
    title: str,
    filename: str,
    plot_results: bool,
    y_unit: str = "eV cm-2 s-1",
) -> None:
    """Helper to plot and save 2-flux energy-squared SED comparison in a non-blocking way."""
    if not plot_results or plt is None:
        return
    fig = plt.figure(figsize=(8, 6))
    E_GeV = E_out.to_value(u.GeV)
    plt.loglog(
        E_GeV,
        (flux_1 * E_out**2).to_value(y_unit),
        label=label_1,
        lw=2,
    )
    plt.loglog(
        E_GeV,
        (flux_2 * E_out**2).to_value(y_unit),
        label=label_2,
        linestyle="--",
        lw=2,
    )
    plt.xlabel("Energy [GeV]")
    plt.ylabel(rf"$E^2 \phi$ [{y_unit}]")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _handle_plot(fig, filename)


def _powerlaw_spectrum(E_cr: u.Quantity, particle: str):
    """dN/dE = 1e5 (E_cr / 1000 GeV)^-2 cm^-3 GeV^-1, and the rest energy of the particle."""
    dn_dE = 1e5 * (E_cr.to_value(u.GeV) / 1000.0) ** -2 * (u.cm**-3 / u.GeV)
    m = (const.m_p if particle == "proton" else const.m_e) * const.c**2
    return dn_dE, m.to(u.GeV)


def _create_powerlaw_state(
    base_args: dict, particle: str = "proton", stage_name: str = "test_naima"
) -> State:
    """Helper to create a homogeneous power-law State, converting dN/dE to psi_p."""
    E_cr = base_args["E_cr_grid"]
    r_grid = base_args["r_grid"]
    dn_dE, m_GeV = _powerlaw_spectrum(E_cr, particle)
    E_tot = E_cr + m_GeV

    p = (np.sqrt(E_tot**2 - m_GeV**2) / const.c).to(su.MOMENTUM)
    dE_dp = (p * const.c**2) / E_tot

    N_p = dn_dE * dE_dp
    f_cr_val = N_p.to_value(su.PSI_P)
    f_cr_2d = np.tile(f_cr_val, (len(r_grid), 1)).T
    particle_spec = Particle.PROTON if particle == "proton" else Particle.ELECTRON
    grid_cr = Grid(
        r_centers=r_grid,
        p_centers=p,
    )
    return State(
        grid=grid_cr,
        psi_p=f_cr_2d * su.PSI_P,
        particle=particle_spec,
        stage_name=stage_name,
    )


def _create_powerlaw_state_and_naima_model(
    base_args: dict, particle: str = "proton", stage_name: str = "test_naima"
):
    """
    Helper to create a homogeneous power-law State and corresponding Naima TableModel.

    Uses the spectrum of :func:`_powerlaw_spectrum`, as a spatial State psi_p and as a
    volume-integrated Naima TableModel.
    """
    from naima.models import TableModel

    state_cr = _create_powerlaw_state(base_args, particle, stage_name)
    r_grid = base_args["r_grid"]
    dn_dE, m_GeV = _powerlaw_spectrum(base_args["E_cr_grid"], particle)
    E_tot = base_args["E_cr_grid"] + m_GeV

    dr = np.gradient(r_grid.to_value(u.cm))
    total_volume_cm3 = np.sum(4 * np.pi * (r_grid.to_value(u.cm) ** 2) * dr)
    dn_dE_total = dn_dE.to_value(u.cm**-3 / u.GeV) * total_volume_cm3

    naima_dist = TableModel(E_tot, dn_dE_total * u.Unit("1/GeV"), amplitude=1)
    return state_cr, naima_dist


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def base_args():
    """Base arguments for the EmissionCalculator."""
    return {
        "E_out_grid": np.logspace(-1, 3, 20) * u.GeV,
        "E_cr_grid": np.logspace(-1, 4, 200) * u.GeV,
        "r_grid": np.linspace(0, 10, 10) * u.pc,
        "n_gas": np.ones(10) * u.cm**-3,
    }


@pytest.fixture
def state_cr(base_args):
    """Dummy cosmic ray state with shape (N_cr, N_r)."""
    # Golden data was generated with psi_p = 1e-10 cm^-3 (GeV/c)^-1: keep that physics.
    f = np.ones((200, 10)) * 1e-10 * u.cm**-3 / su.MOMENTUM
    p_cr = (base_args["E_cr_grid"] / const.c).to(su.MOMENTUM)
    grid = Grid(
        r_centers=base_args["r_grid"],
        p_centers=p_cr,
    )
    return State(grid=grid, psi_p=f, particle=Particle.PROTON, stage_name="test_stage")


@pytest.fixture
def dummy_sigma():
    """Dummy cross-section matrix with shape (N_out, N_cr)."""
    return np.ones((20, 200)) * 1e-27 * u.cm**2 / u.GeV


@pytest.fixture
def dummy_ic_kernel():
    """Dummy IC kernel matrix with shape (N_out, N_cr)."""
    return np.ones((20, 200)) * 1e-15 * u.s**-1 * u.GeV**-1


@pytest.fixture
def golden_data_path():
    """Path to the golden data file on disk."""
    path = Path(__file__).parent / "data" / "golden_emissions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# TEST SUITES
# ==============================================================================


class TestEmissionCalculator:
    """Core functionality and unit tests for EmissionCalculator."""

    def test_initialization(self, base_args):
        calc_p = EmissionCalculator(
            **base_args, particle="proton", distance=1.5 * u.kpc
        )
        assert calc_p.particle == Particle.PROTON
        assert calc_p.distance.unit.is_equivalent(u.kpc)

        # Test valid ndarray inputs without units fail fail-fast boundary
        with pytest.raises(TypeError):
            EmissionCalculator(
                E_out_grid=np.logspace(-1, 3, 20),
                E_cr_grid=np.logspace(-1, 4, 30),
                r_grid=np.linspace(0.1, 10, 10),
                n_gas=np.ones(10),
                particle="electron",
            )

        with pytest.raises(ValueError):
            EmissionCalculator(**base_args, particle="invalid_particle")

    def test_invalid_units(self, base_args):
        wrong_args = base_args.copy()
        wrong_args["E_out_grid"] = np.ones(20) * u.K
        with pytest.raises((u.UnitsError, TypeError, ValueError)):
            EmissionCalculator(**wrong_args, particle="proton")

        with pytest.raises((u.UnitsError, TypeError, ValueError)):
            EmissionCalculator(**base_args, particle="proton", distance=5 * u.kg)

    def test_pion_decay_emission(self, base_args, state_cr, dummy_sigma):
        calc_p = EmissionCalculator(**base_args, particle="proton", distance=1 * u.kpc)
        flux = calc_p.compute_pion_decay_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )
        assert flux.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.GeV**-1)
        assert flux.shape == (20,)

        calc_e = EmissionCalculator(**base_args, particle="electron")
        with pytest.raises(TypeError):
            calc_e.compute_pion_decay_emission(
                state=state_cr, custom_matrix=dummy_sigma
            )

    def test_neutrino_emission(self, base_args, state_cr, dummy_sigma):
        calc_p = EmissionCalculator(**base_args, particle="proton")  # No distance
        spectrum = calc_p.compute_neutrino_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )
        assert spectrum.unit.is_equivalent(u.s**-1 * u.GeV**-1)
        assert spectrum.shape == (20,)

        calc_e = EmissionCalculator(**base_args, particle="electron")
        with pytest.raises(TypeError):
            calc_e.compute_neutrino_emission(state=state_cr, custom_matrix=dummy_sigma)

    def test_bremsstrahlung_emission(self, base_args, state_cr, dummy_sigma):
        calc_e = EmissionCalculator(
            **base_args, particle="electron", distance=2 * u.kpc
        )
        flux = calc_e.compute_bremsstrahlung_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )
        assert flux.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.GeV**-1)

        calc_p = EmissionCalculator(**base_args, particle="proton")
        with pytest.raises(TypeError):
            calc_p.compute_bremsstrahlung_emission(
                state=state_cr, custom_matrix=dummy_sigma
            )

    @pytest.mark.parametrize("ionised", [False, True])
    def test_bremsstrahlung_threshold(self, ionised):
        # The cross-section rises steeply from zero at E_e = E_gamma, never below it
        sigma = AnalyticalBremsstrahlung().compute_matrix(
            np.array([1.0]), np.array([1.0 + 1e-6, 1.01, 2.0]), ionised=ionised
        )
        assert np.all(sigma >= 0.0)

        # so a coarse electron grid must not make the spectrum oscillate with the
        # position of E_gamma between its nodes
        r = np.array([0.0, 1.0]) * u.pc
        E_out = np.logspace(-1.0, 2.0, 61) * u.GeV
        mc2 = (const.m_e * const.c**2).to_value(u.GeV)

        def sed(per_decade, E_out=E_out):
            p = np.logspace(-2.0, 4.0, 6 * per_decade + 1) * su.MOMENTUM
            E_k = (np.sqrt(p.value**2 + mc2**2) - mc2) * u.GeV
            psi = np.outer(p.value**-2.0, np.ones(2)) * su.PSI_P
            calc = EmissionCalculator(
                E_out_grid=E_out,
                E_cr_grid=E_k,
                r_grid=r,
                n_gas=np.ones(2) * u.cm**-3,
                particle="electron",
            )
            state = State(Grid(r_centers=r, p_centers=p), psi_p=psi)
            flux = calc.compute_bremsstrahlung_emission(state, ionised=ionised)
            return (E_out**2 * flux).value

        ratio = sed(10) / sed(400)
        np.testing.assert_allclose(ratio, 1.0, rtol=0.04)
        assert np.max(np.abs(np.diff(ratio, 2))) < 0.03
        # Photons above the highest electron energy are not produced
        assert np.all(sed(10, E_out=[2e4, 5e4] * u.GeV) == 0.0)

    def test_inverse_compton_emission(self, base_args, state_cr, dummy_ic_kernel):
        calc_e = EmissionCalculator(**base_args, particle="electron")
        spectrum = calc_e.compute_inverse_compton_emission(
            state=state_cr, custom_kernel=dummy_ic_kernel
        )
        assert spectrum.unit.is_equivalent(u.s**-1 * u.GeV**-1)

        calc_p = EmissionCalculator(**base_args, particle="proton")
        with pytest.raises(TypeError):
            calc_p.compute_inverse_compton_emission(
                state=state_cr, custom_kernel=dummy_ic_kernel
            )

    def test_total_leptonic_gamma_emission(
        self, base_args, state_cr, dummy_sigma, dummy_ic_kernel
    ):
        calc_e = EmissionCalculator(
            **base_args, particle="electron", distance=1 * u.kpc
        )
        with pytest.raises(RuntimeError):
            calc_e.compute_total_gamma_emission()

        flux_brems = calc_e.compute_bremsstrahlung_emission(
            state_cr, custom_matrix=dummy_sigma
        )
        flux_ic = calc_e.compute_inverse_compton_emission(
            state_cr, custom_kernel=dummy_ic_kernel
        )
        total_flux = calc_e.compute_total_gamma_emission()

        assert total_flux.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.GeV**-1)
        np.testing.assert_allclose(total_flux.value, (flux_brems + flux_ic).value)

    def test_total_hadronic_gamma_emission(
        self, base_args, state_cr, dummy_sigma, plot_results
    ):
        calc_p = EmissionCalculator(**base_args, particle="proton", distance=1 * u.kpc)
        with pytest.raises(RuntimeError):
            calc_p.compute_total_gamma_emission()

        flux_pion = calc_p.compute_pion_decay_emission(
            state_cr, custom_matrix=dummy_sigma
        )
        total_flux = calc_p.compute_total_gamma_emission()

        assert total_flux.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.GeV**-1)
        np.testing.assert_allclose(total_flux.value, flux_pion.value)

        _plot_spectra_comparison(
            E_out=calc_p.E_out_grid,
            flux_1=flux_pion,
            flux_2=total_flux,
            label_1="Pion Decay",
            label_2="Total Hadronic",
            title="Hadronic Emission (Proton Power-law)",
            filename="total_hadronic_gamma_emission.png",
            plot_results=plot_results,
            y_unit="GeV cm-2 s-1",
        )

    def test_model_comparison_kafexhiu_aafrag(self, base_args, plot_results):
        valid_args = base_args.copy()
        valid_args["E_cr_grid"] = np.logspace(1, 4, 30) * u.GeV
        calc_p = EmissionCalculator(**valid_args, particle="proton", distance=1 * u.kpc)
        state_cr_matched = _create_powerlaw_state(
            valid_args, particle="proton", stage_name="test_aafrag_matched"
        )

        try:
            flux_aafrag = calc_p.compute_pion_decay_emission(
                state_cr_matched, model="aafragpy"
            )
        except ImportError:
            pytest.skip("aafragpy not installed, skipping comparison test")

        try:
            flux_kafexhiu = calc_p.compute_pion_decay_emission(
                state_cr_matched, model="kafexhiu"
            )
        except ImportError:
            pytest.skip("Cross_sections_lib not installed, skipping comparison test")

        assert flux_aafrag.shape == flux_kafexhiu.shape
        assert flux_aafrag.unit == flux_kafexhiu.unit

        _plot_spectra_comparison(
            E_out=calc_p.E_out_grid,
            flux_1=flux_aafrag,
            flux_2=flux_kafexhiu,
            label_1="AAFRAG",
            label_2="Kafexhiu",
            title="Debug: Model Comparison (AAFRAG vs Kafexhiu)",
            filename="model_comparison_aafrag_kafexhiu.png",
            plot_results=plot_results,
            y_unit="GeV cm-2 s-1",
        )

    def test_aafrag_uses_total_primary_energy(self):
        aafragpy = pytest.importorskip("aafragpy")
        from saetass.utils.cross_sections import AAFragPyModel

        T_p = np.array([2.0, 3.5, 10.0, 1e3])
        E_gamma = np.array([0.1, 1.0])
        with pytest.warns(UserWarning, match="total energies >= 4 GeV"):
            sigma = AAFragPyModel().compute_matrix(E_gamma, T_p)
        # T_p = 2 GeV is below the tables (E_p = 2.94 GeV < 4 GeV); the others are
        # evaluated at the total energy T_p + m_p c^2
        assert np.all(sigma[:, 0] == 0.0)
        expected = aafragpy.get_cross_section(
            "gam", "p-p", E_primaries=T_p[1:] + 0.938272, E_secondaries=E_gamma
        )[0].T
        np.testing.assert_allclose(sigma[:, 1:], expected, rtol=1e-12)

    def test_model_import_failures(self, base_args, state_cr):
        calc_p = EmissionCalculator(**base_args, particle="proton")

        with pytest.raises(ValueError, match="Cross-section model.*not found"):
            calc_p.compute_pion_decay_emission(state=state_cr, model="fake_model")

        with pytest.raises(ValueError, match="only supports gamma-ray production"):
            calc_p.compute_neutrino_emission(state=state_cr, model="kafexhiu")


class TestEmissionCalculatorEdgeCases:
    """Test suite for edge cases and defensive validation in EmissionCalculator."""

    def test_distance_zero_and_negative(self, base_args):
        with pytest.raises(ValueError, match="strictly positive"):
            EmissionCalculator(**base_args, particle="proton", distance=0 * u.kpc)

        with pytest.raises(ValueError, match="strictly positive"):
            EmissionCalculator(**base_args, particle="proton", distance=-1.5 * u.kpc)

    def test_negative_and_zero_energy_grids(self, base_args):
        bad_args_cr_neg = base_args.copy()
        bad_args_cr_neg["E_cr_grid"] = np.linspace(-5, 100, 200) * u.GeV
        with pytest.raises(ValueError, match="strictly positive"):
            EmissionCalculator(**bad_args_cr_neg, particle="proton")

        bad_args_cr_zero = base_args.copy()
        bad_args_cr_zero["E_cr_grid"] = np.linspace(0, 100, 200) * u.GeV
        with pytest.raises(ValueError, match="strictly positive"):
            EmissionCalculator(**bad_args_cr_zero, particle="proton")

        bad_args_out = base_args.copy()
        bad_args_out["E_out_grid"] = np.linspace(-1, 50, 20) * u.GeV
        with pytest.raises(ValueError, match="strictly positive"):
            EmissionCalculator(**bad_args_out, particle="proton")

    def test_shape_mismatch_n_gas_and_r_grid(self, base_args):
        bad_args = base_args.copy()
        bad_args["n_gas"] = np.ones(15) * u.cm**-3
        with pytest.raises(ValueError, match="Shape mismatch"):
            EmissionCalculator(**bad_args, particle="proton")

    def test_shape_mismatch_state_cr(self, base_args, dummy_sigma, dummy_ic_kernel):
        calc_p = EmissionCalculator(**base_args, particle="proton", distance=1 * u.kpc)
        calc_e = EmissionCalculator(
            **base_args, particle="electron", distance=1 * u.kpc
        )

        grid_bad_cr = Grid(
            r_centers=base_args["r_grid"],
            p_centers=np.logspace(-1, 4, 100) * su.MOMENTUM,
        )
        bad_state_cr = State(
            grid=grid_bad_cr,
            psi_p=np.ones((100, 10)) * su.PSI_P,
            particle=Particle.PROTON,
            stage_name="bad_ncr",
        )
        with pytest.raises(ValueError, match="Shape mismatch in cosmic ray State"):
            calc_p.compute_pion_decay_emission(bad_state_cr, custom_matrix=dummy_sigma)

        grid_bad_r = Grid(
            r_centers=np.linspace(0.1, 10, 5) * su.LENGTH,
            p_centers=(base_args["E_cr_grid"] / const.c).to(su.MOMENTUM),
        )
        bad_state_r = State(
            grid=grid_bad_r,
            psi_p=np.ones((200, 5)) * su.PSI_P,
            particle=Particle.ELECTRON,
            stage_name="bad_nr",
        )
        with pytest.raises(ValueError, match="Shape mismatch in cosmic ray State"):
            calc_e.compute_inverse_compton_emission(
                bad_state_r, custom_kernel=dummy_ic_kernel
            )


class TestEmissionRegression:
    """Strict numerical regression tests using precomputed Golden Data (rtol=1e-5)."""

    def _compute_reference_results(
        self, base_args, state_cr, dummy_sigma, dummy_ic_kernel
    ):
        calc_p = EmissionCalculator(**base_args, particle="proton", distance=1 * u.kpc)
        flux_pion = calc_p.compute_pion_decay_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )
        total_hadronic = calc_p.compute_total_gamma_emission()

        calc_p_nodist = EmissionCalculator(**base_args, particle="proton")
        spec_neutrino = calc_p_nodist.compute_neutrino_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )

        calc_e = EmissionCalculator(
            **base_args, particle="electron", distance=1 * u.kpc
        )
        flux_brems = calc_e.compute_bremsstrahlung_emission(
            state=state_cr, custom_matrix=dummy_sigma
        )
        flux_ic = calc_e.compute_inverse_compton_emission(
            state=state_cr, custom_kernel=dummy_ic_kernel
        )
        total_leptonic = calc_e.compute_total_gamma_emission()

        return {
            "flux_pion": flux_pion.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            "total_hadronic": total_hadronic.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            "spec_neutrino": spec_neutrino.to_value(u.s**-1 * u.GeV**-1),
            "flux_brems": flux_brems.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            "flux_ic": flux_ic.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            "total_leptonic": total_leptonic.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
        }

    def test_golden_data_regression(
        self,
        base_args,
        state_cr,
        dummy_sigma,
        dummy_ic_kernel,
        golden_data_path,
        update_golden,
    ):
        results = self._compute_reference_results(
            base_args, state_cr, dummy_sigma, dummy_ic_kernel
        )

        if update_golden:
            np.savez_compressed(golden_data_path, **results)
            pytest.skip("Golden dataset updated on disk.")
        if not golden_data_path.exists():
            pytest.fail(
                f"Golden dataset {golden_data_path} is missing; "
                "regenerate it deliberately with --update-golden."
            )

        golden = np.load(golden_data_path)
        for key, value in results.items():
            np.testing.assert_allclose(
                value,
                golden[key],
                rtol=1e-5,
                atol=0.0,  # purely relative: an absolute floor hides scale errors
                err_msg=f"Regression mismatch detected for channel '{key}'",
            )


@pytest.mark.skipif(naima is None, reason="naima is required for Naima comparisons")
class TestNaimaComparisons:
    """
    Physical cross-validation against Naima (rtol=0.3).

    Uses broad physical tolerance reflecting differences in cross-section
    parametrizations and integration algorithms.
    """

    def test_naima_comparison_pion_decay(self, base_args, plot_results):
        from naima.models import PionDecay

        state_cr, naima_dist = _create_powerlaw_state_and_naima_model(
            base_args, particle="proton", stage_name="test_naima_pion"
        )
        calc_p = EmissionCalculator(**base_args, particle="proton", distance=1 * u.kpc)
        my_flux = calc_p.compute_pion_decay_emission(state_cr, model="pythia8")

        n_gas_val = base_args["n_gas"][0].to_value(u.cm**-3)
        naima_pion = PionDecay(
            naima_dist, nH=n_gas_val * u.cm**-3, nuclear_enhancement=False
        )
        E_out = base_args["E_out_grid"]
        naima_flux = naima_pion.flux(E_out, distance=1 * u.kpc)

        _plot_spectra_comparison(
            E_out=E_out,
            flux_1=my_flux,
            flux_2=naima_flux,
            label_1="SAETASS",
            label_2="Naima",
            title="Flux Comparison: SAETASS vs Naima (Pion Decay)",
            filename="naima_comparison_pion_decay.png",
            plot_results=plot_results,
        )

        max_flux = np.max(naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1))
        np.testing.assert_allclose(
            my_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            rtol=0.3,
            atol=max_flux * 1e-10,
        )

    def test_naima_comparison_bremsstrahlung(self, base_args, plot_results):
        from naima.models import Bremsstrahlung

        state_cr, naima_dist = _create_powerlaw_state_and_naima_model(
            base_args, particle="electron", stage_name="test_naima_brems"
        )
        valid_args = base_args.copy()
        E_out = np.geomspace(1e-3, 3e6, 40) * u.GeV
        valid_args["E_out_grid"] = E_out

        calc_e = EmissionCalculator(
            **valid_args, particle="electron", distance=1 * u.kpc
        )
        my_flux = calc_e.compute_bremsstrahlung_emission(
            state_cr, model="bremsstrahlung"
        )

        n_gas_val = base_args["n_gas"][0].to_value(u.cm**-3)
        naima_brems = Bremsstrahlung(naima_dist, n0=n_gas_val * u.cm**-3)
        naima_flux = naima_brems.flux(E_out, distance=1 * u.kpc)

        _plot_spectra_comparison(
            E_out=E_out,
            flux_1=my_flux,
            flux_2=naima_flux,
            label_1="SAETASS",
            label_2="Naima",
            title="Flux Comparison: SAETASS vs Naima (Bremsstrahlung)",
            filename="naima_comparison_bremsstrahlung.png",
            plot_results=plot_results,
        )

        max_flux = np.max(naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1))
        np.testing.assert_allclose(
            my_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            rtol=0.3,
            atol=max_flux * 1e-10,
        )

    def test_naima_comparison_synchrotron(self, base_args, plot_results):
        from naima.models import Synchrotron

        state_cr, naima_dist = _create_powerlaw_state_and_naima_model(
            base_args, particle="electron", stage_name="test_naima_synch"
        )
        valid_args = base_args.copy()
        E_out = np.geomspace(1e-12, 1e-3, 40) * u.GeV
        valid_args["E_out_grid"] = E_out

        calc_e = EmissionCalculator(
            **valid_args, particle="electron", distance=1 * u.kpc
        )
        my_flux = calc_e.compute_synchrotron_emission(
            state_cr, B_field=3 * u.uG, pitch_angle="isotropic"
        )

        naima_synch = Synchrotron(naima_dist, B=3 * u.uG)
        naima_flux = naima_synch.flux(E_out, distance=1 * u.kpc)

        _plot_spectra_comparison(
            E_out=E_out,
            flux_1=my_flux,
            flux_2=naima_flux,
            label_1="SAETASS",
            label_2="Naima",
            title="Flux Comparison: SAETASS vs Naima (Synchrotron)",
            filename="naima_comparison_synchrotron.png",
            plot_results=plot_results,
        )

        max_flux = np.max(naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1))
        np.testing.assert_allclose(
            my_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            rtol=0.3,
            atol=max_flux * 1e-10,
        )

    def test_naima_comparison_inverse_compton(self, base_args, plot_results):
        from naima.models import InverseCompton

        state_cr, naima_dist = _create_powerlaw_state_and_naima_model(
            base_args, particle="electron", stage_name="test_naima_ic"
        )
        valid_args = base_args.copy()
        E_out = np.geomspace(1e-3, 3e6, 40) * u.GeV
        valid_args["E_out_grid"] = E_out

        calc_e = EmissionCalculator(
            **valid_args, particle="electron", distance=1 * u.kpc
        )

        eps_grid = np.logspace(-5, -2, 100) * u.eV
        T_cmb = 2.725 * u.K
        kT = (const.k_B * T_cmb).to(u.eV)
        prefactor = (8 * np.pi / (const.h * const.c) ** 3).to(u.cm**-3 * u.eV**-3)
        dn_deps_1d = prefactor * eps_grid**2 / (np.exp(eps_grid / kT) - 1.0)

        my_flux = calc_e.compute_inverse_compton_emission(
            state_cr, model="inverse_compton", eps_grid=eps_grid, dn_deps=dn_deps_1d
        )

        naima_ic = InverseCompton(naima_dist, seed_photon_fields=["CMB"])
        naima_flux = naima_ic.flux(E_out, distance=1 * u.kpc)

        _plot_spectra_comparison(
            E_out=E_out,
            flux_1=my_flux,
            flux_2=naima_flux,
            label_1="SAETASS",
            label_2="Naima",
            title="Flux Comparison: SAETASS vs Naima (Inverse Compton)",
            filename="naima_comparison_inverse_compton.png",
            plot_results=plot_results,
        )

        max_flux = np.max(naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1))
        np.testing.assert_allclose(
            my_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            naima_flux.to_value(u.cm**-2 * u.s**-1 * u.GeV**-1),
            rtol=0.1,
            atol=max_flux * 1e-10,
        )
