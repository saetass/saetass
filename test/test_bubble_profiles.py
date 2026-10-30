import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import Grid
from saetass import units as su
from saetass.utils.bubble_profiles import BubbleModel, BubbleProfileCalculator
from saetass.utils.energy_losses import Particle

FINE = np.linspace(0.1, 100, 1000) * u.pc
COARSE = np.linspace(0.0, 100.0, 11) * u.pc  # 10 pc cells, wider than the 3.9 pc shell
FRAGMENTED = {
    "shell": "fragmented",
    "shell_width": 0.5 * u.pc,
    "shell_penetration": 2.0 * u.pc,
}


@pytest.fixture
def base_kwargs():
    # R_TS = 13.5 pc and R_b = 78.1 pc
    return {
        "L_wind": 1e38 * u.erg / u.s,
        "M_dot": 1e-5 * u.M_sun / u.yr,
        "rho_0": 1e-24 * u.g / u.cm**3,
        "t_b": 1e6 * u.yr,
    }


@pytest.fixture
def r_grid():
    return FINE


def _kinetic_energy(p, mass=const.m_p):
    mc2 = mass * const.c**2
    return np.sqrt((p * const.c) ** 2 + mc2**2) - mc2


def _overlap(faces, r_in, r_out):
    """Volume of each cell between the radii r_in and r_out."""
    lower, upper = np.clip(faces[:-1], r_in, r_out), np.clip(faces[1:], r_in, r_out)
    return 4 / 3 * np.pi * (upper**3 - lower**3)


class TestBubbleProfileCalculator:
    def test_initialization(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        assert calc.R_TS is not None and calc.R_b is not None and calc.v_w is not None
        assert BubbleModel.WEAVER77 == "Weaver77"
        assert BubbleModel.MORLINO21 == "Morlino21"

    @pytest.mark.parametrize(
        "args, kwargs, error, match",
        [
            ((FINE.value,), {}, TypeError, None),
            ((np.linspace(0.1, 100, 100) * u.K,), {}, u.UnitsError, None),
            ((FINE,), {"model": "InvalidModel"}, ValueError, None),
            ((FINE,), {"particle": "neutrino"}, ValueError, "Unknown particle"),
            ((FINE,), {"particle": 5}, TypeError, "Particle or a str"),
            ((FINE,), {"shell": "thin"}, ValueError, "'uniform' or 'fragmented'"),
            (
                (FINE,),
                {"shell": "fragmented", "shell_width": 0.5 * u.pc},
                ValueError,
                "shell_penetration",
            ),
            (
                (FINE,),
                {"shell": "fragmented", "shell_penetration": 2.0 * u.pc},
                ValueError,
                "shell_width",
            ),
            ((FINE,), {"eta_m": 0.0}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 1.5}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 0.5 * u.one}, ValueError, "eta_m"),
            ((FINE,), {"delta_shell": 1.0}, ValueError, "delta_shell"),
            ((FINE,), {"delta_shell": [0.05]}, ValueError, "delta_shell"),
        ],
    )
    def test_invalid_arguments(self, base_kwargs, args, kwargs, error, match):
        with pytest.raises(error, match=match):
            BubbleProfileCalculator(*args, **kwargs, **base_kwargs)

    def test_missing_kwargs(self, r_grid):
        with pytest.raises(ValueError):
            BubbleProfileCalculator(r_grid, model="Weaver77", L_wind=1e38 * u.erg / u.s)

    @pytest.mark.parametrize(
        "method, kwargs, unit",
        [
            ("compute_density_profile", {}, u.cm**-3),
            ("compute_temperature_profile", {}, u.K),
            ("compute_velocity_profile", {}, u.km / u.s),
            ("compute_magnetic_field_profile", {}, u.G),
            *[
                (
                    "compute_diffusion_profile",
                    {"p_or_E_k": 1 * u.TeV, "diffusion_model": model},
                    u.cm**2 / u.s,
                )
                for model in ["kolmogorov", "kraichnan", "bohm"]
            ],
            (
                "compute_source_term",
                {"p_or_E_k": 1 * u.TeV, "E_max": 100 * u.TeV},
                su.SOURCE_PSI_P,
            ),
        ],
    )
    def test_profiles_on_the_radial_grid(
        self, r_grid, base_kwargs, method, kwargs, unit
    ):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        profile = getattr(calc, method)(**kwargs)
        assert profile.unit.is_equivalent(unit)
        assert profile.shape == r_grid.shape

    def test_not_implemented_errors(self, r_grid, base_kwargs):
        # Only Morlino21 models the magnetic field and the transport of cosmic rays
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        with pytest.raises(NotImplementedError):
            calc.compute_magnetic_field_profile()
        with pytest.raises(NotImplementedError):
            calc.compute_diffusion_profile(1 * u.TeV)
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        with pytest.raises(ValueError):
            calc.compute_diffusion_profile(1 * u.TeV, diffusion_model="invalid")


class TestBubbleStructure:
    """Radii and gas of the bubble in its intermediate stage (Weaver et al. 1977)."""

    def test_radii(self, r_grid, base_kwargs):
        L, rho_0, t = base_kwargs["L_wind"], base_kwargs["rho_0"], base_kwargs["t_b"]
        calc = BubbleProfileCalculator(r_grid, **base_kwargs)
        # R_b = 0.7628 (L t^3 / rho_0)^(1/5)
        expected = 0.762865 * (L * t**3 / rho_0) ** (1 / 5)
        assert calc.R_b.to_value(u.pc) == pytest.approx(expected.to_value(u.pc))
        # Wind density just upstream of the termination shock
        rho_w = base_kwargs["M_dot"] / (4 * np.pi * calc.R_TS**2 * calc.v_w)
        assert calc.rho_w.to_value(u.g / u.cm**3) == pytest.approx(
            rho_w.to_value(u.g / u.cm**3)
        )
        # Energy losses reduce the luminosity of the bubble, not that of the wind
        lossy = BubbleProfileCalculator(r_grid, eta_m=0.032, **base_kwargs)
        factor = 0.032 ** (1 / 5)  # 0.5025
        assert (lossy.R_b / calc.R_b).to_value(u.one) == pytest.approx(factor)
        assert (lossy.R_TS / calc.R_TS).to_value(u.one) == pytest.approx(1 / factor)
        assert lossy.v_w == calc.v_w

    @pytest.mark.parametrize("R_c", [0.0, 5.0] * u.pc)
    def test_magnetic_field(self, r_grid, base_kwargs, R_c):
        calc = BubbleProfileCalculator(r_grid, R_c=R_c, **base_kwargs)
        B = calc.compute_magnetic_field_profile(eta_B=0.1).to_value(u.G)
        # r^2 v_w B^2 = eta_B M_dot v_w^2 / 2 in the wind, in Gaussian units
        flux = np.sqrt(
            0.05
            * base_kwargs["M_dot"].to_value(u.g / u.s)
            * calc.v_w.to_value(u.cm / u.s)
        )
        r = r_grid.to_value(u.cm)
        wind = calc.masks["r_wind"]
        np.testing.assert_allclose(B[wind] * r[wind], flux)
        # constant inside the core, as the density, and finite at r = 0 without it
        core = calc.masks["r_core"]
        if R_c > 0:
            np.testing.assert_allclose(B[core], flux / R_c.to_value(u.cm))
        assert np.all(np.isfinite(B))
        # compressed by sqrt(11) downstream of the shock, and zero outside the bubble
        bubble = calc.masks["r_bubble"]
        np.testing.assert_allclose(
            B[bubble], np.sqrt(11) * flux / calc.R_TS.to_value(u.cm)
        )
        assert np.all(B[calc.masks["r_ISM"]] == 0.0)

    @pytest.mark.parametrize("r_grid", [FINE, COARSE], ids=["fine", "coarse"])
    @pytest.mark.parametrize(
        "shell",
        [
            {"shell": "uniform"},
            {"shell": "uniform", "delta_shell": 0.1, "M_shell": 1e4 * u.M_sun},
            {**FRAGMENTED, "M_shell": 2e4 * u.M_sun},
        ],
        ids=["uniform", "uniform-custom", "fragmented"],
    )
    def test_gas_mass_budget(self, base_kwargs, r_grid, shell):
        calc = BubbleProfileCalculator(r_grid, R_c=5 * u.pc, **shell, **base_kwargs)
        n = calc.compute_density_profile()
        n0 = (base_kwargs["rho_0"] / const.m_p).to(u.cm**-3)

        # Core and cold wind, from the mass loss rate
        M_dot, v_w = base_kwargs["M_dot"], calc.v_w
        n_core = M_dot / (4 * np.pi * (5 * u.pc) ** 2 * v_w * const.m_p)
        assert n[calc.masks["r_core"]].to_value(u.cm**-3) == pytest.approx(
            n_core.to_value(u.cm**-3)
        )

        # Hot bubble, filled by the mass evaporated from the shell: 2e-4 M_sun/yr for
        # L = 1e37 erg/s, n0 = 10 cm^-3 and 1 Myr (Menchiari et al. 2024, Eq. 15)
        M_dot_s = (
            2e-4
            * (base_kwargs["L_wind"] / (1e37 * u.erg / u.s)).to_value(u.one)
            ** (27 / 35)
            * (n0 / (10 / u.cm**3)).to_value(u.one) ** (-2 / 35)
            * u.M_sun
            / u.yr
        )
        V_hot = 4 / 3 * np.pi * (calc.R_cd**3 - calc.R_TS**3)
        n_b = (M_dot_s * base_kwargs["t_b"] / (V_hot * const.m_p)).to(u.cm**-3)
        hot = calc.masks["r_hot"]
        if np.any(hot):
            np.testing.assert_allclose(n[hot].to_value(u.cm**-3), n_b.value)

        # Swept-up shell, exactly the requested mass whatever the grid: the cells it
        # touches hold it on top of their hot-bubble and ambient parts
        g = Grid(r_centers=r_grid)
        shell_cells = calc.masks["r_shell"]
        background = n_b * _overlap(g.r_faces, 0 * u.pc, calc.R_cd) + n0 * _overlap(
            g.r_faces, calc.R_b, g.r_faces[-1]
        )
        mass = np.sum((n * g.volumes - background)[shell_cells]) * const.m_p
        expected = shell.get(
            "M_shell", 4 / 3 * np.pi * calc.R_b**3 * base_kwargs["rho_0"]
        )
        assert mass.to_value(u.M_sun) == pytest.approx(expected.to_value(u.M_sun))

        # Uniform between R_cd = (1 - delta_shell) R_b and R_b
        assert calc.R_cd == (1 - shell.get("delta_shell", 0.05)) * calc.R_b
        full = (g.r_faces[:-1] >= calc.R_cd) & (g.r_faces[1:] <= calc.R_b)
        V_shell = 4 / 3 * np.pi * (calc.R_b**3 - calc.R_cd**3)
        n_uniform = (expected / (V_shell * const.m_p)).to(u.cm**-3)
        if np.any(full):
            np.testing.assert_allclose(n[full].to_value(u.cm**-3), n[full][0].value)
        if shell["shell"] == "uniform":
            if np.any(full):
                np.testing.assert_allclose(n[full].to_value(u.cm**-3), n_uniform.value)
        else:
            # Fragmented: part of the mass penetrates 2 pc into the hot bubble, with a
            # density that decreases inwards from R_cd and peaks in the shell
            within = (g.r_faces[:-1] >= calc.R_cd - 2 * u.pc) & (
                g.r_faces[1:] <= calc.R_cd
            )
            assert np.all(shell_cells[within])
            assert np.all(np.diff(n[within]) >= 0)
            assert np.argmax(n) in np.flatnonzero(
                _overlap(g.r_faces, calc.R_cd, calc.R_b)
            )
            assert n.max() < n_uniform

    def test_shell_does_not_change_the_transport(self, r_grid, base_kwargs):
        uniform = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        fragmented = BubbleProfileCalculator(
            r_grid, model="Morlino21", **FRAGMENTED, **base_kwargs
        )
        p = np.logspace(0, 4, 5) * su.MOMENTUM
        for method, kwargs in [
            ("compute_velocity_profile", {}),
            ("compute_magnetic_field_profile", {}),
            ("compute_diffusion_profile", {"p_or_E_k": p}),
            ("compute_source_term", {"p_or_E_k": p, "E_max": 100 * u.TeV}),
        ]:
            np.testing.assert_array_equal(
                getattr(fragmented, method)(**kwargs),
                getattr(uniform, method)(**kwargs),
            )


class TestCosmicRays:
    """Momentum-dependent profiles of the Morlino21 model."""

    @pytest.mark.parametrize(
        "particle, s, diffusion_model, coefficients",
        [
            ("proton", 4.0, "kolmogorov", (10.0, 0.308653, 22.0241, 0.43112)),
            ("proton", 4.4, "kraichnan", (5.0, 0.448549, 12.52, 0.642666)),
            ("electron", 4.0, "bohm", (8.94, 1.29597, 5.31019, 1.13245)),
        ],
    )
    def test_source_term(self, base_kwargs, particle, s, diffusion_model, coefficients):
        r_grid = np.linspace(0.0, 100.0, 1001) * u.pc
        calc = BubbleProfileCalculator(
            r_grid, model="Morlino21", particle=particle, **base_kwargs
        )
        mass = Particle(particle).mass
        p = np.logspace(-1.0, 8.0, 20001) * su.MOMENTUM
        E_max = 100 * u.TeV
        Q = calc.compute_source_term(
            p, E_max=E_max, eta_inj=0.1, s=s, diffusion_model=diffusion_model
        )

        # Injected in the cell that contains R_TS
        faces = Grid(r_centers=r_grid).r_faces
        (cell,) = np.flatnonzero(Q.value.any(axis=0))
        assert faces[cell] <= calc.R_TS < faces[cell + 1]
        # with f ~ p^-s e^-Gamma, so Q ~ p^(2 - s) e^-Gamma, from p_inj = 1 GeV/c, and the
        # cutoff of Morlino et al. (2021) fitted by Menchiari et al. (2024), Eq. 14 and Table 1
        a1, a2, a3, a4 = coefficients
        p_max = np.sqrt(E_max**2 + 2 * E_max * mass * const.c**2) / const.c
        z = (p / p_max).to_value(u.one)
        spectrum = np.where(
            p >= 1 * su.MOMENTUM,
            p.value ** (2 - s) * (1 + a1 * z**a2) * np.exp(-a3 * z**a4),
            0,
        )
        np.testing.assert_allclose(
            Q[:, cell] / Q[:, cell].max(), spectrum / spectrum.max()
        )
        # and a power eta_inj L_w
        power = np.trapezoid(
            Q[:, cell]
            * Grid(r_centers=r_grid).volumes[cell]
            * _kinetic_energy(p, mass),
            p,
        )
        assert (power / (0.1 * base_kwargs["L_wind"])).to_value(u.one) == pytest.approx(
            1.0, rel=1e-3
        )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"eta_inj": 0.0}, "eta_inj"),
            ({"eta_inj": 1.2}, "eta_inj"),
            ({"p_inj": 0.0 * su.MOMENTUM}, "p_inj"),
            ({"diffusion_model": "invalid"}, "Invalid diffusion model"),
        ],
    )
    def test_invalid_source_term(self, r_grid, base_kwargs, kwargs, match):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        with pytest.raises(ValueError, match=match):
            calc.compute_source_term(1 * u.TeV, E_max=100 * u.TeV, **kwargs)
        inside_wind = np.linspace(0.0, 5.0, 51) * u.pc
        calc = BubbleProfileCalculator(inside_wind, model="Morlino21", **base_kwargs)
        with pytest.raises(ValueError, match="outside the radial grid"):
            calc.compute_source_term(1 * u.TeV, E_max=100 * u.TeV)

    def test_particle_kinematics(self, r_grid, base_kwargs):
        protons = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        electrons = BubbleProfileCalculator(
            r_grid, model="Morlino21", particle="Electron", **base_kwargs
        )
        assert electrons.particle is Particle.ELECTRON
        bubble = protons.masks["r_bubble"]
        p = np.array([0.1, 1e3]) * su.MOMENTUM
        # Momenta and kinetic energies are equivalent inputs
        for calc, mass in [(protons, const.m_p), (electrons, const.m_e)]:
            np.testing.assert_allclose(
                calc.compute_diffusion_profile(p),
                calc.compute_diffusion_profile(_kinetic_energy(p, mass)),
                rtol=1e-12,
            )
        # At equal rigidity, D ~ v r_L^(1/3) only differs through the particle velocity
        ratio = (
            electrons.compute_diffusion_profile(p)[:, bubble]
            / protons.compute_diffusion_profile(p)[:, bubble]
        ).to_value(u.one)

        def beta(mass):
            return (
                p * const.c / (_kinetic_energy(p, mass) + mass * const.c**2)
            ).to_value(u.one)

        expected = beta(const.m_e) / beta(const.m_p)  # 3.3 at 0.1 GeV/c, ~1 at 1 TeV/c
        np.testing.assert_allclose(ratio, expected[:, None] * np.ones_like(ratio))

    def test_multi_energy_profiles(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        E_k = np.logspace(0, 4, 5) * u.GeV
        with pytest.raises(ValueError, match="E_max"):
            calc.get_all_profiles(E_k)
        profiles = calc.get_all_profiles(E_k, E_max=100 * u.TeV, s=4.2)
        single = calc.get_all_profiles(E_k[2], E_max=100 * u.TeV, s=4.2)
        # Spatial profiles are 1D; momentum-dependent ones have one row per momentum
        for key in ["n_gas", "T_gas", "v_field", "B_field"]:
            assert profiles[key].shape == r_grid.shape
        for key in ["D_values", "Q"]:
            assert profiles[key].shape == (E_k.size, r_grid.size)
            np.testing.assert_allclose(profiles[key][2], single[key])
        CR = calc.compute_analytical_CR_profile(profiles["D_values"])
        np.testing.assert_allclose(
            CR[2], calc.compute_analytical_CR_profile(single["D_values"])
        )

    @pytest.mark.parametrize(
        "r_grid",
        [
            np.linspace(0.0, 100.0, 6) * u.pc,  # a single wind cell, r = 0
            np.linspace(0.0, 10.0, 21) * u.pc,  # only wind cells
            np.sort(np.append(np.linspace(0.0, 100.0, 101), 78.0835803851831)) * u.pc,
        ],
        ids=["coarse", "wind-only", "point-at-R_b"],
    )
    def test_analytical_profile_on_limiting_grids(self, base_kwargs, r_grid):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        assert not np.any(calc.masks["r_bubble"] & calc.masks["r_ISM"])
        CR = calc.compute_analytical_CR_profile(
            calc.compute_diffusion_profile(1 * u.TeV)
        )
        assert CR.shape == r_grid.shape
