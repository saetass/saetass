import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import Grid, Particle, State
from saetass import units as su


def test_grid_strict_quantity_input():
    # Passing raw floats to uniform constructor must fail with TypeError
    with pytest.raises(TypeError):
        Grid.uniform(r_min=0.0, r_max=100.0, num_r_cells=10)

    # Passing compatible units (e.g. kpc, yr) must succeed and convert to canonical pc, Myr
    grid = Grid.uniform(
        r_min=0.0 * u.kpc,
        r_max=1.0 * u.kpc,
        num_r_cells=100,
        t_min=0.0 * u.yr,
        t_max=1e6 * u.yr,
        num_timesteps=10,
    )
    assert grid.r_faces[0].unit == su.LENGTH
    assert np.isclose(grid.r_centers[-1].to_value(u.pc), 1000.0)
    assert grid.t_grid.unit == su.TIME
    assert np.isclose(grid.t_grid[-1].to_value(u.Myr), 1.0)


def test_state_strict_quantity_input():
    grid = Grid.uniform(
        r_min=0.0 * u.pc,
        r_max=100.0 * u.pc,
        num_r_cells=50,
    )

    # Raw numpy array without units must fail with TypeError
    raw_arr = np.ones(50)
    with pytest.raises(TypeError):
        State(grid=grid, psi_p=raw_arr)

    # Incompatible unit (e.g. meters instead of differential density) must fail with UnitsError
    with pytest.raises((u.UnitsError, TypeError)):
        State(grid=grid, psi_p=raw_arr * u.m)

    # Valid Astropy Quantity must succeed
    valid_qty = raw_arr * 1e-10 * (u.cm**-3 / (u.GeV / const.c))
    state = State(grid=grid, psi_p=valid_qty, particle=Particle.PROTON)

    # Internal values are stored in canonical units (pc^-3 (GeV/c)^-1)
    expected_canon = valid_qty.to_value(su.PSI_P)
    np.testing.assert_allclose(state._values[0], expected_canon)

    # User property psi_p returns Quantity in canonical units
    assert isinstance(state.psi_p, u.Quantity)
    assert state.psi_p.unit == su.PSI_P


def test_subsolvers_reject_quantities():
    from saetass.solvers.advection_solver import AdvectionSolver
    from saetass.solvers.diffusion_solver import DiffusionSolver
    from saetass.solvers.loss_solver import LossSolver
    from saetass.solvers.source_solver import SourceSolver
    from saetass.splitting import StrangSplitting

    grid = Grid.uniform(
        r_min=0.0 * u.pc,
        r_max=100.0 * u.pc,
        num_r_cells=10,
        t_min=0.0 * u.Myr,
        t_max=1.0 * u.Myr,
        num_timesteps=5,
    )
    t_grid_nude = np.linspace(0.0, 1.0, 6)

    # 1. Splitting scheme rejects Quantity in t_grid
    scheme = StrangSplitting()
    with pytest.raises(TypeError, match="t_grid must be a bare number or array"):
        scheme._store_t_grid(grid.t_grid)

    # 2. AdvectionSolver rejects Quantity in params or t_grid
    with pytest.raises(TypeError, match="t_grid must be a bare number or array"):
        AdvectionSolver(
            grid,
            grid.t_grid,
            params={
                "v_centers": np.ones(10),
                "limiter": "minmod",
                "cfl": 0.5,
                "order": 1,
                "inflow_value_U": 0.0,
            },
        )

    with pytest.raises(
        TypeError, match=r"parameter '\w+' must be a bare number or array"
    ):
        AdvectionSolver(
            grid,
            t_grid_nude,
            params={
                "v_centers": np.ones(10) * (u.km / u.s),
                "limiter": "minmod",
                "cfl": 0.5,
                "order": 1,
                "inflow_value_U": 0.0,
            },
        )

    # 3. DiffusionSolver rejects Quantity in params
    with pytest.raises(
        TypeError, match=r"parameter '\w+' must be a bare number or array"
    ):
        DiffusionSolver(
            grid,
            t_grid_nude,
            params={"D_values": np.ones(10) * (u.cm**2 / u.s)},
        )

    # 4. SourceSolver rejects Quantity in params
    with pytest.raises(
        TypeError, match=r"parameter '\w+' must be a bare number or array"
    ):
        SourceSolver(
            grid,
            t_grid_nude,
            params={"source": np.ones(10) * su.SOURCE_PSI_P},
        )

    # 5. LossSolver rejects Quantity in params
    grid_p = Grid(
        p_centers=np.logspace(0, 2, 10) * su.MOMENTUM,
        t_grid=grid.t_grid,
    )
    with pytest.raises(
        TypeError, match=r"parameter '\w+' must be a bare number or array"
    ):
        LossSolver(
            grid_p,
            t_grid_nude,
            params={
                "P_dot": np.ones(10) * su.MOMENTUM_LOSS_RATE,
                "limiter": "minmod",
                "cfl": 0.5,
                "order": 1,
                "inflow_value_psi": 0.0,
            },
        )


class TestSolverParameterConversion:
    """The Solver boundary converts declared parameters and rejects everything else."""

    @staticmethod
    def _grid_2d():
        return Grid(
            r_centers=np.linspace(0.0, 10.0, 5) * u.pc,
            p_centers=np.logspace(0, 2, 11) * su.MOMENTUM,
            t_grid=np.linspace(0.0, 1.0, 3) * u.Myr,
        )

    def test_quantities_are_converted_to_canonical_floats(self):
        from saetass.solvers.advection_solver import AdvectionSolver

        grid = self._grid_2d()
        out = AdvectionSolver.convert_params(
            {"v_centers": np.ones((11, 5)) * u.km / u.s, "cfl": 0.5}, grid
        )
        expected = (1.0 * u.km / u.s).to_value(su.VELOCITY)
        np.testing.assert_allclose(out["v_centers"], expected)
        assert not isinstance(out["v_centers"], u.Quantity)
        assert out["cfl"] == 0.5

    def test_unknown_parameter_rejected(self):
        from saetass.solvers.diffusion_solver import DiffusionSolver

        with pytest.raises(
            ValueError, match=r"Unknown DiffusionSolver parameter\(s\) \['f_end'\]"
        ):
            DiffusionSolver.convert_params({"f_end": 0.0 * su.PSI_P}, self._grid_2d())

    def test_physical_parameter_without_units_rejected(self):
        from saetass.solvers.loss_solver import LossSolver

        with pytest.raises(
            TypeError, match="'v_centers_physical' has no 'unit' attribute"
        ):
            LossSolver.convert_params(
                {"v_centers_physical": np.ones(5)}, self._grid_2d()
            )

    def test_incompatible_units_rejected(self):
        from saetass.solvers.diffusion_solver import DiffusionSolver

        with pytest.raises(
            u.UnitsError, match="'psi_end' must be in units convertible to"
        ):
            DiffusionSolver.convert_params({"psi_end": 1.0 * su.F_PS}, self._grid_2d())

    def test_non_physical_parameter_with_units_rejected(self):
        from saetass.solvers.advection_solver import AdvectionSolver

        with pytest.raises(TypeError, match="'cfl' must be a bare number or array"):
            AdvectionSolver.convert_params({"cfl": 0.5 * u.s}, self._grid_2d())

    def test_callable_rejected_for_static_parameter(self):
        from saetass.solvers.advection_solver import AdvectionSolver

        with pytest.raises(
            TypeError, match="'inflow_value_U' does not accept a callable"
        ):
            AdvectionSolver.convert_params(
                {"inflow_value_U": lambda t: 0.0 * su.AREA * su.PSI_P}, self._grid_2d()
            )

    def test_callable_must_return_quantity(self):
        from saetass.solvers.diffusion_solver import DiffusionSolver

        out = DiffusionSolver.convert_params(
            {"D_values": lambda t: np.ones((11, 5))}, self._grid_2d()
        )
        with pytest.raises(
            TypeError, match=r"'D_values' \(callable\) has no 'unit' attribute"
        ):
            out["D_values"](0.0)

    def test_source_callable_receives_physical_coordinates(self):
        from saetass.solvers.source_solver import SourceSolver

        grid = self._grid_2d()
        seen = {}

        def source(r, p, t):
            seen.update(r=r, p=p, t=t)
            return np.zeros((11, 5)) * su.SOURCE_PSI_P

        numeric = SourceSolver.convert_params({"source": source}, grid)["source"]
        numeric(0.5)

        # Physical momentum (not log10(p)) in GeV/c, radius and time as Quantities.
        np.testing.assert_allclose(
            seen["p"].to_value(u.GeV / const.c), np.logspace(0, 2, 11)
        )
        np.testing.assert_allclose(seen["r"].to_value(u.pc), np.linspace(0.0, 10.0, 5))
        assert seen["t"] == 0.5 * u.Myr


class TestUtilityInputs:
    """Physical inputs of the utility calculators are validated consistently."""

    bubble_kwargs = {
        "L_wind": 1e38 * u.erg / u.s,
        "M_dot": 1e-5 * u.M_sun / u.yr,
        "rho_0": 1e-24 * u.g / u.cm**3,
        "t_b": 1e6 * u.yr,
    }

    def test_bubble_model_kwargs_validated(self):
        from saetass.utils.bubble_profiles import BubbleProfileCalculator

        r_grid = np.linspace(0.1, 100, 50) * u.pc
        bare = {**self.bubble_kwargs, "L_wind": 1e38}
        with pytest.raises(TypeError, match="'L_wind' has no 'unit' attribute"):
            BubbleProfileCalculator(r_grid, **bare)
        wrong = {**self.bubble_kwargs, "t_b": 1.0 * u.pc}
        with pytest.raises(u.UnitsError, match="'t_b' must be in units convertible"):
            BubbleProfileCalculator(r_grid, **wrong)

    def test_bubble_methods_validated(self):
        from saetass.utils.bubble_profiles import BubbleProfileCalculator

        calc = BubbleProfileCalculator(
            np.linspace(0.1, 100, 50) * u.pc, **self.bubble_kwargs
        )
        with pytest.raises(TypeError):
            calc.compute_diffusion_profile(p_or_E_k=10.0)
        with pytest.raises(u.UnitsError):
            calc.compute_temperature_profile(T_w=200 * u.m)

    def test_energy_loss_methods_validated(self):
        from saetass.utils.energy_losses import EnergyLossCalculator

        calc = EnergyLossCalculator(
            E_grid=np.logspace(-1, 3, 10) * u.GeV,
            r_grid=np.linspace(0.1, 10, 5) * u.pc,
            n_gas=np.ones(5) * u.cm**-3,
            particle="electron",
        )
        with pytest.raises(TypeError):
            calc.compute_sychrotron_losses(B_field=np.full(5, 10.0))
        with pytest.raises(u.UnitsError):
            calc.compute_inverse_compton_losses(
                eps_grid=np.logspace(-4, 1, 20) * u.eV,
                dn_deps=np.ones((20, 5)) * u.cm**-3,
            )

    def test_cross_section_kwargs_validated(self):
        from saetass.utils.cross_sections import AnalyticalSynchrotron

        E = np.logspace(-1, 2, 4)
        with pytest.raises(TypeError, match="'B_field' has no 'unit' attribute"):
            AnalyticalSynchrotron().compute_matrix(E, E, B_field=10.0)
