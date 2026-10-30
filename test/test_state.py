import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import units as su
from saetass.grid import Grid
from saetass.state import (
    CANONICAL_F_PS_UNIT,
    CANONICAL_PSI_E_UNIT,
    CANONICAL_PSI_P_UNIT,
    State,
)
from saetass.utils.energy_losses import Particle


class TestState:
    def test_mandatory_grid(self):
        # Missing grid parameter -> TypeError
        with pytest.raises(TypeError):
            State(psi_p=np.ones(10) * su.PSI_P)

        # grid=None -> TypeError
        with pytest.raises(TypeError, match="requires a valid Grid"):
            State(grid=None, psi_p=np.ones(10) * su.PSI_P)

        # Invalid grid type -> TypeError
        with pytest.raises(TypeError, match="requires a valid Grid"):
            State(grid="invalid_grid", psi_p=np.ones(10) * su.PSI_P)

    def test_mutual_exclusivity(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )

        # 0 parameters provided -> ValueError
        with pytest.raises(ValueError, match="exactly one representation"):
            State(grid=grid)

        # 2 parameters provided -> ValueError
        with pytest.raises(ValueError, match="mutually exclusive"):
            State(
                grid=grid,
                psi_p=np.ones((10, 20)) * su.PSI_P,
                f_ps=np.ones((10, 20)) * su.F_PS,
            )

    def test_no_values_parameter(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )

        # Passing legacy 'values' keyword argument must raise TypeError
        with pytest.raises(TypeError):
            State(grid=grid, values=np.ones((10, 20)))

    def test_named_parameters_unitless_and_quantities(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM
        grid = Grid(r_centers=r_grid, p_centers=p_grid)

        # 1. psi_p as raw array must fail with TypeError
        with pytest.raises(TypeError):
            State(grid=grid, psi_p=np.full((10, 20), 2.5))

        # 2. psi_p as Quantity succeeds
        s2 = State(grid=grid, psi_p=np.full((10, 20), 2.5) * CANONICAL_PSI_P_UNIT)
        np.testing.assert_allclose(s2.psi_p.to_value(CANONICAL_PSI_P_UNIT), 2.5)
        assert isinstance(s2._values, np.ndarray)

        # 3. f_ps as Quantity (converts to psi_p = 4*pi*p^2 * f_ps)
        s4 = State(grid=grid, f_ps=np.ones((10, 20)) * CANONICAL_F_PS_UNIT)
        expected_psi = np.tile(
            (4.0 * np.pi * (p_grid.to_value(su.MOMENTUM) ** 2))[:, np.newaxis], (1, 20)
        )
        np.testing.assert_allclose(
            s4.psi_p.to_value(CANONICAL_PSI_P_UNIT), expected_psi, rtol=1e-12
        )
        np.testing.assert_allclose(
            s4.f_ps.to_value(CANONICAL_F_PS_UNIT), 1.0, rtol=1e-12
        )

        # 4. psi_E as Quantity
        s6 = State(
            grid=grid,
            psi_E=np.ones((10, 20)) * CANONICAL_PSI_E_UNIT,
            particle=Particle.PROTON,
        )
        np.testing.assert_allclose(
            s6.psi_E.to_value(CANONICAL_PSI_E_UNIT), 1.0, rtol=1e-12
        )

        # Incompatible units
        with pytest.raises((u.UnitsError, TypeError)):
            State(grid=grid, psi_p=np.ones((10, 20)) * u.m)
        with pytest.raises((u.UnitsError, TypeError)):
            State(grid=grid, f_ps=np.ones((10, 20)) * u.m)
        with pytest.raises((u.UnitsError, TypeError)):
            State(grid=grid, psi_E=np.ones((10, 20)) * u.m)

    def test_grid_compatibility_checks(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM

        # 1. 2D grid
        grid_2d = Grid(r_centers=r_grid, p_centers=p_grid)
        # Valid 2D shape (10, 20)
        s_2d = State(grid=grid_2d, psi_p=np.ones((10, 20)) * su.PSI_P)
        assert s_2d.grid_shape == (10, 20)
        # Incompatible 2D shape
        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            State(grid=grid_2d, psi_p=np.ones((10, 15)) * su.PSI_P)
        # 1D array with 2D grid
        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            State(grid=grid_2d, psi_p=np.ones(20) * su.PSI_P)

        # 2. 1D spatial grid
        grid_r = Grid(r_centers=r_grid)
        # Valid 1D array (20,)
        s_r = State(grid=grid_r, psi_p=np.ones(20) * su.PSI_P)
        assert s_r.ndim == 1
        # Valid (1, 20) 2D array
        s_r_2d = State(grid=grid_r, psi_p=np.ones((1, 20)) * su.PSI_P)
        assert s_r_2d.n_r == 20
        # Incompatible size
        with pytest.raises(ValueError, match="incompatible with spatial Grid"):
            State(grid=grid_r, psi_p=np.ones(15) * su.PSI_P)
        with pytest.raises(ValueError, match="incompatible with 1D spatial Grid"):
            State(grid=grid_r, psi_p=np.ones((2, 20)) * su.PSI_P)

        # 3. 1D momentum grid
        grid_p = Grid(p_centers=p_grid)
        # Valid 1D array (10,)
        s_p = State(grid=grid_p, psi_p=np.ones(10) * su.PSI_P)
        assert s_p.ndim == 1
        # Conversion on 1D momentum grid
        f_ps_1d = s_p.to_f_ps()
        assert f_ps_1d.shape == (10,)
        psi_E_1d = s_p.to_psi_E()
        assert psi_E_1d.shape == (10,)
        # Incompatible size
        with pytest.raises(ValueError, match="incompatible with momentum Grid"):
            State(grid=grid_p, psi_p=np.ones(15) * su.PSI_P)
        with pytest.raises(ValueError, match="incompatible with 1D momentum Grid"):
            State(grid=grid_p, psi_p=np.ones((2, 10)) * su.PSI_P)

    def test_initialization_metadata(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        vals = np.ones((10, 20)) * su.PSI_P
        state = State(grid=grid, psi_p=vals)
        assert state.n_p == 10
        assert state.n_r == 20
        assert state.grid_shape == (10, 20)
        assert state.t.to_value(su.TIME) == 0.0
        assert state.particle == Particle.PROTON

        # 1D initialization
        grid_1d = Grid(r_centers=np.linspace(0, 10, 20) * su.LENGTH)
        vals_1d = np.ones((20,)) * su.PSI_P
        state_1d = State(grid=grid_1d, psi_p=vals_1d)
        assert state_1d.grid_shape == (1, 20)
        assert state_1d.ndim == 1

        # Particle string initialization
        state_elec = State(grid=grid, psi_p=vals, particle="leptonic")
        assert state_elec.particle == Particle.ELECTRON

    def test_clone(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        vals = np.ones((10, 20)) * su.PSI_P
        state = State(grid=grid, psi_p=vals, t=1.0 * su.TIME)
        state.record_substep("first")

        # Clone without history
        clone1 = state.clone(copy_history=False)
        assert clone1.t.to_value(su.TIME) == 1.0
        assert len(clone1.history) == 0
        assert clone1.grid is grid

        # Clone with history
        clone2 = state.clone(copy_history=True)
        assert len(clone2.history) == 1
        assert "values" in clone2.history[0]
        assert clone2.grid is grid

    def test_internal_numerical_methods(self):
        grid_1d = Grid(r_centers=np.linspace(0, 10, 20) * su.LENGTH)
        vals_1d = np.ones((20,)) * su.PSI_P
        state = State(grid=grid_1d, psi_p=vals_1d)
        assert state._get_values().shape == (20,)  # Returns natural dimensionality (1D)

        # update with 1D
        new_vals = np.zeros((20,))
        state._update_values(new_vals)
        assert np.all(state._get_values() == 0)
        assert isinstance(state._values, np.ndarray)

        # invalid shape update
        with pytest.raises(ValueError):
            state._update_values(np.zeros((10,)))

        # 2D update
        grid_2d = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        state2 = State(grid=grid_2d, psi_p=np.ones((10, 20)) * su.PSI_P)
        state2._update_values(np.zeros((10, 20)))
        assert np.all(state2._get_values() == 0)

    def test_property_setters(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM
        grid = Grid(r_centers=r_grid, p_centers=p_grid)

        state = State(
            grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P, particle=Particle.PROTON
        )

        # 1. Set via state.psi_p
        state.psi_p = np.full((10, 20), 3.0) * su.PSI_P
        np.testing.assert_allclose(state.psi_p.to_value(su.PSI_P), 3.0)

        # Incompatible grid shape -> ValueError
        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.psi_p = np.ones((5, 10)) * su.PSI_P

        # 2. Set via state.f_ps
        state.f_ps = np.full((10, 20), 1.0) * su.F_PS
        np.testing.assert_allclose(state.f_ps.to_value(su.F_PS), 1.0, rtol=1e-12)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.f_ps = np.ones(20) * su.F_PS

        # 3. Set via state.psi_E / state.dndE
        state.psi_E = np.full((10, 20), 4.0) * su.PSI_E
        np.testing.assert_allclose(state.psi_E.to_value(su.PSI_E), 4.0, rtol=1e-12)
        state.dndE = np.full((10, 20), 5.0) * su.PSI_E
        np.testing.assert_allclose(state.psi_E.to_value(su.PSI_E), 5.0, rtol=1e-12)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.psi_E = np.ones((10, 15)) * su.PSI_E

    def test_explicit_update_methods(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM
        grid = Grid(r_centers=r_grid, p_centers=p_grid)

        state = State(
            grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P, particle=Particle.PROTON
        )

        # update_psi_p
        state.update_psi_p(
            np.full((10, 20), 2.0) * su.PSI_P,
            dt=0.1 * su.TIME,
            stage=1,
            stage_name="step1",
        )
        assert state.dt.to_value(su.TIME) == 0.1
        assert state.stage == 1
        assert state.stage_name == "step1"
        np.testing.assert_allclose(state.psi_p.to_value(su.PSI_P), 2.0)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.update_psi_p(np.ones(10) * su.PSI_P)

        # update_f_ps
        state.update_f_ps(np.full((10, 20), 1.0) * su.F_PS)
        np.testing.assert_allclose(state.f_ps.to_value(su.F_PS), 1.0, rtol=1e-12)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.update_f_ps(np.ones((10, 15)) * su.F_PS)

        # update_psi_E
        state.update_psi_E(np.full((10, 20), 3.0) * su.PSI_E)
        np.testing.assert_allclose(state.psi_E.to_value(su.PSI_E), 3.0, rtol=1e-12)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.update_psi_E(np.ones(20) * su.PSI_E)

        # generic update
        state.update(psi_p=np.full((10, 20), 5.0) * su.PSI_P)
        np.testing.assert_allclose(state.psi_p.to_value(su.PSI_P), 5.0)

        with pytest.raises(ValueError, match="incompatible with 2D Grid"):
            state.update(f_ps=np.ones((5, 5)) * su.F_PS)

        with pytest.raises(ValueError):
            state.update(
                psi_p=np.ones((10, 20)) * su.PSI_P, f_ps=np.ones((10, 20)) * su.F_PS
            )

    def test_transformations_and_properties(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM  # in GeV/c
        grid = Grid(r_centers=r_grid, p_centers=p_grid)

        # Create state with grid and particle
        psi_vals = np.ones((10, 20))
        state = State(grid=grid, psi_p=psi_vals * su.PSI_P, particle=Particle.PROTON)

        # 1. Phase-space conversion: f_ps = psi / (4*pi*p^2)
        f_ps = state.to_phase_space()
        p_vals = p_grid.to_value(su.MOMENTUM)
        expected_f_ps = psi_vals / (4.0 * np.pi * (p_vals**2)[:, np.newaxis])
        np.testing.assert_allclose(f_ps.to_value(su.F_PS), expected_f_ps, rtol=1e-12)
        np.testing.assert_allclose(
            state.f_ps.to_value(su.F_PS), expected_f_ps, rtol=1e-12
        )

        # With unit request
        f_ps_quant = state.to_f_ps(unit=u.m**-3 / (u.GeV / const.c) ** 3)
        assert isinstance(f_ps_quant, u.Quantity)
        np.testing.assert_allclose(
            f_ps_quant.to_value(CANONICAL_F_PS_UNIT), expected_f_ps, rtol=1e-12
        )

        # 2. Energy-space differential density: dn/dE = psi * dp/dE
        dp_dE = state.dp_dE
        expected_dndE = psi_vals * dp_dE[:, np.newaxis]

        dndE = state.to_dndE()
        np.testing.assert_allclose(dndE.to_value(su.PSI_E), expected_dndE, rtol=1e-12)
        np.testing.assert_allclose(
            state.psi_E.to_value(su.PSI_E), expected_dndE, rtol=1e-12
        )
        np.testing.assert_allclose(
            state.dndE.to_value(su.PSI_E), expected_dndE, rtol=1e-12
        )

        # 3. to_psi_p with unit
        psi_p_quant = state.to_psi_p(unit=u.m**-3 / (u.GeV / const.c))
        assert isinstance(psi_p_quant, u.Quantity)
        np.testing.assert_allclose(
            psi_p_quant.to_value(CANONICAL_PSI_P_UNIT), psi_vals, rtol=1e-12
        )

        # Test with spatial-only grid raises ValueError for momentum conversions
        grid_spatial_only = Grid(r_centers=r_grid)
        state_spatial_only = State(grid=grid_spatial_only, psi_p=np.ones(20) * su.PSI_P)
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            state_spatial_only.to_phase_space()
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            state_spatial_only.to_dndE()

    def test_kinematics(self):
        r_grid = np.linspace(0, 10, 20) * su.LENGTH
        p_grid = np.logspace(0, 3, 10) * su.MOMENTUM  # in GeV/c
        grid = Grid(r_centers=r_grid, p_centers=p_grid)

        # Protons
        state_p = State(
            grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P, particle=Particle.PROTON
        )
        m_p_gev = (const.m_p * const.c**2).to_value(u.GeV)
        p_vals_gev_c = (p_grid * const.c).to_value(u.GeV)
        exp_E_tot = np.sqrt(p_vals_gev_c**2 + m_p_gev**2)
        exp_E = exp_E_tot - m_p_gev
        exp_gamma = exp_E_tot / m_p_gev
        exp_beta = p_vals_gev_c / exp_E_tot
        exp_v = exp_beta * const.c.to_value(u.pc / u.Myr)

        np.testing.assert_allclose(state_p.E_tot.to_value(u.GeV), exp_E_tot)
        np.testing.assert_allclose(state_p.E.to_value(u.GeV), exp_E)
        np.testing.assert_allclose(state_p.gamma, exp_gamma)
        np.testing.assert_allclose(state_p.beta, exp_beta)
        np.testing.assert_allclose(state_p.v.to_value(u.pc / u.Myr), exp_v)

        # Spatial-only grid raises ValueError
        grid_spatial_only = Grid(r_centers=r_grid)
        state_spatial = State(grid=grid_spatial_only, psi_p=np.ones(20) * su.PSI_P)
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.E_tot
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.E
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.gamma
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.beta
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.v
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.dp_dE
        with pytest.raises(
            ValueError, match="requires a Grid with momentum coordinates"
        ):
            _ = state_spatial.dE_dp

    def test_time_and_stage(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        state = State(grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P)
        state.set_time(5.0 * su.TIME)
        assert state.t.to_value(su.TIME) == 5.0
        assert state.dt.to_value(su.TIME) == 5.0

        state.step_stage("split1")
        assert state.stage == 1
        assert state.stage_name == "split1"

    def test_time_single_source_of_truth(self):
        grid = Grid(r_centers=np.linspace(0, 10, 20) * su.LENGTH)
        state = State(grid=grid, psi_p=np.ones(20) * su.PSI_P)

        # Assigning the public Quantity keeps the float view used by solvers in sync.
        state.t = 2.0 * u.kyr
        assert state.t_val == pytest.approx(2e-3)
        assert state.t.unit == su.TIME

        # Float (solver) path updates the Quantity view and dt.
        state.set_time(0.5)
        assert state.t == 0.5 * su.TIME
        assert state.dt.to_value(su.TIME) == pytest.approx(0.498)

        with pytest.raises(TypeError):
            state.t = 1.0
        with pytest.raises(u.UnitsError):
            state.dt = 1.0 * u.pc

    def test_history_management(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        state = State(grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P, t=0.0 * su.TIME)
        state.record_substep("init")

        state.set_time(1.0 * su.TIME)
        state.update_psi_p(np.zeros((10, 20)) * su.PSI_P)
        state.record_substep("step1")

        assert len(state.history) == 2

        sub = state.get_substep(0)
        assert sub["t"].to_value(su.TIME) == 0.0
        assert np.all(sub["values"] == 1)

        with pytest.raises(IndexError):
            state.get_substep(5)

        # Restore by name
        state.restore_substep("init")
        assert state.t.to_value(su.TIME) == 0.0
        assert np.all(state.psi_p.to_value(su.PSI_P) == 1)

        # Restore by index
        state.restore_substep(1)
        assert state.t.to_value(su.TIME) == 1.0
        assert np.all(state.psi_p.to_value(su.PSI_P) == 0)

        with pytest.raises(ValueError):
            state.restore_substep("invalid")

        with pytest.raises(IndexError):
            state.restore_substep(5)

        state.clear_history()
        assert len(state.history) == 0

    def test_str_repr(self):
        grid = Grid(
            r_centers=np.linspace(0, 10, 20) * su.LENGTH,
            p_centers=np.logspace(0, 3, 10) * su.MOMENTUM,
        )
        state = State(
            grid=grid, psi_p=np.ones((10, 20)) * su.PSI_P, particle=Particle.PROTON
        )
        assert "State" in repr(state)
        assert "hadronic" in repr(state)


class TestStateInputValidation:
    @staticmethod
    def _grid_r():
        return Grid(r_centers=np.linspace(0, 10, 5) * su.LENGTH)

    def test_particle_parsing(self):
        grid = self._grid_r()
        psi = np.ones(5) * su.PSI_P
        assert State(grid, psi_p=psi, particle="leptonic").particle == Particle.ELECTRON
        assert State(grid, psi_p=psi, particle="Proton").particle == Particle.PROTON
        with pytest.raises(ValueError):
            State(grid, psi_p=psi, particle="muon")
        with pytest.raises(TypeError, match="particle must be a Particle or a str"):
            State(grid, psi_p=psi, particle=None)

    def test_rejects_arrays_with_more_than_two_dimensions(self):
        with pytest.raises(NotImplementedError, match="must be 1D or 2D"):
            State(self._grid_r(), psi_p=np.ones((2, 2, 5)) * su.PSI_P)

    def test_f_ps_requires_momentum_grid(self):
        with pytest.raises(ValueError, match="requires a Grid with momentum"):
            State(self._grid_r(), f_ps=np.ones(5) * su.F_PS)
