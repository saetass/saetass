"""Tests of the Solver orchestrator: its arguments and terminal output."""

import numpy as np
import pytest

from saetass import Grid, Solver, State
from saetass import units as su

R = np.linspace(0.0, 1.0, 11)
P = np.logspace(0, 1, 5)
T = np.linspace(0.0, 1.0, 3)


def _grid(r=R, p=P, t=T):
    return Grid(
        r_centers=r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=t * su.TIME,
    )


def _solver(grid, **kwargs):
    state = State(grid, psi_p=np.ones(grid.shape) * su.PSI_P)
    D = np.ones(grid.shape) * su.DIFFUSION_COEFFICIENT
    return Solver(state, "diffusion", {"diffusion": {"D_values": D}}, **kwargs)


class TestArguments:
    def test_run_after_step_reaches_the_final_time(self):
        solver = _solver(_grid())
        solver.step(1)
        solver.run()
        assert solver.global_step == solver.total_steps
        assert solver.state.t_val == pytest.approx(T[-1])

    def test_grid_is_that_of_the_state(self):
        grid = _grid()
        solver = _solver(grid)
        assert solver.grid is grid
        assert solver.total_steps == T.size - 1

    def test_grid_as_first_argument_is_rejected(self):
        grid = _grid()
        state = State(grid, psi_p=np.ones(grid.shape) * su.PSI_P)
        with pytest.raises(TypeError, match="takes the initial State as first"):
            Solver(grid, state, "diffusion")
