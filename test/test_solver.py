"""Tests of the Solver orchestrator: its arguments and terminal output."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from saetass import Grid, Solver, State
from saetass import units as su
from saetass.cli import create_progress_bar, is_quiet, set_quiet
from saetass.cli.banner import BannerManager

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


@pytest.fixture
def default_output(monkeypatch):
    """Start from the default output, and restore it afterwards."""
    monkeypatch.delenv("SAETASS_QUIET", raising=False)
    set_quiet(None)
    yield
    set_quiet(None)


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


class TestProgressBar:
    def test_progress_bar_stops_on_errors(self, monkeypatch, capsys):
        solver = _solver(_grid())

        def fail(*args):
            raise RuntimeError("step failed")

        monkeypatch.setattr(solver.splitting_scheme, "apply", fail)
        with pytest.raises(RuntimeError, match="step failed"):
            solver.run()
        assert create_progress_bar()._progress is None
        # Later simulations print their own bar
        capsys.readouterr()
        _solver(_grid()).run()
        assert "Solving" in capsys.readouterr().out


class TestQuiet:
    def _output(self, capsys, **kwargs):
        BannerManager().reset()
        capsys.readouterr()
        _solver(_grid(), **kwargs).run()
        return capsys.readouterr().out

    def test_output_is_shown_by_default(self, capsys, default_output):
        output = self._output(capsys)
        assert "Equation of Transport" in output  # banner
        assert "Solving" in output  # progress bar

    def test_quiet_solver_prints_nothing(self, capsys, default_output):
        assert self._output(capsys, quiet=True) == ""

    def test_set_quiet(self, capsys, default_output):
        set_quiet(True)
        assert is_quiet()
        assert self._output(capsys) == ""
        assert "Solving" in self._output(capsys, quiet=False)
        set_quiet(False)
        assert not is_quiet()

    @pytest.mark.parametrize(
        "value, quiet",
        [("1", True), ("TRUE", True), (" yes ", True), ("on", True)]
        + [("0", False), ("false", False), ("No", False), ("off", False)]
        + [("", False)],
    )
    def test_environment_variable(self, monkeypatch, default_output, value, quiet):
        monkeypatch.setenv("SAETASS_QUIET", value)
        assert is_quiet() is quiet
        set_quiet(not quiet)  # takes precedence
        assert is_quiet() is (not quiet)

    def test_invalid_environment_variable(self, monkeypatch, default_output):
        monkeypatch.setenv("SAETASS_QUIET", "maybe")
        with pytest.raises(ValueError, match="SAETASS_QUIET must be one of"):
            is_quiet()

    def test_child_processes_are_quiet(self, default_output):
        assert not is_quiet()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            assert pool.submit(is_quiet).result()
