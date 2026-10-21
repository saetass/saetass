"""Tests of the Solver orchestrator: its arguments and terminal output."""

import logging
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
import pytest

from saetass import Grid, Separable, Solver, State, parallel
from saetass import units as su
from saetass.cli import (
    create_progress_bar,
    is_quiet,
    output,
    set_log_level,
    set_quiet,
    set_rich_logging,
)
from saetass.cli.banner import (
    BannerManager,
)
from saetass.cli.output import DEFAULT_LOG_LEVEL, LOGGER_NAME
from saetass.solver import _describe_parameter

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
        solver = _solver(_grid(), quiet=True)
        solver.step(1)
        solver.run()
        assert solver.global_step == solver.total_steps
        assert solver.state.t_val == pytest.approx(T[-1])

    def test_grid_is_that_of_the_state(self):
        grid = _grid()
        solver = _solver(grid, quiet=True)
        assert solver.grid is grid
        assert solver.total_steps == T.size - 1

    def test_grid_as_first_argument_is_rejected(self):
        grid = _grid()
        state = State(grid, psi_p=np.ones(grid.shape) * su.PSI_P)
        with pytest.raises(TypeError, match="takes the initial State as first"):
            Solver(grid, state, "diffusion")


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

    def test_quiet_solver_prints_only_log_messages(self, capsys, default_output):
        output = self._output(capsys, quiet=True)
        assert "Equation of Transport" not in output and "Solving" not in output
        assert "Solver finished" in output  # log messages are not affected

    def test_set_quiet(self, capsys, default_output):
        set_quiet(True)
        assert is_quiet()
        assert "Solving" not in self._output(capsys)
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

    def test_progress_bar_stops_on_errors(self, monkeypatch, capsys, default_output):
        solver = _solver(_grid(), quiet=False)

        def fail(*args):
            raise RuntimeError("step failed")

        monkeypatch.setattr(solver.splitting_scheme, "apply", fail)
        with pytest.raises(RuntimeError, match="step failed"):
            solver.run()
        assert create_progress_bar()._progress is None
        # Later simulations print their own bar
        assert "Solving" in self._output(capsys)


@pytest.fixture
def debug_log(saetass_caplog):
    """Log messages of SAETASS down to DEBUG, restoring the level afterwards."""
    with saetass_caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        yield saetass_caplog


def _double_rows(x):
    for i in numba.prange(x.size):
        x[i] *= 2.0


class TestLogging:
    def test_messages_go_through_the_rich_console(self, capsys):
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO and not logger.propagate
        capsys.readouterr()
        _solver(_grid(), quiet=True).run()
        output = capsys.readouterr().out
        assert "Solver: diffusion with Strang splitting, proton on 11 r cells" in output
        assert "Solver finished 2 steps" in output

    def test_set_rich_logging(self, caplog):
        logger = logging.getLogger(LOGGER_NAME)
        set_rich_logging(False)
        try:
            assert logger.propagate and output._handler not in logger.handlers
            _solver(_grid(), quiet=True)
            assert "Solver: diffusion" in caplog.text  # handled by the root logger
        finally:
            set_rich_logging(True)
        set_rich_logging(True)  # idempotent
        assert logger.handlers.count(output._handler) == 1 and not logger.propagate

    @pytest.mark.parametrize(
        "level, expected",
        [("debug", logging.DEBUG), (" WARNING ", logging.WARNING), ("15", 15)]
        + [(logging.ERROR, logging.ERROR)],
    )
    def test_set_log_level(self, level, expected):
        logger = logging.getLogger(LOGGER_NAME)
        try:
            set_log_level(level)
            assert logger.level == expected
        finally:
            set_log_level(DEFAULT_LOG_LEVEL)

    @pytest.mark.parametrize("level", ["verbose", 1.5, True, None])
    def test_invalid_log_level(self, level):
        with pytest.raises(ValueError, match="must be a logging level"):
            set_log_level(level)

    def test_environment_variable_sets_the_level(self):
        script = (
            "import logging, saetass; "
            "print(logging.getLogger('saetass').getEffectiveLevel())"
        )
        env = {**os.environ, "SAETASS_LOG_LEVEL": "warning"}
        out = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == str(logging.WARNING)

    def test_parameter_descriptions(self):
        separable = Separable([np.ones(3) * su.VELOCITY], [lambda t: 1.0])
        canonical = separable.canonical(su.VELOCITY, "v")
        unit = su.VELOCITY
        assert _describe_parameter(canonical, unit) == "separable, 1 component pc / Myr"
        assert (
            _describe_parameter(lambda t: 0.0, unit)
            == "time-dependent callable pc / Myr"
        )
        assert _describe_parameter("minmod", None) == "'minmod'"
        assert _describe_parameter(0.5, None) == "0.5"
        assert _describe_parameter(np.arange(4.0).reshape(2, 2), unit) == (
            "array of shape (2, 2) in [0, 3] pc / Myr"
        )

    @pytest.mark.parametrize(
        "velocity, expected",
        [
            (np.full(R.size, 0.1), "static velocities, CFL step"),
            (np.zeros(R.size), "zero velocities"),
            (lambda t: np.full(R.size, 0.1) * su.VELOCITY, "time-dependent velocities"),
            (
                Separable([np.full(R.size, 0.1) * su.VELOCITY], [lambda t: 1.0]),
                "separable velocities with 1 component",
            ),
        ],
        ids=["static", "zero", "callable", "separable"],
    )
    def test_operator_messages(self, debug_log, velocity, expected):
        grid = _grid(p=None)
        if isinstance(velocity, np.ndarray):
            velocity = velocity * su.VELOCITY
        params = {
            "advection": {
                "v_centers": velocity,
                "limiter": "minmod",
                "cfl": 0.5,
                "order": 2,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "diffusion": {"D_values": np.ones(R.size) * su.DIFFUSION_COEFFICIENT},
            "source": {"source": np.ones(R.size) * su.SOURCE_PSI_P},
        }
        state = State(grid, psi_p=np.ones(R.size) * su.PSI_P)
        Solver(state, "advection-diffusion-source", params, quiet=True).run()
        text = debug_log.text
        assert "AdvectionSolver: order 2, minmod limiter, CFL number 0.5" in text
        assert expected in text
        assert "AdvectionSolver parameter limiter: 'minmod'" in text
        assert "DiffusionSolver: 1 row of 11 cells, dirichlet boundary" in text
        assert "SourceSolver: static source" in text
        assert "Created grid: 11 r cells" in text
        assert "Advance finished at step 2/2" in text
        if expected.startswith("static"):
            assert "1 CFL sub-cycle for 1 step from t = 0 Myr" in text

    def test_kernel_messages(self, debug_log, monkeypatch):
        kernel = parallel.TwinKernel(_double_rows)
        x = np.ones(4)
        kernel(1, x)
        kernel(1, x)
        np.testing.assert_array_equal(x, 4.0)
        assert debug_log.text.count("Kernel _double_rows (serial): first call") == 1

        monkeypatch.setattr(parallel, "_load_sample", None)
        parallel._idle_cores()
        assert "Cores kept busy by other processes" in debug_log.text

    def test_thread_team_message(self, debug_log, monkeypatch):
        monkeypatch.setattr(parallel, "team_size", lambda work_ns: 3)
        monkeypatch.setattr(parallel, "_logged_team", 1)
        parallel.reset_team()
        assert parallel.threads_for(4, 1e6) == 3
        assert parallel.threads_for(4, 1e6) == 3
        assert debug_log.text.count("Kernel thread team: 3 threads") == 1
        parallel.reset_team()
