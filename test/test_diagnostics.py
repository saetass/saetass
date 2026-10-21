try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import math
import warnings

import numpy as np
import pytest

from saetass import Grid, Solver, State
from saetass import units as su
from saetass.diagnostics import WARNING_SPLITTING_RATIO, DiagnosticsWarning, diagnose

R = np.linspace(0.0, 10.0, 101)  # cell centers 0.1 pc apart
ADVECTION = {
    "limiter": "minmod",
    "order": 2,
    "cfl": 0.5,
    "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
}


def _solver(problem_type, params, t_end=1.0, num_timesteps=10, p=None):
    grid = Grid(
        r_centers=R * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=np.linspace(0.0, t_end, num_timesteps + 1) * su.TIME,
    )
    shape = R.shape if p is None else (p.size, R.size)
    psi = np.broadcast_to(np.exp(-((R - 5.0) ** 2)), shape)
    return Solver(
        state=State(grid=grid, psi_p=psi * su.PSI_P),
        problem_type=problem_type,
        operator_params=params,
    )


def _advection(v):
    return {**ADVECTION, "v_centers": np.full(R.size, v) * su.VELOCITY}


def _diffusion(D):
    return {"D_values": np.full(R.size, D) * su.DIFFUSION_COEFFICIENT}


def test_advection_courant_number_and_timescale(plot_results):
    solver = _solver("advection", {"advection": _advection(2.0)})
    psi_initial = solver.state.psi_p.to_value(su.PSI_P).copy()
    diag = solver.diagnostics()["advection"]
    # dt = 0.1 Myr, |v| = 2 pc/Myr, dr = 0.1 pc
    assert diag.time_step.to_value(su.TIME) == pytest.approx(0.1)
    assert diag.numbers["courant_number"] == pytest.approx(2.0)
    assert diag.numbers["cfl_subcycles"] == 4.0  # ceil(Courant / cfl)
    # Domain from the first to the last face, 10.1 pc, crossed at 2 pc/Myr
    assert diag.timescale.to_value(su.TIME) == pytest.approx(10.1 / 2.0)

    # The reported sub-cycles are those the solver actually performs per step
    subsolver = solver.operator_subsolvers[0]
    solver.step(1)
    assert subsolver._cfl_subcycles == diag.numbers["cfl_subcycles"]

    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(R, psi_initial, "k--", label="Initial")
        plt.plot(R, solver.state.psi_p.to_value(su.PSI_P), label="After one step")
        plt.title(
            f"Advection over one step: Courant number "
            f"{diag.numbers['courant_number']:.2f}, "
            f"{subsolver._cfl_subcycles} CFL sub-cycles"
        )
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


@pytest.mark.parametrize("D", [0.02, 0.5])
def test_fourier_number_is_the_crank_nicolson_positivity_condition(D, plot_results):
    solver = _solver("diffusion", {"diffusion": _diffusion(D)})
    fourier = solver.diagnostics(warn=False)["diffusion"].numbers["fourier_number"]

    # Explicit half of Crank-Nicolson: B_ii = 2 V_i / dt - (G_i-1/2 + G_i+1/2)
    sub = solver.operator_subsolvers[0]
    dt = float(np.diff(sub.t_grid)[0])
    sub._build_matrices(dt)
    integrated = slice(None, -1)  # the Dirichlet boundary cell is not integrated
    expected = np.max(
        1.0 - sub._B_diag[:, integrated] * dt / (2.0 * sub.V_b[:, integrated])
    )
    assert fourier == pytest.approx(expected)
    assert (fourier <= 1.0) == bool(np.all(sub._B_diag[:, integrated] >= 0.0))

    if plot_results:
        coupling = 1.0 - sub._B_diag[0] * dt / (2.0 * sub.V_b[0])
        plt.figure(figsize=(10, 6))
        plt.plot(R[integrated], coupling[integrated], label="Cell coupling")
        plt.axhline(fourier, color="r", ls="--", label=f"Fourier number {fourier:.3g}")
        plt.axhline(1.0, color="k", ls=":", label="Positivity limit")
        plt.title(f"Crank-Nicolson cell coupling, D = {D} pc^2/Myr")
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_loss_timescale_is_shortest_loss_time(plot_results):
    p = np.logspace(0, 3, 40)
    solver = _solver(
        "loss",
        {
            "loss": {
                **ADVECTION,
                "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
                "P_dot": np.broadcast_to(-0.01 * p[:, None] ** 2, (p.size, R.size))
                * su.MOMENTUM_LOSS_RATE,
            }
        },
        p=p,
    )
    diag = solver.diagnostics()["loss"]
    # p / |p_dot| = 1 / (0.01 p), shortest at the highest momentum
    assert diag.timescale.to_value(su.TIME) == pytest.approx(1.0 / (0.01 * p[-1]))

    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.loglog(p, 1.0 / (0.01 * p), label="Loss time p / |p_dot|")
        plt.axhline(
            diag.timescale.to_value(su.TIME),
            color="r",
            ls="--",
            label=f"Reported timescale {diag.timescale:.3g}",
        )
        plt.title("Loss timescale")
        plt.xlabel("p [GeV/c]")
        plt.ylabel("t [Myr]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_fourier_number_scales_with_the_time_step_without_warning(plot_results):
    coarse = _solver("diffusion", {"diffusion": _diffusion(0.5)}, num_timesteps=10)
    fine = _solver("diffusion", {"diffusion": _diffusion(0.5)}, num_timesteps=20)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # large Fourier numbers are reported, not warned about
        fourier_coarse = coarse.diagnostics()["diffusion"].numbers["fourier_number"]
    fourier_fine = fine.diagnostics()["diffusion"].numbers["fourier_number"]
    assert fourier_coarse > 1.0
    assert fourier_coarse == pytest.approx(2.0 * fourier_fine)

    if plot_results:
        psi_initial = coarse.state.psi_p.to_value(su.PSI_P).copy()
        plt.figure(figsize=(10, 6))
        plt.plot(R, psi_initial, "k--", label="Initial")
        plt.plot(
            R,
            coarse.run().psi_p.to_value(su.PSI_P),
            label=f"10 steps (Fourier number {fourier_coarse:.3g})",
            lw=3,
            alpha=0.6,
        )
        plt.plot(
            R,
            fine.run().psi_p.to_value(su.PSI_P),
            "r:",
            label=f"20 steps (Fourier number {fourier_fine:.3g})",
            lw=2,
        )
        plt.title("Diffusion with large Fourier numbers")
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_splitting_warning_suggests_enough_timesteps(plot_results):
    params = {
        "advection": _advection(2.0),
        "source": {"source": np.ones(R.size) * su.SOURCE_PSI_P},
    }
    # dt = 1 Myr against a crossing time of 5.05 Myr
    solver = _solver("advection-source", params, t_end=20.0, num_timesteps=20)
    ratio = 1.0 / 5.05
    suggested = math.ceil(20 * ratio / WARNING_SPLITTING_RATIO)
    with pytest.warns(DiagnosticsWarning, match=f"at least {suggested} timesteps"):
        diag = solver.diagnostics()
    assert diag.numbers["splitting_ratio"] == pytest.approx(ratio)

    refined = _solver("advection-source", params, t_end=20.0, num_timesteps=suggested)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        refined_diag = refined.diagnostics()
    assert refined_diag.warnings == ()
    assert refined_diag.numbers["splitting_ratio"] <= WARNING_SPLITTING_RATIO

    if plot_results:
        reference = _solver("advection-source", params, t_end=20.0, num_timesteps=400)
        plt.figure(figsize=(10, 6))
        plt.plot(R, reference.run().psi_p.to_value(su.PSI_P), "k--", label="Reference")
        plt.plot(
            R,
            solver.run().psi_p.to_value(su.PSI_P),
            label=f"20 steps (ratio {ratio:.3g}, warned)",
        )
        plt.plot(
            R,
            refined.run().psi_p.to_value(su.PSI_P),
            label=f"{suggested} steps (suggested)",
        )
        plt.title("Advection with a source after 20 Myr")
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_strang_outer_operators_integrate_half_steps():
    diag = _solver(
        "advection-diffusion",
        {"advection": _advection(2.0), "diffusion": _diffusion(0.02)},
    ).diagnostics()
    assert diag["advection"].time_step.to_value(su.TIME) == pytest.approx(0.05)
    assert diag["diffusion"].time_step.to_value(su.TIME) == pytest.approx(0.1)


def test_cell_peclet_number():
    diag = _solver(
        "advection-diffusion",
        {"advection": _advection(2.0), "diffusion": _diffusion(0.5)},
    ).diagnostics(warn=False)
    assert diag.numbers["cell_peclet"] == pytest.approx(2.0 * 0.1 / 0.5)
    assert "cell_peclet" in str(diag) and "splitting_ratio" in str(diag)


def test_time_dependent_parameters_report_the_worst_case(plot_results):
    params = {
        "diffusion": {
            "D_values": lambda t: (
                np.full(R.size, 0.01 * (1.0 + t.to_value(su.TIME)))
                * su.DIFFUSION_COEFFICIENT
            )
        }
    }
    start = _solver("diffusion", params).diagnostics(n_samples=1)
    worst = _solver("diffusion", params).diagnostics(n_samples=3)
    assert worst.sampled_times.to_value(su.TIME) == pytest.approx([0.0, 0.5, 1.0])
    assert worst["diffusion"].numbers["fourier_number"] == pytest.approx(
        2.0 * start["diffusion"].numbers["fourier_number"]
    )

    if plot_results:
        sub = _solver("diffusion", params).operator_subsolvers[0]
        times = np.linspace(0.0, 1.0, 21)
        dt = float(np.diff(sub.t_grid)[0])
        fourier = [
            sub.characteristic_scales(t, dt).numbers["fourier_number"] for t in times
        ]
        plt.figure(figsize=(10, 6))
        plt.plot(times, fourier, label="Fourier number at time t")
        plt.plot(
            worst.sampled_times.to_value(su.TIME),
            [
                sub.characteristic_scales(t, dt).numbers["fourier_number"]
                for t in worst.sampled_times.to_value(su.TIME)
            ],
            "o",
            label="Sampled times",
        )
        plt.axhline(
            worst["diffusion"].numbers["fourier_number"],
            color="r",
            ls="--",
            label="Reported worst case",
        )
        plt.title("Fourier number with a growing diffusion coefficient")
        plt.xlabel("t [Myr]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_diagnostics_do_not_change_the_simulation(plot_results):
    params = {
        "advection": {
            **ADVECTION,
            "v_centers": lambda t: (
                np.full(R.size, 1.0 + t.to_value(su.TIME)) * su.VELOCITY
            ),
        },
        "diffusion": {
            "D_values": lambda t: (
                np.full(R.size, 0.05 * (1.0 + t.to_value(su.TIME)))
                * su.DIFFUSION_COEFFICIENT
            )
        },
    }
    reference = _solver("advection-diffusion", params).run().psi_p
    diagnosed = _solver("advection-diffusion", params)
    diagnosed.diagnostics(warn=False)
    result = diagnosed.run().psi_p
    np.testing.assert_array_equal(result, reference)

    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(R, reference.to_value(su.PSI_P), label="Without diagnostics", lw=3)
        plt.plot(R, result.to_value(su.PSI_P), "r:", label="With diagnostics", lw=2)
        plt.title("Running the diagnostics does not change the simulation")
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_operators_without_scales():
    diag = _solver(
        "advection-source",
        {
            "advection": _advection(0.0),
            "source": {"source": np.ones(R.size) * su.SOURCE_PSI_P},
        },
    ).diagnostics()
    assert diag["source"].timescale is None and dict(diag["source"].numbers) == {}
    assert diag["advection"].numbers["courant_number"] == 0.0
    assert diag["advection"].numbers["cfl_subcycles"] == 1.0
    assert np.isinf(diag["advection"].timescale)
    assert "splitting_ratio" not in diag.numbers  # no finite timescale to compare with

    still = _solver("diffusion", {"diffusion": _diffusion(0.0)}).diagnostics()
    assert np.isinf(still["diffusion"].timescale)
    assert "splitting_ratio" not in still.numbers  # a single operator is not split


def test_vanishing_diffusion_coefficients_do_not_warn():
    # The harmonic mean of a face next to a zero coefficient divides by zero
    D = np.where(R < 5.0, 0.0, 1.0) * su.DIFFUSION_COEFFICIENT
    solver = _solver("diffusion", {"diffusion": {"D_values": D}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        diag = solver.diagnostics(warn=False)

    D_face = np.empty_like(solver.operator_subsolvers[0]._D_face)
    G = np.empty_like(solver.operator_subsolvers[0]._G)
    solver.operator_subsolvers[0]._compute_conductances(D.value[None, :], D_face, G)
    assert np.all(D_face[0, 1:-1][R[1:] <= 5.0] == 0.0)
    assert np.all(D_face[0, 1:-1][R[:-1] >= 5.0] > 0.0)
    assert np.isfinite(diag["diffusion"].numbers["fourier_number"])


def test_report_and_lookup():
    params = {
        "advection": _advection(2.0),
        "source": {"source": np.ones(R.size) * su.SOURCE_PSI_P},
    }
    diag = _solver("advection-source", params, t_end=20.0, num_timesteps=2).diagnostics(
        warn=False
    )
    report = str(diag)
    assert "advection" in report and "courant_number" in report and "WARNING" in report
    with pytest.raises(KeyError):
        diag["diffusion"]
    with pytest.raises(ValueError, match="n_samples"):
        diagnose(_solver("diffusion", {"diffusion": _diffusion(0.5)}), n_samples=0)


def test_hyperbolic_operators_have_no_timescale_by_default():
    from saetass.solvers.hyperbolic_solver import HyperbolicSolver

    advection = _solver(
        "advection", {"advection": _advection(2.0)}
    ).operator_subsolvers[0]
    assert (
        HyperbolicSolver._characteristic_timescale(advection, advection.V_centers)
        is None
    )


def test_cell_damkohler_numbers_compare_losses_with_transport(plot_results):
    p = np.logspace(0, 2, 30)
    shape = (p.size, R.size)
    v = np.broadcast_to(np.where(R < 1.0, 0.0, 2.0), shape)  # no advection near r = 0
    D = np.broadcast_to(np.where(R < 1.0, 0.0, 0.5), shape)  # nor diffusion
    params = {
        "advection": {**ADVECTION, "v_centers": v * su.VELOCITY},
        "diffusion": {"D_values": D * su.DIFFUSION_COEFFICIENT},
        "loss": {
            **ADVECTION,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            "P_dot": np.broadcast_to(-0.01 * p[:, None] ** 2, shape)
            * su.MOMENTUM_LOSS_RATE,
        },
    }
    solver = _solver("advection-diffusion-loss", params, p=p)
    diag = solver.diagnostics(warn=False)
    # |p_dot| / p = 0.01 p, fastest at the highest momentum; dr = 0.1 pc
    loss_rate = 0.01 * p[-1]
    assert diag.numbers["cell_damkohler_advection"] == pytest.approx(
        0.1 * loss_rate / 2.0
    )
    assert diag.numbers["cell_damkohler_diffusion"] == pytest.approx(
        0.1**2 * loss_rate / 0.5
    )
    # Cells where a process does not act are left out instead of giving infinities
    assert diag.numbers["cell_peclet"] == pytest.approx(2.0 * 0.1 / 0.5)

    if plot_results:
        rates = solver.operator_subsolvers[2].loss_rates(0.0)[:, -1]
        plt.figure(figsize=(10, 6))
        plt.loglog(p, 0.1 * rates / 2.0, label="Advection, (dr / |v|) |p_dot| / p")
        plt.loglog(p, 0.1**2 * rates / 0.5, label="Diffusion, (dr^2 / D) |p_dot| / p")
        plt.axhline(1.0, color="k", ls=":", label="Unresolved above")
        plt.title("Cell Damköhler numbers by momentum")
        plt.xlabel("p [GeV/c]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def _oscillating_source(period, amplitude=1.0):
    def source(r, p, t):
        phase = 2.0 * np.pi * t.to_value(su.TIME) / period
        return amplitude * (1.0 + np.sin(phase)) * np.ones(R.size) * su.SOURCE_PSI_P

    return {"source": {"source": source}}


def test_source_timescale_depends_on_its_variation_not_its_magnitude(plot_results):
    def scales(period, amplitude=1.0):
        diag = _solver("source", _oscillating_source(period, amplitude)).diagnostics()
        return (
            diag["source"].timescale.to_value(su.TIME),
            diag["source"].numbers["source_variation"],
        )

    slow, fast = scales(100.0), scales(1.0)
    assert slow[1] < 0.1 < fast[1]
    assert slow[0] > 10.0 * fast[0]
    assert scales(1.0, amplitude=1e6) == pytest.approx(fast)

    # Variations faster than the step are resolved, not aliased by its ends
    assert scales(0.2)[1] > 1.0

    constant = _solver(
        "source",
        {"source": {"source": lambda r, p, t: np.ones(R.size) * su.SOURCE_PSI_P}},
    )
    assert np.isinf(constant.diagnostics()["source"].timescale)
    off = _solver("source", _oscillating_source(1.0, amplitude=0.0))
    assert off.diagnostics()["source"].numbers["source_variation"] == 0.0

    if plot_results:
        times = np.linspace(0.0, 1.0, 1001)
        plt.figure(figsize=(10, 6))
        for period in (100.0, 1.0, 0.2):
            variation = scales(period)[1]
            plt.plot(
                times,
                1.0 + np.sin(2.0 * np.pi * times / period),
                label=f"Period {period} Myr, source_variation {variation:.3g}",
            )
        for t_step in np.linspace(0.0, 1.0, 11):
            plt.axvline(t_step, color="grey", lw=0.5)
        plt.title("Source time profiles against the time steps (grey)")
        plt.xlabel("t [Myr]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()


def test_fast_source_variation_triggers_the_splitting_warning(plot_results):
    params = {"diffusion": _diffusion(0.02), **_oscillating_source(0.37)}
    with pytest.warns(DiagnosticsWarning, match=r"\(source"):
        diag = _solver("diffusion-source", params).diagnostics()
    assert diag["source"].timescale < diag["diffusion"].timescale

    if plot_results:
        coarse = _solver("diffusion-source", params).run().psi_p
        fine = _solver("diffusion-source", params, num_timesteps=400).run().psi_p
        plt.figure(figsize=(10, 6))
        plt.plot(R, fine.to_value(su.PSI_P), "k--", label="400 steps")
        plt.plot(R, coarse.to_value(su.PSI_P), label="10 steps (warned)")
        plt.title("Diffusion with a source oscillating faster than the time step")
        plt.xlabel("r [pc]")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()
