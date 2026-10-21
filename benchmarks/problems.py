r"""
Problems of the SAETASS performance benchmarks, in three groups.

* Representative problems: a stellar wind bubble in canonical units (pc, Myr, GeV/c), with
  a wind of :math:`v_w \approx 2000` km/s up to the termination shock and a subsonic
  :math:`v \propto r^{-2}` flow beyond it, Kolmogorov-like diffusion
  :math:`D \propto p^{1/3}` and synchrotron-like losses :math:`\dot p \propto -p^2`. Their
  time steps need a few CFL sub-cycles per step, as in typical runs.
* Single operators (:py:func:`operator_case`): one subsolver advanced by one CFL
  sub-cycle, to measure the cost of each kernel.
* Stress problems: parameters at the restrictive end of the physically valid range (fast
  wind, strong magnetic field, dense gas, wide momentum range), where the CFL condition
  forces up to tens of thousands of sub-cycles per step.

Every builder returns fresh objects, so a benchmark round always starts from the same
initial state. Apart from the variants with ``dynamic="separable"``, the problems only use
the API of SAETASS 0.5, so that they also measure that release.
"""

from dataclasses import dataclass
from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np

from saetass import Grid, Particle, Solver, State
from saetass import units as su
from saetass.solvers import (
    AdvectionSolver,
    DiffusionSolver,
    LossSolver,
    SourceSolver,
)
from saetass.utils.energy_losses import EnergyLossCalculator

#: Wind velocity, about 2000 km/s, in pc/Myr.
V_WIND = 2045.0
#: Diffusion coefficient at 1 GeV/c, about 1e28 cm^2/s, in pc^2/Myr.
D_0 = 33.0
#: Loss rate coefficient of p_dot = -B_LOSS p^2, in 1/(GeV/c Myr).
B_LOSS = 1e-5


@dataclass
class Problem:
    """A solver ready to run, with the number of global steps a benchmark takes."""

    solver: Solver
    n_steps: int

    def run(self) -> np.ndarray:
        """Advance ``n_steps`` global steps and return the bare final distribution."""
        self.solver.step(self.n_steps)
        return self.solver.state._values


def _r_grid(n_r: int, r_max: float) -> np.ndarray:
    # The diffusion operator requires the first cell center at r = 0
    return np.linspace(0.0, r_max, n_r)


def _p_grid(n_p: int) -> np.ndarray:
    return np.logspace(0.0, 7.0, n_p)


def wind_velocity(r: np.ndarray, r_ts: float, v_wind: float = V_WIND) -> np.ndarray:
    """Free wind inside the termination shock, subsonic ``r^-2`` flow beyond it."""
    r_safe = np.maximum(r, r_ts)
    return np.where(r < r_ts, v_wind, 0.25 * v_wind * (r_ts / r_safe) ** 2)


def _initial_profile(r: np.ndarray, r_max: float) -> np.ndarray:
    """Smooth, strictly positive profile, so that every limiter branch is exercised."""
    return np.exp(-((r / (0.2 * r_max)) ** 2)) + 1e-3


def _make_solver(
    r,
    p,
    t_grid,
    problem_type,
    operator_params,
    psi,
    scheme="strang",
    particle=Particle.ELECTRON,
):
    """A :py:class:`Solver` on the bare canonical grids ``r`` and ``p`` (either may be ``None``)."""
    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
        t_grid=t_grid * su.TIME,
    )
    state = State(grid=grid, psi_p=psi * su.PSI_P, particle=particle)
    return Solver(
        grid=grid,
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        splitting_scheme=scheme,
    )


def _separable(profiles, factors):
    """
    A :py:class:`saetass.Separable` parameter, imported here so that the module (and the
    benchmarks that do not use it) also runs with SAETASS 0.5, which does not have it.
    """
    from saetass import Separable

    return Separable(profiles, factors)


def _time_factor(function):
    """A :py:class:`saetass.TimeFactor`, imported here for the same reason."""
    from saetass import TimeFactor

    return TimeFactor(function)


def _hyperbolic_options(order: int = 2, limiter: str = "minmod") -> dict:
    return {"order": order, "limiter": limiter, "cfl": 0.8}


# ---------------------------------------------------------------------------
# Representative problems
# ---------------------------------------------------------------------------


def diffusion_source_1d(n_r: int = 1000, n_steps: int = 2000) -> Problem:
    """
    The diffusion-source steady state of the test suite: a 1D grid and many cheap steps.

    The Python overhead of every step dominates this problem.
    """
    r = _r_grid(n_r, 1.0)
    t_grid = np.linspace(0.0, 20.0 * n_steps / 1e5, n_steps + 1)
    params = {
        "diffusion": {
            "D_values": (r + 0.01) ** 2 * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        },
        "source": {"source": 4.0 * r * su.SOURCE_PSI_P},
    }
    solver = _make_solver(r, None, t_grid, "diffusion-source", params, 0.0 * r)
    return Problem(solver, n_steps)


def wind_bubble_1d(
    n_r: int = 600, n_steps: int = 400, D: float = D_0, order: int = 2
) -> Problem:
    """
    One energy of the multi-energy example: advection, source and diffusion in 1D.

    The time step needs one CFL sub-cycle per advection half-step.
    """
    r_max, r_ts = 300.0, 20.0
    r = _r_grid(n_r, r_max)
    dr = r[1] - r[0]
    dt = 0.8 * dr / V_WIND  # two half-steps of half the CFL limit
    t_grid = dt * np.arange(n_steps + 1)
    source = np.zeros(n_r)
    source[np.searchsorted(r, r_ts)] = 1.0 / dr
    params = {
        "advection": {
            "v_centers": wind_velocity(r, r_ts) * su.VELOCITY,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(order),
        },
        "source": {"source": source * su.SOURCE_PSI_P},
        "diffusion": {
            "D_values": np.full(n_r, D) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        },
    }
    psi = _initial_profile(r, r_max)
    solver = _make_solver(r, None, t_grid, "advection-source-diffusion", params, psi)
    return Problem(solver, n_steps)


def full_2d(
    n_r: int = 400, n_p: int = 71, n_steps: int = 20, dynamic: bool | str = False
) -> Problem:
    r"""
    The full transport equation on an :math:`(n_p, n_r)` grid.

    Advection needs three CFL sub-cycles per half-step and losses one. With
    ``dynamic=True`` the velocity and the loss rate are time-dependent callables, which
    adds the cost of evaluating them and interpolating them to the faces every sub-cycle;
    ``dynamic="separable"`` declares the same time dependence with
    :py:class:`saetass.Separable` parameters.
    """
    r_max, r_ts = 18.0, 2.0
    r, p = _r_grid(n_r, r_max), _p_grid(n_p)
    dt = 1e-4 * (400 / n_r)
    t_grid = dt * np.arange(n_steps + 1)

    v = np.broadcast_to(wind_velocity(r, r_ts), (n_p, n_r))
    D = D_0 * np.outer(p ** (1.0 / 3.0), np.ones(n_r))
    P_dot = np.broadcast_to((-B_LOSS * p**2)[:, None], (n_p, n_r))
    source = np.zeros((n_p, n_r))
    source[:, np.searchsorted(r, r_ts)] = p**-2.0 * np.exp(-p / 1e6)

    if dynamic == "separable":
        v_param = _separable(
            [v * su.VELOCITY], [lambda t: 1.0 + 0.1 * np.sin(t.to_value(su.TIME))]
        )
        P_dot_param = _separable(
            [P_dot * su.MOMENTUM_LOSS_RATE],
            [lambda t: 1.0 + 0.1 * np.cos(t.to_value(su.TIME))],
        )
    elif dynamic:

        def v_param(t):
            return (1.0 + 0.1 * np.sin(t.to_value(su.TIME))) * v * su.VELOCITY

        def P_dot_param(t):
            return (
                (1.0 + 0.1 * np.cos(t.to_value(su.TIME)))
                * P_dot
                * su.MOMENTUM_LOSS_RATE
            )

    else:
        v_param = v * su.VELOCITY
        P_dot_param = P_dot * su.MOMENTUM_LOSS_RATE

    params = {
        "advection": {
            "v_centers": v_param,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(),
        },
        "loss": {
            "P_dot": P_dot_param,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic_options(),
        },
        "source": {"source": source * su.SOURCE_PSI_P},
        "diffusion": {
            "D_values": D * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        },
    }
    psi = np.outer(p**-2.0, _initial_profile(r, r_max))
    solver = _make_solver(r, p, t_grid, "advection-loss-source-diffusion", params, psi)
    return Problem(solver, n_steps)


def energy_sweep(n_runs: int = 8, n_r: int = 600, n_steps: int = 200) -> list[Problem]:
    """Independent 1D runs at different energies, as in a parameter sweep."""
    energies = np.logspace(0.0, 5.0, n_runs)
    return [
        wind_bubble_1d(n_r=n_r, n_steps=n_steps, D=D_0 * E ** (1.0 / 3.0))
        for E in energies
    ]


def sweep_energies(n_runs: int) -> np.ndarray:
    """Energies of a sweep, in GeV."""
    return np.logspace(0.0, 5.0, n_runs)


def run_wind_bubble(energy: float, n_steps: int = 2000) -> np.ndarray:
    """
    Build and run one energy of the wind bubble, returning its final distribution.

    A top-level function, so that the workers of :py:func:`saetass.sweep.run_sweep` can
    import it.
    """
    return wind_bubble_1d(n_steps=n_steps, D=D_0 * energy ** (1.0 / 3.0)).run().copy()


#: Steps of each run of the larger sweep benchmark.
SWEEP_STEPS = 6000


def run_wind_bubble_long(energy: float) -> np.ndarray:
    """:py:func:`run_wind_bubble` with :py:data:`SWEEP_STEPS` steps, for sweep workers."""
    return run_wind_bubble(energy, n_steps=SWEEP_STEPS)


# ---------------------------------------------------------------------------
# Single operators
# ---------------------------------------------------------------------------


@dataclass
class OperatorCase:
    """A subsolver and a state whose distribution can be reset between rounds."""

    subsolver: object
    state: State
    initial: np.ndarray

    def reset(self) -> None:
        self.state._update_values(self.initial.copy())

    def advance(self) -> None:
        """Advance one step of the subsolver time grid."""
        self.subsolver.advance(1, self.state)


def operator_case(operator: str, shape: tuple[int, ...], **options) -> OperatorCase:
    r"""
    One subsolver on a grid of ``shape``, ``(n_r,)`` or ``(n_p,)`` in 1D and ``(n_p, n_r)`` in 2D.

    Its time step is a single CFL sub-cycle, so one call to ``advance`` evaluates the
    kernel exactly once. ``options`` override the hyperbolic options (``order``,
    ``limiter``); ``dynamic=True`` makes the parameters time-dependent callables, and
    ``dynamic="separable"`` :py:class:`saetass.Separable` parameters.
    """
    dynamic = options.pop("dynamic", False)
    is_loss = operator == "loss"
    if len(shape) == 2:
        n_p, n_r = shape
        r, p = _r_grid(n_r, 18.0), _p_grid(n_p)
    elif is_loss:
        (n_p,) = shape
        r, p = None, _p_grid(n_p)
    else:
        (n_r,) = shape
        r, p = _r_grid(n_r, 18.0), None

    grid = Grid(
        r_centers=None if r is None else r * su.LENGTH,
        p_centers=None if p is None else p * su.MOMENTUM,
    )
    full_shape = (1 if p is None else len(p), 1 if r is None else len(r))

    def broadcast(values_r, values_p):
        out = np.ones(full_shape)
        if values_r is not None:
            out = out * values_r[None, :]
        if values_p is not None:
            out = out * values_p[:, None]
        return out.reshape(shape)

    def maybe_dynamic(values, unit, coords=False):
        if not dynamic:
            return values * unit
        if dynamic == "separable":
            return _separable(
                [values * unit], [lambda t: 1.0 + 0.1 * np.sin(t.to_value(su.TIME))]
            )

        def fn(t):
            return (1.0 + 0.1 * np.sin(t.to_value(su.TIME))) * values * unit

        # Sources receive the cell-center coordinates first
        return (lambda r, p, t: fn(t)) if coords else fn

    if operator == "advection":
        v = broadcast(wind_velocity(r, 2.0), None)
        dt = 0.8 * (r[1] - r[0]) / V_WIND
        solver_class = AdvectionSolver
        params = {
            "v_centers": maybe_dynamic(v, su.VELOCITY),
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(),
        }
    elif is_loss:
        P_dot = broadcast(None, -B_LOSS * p**2)
        # The CFL step of the loss operator: the smallest cell width (in log10 p) over
        # the largest generalized velocity B p_max / ln 10, slightly reduced so that
        # rounding never adds a second, tiny sub-cycle
        width = np.min(np.asarray(grid.dp, dtype=float))
        dt = (1.0 - 1e-9) * 0.8 * width * np.log(10.0) / (B_LOSS * p[-1])
        solver_class = LossSolver
        params = {
            "P_dot": maybe_dynamic(P_dot, su.MOMENTUM_LOSS_RATE),
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic_options(),
        }
    elif operator == "diffusion":
        D = broadcast(np.ones_like(r), None if p is None else D_0 * p ** (1 / 3))
        dt = 1e-4
        solver_class = DiffusionSolver
        params = {
            "D_values": maybe_dynamic(D, su.DIFFUSION_COEFFICIENT),
            "psi_end": 0.0 * su.PSI_P,
        }
    elif operator == "source":
        Q = broadcast(_initial_profile(r, 18.0), None if p is None else p**-2.0)
        dt = 1e-4
        solver_class = SourceSolver
        params = {"source": maybe_dynamic(Q, su.SOURCE_PSI_P, coords=True)}
    else:
        raise ValueError(f"Unknown operator {operator!r}")

    for key in ("order", "limiter"):
        if key in options:
            params[key] = options.pop(key)
    if options:
        raise TypeError(f"Unexpected options {sorted(options)}")

    t_grid = np.array([0.0, dt])
    params = solver_class.convert_params(params, grid)
    subsolver = solver_class(grid, t_grid, params)

    psi = broadcast(
        None if r is None else _initial_profile(r, 18.0),
        None if p is None else p**-2.0,
    )
    state = State(grid=grid, psi_p=psi * su.PSI_P, particle=Particle.ELECTRON)
    return OperatorCase(subsolver, state, state._values.copy())


# ---------------------------------------------------------------------------
# Stress problems
# ---------------------------------------------------------------------------

#: Termination shock radius, and outer radius of the grid, in pc.
R_TS = 1.0
R_MAX = 18.0
#: Radius of the bubble (ionised gas inside, neutral shell beyond), in pc.
R_BUBBLE = 15.0
#: Wind velocity, 3000 km/s: the fast end of the winds of massive stars, in pc/Myr.
V_FAST = (3000.0 * u.km / u.s).to_value(su.VELOCITY)
#: Magnetic field of the bubble, 100 microgauss: strong, so synchrotron losses and the
#: Bohm diffusion coefficient are large at high momenta.
B_FIELD = (100.0 * u.uG).to(u.uG)
#: Gas density of the ionised bubble and of the neutral (or HII) shell, in cm^-3.
N_BUBBLE = 1e-2
N_SHELL = 1e3
#: Gas temperature of the ionised gas, in K, for the Coulomb losses.
T_GAS = 1e4
#: Period of the time-dependent parameters of the dynamic variants, in Myr.
PERIOD = 1e-3
#: Photon fields: CMB, infrared of the shell and starlight, as (T [K], U [eV/cm^3]).
PHOTON_FIELDS = ((2.725, 0.261), (70.0, 9.0), (3e4, 50.0))
#: Photon energies of the inverse Compton integral, in eV.
EPS_GRID = np.logspace(-7.0, 1.0, 281) * u.eV
#: Exponential cutoff of the injected spectra, in GeV/c.
P_CUTOFF = 1e6
#: Rest energies of the electron and the proton, in GeV.
MEC2 = (const.m_e * const.c**2).to_value(u.GeV)
MPC2 = (const.m_p * const.c**2).to_value(u.GeV)


def _greybody(T: float, U: float) -> u.Quantity:
    """Dilute blackbody of temperature T [K] and energy density U [eV/cm^3], dn/deps."""
    T, U = T * u.K, U * u.eV / u.cm**3
    planck = 8 * np.pi / (const.h * const.c) ** 3 * EPS_GRID**2
    x = (EPS_GRID / (const.k_B * T)).to_value(u.one)
    planck = planck / np.expm1(np.minimum(x, 700.0))
    return ((U / (4 * const.sigma_sb * T**4 / const.c)) * planck).to(u.cm**-3 / u.eV)


def photon_number_density() -> u.Quantity:
    """Spectral number density of the :py:data:`PHOTON_FIELDS` on :py:data:`EPS_GRID`."""
    return sum(_greybody(T, U) for T, U in PHOTON_FIELDS)


def _kinetic_energy(p: np.ndarray, mc2: float) -> u.Quantity:
    """Kinetic energy of momenta p [GeV/c] for a rest energy mc2 [GeV]."""
    return (np.sqrt(p**2 + mc2**2) - mc2) * u.GeV


@lru_cache(maxsize=None)
def _electron_loss_columns(
    log_p_min: float, log_p_max: float, n_p: int, zones: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loss rates of electrons in a few uniform zones, as ``(synchrotron, other)``.

    ``zones`` is a tuple of ``(n_gas [cm^-3], ionised, B [uG])``. Both outputs have shape
    ``(n_p, len(zones))`` in canonical units; synchrotron is separate because it scales
    as :math:`B^2`. The photon fields and the gas are uniform within a zone, so the
    expensive inverse Compton integral is evaluated once per zone and the callers
    expand the columns to their radial grids. The result is cached and must not be
    modified.
    """
    p = np.logspace(log_p_min, log_p_max, n_p)
    E = _kinetic_energy(p, MEC2)
    r_zones = np.ones(len(zones)) * u.pc
    n_gas = np.array([z[0] for z in zones]) * u.cm**-3
    ionised = np.array([z[1] for z in zones])
    B = np.array([z[2] for z in zones]) * u.uG

    def calculator(n):
        return EnergyLossCalculator(
            E_grid=E, r_grid=r_zones, n_gas=n, particle="electron"
        )

    sync = calculator(n_gas)
    sync.compute_synchrotron_losses(B_field=B)

    calc = calculator(n_gas)
    dn = photon_number_density()
    calc.compute_inverse_compton_losses(
        eps_grid=EPS_GRID, dn_deps=np.repeat(dn[:, None], len(zones), axis=1)
    )
    calc.compute_bremsstrahlung_losses(ionised_mask=ionised)
    P_dot = calc.get_momentum_loss_rate()

    # Ionisation of the neutral gas and Coulomb scattering in the ionised gas
    neutral = calculator(np.where(ionised, 0.0, n_gas.value) * u.cm**-3)
    plasma = calculator(np.where(ionised, n_gas.value, 0.0) * u.cm**-3)
    E_dot = neutral.compute_ionization_losses().to_value(u.GeV / u.s)
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb = plasma.compute_coulomb_losses(
            T_gas=np.full(len(zones), T_GAS) * u.K
        ).to_value(u.GeV / u.s)
    E_dot = E_dot + np.where(ionised & (n_gas.value > 0), np.nan_to_num(coulomb), 0.0)
    P_dot = P_dot + E_dot * u.GeV / u.s * calc.dp_dE[:, None]

    P_sync = sync.get_momentum_loss_rate().to_value(su.MOMENTUM_LOSS_RATE)
    return P_sync, P_dot.to_value(su.MOMENTUM_LOSS_RATE)


@lru_cache(maxsize=None)
def _proton_loss_columns(
    log_p_min: float, log_p_max: float, n_p: int, zones: tuple
) -> np.ndarray:
    """
    Loss rates of protons in a few uniform zones, shape ``(n_p, len(zones))``.

    ``zones`` is a tuple of ``(n_gas [cm^-3], ionised)``. Pion production in all zones
    and Coulomb losses in the ionised ones. The result is cached.
    """
    p = np.logspace(log_p_min, log_p_max, n_p)
    n_gas = np.array([z[0] for z in zones]) * u.cm**-3
    ionised = np.array([z[1] for z in zones])
    calc = EnergyLossCalculator(
        E_grid=_kinetic_energy(p, MPC2),
        r_grid=np.ones(len(zones)) * u.pc,
        n_gas=n_gas,
        particle="proton",
    )
    pion = calc.compute_pion_production_losses().to_value(u.GeV / u.s)
    with np.errstate(divide="ignore", invalid="ignore"):
        coulomb = calc.compute_coulomb_losses(
            T_gas=np.full(len(zones), T_GAS) * u.K
        ).to_value(u.GeV / u.s)
    # Coulomb losses only in the ionised zones, without the NaN of empty ones
    E_dot = pion + np.where(ionised, np.nan_to_num(coulomb), 0.0)
    P_dot = E_dot * u.GeV / u.s * calc.dp_dE[:, None]
    return P_dot.to_value(su.MOMENTUM_LOSS_RATE)


def _zone_index(r: np.ndarray) -> np.ndarray:
    """Zone of each cell: 0 inside the bubble, 1 in the shell."""
    return (r >= R_BUBBLE).astype(int)


def _bohm_diffusion(p: np.ndarray, B: u.Quantity) -> np.ndarray:
    r"""Bohm coefficient :math:`D = r_L c / 3` at momenta p [GeV/c], in pc^2/Myr."""
    p_si = (p * su.MOMENTUM).to(u.kg * u.m / u.s)
    r_L = p_si / (const.e.si * B.to(u.T))
    return (r_L.to(u.pc) * const.c / 3).to_value(su.DIFFUSION_COEFFICIENT)


def _shock_source(r: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Source of shape ``(len(spectrum), len(r))`` injected in the cell of the shock."""
    source = np.zeros((spectrum.size, r.size))
    source[:, np.searchsorted(r, R_TS)] = spectrum
    return source


def electrons_2d(n_steps: int | None = None, dynamic: bool | str = False) -> Problem:
    r"""
    Electrons in a strong-field, dense bubble: the full equation with every loss process.

    Restrictive choices: a 3000 km/s wind, 400 cells and a time step of
    :math:`10^{-4}` Myr (about 5 advection sub-cycles per half-step), momenta from 1 MeV/c
    to 10 PeV/c (ten decades), :math:`B = 100\,\mu\mathrm{G}` and a dense neutral shell, so
    that synchrotron, inverse Compton, bremsstrahlung, ionisation and Coulomb losses
    make :math:`\dot p` span many orders of magnitude and the loss operator needs about
    2400 sub-cycles per half-step. Diffusion is Bohm. With ``dynamic=True`` the field,
    the velocity and hence the loss rate and diffusion coefficient oscillate with a
    period of :math:`10^{-3}` Myr; ``dynamic="separable"`` declares the same time
    dependence with :py:class:`saetass.Separable` parameters, written with
    :py:class:`saetass.TimeFactor`.
    """
    if n_steps is None:
        n_steps = 1 if dynamic else 2
    n_r, n_p = 400, 101
    r = _r_grid(n_r, R_MAX)
    p = np.logspace(-3.0, 7.0, n_p)
    t_grid = 1e-4 * np.arange(n_steps + 1)
    zone = _zone_index(r)

    zones = ((N_BUBBLE, True, B_FIELD.value), (N_SHELL, False, B_FIELD.value))
    P_sync, P_other = _electron_loss_columns(-3.0, 7.0, n_p, zones)
    P_sync, P_other = P_sync[:, zone], P_other[:, zone]
    P_total = P_sync + P_other
    D0 = np.repeat(_bohm_diffusion(p, B_FIELD)[:, None], n_r, axis=1)
    v0 = np.broadcast_to(wind_velocity(r, R_TS, V_FAST), (n_p, n_r))
    source = _shock_source(r, p**-2.0 * np.exp(-p / P_CUTOFF))

    omega = 2.0 * np.pi / PERIOD

    def field(t):
        return 1.0 + 0.5 * np.sin(omega * t.to_value(su.TIME))

    if dynamic == "separable":
        # The same time dependence, written as its formula
        wind = _time_factor(lambda t: 1.0 + 0.1 * np.sin(omega * t.to_value(su.TIME)))
        B_squared = _time_factor(lambda t: field(t) ** 2)
        inverse_B = _time_factor(lambda t: 1.0 / field(t))
        v_param = v0 * su.VELOCITY * wind
        P_dot_param = (
            P_sync * su.MOMENTUM_LOSS_RATE * B_squared + P_other * su.MOMENTUM_LOSS_RATE
        )
        D_param = D0 * su.DIFFUSION_COEFFICIENT * inverse_B
    elif dynamic:

        def v_param(t):
            return (1.0 + 0.1 * np.sin(omega * t.to_value(su.TIME))) * v0 * su.VELOCITY

        def P_dot_param(t):
            return (field(t) ** 2 * P_sync + P_other) * su.MOMENTUM_LOSS_RATE

        def D_param(t):
            return D0 / field(t) * su.DIFFUSION_COEFFICIENT

    else:
        v_param = v0 * su.VELOCITY
        P_dot_param = P_total * su.MOMENTUM_LOSS_RATE
        D_param = D0 * su.DIFFUSION_COEFFICIENT

    params = {
        "advection": {
            "v_centers": v_param,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(),
        },
        "loss": {
            "P_dot": P_dot_param,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic_options(),
        },
        "source": {"source": source * su.SOURCE_PSI_P},
        "diffusion": {"D_values": D_param, "psi_end": 0.0 * su.PSI_P},
    }
    psi = np.outer(p**-2.0, _initial_profile(r, R_MAX))
    solver = _make_solver(r, p, t_grid, "advection-loss-source-diffusion", params, psi)
    return Problem(solver, n_steps)


def protons_2d(n_steps: int = 4) -> Problem:
    r"""
    Protons in an HII-region shell: the full equation from 1 MeV kinetic energy to 1 PeV.

    Restrictive choices: 20 momenta per decade (half the CFL step of the usual grid), a
    3000 km/s wind, ionised gas of 10 and :math:`10^3` cm\ :sup:`-3` in the bubble and
    the shell (strong Coulomb losses at low energy, and pion production at high
    energy) and diffusion
    :math:`D = 10^{28}\,\beta\,(p/\mathrm{GeV}\,c^{-1})^{1/2}` cm\ :sup:`2`/s. Advection
    needs about 5 CFL sub-cycles per half-step and losses about 85.
    """
    n_r, per_decade = 400, 20
    r = _r_grid(n_r, R_MAX)
    p_min = np.sqrt(1e-3**2 + 2e-3 * MPC2)  # 1 MeV kinetic energy
    log_p_min, log_p_max = np.log10(p_min), 6.0
    n_p = round((log_p_max - log_p_min) * per_decade) + 1
    p = np.logspace(log_p_min, log_p_max, n_p)
    t_grid = 1e-4 * np.arange(n_steps + 1)
    zone = _zone_index(r)

    zones = ((10.0, True), (N_SHELL, True))
    P_dot = _proton_loss_columns(log_p_min, log_p_max, n_p, zones)[:, zone]
    beta = p / np.sqrt(p**2 + MPC2**2)
    D_cgs = 1e28 * beta * p**0.5 * u.cm**2 / u.s
    D = np.repeat(D_cgs.to_value(su.DIFFUSION_COEFFICIENT)[:, None], n_r, axis=1)
    v = np.broadcast_to(wind_velocity(r, R_TS, V_FAST), (n_p, n_r))
    source = _shock_source(r, p**-2.0 * np.exp(-p / 1e5))

    params = {
        "advection": {
            "v_centers": v * su.VELOCITY,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(),
        },
        "loss": {
            "P_dot": P_dot * su.MOMENTUM_LOSS_RATE,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic_options(),
        },
        "source": {"source": source * su.SOURCE_PSI_P},
        "diffusion": {
            "D_values": D * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        },
    }
    psi = np.outer(p**-2.0, _initial_profile(r, R_MAX))
    solver = _make_solver(
        r,
        p,
        t_grid,
        "advection-loss-source-diffusion",
        params,
        psi,
        particle=Particle.PROTON,
    )
    return Problem(solver, n_steps)


def electron_cooling_1d(n_steps: int = 2) -> Problem:
    r"""
    Cooling of electrons in a single dense, magnetized zone: only the loss operator.

    Restrictive choices: 20 momenta per decade from 1 MeV/c to 10 PeV/c,
    :math:`B = 100\,\mu\mathrm{G}`, neutral gas of :math:`10^3` cm\ :sup:`-3` and the
    three photon fields, so that the loss rate at the top of the grid is huge and a step
    of :math:`10^{-3}` Myr needs about 90 000 CFL sub-cycles.
    """
    n_p = 201
    p = np.logspace(-3.0, 7.0, n_p)
    t_grid = 1e-3 * np.arange(n_steps + 1)
    zones = ((N_SHELL, False, B_FIELD.value),)
    P_sync, P_other = _electron_loss_columns(-3.0, 7.0, n_p, zones)
    params = {
        "source": {"source": p**-2.0 * np.exp(-p / P_CUTOFF) * su.SOURCE_PSI_P},
        "loss": {
            "P_dot": (P_sync + P_other)[:, 0] * su.MOMENTUM_LOSS_RATE,
            "inflow_value_U": 0.0 * su.MOMENTUM * su.PSI_P,
            **_hyperbolic_options(),
        },
    }
    psi = p**-2.0 + 1e-30
    solver = _make_solver(None, p, t_grid, "source-loss", params, psi)
    return Problem(solver, n_steps)


def wind_bubble_1d_stress(n_steps: int = 200, dynamic: bool | str = False) -> Problem:
    r"""
    Advection, source and diffusion in 1D with a 3000 km/s wind on 2000 cells.

    The time step of :math:`10^{-4}` Myr needs about 22 advection sub-cycles per
    half-step. Diffusion is Bohm at :math:`10^5` GeV/c in 100 :math:`\mu`G. With
    ``dynamic=True`` the wind velocity oscillates by 20% with a period of
    :math:`10^{-3}` Myr; ``dynamic="separable"`` declares the same time dependence with
    a :py:class:`saetass.Separable` parameter.
    """
    n_r = 2000
    r = _r_grid(n_r, R_MAX)
    dr = r[1] - r[0]
    t_grid = 1e-4 * np.arange(n_steps + 1)
    source = np.zeros(n_r)
    source[np.searchsorted(r, R_TS)] = 1.0 / dr
    v = wind_velocity(r, R_TS, V_FAST)
    D = _bohm_diffusion(np.array([1e5]), B_FIELD)[0]

    if dynamic == "separable":
        v_param = _separable(
            [v * su.VELOCITY],
            [lambda t: 1.0 + 0.2 * np.sin(2.0 * np.pi * t.to_value(su.TIME) / PERIOD)],
        )
    elif dynamic:

        def v_param(t):
            return (
                (1.0 + 0.2 * np.sin(2.0 * np.pi * t.to_value(su.TIME) / PERIOD))
                * v
                * su.VELOCITY
            )

    else:
        v_param = v * su.VELOCITY

    params = {
        "advection": {
            "v_centers": v_param,
            "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            **_hyperbolic_options(),
        },
        "source": {"source": source * su.SOURCE_PSI_P},
        "diffusion": {
            "D_values": np.full(n_r, D) * su.DIFFUSION_COEFFICIENT,
            "psi_end": 0.0 * su.PSI_P,
        },
    }
    psi = _initial_profile(r, R_MAX)
    solver = _make_solver(r, None, t_grid, "advection-source-diffusion", params, psi)
    return Problem(solver, n_steps)


def full_2d_large(n_steps: int = 4) -> Problem:
    """:py:func:`full_2d` on a large grid (256 x 2000 cells), for thread scaling."""
    return full_2d(n_r=2000, n_p=256, n_steps=n_steps)


def full_2d_huge(n_steps: int = 2) -> Problem:
    """:py:func:`full_2d` on a huge grid (1024 x 4000 cells), for thread scaling."""
    return full_2d(n_r=4000, n_p=1024, n_steps=n_steps)
