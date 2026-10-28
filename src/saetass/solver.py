"""
The solver module contains the main interface for running simulations.

It defines the public API that all physical operators must expose through the abstract class :py:class:`~saetass.solver.SubSolver`.

It also defines the :py:class:`~saetass.solver.Solver` class, which manages the overall time loop and coordinates the advancement of the solution according to the specified problem type and parameters.
Hence, it serves as the central orchestrator of the simulation workflow, while delegating the actual numerical updates to the specialized operators (:py:class:`~saetass.solvers.advection_solver.AdvectionSolver`, :py:class:`~saetass.solvers.diffusion_solver.DiffusionSolver`, :py:class:`~saetass.solvers.loss_solver.LossSolver`, :py:class:`~saetass.solvers.source_solver.SourceSolver`...).

--------------
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

import astropy.units as u
import numpy as np

from . import units as su
from .cli.banner import print_banner
from .cli.output import is_quiet, plural
from .cli.progress import create_progress_bar
from .diagnostics import CharacteristicScales, SimulationDiagnostics, diagnose
from .diagnostics import warn as _warn
from .grid import Grid
from .parameters import CanonicalSeparable, Separable
from .splitting import create_splitting_scheme
from .state import State

#: Seconds between updates of the extrema shown by the progress bar, its refresh period.
_PROGRESS_METRICS_INTERVAL = 0.1


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """
    Declaration of a single user-facing :py:class:`~saetass.solver.SubSolver` parameter.

    Each :py:class:`~saetass.solver.SubSolver` declares its accepted parameters in
    :py:attr:`~saetass.solver.SubSolver.PARAM_SPECS`. :py:class:`~saetass.solver.Solver`
    uses these declarations to convert physical inputs into the bare canonical floats
    consumed by the numerical kernels, rejecting anything undeclared or missing.

    Parameters
    ----------
    unit : astropy.units.UnitBase or None, optional
        Canonical unit of a physical parameter, which must then be given as a
        :py:class:`~astropy.units.Quantity` (or a callable returning one).
        ``None`` marks a non-physical option (e.g. ``cfl``, ``limiter``), passed through unchanged.
    dynamic : bool, optional
        Whether a callable may be given instead of a fixed value. It is called with the
        time as a Quantity, ``fn(t)``, and must return a Quantity.
    coords : bool, optional
        Whether a dynamic callable also receives the physical cell-center coordinates,
        ``fn(r, p, t)``, where ``r`` or ``p`` is ``None`` for 1D grids.
    required : bool, optional
        Whether the parameter must always be given. Default is ``False``.
    """

    unit: u.UnitBase | None = None
    dynamic: bool = False
    coords: bool = False
    required: bool = False


class SubSolver(ABC):
    """
    Abstract base class defining the minimal interface every operator must expose.

    Implementations must properly initialize from a domain :py:class:`~saetass.grid.Grid`, a specific time grid,
    and a parameter dictionary, and must provide an :py:meth:`~saetass.solver.SubSolver.advance` method to update the
    global :py:class:`~saetass.state.State`.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        The :py:class:`~saetass.grid.Grid` object containing spatial and/or momentum nodes.
    t_grid : numpy.ndarray
        The refined time grid for this specific operator's integration steps, in canonical :py:data:`~saetass.units.TIME` units.
    params : dict
        Dictionary containing numerical parameters specific to this :py:class:`~saetass.solver.SubSolver`, already converted to bare canonical floats (see :py:meth:`~saetass.solver.SubSolver.convert_params`).
    """

    #: Accepted user-facing parameters and their canonical units.
    PARAM_SPECS: ClassVar[Mapping[str, ParamSpec]] = MappingProxyType({})

    @property
    def t_grid(self) -> np.ndarray:
        """Time grid of the operator, in canonical :py:data:`~saetass.units.TIME` units."""
        return self._t_grid

    @t_grid.setter
    def t_grid(self, value: np.ndarray) -> None:
        self._t_grid = np.asarray(value, dtype=float)
        # advance() runs every step: difference the (long) grid once, not per call
        self._dt = float(self._t_grid[1] - self._t_grid[0])

    @classmethod
    def convert_params(cls, params: Mapping[str, Any], grid: Grid) -> dict[str, Any]:
        """
        Convert user-facing parameters into the bare canonical floats consumed by this subsolver.

        Physical parameters must be Quantities (or, where declared dynamic, callables returning Quantities, or :py:class:`~saetass.parameters.Separable` parameters) with units equivalent to those declared in :py:attr:`PARAM_SPECS`.
        Dynamic callables are wrapped into ``fn(t: float) -> ndarray`` in canonical units, and separable parameters into :py:class:`~saetass.parameters.CanonicalSeparable`, which is also such a callable.

        Parameters
        ----------
        params : dict
            User-facing parameters.
        grid : :py:class:`~saetass.grid.Grid`
            Grid providing the physical coordinates passed to coordinate-dependent callables.

        Returns
        -------
        dict
            Parameters ready to be passed to the subsolver constructor.

        Raises
        ------
        ValueError
            If a parameter is not declared in :py:attr:`PARAM_SPECS`, or a required one is missing.
        TypeError
            If a physical parameter lacks units, a non-physical one has them, or a callable is given for a non-dynamic parameter.
        astropy.units.UnitsError
            If a parameter has units incompatible with its declared unit.
        """
        unknown = params.keys() - cls.PARAM_SPECS.keys()
        if unknown:
            raise ValueError(
                f"Unknown {cls.__name__} parameter(s) {sorted(unknown)}. "
                f"Accepted parameters are {sorted(cls.PARAM_SPECS)}."
            )

        converted = {}
        for key, value in params.items():
            spec = cls.PARAM_SPECS[key]
            name = f"{cls.__name__} parameter '{key}'"
            if spec.unit is None:
                su.require_bare(value, name)
                converted[key] = value
            elif isinstance(value, Separable):
                if not spec.dynamic:
                    raise TypeError(f"{name} does not accept a Separable.")
                converted[key] = value.canonical(spec.unit, name)
            elif callable(value):
                if not spec.dynamic:
                    raise TypeError(f"{name} does not accept a callable.")
                coords = (grid.r_centers, grid.p_centers_phys) if spec.coords else ()
                converted[key] = _canonical_callable(value, spec.unit, coords, name)
            else:
                converted[key] = su.validate_quantity(value, spec.unit, name).value

        missing = [
            key
            for key, spec in cls.PARAM_SPECS.items()
            if spec.required and key not in params
        ]
        if missing:
            raise ValueError(
                f"Missing required {cls.__name__} parameter(s) {sorted(missing)}."
            )
        return converted

    @classmethod
    def _validate_unit_free_inputs(
        cls, t_grid: np.ndarray, params: dict[str, Any]
    ) -> None:
        """
        Validate that subsolvers receive bare canonical floats, since unit conversions are performed by :py:meth:`convert_params`.
        """
        su.require_bare(t_grid, f"{cls.__name__} t_grid")
        for key, value in (params or {}).items():
            su.require_bare(value, f"{cls.__name__} parameter '{key}'")

    @abstractmethod
    def __init__(
        self, grid: Grid, t_grid: np.ndarray, params: dict[str, Any], **kwargs
    ):
        pass

    @abstractmethod
    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the provided :py:class:`~saetass.state.State` by a specified number of internal steps.

        Parameters
        ----------
        n_steps : int
            The number of internal substeps to advance based on :py:attr:`~saetass.solver.SubSolver.t_grid`.
        state : :py:class:`~saetass.state.State`
            The global tracking :py:class:`~saetass.state.State` to be updated.
        """
        pass

    def characteristic_scales(self, t: float, dt: float) -> CharacteristicScales:
        """
        Characteristic timescale and dimensionless numbers of the operator, used by :py:mod:`saetass.diagnostics`.

        Operators without meaningful scales keep this default, which reports none.

        Parameters
        ----------
        t : float
            Time at which time-dependent parameters are evaluated, as a bare float in canonical :py:data:`~saetass.units.TIME` units.
        dt : float
            Time step integrated by the operator in each call, as a bare float in canonical :py:data:`~saetass.units.TIME` units.

        Returns
        -------
        :py:class:`~saetass.diagnostics.CharacteristicScales`
            The characteristic scales of the operator.
        """
        return CharacteristicScales()


def _canonical_callable(
    fn: Callable[..., u.Quantity],
    unit: u.UnitBase,
    coords: tuple[u.Quantity | None, ...],
    name: str,
) -> Callable[[float], Any]:
    """Wrap a physical callable into ``f(t: float) -> bare floats in unit``."""

    def numeric(t: float) -> Any:
        return su.validate_quantity(
            fn(*coords, t * su.TIME), unit, f"{name} (callable)"
        ).value

    return numeric


def _describe_parameter(value: Any, unit: u.UnitBase | None) -> str:
    """Short description of a converted parameter for the debug log: its kind, shape and range."""
    unit_text = "" if unit is None or unit == u.dimensionless_unscaled else f" {unit}"
    if isinstance(value, CanonicalSeparable):
        return f"separable, {plural(len(value.profiles), 'component')}{unit_text}"
    if callable(value):
        return f"time-dependent callable{unit_text}"
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        return repr(value)
    if array.size == 1:
        return f"{array.item():.4g}{unit_text}"
    return (
        f"array of shape {array.shape} in [{np.min(array):.4g}, {np.max(array):.4g}]"
        f"{unit_text}"
    )


from .solvers.advection_solver import AdvectionSolver  # noqa: E402
from .solvers.diffusion_solver import DiffusionSolver  # noqa: E402
from .solvers.loss_solver import LossSolver  # noqa: E402
from .solvers.source_solver import SourceSolver  # noqa: E402

logger = logging.getLogger(__name__)


class OperatorType(StrEnum):
    """Auxiliary class for correct operator type handling.

    .. note::
       Currently, the supported operator types are: "advection", "diffusion", "loss" and "source".


    Parameters
    ----------
    operator_type : str
        String identifier for the operator type (e.g., "advection", "diffusion"). This
        will raise a ``ValueError`` if an unsupported operator type is provided.
    """

    ADVECTION = ("advection", AdvectionSolver)
    DIFFUSION = ("diffusion", DiffusionSolver)
    LOSS = ("loss", LossSolver)
    SOURCE = ("source", SourceSolver)

    def __new__(cls, operator_type: str, solver_class: type):
        obj = str.__new__(cls, operator_type)
        obj._value_ = operator_type
        obj.solver_class = solver_class
        return obj


class Solver:
    """
    Main class to manage the simulation workflow. It initializes operators based on the specified problem type and coordinates the time advancement of the solution.

    In the initialization phase, the :py:class:`~saetass.solver.Solver` class takes in the initial :py:class:`~saetass.state.State`, whose :py:class:`~saetass.grid.Grid` defines the cells and the time grid of the simulation, the problem type, operator parameters, substep counts and splitting scheme.
    It parses the problem type to determine which operators are involved and initializes them with their own refined time grids based on the specified splitting type.

    Any object of the :py:class:`~saetass.solver.Solver` class exposes two main methods: :py:meth:`~saetass.solver.Solver.step` for advancing the solution by some amount of time steps and :py:meth:`~saetass.solver.Solver.run` for running the entire simulation.
    After each advancement, :py:class:`~saetass.solver.Solver` updates the :py:class:`~saetass.state.State` object associated with the solution.

    Parameters
    ----------
    state : :py:class:`~saetass.state.State`
        :py:class:`~saetass.state.State` object containing the initial distribution. Its :py:class:`~saetass.grid.Grid`, :py:attr:`~saetass.solver.Solver.grid`, provides the spatial and momentum cells and the time grid of the simulation.
    problem_type : str
        String specifying the type of problem and which operators to include. Valid operators are defined in
        :py:class:`~saetass.solver.OperatorType` (e.g., "advection-diffusion-loss-source"). The order of the operators will affect the behaviour of :py:class:`~saetass.splitting.SplittingScheme`.
    operator_params : dict, optional
        Dictionary mapping operator names to their specific parameters (e.g., ``{"advection": {...}, "diffusion": {...}}``).
        These parameters will be passed to the corresponding subsolvers during initialization and should be structured accordingly.
        For further details on expected parameters for each operator, refer to the documentation of the respective subsolver classes.
    substeps : dict, optional
        Dictionary specifying the number of substeps for each operator (e.g., ``{"advection": 2, "diffusion": 1}``).
        Default is ``None``, i.e. one substep per operator.
    splitting_scheme : str or :py:class:`~saetass.splitting.SplittingSchemeType`, optional
        String or :py:class:`~saetass.splitting.SplittingSchemeType` specifying the :py:class:`~saetass.splitting.SplittingScheme` to use. Valid schemes are defined in
        :py:class:`~saetass.splitting.SplittingSchemeType` (e.g., ``"strang"``, ``"lie"``). Default is ``"strang"``.
    quiet : bool or None, optional
        Whether to hide the banner and the progress bar. Default is ``None``, meaning :py:func:`~saetass.cli.output.is_quiet`: shown in the main process and hidden in child processes, unless :py:func:`~saetass.cli.output.set_quiet` or ``SAETASS_QUIET`` say otherwise.
    """

    def __init__(
        self,
        state: State,
        problem_type: str,
        operator_params: dict = None,
        substeps: dict = None,
        splitting_scheme: str = None,
        quiet: bool | None = None,
        **kwargs,
    ):
        self.quiet = is_quiet() if quiet is None else bool(quiet)
        if not self.quiet:
            print_banner()

        if not isinstance(state, State):
            raise TypeError(
                f"Solver takes the initial State as first argument, not "
                f"{type(state).__name__}: the Grid of the simulation is that of the State."
            )
        self.state = state
        #: :py:class:`~saetass.grid.Grid` of the simulation, that of :py:attr:`state`.
        self.grid = state.grid
        self.problem_type = problem_type.lower()

        # Store operator parameters (nested dict)
        self.operator_params = operator_params or {}
        self.substeps = substeps or {}

        # Determine which operators are present and print for debugging

        # Parse operators from problem_type, handling exact operator names split by '-'
        self.operator_list = []
        ops_in_type = [op.strip().lower() for op in self.problem_type.split("-")]

        for op in ops_in_type:
            # This implicitly validates standard operator strings (raises ValueError if invalid)
            operator_type = OperatorType(op)
            self.operator_list.append(operator_type)

        self.n_os = len(self.operator_list)

        if self.n_os == 0:
            raise ValueError("No valid operators specified in problem_type.")

        # Substeps for each operator (default 1)
        self.substeps_per_op = {
            op: self.substeps.get(op, 1) for op in self.operator_list
        }

        self.global_step = 0
        self.total_steps = self.grid.num_timesteps

        # Initialize splitting scheme
        if splitting_scheme is None:
            splitting_scheme = "strang"
        self.splitting_scheme = create_splitting_scheme(splitting_scheme)

        # Prepare subsolvers using the mapping
        self.operator_subsolvers = []
        self._initialize_subsolvers(**kwargs)

        logger.info(
            "Solver: %s with %s splitting, %s on %s",
            "-".join(op.value for op in self.operator_list),
            type(self.splitting_scheme).__name__.removesuffix("Splitting"),
            self.state.particle.value,
            self.grid.summary(),
        )

    def _initialize_subsolvers(self, **kwargs):
        """Initialize subsolvers with appropriate t_grids and canonical numeric parameters."""
        # Grid enforces Quantity time grids: this is the single unit-stripping point for time.
        t_grid_canonical = self.grid.t_grid.to_value(su.TIME)
        refined_t_grids = self.splitting_scheme.initialize_t_grid(
            self.operator_list, self.substeps_per_op, t_grid_canonical
        )

        logger.debug(
            "Time grid points per operator: %s",
            {op.value: len(refined_t_grids[op]) for op in self.operator_list},
        )

        for op in self.operator_list:
            solver_class = op.solver_class
            op_params = solver_class.convert_params(
                self._operator_params_of(op), self.grid
            )

            self.operator_subsolvers.append(
                solver_class(
                    self.grid,
                    refined_t_grids[op],
                    op_params,
                    **kwargs,
                )
            )

            if logger.isEnabledFor(logging.DEBUG):
                for key, value in op_params.items():
                    logger.debug(
                        "%s parameter %s: %s",
                        solver_class.__name__,
                        key,
                        _describe_parameter(value, solver_class.PARAM_SPECS[key].unit),
                    )

    def _operator_params_of(self, op: OperatorType) -> dict:
        """
        Parameters given by the user for the operator ``op``.

        The adiabatic losses take the velocity of the advection operator when they are given none, so that both operators see the same flow.
        """
        params = self.operator_params.get(op.value, {})
        advection = self.operator_params.get(OperatorType.ADVECTION.value, {})
        if (
            op is OperatorType.LOSS
            and params.get("adiabatic_losses", False)
            and "v_centers_physical" not in params
            and OperatorType.ADVECTION in self.operator_list
            and "v_centers" in advection
        ):
            logger.debug("Adiabatic losses use the velocity of the advection operator")
            params = {**params, "v_centers_physical": advection["v_centers"]}
        return params

    def _advance(self, n_steps):
        """Advance the solution by n_steps using operator splitting."""
        manage_progress = False
        if not self.quiet and getattr(self, "_progress", None) is None:
            self._progress = create_progress_bar()
            self._progress.start()
            self._task_id = self._progress.add_task(
                "[bold cyan]Solving...",
                total=self.total_steps,
                completed=self.global_step,
                metrics="max=0.0 min=0.0",
            )
            manage_progress = True

        next_metrics = 0.0
        try:
            for _ in range(n_steps):
                self.global_step += 1

                if not self.quiet:
                    # The bar redraws a few times per second: computing the extrema of the
                    # state for every step would cost more than a step on small grids
                    update = {"completed": self.global_step}
                    now = time.monotonic()
                    if now >= next_metrics:
                        next_metrics = now + _PROGRESS_METRICS_INTERVAL
                        val_max = np.max(self.state._values)
                        val_min = np.min(self.state._values)
                        update["metrics"] = f"max={val_max:.4g} min={val_min:.4g}"
                    self._progress.update(self._task_id, **update)

                self.splitting_scheme.apply(
                    self.operator_list,
                    self.operator_subsolvers,
                    self.substeps_per_op,
                    self.state,
                )
        finally:
            # Also on errors: a live bar left running would swallow later output
            if manage_progress or self.global_step >= self.total_steps:
                if getattr(self, "_progress", None) is not None:
                    self._progress.stop()
                    self._progress = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Advance finished at step {self.global_step}/{self.total_steps}, "
                f"t = {self.state.t_val:.4g} {su.TIME} | "
                f"max(psi)={np.max(self.state._values):.4g} min(psi)={np.min(self.state._values):.4g}"
            )

    def run(self) -> State:
        """Advances the solution to the final time, updating and returning the final :py:class:`~saetass.state.State` object.

        Returns
        -------
        State
            The final state of the solution after advancing to the final time.
        """
        start = time.perf_counter()
        steps = self.total_steps - self.global_step
        self._advance(steps)
        elapsed = time.perf_counter() - start
        subcycles = {
            op.value: sub.cfl_subcycles_total
            for op, sub in zip(self.operator_list, self.operator_subsolvers)
            if hasattr(sub, "cfl_subcycles_total")
        }
        logger.info(
            "Solver finished %s in %.3g s (%.3g steps/s), at t = %.4g %s%s",
            plural(steps, "step"),
            elapsed,
            steps / elapsed if elapsed > 0.0 else float("inf"),
            self.state.t_val,
            su.TIME,
            f"; CFL sub-cycles: {subcycles}" if subcycles else "",
        )
        return self.state

    def step(self, n_steps=1) -> State:
        """Advances the solution by n_steps global time steps, updating and returning the current :py:class:`~saetass.state.State` object.

        Parameters
        ----------
        n_steps : int, optional
            The number of global time steps to advance. Default is ``1``.
        """
        self._advance(n_steps)
        return self.state

    def diagnostics(
        self, n_samples: int = 3, warn: bool = True
    ) -> SimulationDiagnostics:
        """
        Analyze the characteristic scales of the simulation to detect time steps that compromise its accuracy.

        See :py:mod:`saetass.diagnostics` for the quantities reported.

        Parameters
        ----------
        n_samples : int, optional
            Number of times, evenly spread over the time grid, at which time-dependent parameters are evaluated. Default is ``3``.
        warn : bool, optional
            Whether to log a warning for each problem found. Default is ``True``.

        Returns
        -------
        :py:class:`~saetass.diagnostics.SimulationDiagnostics`
            The diagnostics of the simulation.
        """
        result = diagnose(self, n_samples=n_samples)
        if logger.isEnabledFor(logging.DEBUG):
            # The warnings are logged on their own below
            logger.debug("%s", result.report(include_warnings=not warn))
        if warn:
            _warn(result)
        return result
