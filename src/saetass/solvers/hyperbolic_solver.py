r"""
The :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver` class implements a finite volume method for solving hyperbolic PDEs of the general form

.. math::

    \frac{\partial U}{\partial t} + \frac{\partial}{\partial y}\big(V(t,y)\, U\big) = 0,

where :math:`V(t,y)` is a generalized velocity that can depend on time and on the variable :math:`y`.
It supports both first-order upwind and second-order MUSCL-Hancock schemes with various slope limiters (minmod, van Leer, MC) to ensure stability and non-oscillatory behavior.
The solver can handle both 1D and 2D problems by treating the secondary axis as independent slices.
It automatically computes stable time steps based on the CFL condition and updates the associated :py:class:`~saetass.state.State` object accordingly.
The finite volume update itself runs in the fused, compiled kernels of :py:mod:`~saetass.solvers.hyperbolic_kernels`, one pass over each row of the grid per CFL sub-cycle.

This class serves as a flexible base for specific hyperbolic problems, such as advection or loss terms, by defining the appropriate generalized variable transformations and velocity fields.
Thus, :py:class:`~saetass.solvers.advection_solver.AdvectionSolver` and :py:class:`~saetass.solvers.loss_solver.LossSolver` inherit from this base class and implement the problem-specific logic while reusing the core finite volume update mechanism.
"""

import logging
import math
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Literal

import numpy as np

from ..diagnostics import CharacteristicScales
from ..grid import Grid
from ..parallel import reset_team, threads_for
from ..parameters import CanonicalSeparable
from ..solver import ParamSpec, SubSolver
from ..state import State
from . import hyperbolic_kernels as kernels

logger = logging.getLogger(__name__)

#: Fraction of a step below which the remaining time of a CFL sub-cycling loop is rounding residue.
_TIME_TOLERANCE = 1e-12

#: Factor on the work a single sub-cycle of time-dependent velocities needs to use threads.
#: Each such call alternates with serial NumPy work (the user callable, the interpolation
#: to faces, the CFL step), which runs slower in a process whose threads just ran; with
#: short parallel calls, threads made these runs slower than serial ones.
_DYNAMIC_PARALLEL_FACTOR = 10.0


class HyperbolicSolver(SubSolver, ABC):
    """
    Base class for hyperbolic PDE subsolvers using finite volume methods.

    It exposes a public API for advancing the solution in time and relies on subclasses to define problem-specific transformations and velocity fields.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        A grid object containing both spatial and momentum grids.
    t_grid : numpy.ndarray
        Time grid for integration, in canonical :py:data:`~saetass.units.TIME` units.
        In the standard SAETASS workflow, this is typically subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration as bare floats in canonical units. Accepted keys are:

        V_centers : numpy.ndarray or callable
            Generalized velocities at cell centers. Shape must match
            grid dimensions. A callable must have signature ``V_centers(t: float) -> numpy.ndarray``.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL (Courant-Friedrichs-Lewy) number for stable time
            step calculation.
        inflow_value_U : float or numpy.ndarray
            Value of the conservative variable :math:`U` at the outer boundary
            for inflow conditions.
        order : ``{1, 2}``
            Order of the numerical scheme:
            * 1: First-order upwind.
            * 2: Second-order MUSCL-Hancock.
        axis : ``{0, 1}``
            Integer indicating the main axis of the problem:
            * 0: Momentum axis.
            * 1: Spatial axis.
    """

    #: Numerical options shared by all hyperbolic subsolvers (extended by subclasses).
    PARAM_SPECS = MappingProxyType(
        {"limiter": ParamSpec(), "cfl": ParamSpec(), "order": ParamSpec()}
    )

    #: Whether slopes are limited on :math:`U / r^2` instead of :math:`U` (spherical advection).
    _SPHERICAL_SLOPES = False

    #: How the kernels convert between the distribution and the conservative variable
    #: :math:`U`, a ``TRANSFORM_*`` code of :py:mod:`~saetass.solvers.hyperbolic_kernels`.
    #: :py:class:`~saetass.solvers.advection_solver.AdvectionSolver` (:math:`U = r^2 \psi`)
    #: and :py:class:`~saetass.solvers.loss_solver.LossSolver` (:math:`U = p\,\psi`) set their
    #: own code, so that the kernels convert as they read and write the distribution. The
    #: default, ``TRANSFORM_NONE``, is for other subclasses, whose conservative variable
    #: the kernels do not know: :py:meth:`advance` then converts with
    #: :py:meth:`_generalized_variable` and :py:meth:`_inverse_generalized_variable`.
    _KERNEL_TRANSFORM = kernels.TRANSFORM_NONE

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict,
        **kwargs,
    ) -> None:
        self._validate_unit_free_inputs(t_grid, params)
        # Size the threads of the kernels by the calls of this simulation
        reset_team()
        self._unpack_params(params)
        self._unpack_grid(grid)
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.params = params or {}
        self._init_kernel_arguments()
        #: CFL sub-cycles performed by the last call to advance().
        self._cfl_subcycles = 0
        #: Velocities with a separable time dependence, if any (see :py:mod:`saetass.parameters`).
        self._separable = None

        # Determine if V_centers is static or dynamic
        V_input = params.get("V_centers")
        if callable(V_input):
            self.is_V_dynamic = True

            # Helper to fetch and orient V_centers properly
            def _get_V_centers_dynamic(t):
                V = np.asarray(V_input(t), dtype=float)
                if self.axis == 0:
                    V = V.T
                return V

            self._get_V_centers = _get_V_centers_dynamic

            # Provide an initial value just for shape/metrics purposes
            self.V_centers = self._get_V_centers(self.t_grid[0])
            if isinstance(V_input, CanonicalSeparable):
                self._init_separable(V_input)

        else:
            self.is_V_dynamic = False
            self.V_centers = np.asarray(V_input, dtype=float)
            if self.axis == 0:
                self.V_centers = self.V_centers.T

            # Caching strategy for static case
            self.V_centers_static = self.V_centers
            self.V_faces_static = self._face_generalized_velocity_interpolated(
                self.V_centers
            )
            self._get_V_centers = lambda t: self.V_centers_static
            self._dt_cfl_static = float(self._compute_dt(V_faces=self.V_faces_static))
            self._V_centers_rows = self._as_rows(self.V_centers, self._rows_shape)
            self._V_faces_rows = self._as_rows(self.V_faces_static, self._faces_shape)

    @property
    def _main_centers(self) -> np.ndarray:
        """Return the main axis centers based on axis."""
        if self.axis == 0:
            return self.p_centers
        elif self.axis == 1:
            return self.r_centers
        else:
            raise ValueError("Invalid axis for main axis.")

    @property
    def _main_faces(self) -> np.ndarray:
        """Return the main axis faces based on axis."""
        if self.axis == 0:
            return self.p_faces
        elif self.axis == 1:
            return self.r_faces
        else:
            raise ValueError("Invalid axis for main axis.")

    @property
    def _other_axis(self) -> int:
        """Return the non-active axis index (0 or 1)."""
        return 1 - self.axis

    def advance(self, n_steps: int, state: State) -> None:
        r"""
        Advance the :py:class:`~saetass.state.State` by ``n_steps`` in :py:attr:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver.t_grid`.

        The inner loop may subdivide each requested step further to satisfy the CFL stability condition, so the number of actual flux evaluations can exceed ``n_steps``.
        The total elapsed time is always exactly ``n_steps * dt``, where ``dt`` is the time step size inferred from :py:attr:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver.t_grid`.

        .. note::
            A positivity floor :math:`\psi \geq 0` is enforced after the update to suppress negatives that may arise at steep gradient fronts when using the MUSCL-Hancock scheme.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        state : :py:class:`~saetass.state.State`
            Current simulation state. The differential density is updated in-place at the end of the call.
        """
        values = np.asarray(state._get_values(), dtype=float)
        if self._KERNEL_TRANSFORM == kernels.TRANSFORM_NONE:
            U = self._generalized_variable(
                values.T if self.axis == 0 else values, self.grid
            )
            U_rows = np.ascontiguousarray(U).reshape(self._rows_shape)
            values_in = values_out = U_rows  # unused by the kernels
        else:
            # The kernels read and write the rows of the state directly
            values_new = np.empty_like(values)
            values_in, values_out = self._rows(values), self._rows(values_new)
            U_rows = np.empty(self._rows_shape)

        total_time = float(n_steps) * self._dt
        t_local = state.t_val
        # Remaining time below this relative tolerance is rounding residue, not a step to take
        time_tolerance = _TIME_TOLERANCE * total_time
        kernel_rows = (values_in, U_rows, values_out)

        if self._separable is not None:
            # Separable velocities: the factors of every sub-cycle first, then all the
            # sub-cycles in a single kernel call that combines the components
            steps, factors = [], []
            while total_time > time_tolerance:
                values_k = self._separable.factor_values(t_local)
                largest = kernels.separable_largest_speed(
                    values_k, self._separable_faces, self._separable_last
                )
                dt_cfl = self._cfl_dx / largest if largest > 0.0 else np.inf
                dt_step = min(total_time, dt_cfl)
                steps.append(dt_step)
                factors.append(values_k)
                total_time -= dt_step
                t_local += dt_step
            n_components = self._separable_centers.shape[0]
            self._run_kernel(
                kernel_rows,
                self._separable_centers,
                self._separable_faces,
                np.array(steps),
                load=True,
                store=True,
                factors=np.array(factors).reshape(len(steps), n_components),
                combine=True,
            )
            self._cfl_subcycles = len(steps)
        elif self.is_V_dynamic:
            # The velocities change with time: one kernel call per sub-cycle, loading the
            # distribution in the first and storing it in the last
            self._cfl_subcycles = 0
            while total_time > time_tolerance:
                self.V_centers = self._get_V_centers(t_local)
                V_rows = self._as_rows(self.V_centers, self._rows_shape)
                # Faces and CFL step in one compiled pass, with the arithmetic of
                # _face_generalized_velocity_interpolated and _compute_dt
                largest = kernels.faces_and_max(
                    V_rows, *self._face_weights, self._faces_buffer
                )
                largest = max(largest, self._last_center_speed(self.V_centers))
                dt_cfl = self._cfl_dx / largest if largest > 0.0 else np.inf
                dt_step = min(total_time, dt_cfl)
                self._run_kernel(
                    kernel_rows,
                    V_rows,
                    self._faces_buffer,
                    np.array([dt_step]),
                    load=self._cfl_subcycles == 0,
                    store=total_time - dt_step <= time_tolerance,
                    work_factor=_DYNAMIC_PARALLEL_FACTOR,
                )
                total_time -= dt_step
                t_local += dt_step
                self._cfl_subcycles += 1
            if self._cfl_subcycles == 0:
                self._run_kernel(
                    kernel_rows,
                    self._as_rows(self.V_centers, self._rows_shape),
                    self._as_rows(
                        self._face_generalized_velocity_interpolated(self.V_centers),
                        self._faces_shape,
                    ),
                    np.empty(0),
                    load=True,
                    store=True,
                )
        else:
            # Static velocities: every sub-cycle in a single kernel call
            steps = []
            while total_time > time_tolerance:
                dt_step = min(total_time, self._dt_cfl_static)
                steps.append(dt_step)
                total_time -= dt_step
            self._run_kernel(
                kernel_rows,
                self._V_centers_rows,
                self._V_faces_rows,
                np.array(steps),
                load=True,
                store=True,
            )
            self._cfl_subcycles = len(steps)

        if self._KERNEL_TRANSFORM == kernels.TRANSFORM_NONE:
            U = U_rows.reshape(U.shape)
            values_new = self._inverse_generalized_variable(U, self.grid)
            # Positivity floor: MUSCL-Hancock is TVD but not strictly positive-definite;
            # clip machine-precision negatives that arise at steep gradient fronts.
            values_new = np.maximum(values_new, 0.0)
            if self.axis == 0:
                values_new = values_new.T
        state._update_values(values_new)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"max(|values|) after step: {np.max(np.abs(values_new)):.4g}")

    # ---------------- internal helpers ----------------
    def _init_kernel_arguments(self) -> None:
        """Arrays and options of the fused kernels that do not change between calls."""
        rows = 1 if self.M is None else self.M
        self._rows_shape = (rows, self.N)
        # Weights of the interpolation of the velocities to the internal faces, which
        # run every sub-cycle when the velocities depend on time
        dist_left, dist_right = self.dx_L[1:], self.dx_R[:-1]
        denom = dist_left + dist_right
        self._face_weights = (
            dist_left,
            dist_right,
            np.where(denom == 0.0, 1.0, denom),
        )
        self._faces_shape = (rows, max(self.N - 1, 0))
        # Faces of time-dependent velocities, refilled every sub-cycle
        self._faces_buffer = np.empty(self._faces_shape)
        self._cfl_dx = float(self.cfl * np.min(self.dx))
        centers = np.ascontiguousarray(self._main_centers)
        if self._SPHERICAL_SLOPES:
            r, r2 = centers, centers**2
        else:
            r = r2 = np.empty(0)
        scale, inv_scale = self._kernel_scales()
        self._kernel_geometry = (
            centers,
            np.ascontiguousarray(self._main_faces),
            kernels.reciprocal_spacings(
                centers,
                np.asarray(self.dx, dtype=float),
                r2 if self._SPHERICAL_SLOPES else None,
            ),
            np.ascontiguousarray(self.dx_L),
            np.ascontiguousarray(self.dx_R),
            r,
            r2,
            np.ascontiguousarray(scale, dtype=float),
            np.ascontiguousarray(inv_scale, dtype=float),
        )
        if self.axis == 1 and self._main_faces[0] <= 0.0:
            self._left_mode = kernels.LEFT_ZERO_FLUX  # symmetry at r = 0
        elif self.axis == 0:
            self._left_mode = kernels.LEFT_EXTRAPOLATED  # momentum inflow
        else:
            self._left_mode = None
        if self.inflow_value_U.size not in (1, rows):
            raise ValueError(
                f"inflow_value_U must have 1 or {rows} values, got {self.inflow_value_U.size}."
            )

    def _kernel_scales(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
        :math:`U / \psi` along the active axis and its inverse, for the conversion done by the kernels.

        Only used when :py:attr:`_KERNEL_TRANSFORM` is not ``TRANSFORM_NONE``.
        """
        return np.empty(0), np.empty(0)

    def _rows(self, values: np.ndarray) -> np.ndarray:
        """View of the distribution as the rows of the kernels, the active axis last."""
        if values.ndim == 1:
            return values.reshape(self._rows_shape)
        return values.T if self.axis == 0 else values

    @staticmethod
    def _last_center_speed(V_centers: np.ndarray) -> float:
        """
        Largest speed of the last cells along the active axis, the proxy of the outer face in the CFL condition.

        Velocities are kept with the active axis last, whatever :py:attr:`axis` is.
        """
        return float(np.max(np.abs(V_centers[..., -1])))

    def _init_separable(self, V_input: CanonicalSeparable) -> None:
        """Components of separable velocities as kernel rows, at the centers, faces and last cells."""
        components = [np.asarray(V, dtype=float) for V in V_input.profiles]
        if self.axis == 0:
            components = [V.T for V in components]
        self._separable_centers = np.stack(
            [self._as_rows(V, self._rows_shape) for V in components]
        )
        self._separable_faces = np.stack(
            [
                self._as_rows(
                    self._face_generalized_velocity_interpolated(V), self._faces_shape
                )
                for V in components
            ]
        )
        if components[0].ndim == 1:
            last = [V[-1:] for V in components]
        else:
            last = [V[..., -1] for V in components]
        self._separable_last = np.ascontiguousarray(np.stack(last), dtype=float)
        self._separable = V_input

    @staticmethod
    def _as_rows(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Contiguous copy of ``values`` broadcast to the rows of the kernels."""
        return np.ascontiguousarray(np.broadcast_to(values, shape), dtype=float)

    def _run_kernel(
        self,
        kernel_rows: tuple[np.ndarray, np.ndarray, np.ndarray],
        V_centers: np.ndarray,
        V_faces: np.ndarray,
        steps: np.ndarray,
        load: bool,
        store: bool,
        work_factor: float = 1.0,
        factors: np.ndarray | None = None,
        combine: bool = False,
    ) -> None:
        """
        Advance the conservative variable through the sub-cycles ``steps``.

        ``work_factor`` divides the estimated work when choosing the threads, so that a call runs in parallel only if it has ``work_factor`` times the usual minimum work.
        The velocities are the rows of a single component ``(M, N)`` or, with ``combine``, components ``(K, M, N)`` combined with ``factors`` of shape ``(len(steps), K)``.

        ``kernel_rows`` holds the rows of the distribution to load, of :math:`U`, and of the distribution to store; the conversions only happen when the kernels do them (:py:attr:`_KERNEL_TRANSFORM`).
        """
        if self.order == 2 and self._left_mode is None:
            raise ValueError("Invalid left face position for the given axis.")
        values_in, U_rows, values_out = kernel_rows
        transform = self._KERNEL_TRANSFORM
        work_ns = U_rows.size * max(steps.size, 1) * kernels.NS_PER_CELL[self.order]
        work_ns /= work_factor
        n_threads = threads_for(U_rows.shape[0], work_ns)
        # A few blocks of rows per thread balance the threads; serially, a single block
        n_blocks = 1 if n_threads == 1 else min(U_rows.shape[0], 4 * n_threads)
        arguments = (
            values_in,
            U_rows,
            values_out,
            V_centers,
            V_faces,
            *self._kernel_geometry,
            self.inflow_value_U,
            steps,
            self.order,
            kernels.LIMITERS[self.limiter],
            self._left_mode or kernels.LEFT_ZERO_FLUX,
            self._SPHERICAL_SLOPES,
            transform,
            load and transform != kernels.TRANSFORM_NONE,
            store and transform != kernels.TRANSFORM_NONE,
        )
        if combine:
            kernels.advance_rows_combined(n_threads, *arguments, factors, n_blocks)
        else:
            kernels.advance_rows(n_threads, *arguments, n_blocks)

    def _unpack_params(self, params: dict = None) -> None:
        """Unpack and strictly validate parameters."""
        # expected keys
        # expected_keys = {
        #     "V_centers",
        #     "limiter",
        #     "cfl",
        #     "inflow_value_U",
        #     "order",
        #     "axis",
        # }
        # provided_keys = set(params.keys())

        # check missing / extra
        # missing = expected_keys - provided_keys
        # extra = provided_keys - expected_keys
        # if missing:
        #    raise ValueError(f"Missing required params: {missing}")
        # if extra:
        #    raise ValueError(f"Unexpected params provided: {extra}")

        # --- axis ---
        try:
            self.axis: int = int(params["axis"])
        except Exception:
            raise ValueError("axis must be an integer")
        if not (0 <= self.axis < 2):
            raise ValueError(
                "axis must be 0 (for momentum problems) or 1 (for spatial problems)"
            )

        # Note: V_centers is handled locally in __init__ now due to static/callable checks

        # --- limiter ---
        limiter = params["limiter"]
        if limiter not in {"minmod", "vanleer", "mc"}:
            raise ValueError("limiter must be 'minmod', 'vanleer' or 'mc'")
        self.limiter: Literal["minmod", "vanleer", "mc"] = limiter

        # --- cfl ---
        try:
            self.cfl: float = float(params["cfl"])
        except Exception:
            raise ValueError("cfl must be a float")

        # --- inflow_value_U ---
        self.inflow_value_U = params["inflow_value_U"]
        # convert to 1D plain row array
        self.inflow_value_U = np.asarray(self.inflow_value_U, dtype=float).flatten()

        # --- order ---
        order = int(params["order"])
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        self.order: Literal[1, 2] = order

    def _unpack_grid(self, grid: Grid):
        """Unpack and validate grid object."""
        self.grid = grid

        def _load_axis(name: str) -> bool:
            """Loads an axis ('r' or 'p') from the grid and returns True if it exists."""
            centers = getattr(grid, f"{name}_centers", None)
            faces = getattr(grid, f"{name}_faces", None)

            if centers is not None:
                centers = np.asarray(centers, dtype=float)
                faces = np.asarray(faces, dtype=float)
                setattr(self, f"{name}_centers", centers)
                setattr(self, f"{name}_faces", faces)
                logger.info(f"Using {name}-grid with {len(centers)} points")
                return True
            else:
                setattr(self, f"{name}_centers", None)
                setattr(self, f"{name}_faces", None)
                return False

        def _init_axis_metrics(name: str):
            """Initialize mesh metrics for the main axis."""
            centers = getattr(self, f"{name}_centers")
            faces = getattr(self, f"{name}_faces")
            self.N = len(centers)
            dx = getattr(grid, f"d{name}", None)  # for compatibility (dr or dp)
            self.dx = np.asarray(dx, dtype=float) if dx is not None else None
            self.dx_c = centers[1:] - centers[:-1]  # Δ between centers
            self.dx_R = faces[1:] - centers  # distance center to right face
            self.dx_L = centers - faces[:-1]  # distance center to left face

        # --- main logic ---
        axes = {0: "p", 1: "r"}
        main_axis = axes.get(self.axis)

        if main_axis is None:
            raise ValueError(
                f"Invalid axis {self.axis}: expected 0 (momentum) or 1 (spatial)"
            )

        if not _load_axis(main_axis):
            raise ValueError(
                f"Grid must include {main_axis}_centers for {main_axis}-space problems"
            )

        _init_axis_metrics(main_axis)

        other_axis = "p" if main_axis == "r" else "r"
        if _load_axis(other_axis):
            logger.info(f"Found {other_axis}-grid")
            self.M = len(getattr(self, f"{other_axis}_centers"))
        else:
            logger.info(f"No {other_axis}-grid found, operating in 1D mode")
            self.M = None

    def _face_generalized_velocity_interpolated(
        self, V_centers: np.ndarray
    ) -> np.ndarray:
        """
        Linear interpolation of generalized velocities from cell centers to internal face positions x_{i+1/2}.
        Uses distance-weighted linear interpolation (works for non-uniform grids).

        Returns
        -------
        V_face : numpy.ndarray
            Interpolated generalized velocities at faces, length N-1.
        """
        if self.N <= 1:
            return np.zeros(0)

        # Distances from each internal face to its two centers, broadcast over the rows
        dist_left, dist_right, denom = self._face_weights
        V_left = V_centers[..., :-1]
        V_right = V_centers[..., 1:]

        # Linear interpolation: closer center gets higher weight
        return (V_left * dist_right + V_right * dist_left) / denom

    def _compute_dt(self, V_faces: np.ndarray, V_centers: np.ndarray = None) -> float:
        """
        Compute a stable time step based on the CFL condition:

            dt = cfl * min(dx) / max(|V|)

        Works for both 1D and 2D problems, with the active axis last in ``V_faces`` and ``V_centers``.
        """
        if V_centers is None:
            V_centers = self.V_centers

        # Take absolute max over all entries (works for scalar, 1D, or 2D)
        if V_faces.size:
            Vmax = float(np.max(np.abs(V_faces)))
        else:
            Vmax = 0.0

        # Include the outermost face, using the last cells along the active axis as proxy
        Vmax = max(Vmax, self._last_center_speed(V_centers))

        # If velocity is zero everywhere → infinite dt
        if Vmax <= 0.0:
            logger.debug("Max velocity is zero, returning infinite dt_cfl")
            return np.inf

        dx_min = np.min(self.dx)

        return float(self.cfl * dx_min / Vmax)

    def velocities(self, t: float) -> np.ndarray:
        """
        Generalized velocities at cell centers at time ``t``, without modifying the solver state.

        Parameters
        ----------
        t : float
            Time, as a bare float in canonical :py:data:`~saetass.units.TIME` units.

        Returns
        -------
        numpy.ndarray
            Generalized velocities :math:`V` at cell centers, in canonical units.
        """
        return self._get_V_centers(t)

    def characteristic_scales(self, t: float, dt: float) -> CharacteristicScales:
        """
        Courant number and CFL sub-cycles per call, and the timescale defined by the subclass.

        See :py:meth:`~saetass.solver.SubSolver.characteristic_scales`.
        """
        V_centers = self.velocities(t)
        dt_cfl = self._compute_dt(
            self._face_generalized_velocity_interpolated(V_centers), V_centers
        )
        if np.isfinite(dt_cfl):
            courant = self.cfl * dt / dt_cfl
            # Same count as the sub-cycling loop in advance(), which ignores rounding residues
            subcycles = max(1, math.ceil(dt * (1.0 - _TIME_TOLERANCE) / dt_cfl))
        else:
            courant, subcycles = 0.0, 1
        return CharacteristicScales(
            timescale=self._characteristic_timescale(V_centers),
            numbers={"courant_number": courant, "cfl_subcycles": float(subcycles)},
        )

    def _characteristic_timescale(self, V_centers: np.ndarray) -> float | None:
        """Characteristic timescale of the operator for the given velocities, if it has one."""
        return None

    @abstractmethod
    def _generalized_variable(self, f: np.ndarray, centers: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _inverse_generalized_variable(
        self, U: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        pass
