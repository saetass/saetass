import logging
from types import MappingProxyType

import numpy as np

from .. import units as su
from ..cli.output import plural
from ..diagnostics import CharacteristicScales
from ..grid import Grid
from ..solver import ParamSpec, SubSolver
from ..state import State

logger = logging.getLogger(__name__)

#: Evaluations of a time-dependent source per step used to estimate its variation.
_VARIATION_SAMPLES = 5


class SourceSolver(SubSolver):
    r"""
    Explicit Euler operator for a source term, inheriting from :py:class:`~saetass.solver.SubSolver`.

    Advances the differential density according to:

    .. math::

        \frac{\partial \psi}{\partial t} = Q(t, r, p),

    using a single first-order explicit Euler step over the total requested
    time ``n_steps * dt``.

    Because the source operator does not involve any spatial derivatives, no CFL condition applies and the entire ``n_steps * dt`` interval is consumed in one evaluation of :math:`Q`.
    The source function :math:`Q` may be time-dependent or a fixed array.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` providing ``r_centers`` and/or ``p_centers`` depending on the problem dimension.
    t_grid : numpy.ndarray
        Subproblem time grid.
        In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        source : astropy.units.Quantity or callable
            Source term, :math:`Q`. Units compatible with :py:data:`~saetass.units.SOURCE_PSI_P`.
            If callable, the signature must be ``source(r, p, t) -> Quantity``, where ``r`` and ``p`` are the physical cell-center coordinates (either may be ``None`` for 1D problems) and ``t`` is the time, all as Quantities.
            Its shape must match the :py:class:`~saetass.state.State` differential density array.
    """

    PARAM_SPECS = MappingProxyType(
        {"source": ParamSpec(su.SOURCE_PSI_P, dynamic=True, coords=True, required=True)}
    )

    def __init__(self, grid: Grid, t_grid: np.ndarray, params: dict, **kwargs):
        self._validate_unit_free_inputs(t_grid, params)
        self.grid = grid
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.params = params or {}

        source_input = self.params.get("source", None)

        if source_input is None:
            raise ValueError("A source function or array must be provided.")

        if callable(source_input):
            self.is_source_dynamic = True

            # Solver has already bound the physical coordinates: source_input(t) -> ndarray
            def _get_source_dynamic(t):
                return np.asarray(source_input(t), dtype=float)

            self._get_source = _get_source_dynamic
        else:
            self.is_source_dynamic = False
            self.source_static = np.asarray(source_input, dtype=float)
            self._get_source = lambda t: self.source_static

        logger.debug(
            "SourceSolver: %s source, time step %.4g %s",
            "time-dependent" if self.is_source_dynamic else "static",
            self._dt,
            su.TIME,
        )

    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the :py:class:`~saetass.state.State` by ``n_steps`` in :py:attr:`~saetass.solvers.source_solver.SourceSolver.t_grid`.

        Applies a single explicit Euler step with total time ``total_dt = n_steps * dt``.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        state : :py:class:`~saetass.state.State`
            Current simulation state. The differential density is updated in-place at the end of the call.
        """
        total_dt = float(n_steps) * self._dt

        # Process source term efficiently
        if self.is_source_dynamic:
            S = self._get_source(state.t_val)
        else:
            S = self.source_static

        # Ensure shape compatibility
        if S.shape != state._get_values().shape:
            raise ValueError(
                f"Source shape {S.shape} does not match state shape {state._values.shape}"
            )

        # Explicit update
        values_new = state._values + total_dt * S
        state._update_values(values_new)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SourceSolver: advanced by %s, %.4g %s",
                plural(n_steps, "step"),
                total_dt,
                su.TIME,
            )

    def characteristic_scales(self, t: float, dt: float) -> CharacteristicScales:
        r"""
        Variation timescale of a time-dependent source, :math:`\tau_Q = \max |Q| / \max |\partial_t Q|`.

        Each call injects :math:`Q` evaluated at the start of the step, so its error is controlled by how much :math:`Q` changes within the step, not by its magnitude: since the transport equation is linear, scaling :math:`Q` only scales the part of the solution it produces.
        The time derivative is estimated by finite differences over quarters of the step, and ``source_variation`` is :math:`\Delta t / \tau_Q`, the relative change of the source over one step.
        Static sources report no scales.
        See :py:meth:`~saetass.solver.SubSolver.characteristic_scales`.
        """
        if not self.is_source_dynamic:
            return CharacteristicScales()
        times = t + dt * np.linspace(0.0, 1.0, _VARIATION_SAMPLES)
        samples = np.stack([self._get_source(time) for time in times])
        magnitude = float(np.max(np.abs(samples)))
        change = float(np.max(np.abs(np.diff(samples, axis=0)), initial=0.0))
        # Relative change per step, extrapolated from the largest change between samples
        variation = (
            change / magnitude * (_VARIATION_SAMPLES - 1) if magnitude > 0.0 else 0.0
        )
        return CharacteristicScales(
            timescale=dt / variation if variation > 0.0 else np.inf,
            numbers={"source_variation": variation},
        )
