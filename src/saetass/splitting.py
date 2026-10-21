"""
The splitting module defines the operator splitting schemes used by the simulation.

SAETASS uses operator splitting schemes to advance the state of the system in time by
combining different physical operators (e.g. advection, diffusion, sources, losses).
The splitting module provides the abstract base class :py:class:`~saetass.splitting.SplittingScheme`
and concrete implementations such as :py:class:`~saetass.splitting.StrangSplitting` and
:py:class:`~saetass.splitting.LieSplitting`.

These schemes coordinate with the main :py:class:`~saetass.solver.Solver` to refine time
grids per operator and explicitly order the execution sequence of
:py:class:`~saetass.solver.SubSolver` instances.
"""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np

from . import units as su
from .state import State

logger = logging.getLogger(__name__)


class SplittingScheme(ABC):
    """
    Abstract base class for operator splitting schemes.

    A splitting scheme dictates how a global macro-timestep is divided
    among the constituent physical operators (e.g., advection and diffusion) and in what
    order those operators are applied.

    Subclasses must call ``super().__init__()`` to initialize internal tracking. Furthermore,
    they are contractually required to invoke :py:meth:`~saetass.splitting.SplittingScheme._store_t_grid`
    inside their :py:meth:`~saetass.splitting.SplittingScheme.initialize_t_grid` implementation.
    At the end of every :py:meth:`~saetass.splitting.SplittingScheme.apply` call, they must invoke
    :py:meth:`~saetass.splitting.SplittingScheme._advance_global_time` so that the underlying
    :py:attr:`saetass.state.State.t` scalar always reflects the exact canonical grid value, preventing
    floating-point accumulation drift over millions of substeps.
    """

    def __init__(self):
        self.t_grid: np.ndarray = None
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _store_t_grid(self, t_grid: np.ndarray) -> None:
        """
        Store the global time grid and reset the internal step counter.

        This method must be called by every concrete ``initialize_t_grid`` implementation
        so the scheme tracks exactly which macro-step is currently active.

        Parameters
        ----------
        t_grid : numpy.ndarray
            1D array of global simulation times as bare floats, in canonical :py:data:`~saetass.units.TIME` units.
        """
        su.require_bare(t_grid, "Splitting scheme t_grid")
        self.t_grid = np.asarray(t_grid, dtype=float)
        self._global_step = 0

    def _advance_global_time(self, state: State) -> None:
        """
        Increment the internal step counter and snap the state's time to the grid.

        By using :py:meth:`~saetass.state.State.set_time` via direct assignment rather than accumulating
        `dt` repeatedly, this guarantees that ``state.t`` naturally stays perfectly aligned with the grid array
        and immune to floating-point drift.

        Parameters
        ----------
        state : :py:class:`~saetass.state.State`
            The global distribution tracking state object to mutate.
        """
        self._global_step += 1
        state.set_time(self.t_grid[self._global_step])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize_t_grid(
        self, operator_list: list, substeps_per_op: dict, t_grid: np.ndarray
    ) -> dict:
        """
        Build per-operator refined time grids based on requested sub-stepping configurations.

        Implementations must call ``self._store_t_grid(t_grid)`` to track global time internally.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            The ordered sequence of split operators to simulate.
        substeps_per_op : dict
            Dictionary mapping :py:class:`~saetass.solver.OperatorType` elements to their integer sub-step multipliers.
        t_grid : numpy.ndarray
            The global time grid of the main :py:class:`~saetass.grid.Grid` as bare floats, in canonical :py:data:`~saetass.units.TIME` units.

        Returns
        -------
        dict
            A dictionary mapping :py:class:`~saetass.solver.OperatorType` operators to their specific refined 1D time grids.
        """
        pass

    @abstractmethod
    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ) -> None:
        """
        Apply exactly one global macro-step of the splitting scheme.

        Sequentially commands the provided ``operator_subsolvers`` to integrate their independent
        physical phenomena on the transport domain. Implementations must call ``self._advance_global_time(state)``
        as their final action.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            Ordered sequence indicating which physical mechanism each subsolver targets.
        operator_subsolvers : list of :py:class:`~saetass.solver.SubSolver`
            Corresponding list of instantiated numerical physics operators.
        substeps_per_op : dict
            Dictionary mapping operators to their sub-step configuration.
        state : :py:class:`~saetass.state.State`
            The global state object to consecutively mutate inplace.
        """
        pass


class StrangSplitting(SplittingScheme):
    """
    Implements second-order Strang Operator Splitting.

    For an operator sequence ``[A, B, C]``, Strang splitting evaluates the operators
    symmetrically over a single timestep: it steps half intervals for all preceding operators,
    a full interval for the final operator, and then half intervals back out in reversed order.
    """

    def __init__(self):
        super().__init__()

    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ) -> dict:
        """
        Refine the macroscopic time grid for Strang splitting topologies.

        Because symmetric operators in the Strang hierarchy are stepped *twice* per global macro-step,
        their sub-stepping grid structures inherently receive exactly double their normally specified
        number of partitions.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            Ordered sequence of operators.
        substeps_per_op : dict
            User configurations for base sub-steps per operator.
        t_grid : numpy.ndarray
            1D array of the base canonical macro-timesteps.

        Returns
        -------
        dict
            Dictionary mapping each operator to its refined time grid array.
        """
        self._store_t_grid(t_grid)
        refined_t_grids = {}

        for i, op in enumerate(operator_list):
            n_sub = substeps_per_op[op]
            if i < len(operator_list) - 1:
                n_sub = (
                    n_sub * 2
                )  # Double substeps for all but the central (last) operator

            refined_t_grids[op] = _refine_t_grid(t_grid, n_sub)

        return refined_t_grids

    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ) -> None:
        """
        Perform a single macroscopic advancement using Strang splitting evaluating logic.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            Ordered operators sequence.
        operator_subsolvers : list of :py:class:`~saetass.solver.SubSolver`
            Associated physics solver objects.
        substeps_per_op : dict
            Operator sub-step parameter lookup.
        state : :py:class:`~saetass.state.State`
            State wrapper passed in-place down the subsolver chain.
        """
        if np.min(state._values) < 0:
            logger.warning(
                "Strang splitting: Negative values detected in state._values before applying operators."
            )

        # First half-step for all but the last operator
        for op, subsolver in zip(operator_list[:-1], operator_subsolvers[:-1]):
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Full step for the last operator
        n_sub = substeps_per_op[operator_list[-1]]
        operator_subsolvers[-1].advance(n_sub, state)

        # Second half-step for all but the last operator (reversed)
        for op, subsolver in zip(
            reversed(operator_list[:-1]), reversed(operator_subsolvers[:-1])
        ):
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Advance state.t to the exact end-of-step grid value
        self._advance_global_time(state)


class LieSplitting(SplittingScheme):
    """
    Implements first-order Lie-Trotter Operator Splitting.

    For an operator sequence ``[A, B, C]``, Lie splitting simply evaluates the operators
    sequentially for the full timestep interval.
    """

    def __init__(self):
        super().__init__()

    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ) -> dict:
        """
        Refine the time grid strictly using sub-step values defined for Lie topologies.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            Ordered sequence of operators.
        substeps_per_op : dict
            User configurations for sub-steps per operator.
        t_grid : numpy.ndarray
            1D array of the base canonical macro-timesteps.

        Returns
        -------
        dict
            Dictionary grouping each respective operator to its sub-refined time grid array.
        """
        self._store_t_grid(t_grid)
        refined_t_grids = {}

        for op in operator_list:
            n_sub = substeps_per_op[op]
            refined_t_grids[op] = _refine_t_grid(t_grid, n_sub)

        return refined_t_grids

    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ) -> None:
        """
        Perform a single macroscopic advancement exclusively evaluating physics sequentially.

        Parameters
        ----------
        operator_list : list of :py:class:`~saetass.solver.OperatorType`
            Ordered operators sequence.
        operator_subsolvers : list of :py:class:`~saetass.solver.SubSolver`
            Associated physics solver objects.
        substeps_per_op : dict
            Operator sub-step parameter lookup.
        state : :py:class:`~saetass.state.State`
            State wrapper iterated across consecutively.
        """
        for op, subsolver in zip(operator_list, operator_subsolvers):
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Advance state.t to the exact end-of-step grid value
        self._advance_global_time(state)


class SplittingSchemeType(StrEnum):
    """
    Auxiliary class enumerating registered operator splitting schema choices.

    .. note::
        Currently, the supported splitting types are: ``"strang"`` and ``"lie"``.

    Parameters
    ----------
    scheme_type : str
        String identifier for the requested scheme context. Raising ``ValueError`` silently for unavailable schemes.
    """

    STRANG = ("strang", StrangSplitting)
    LIE = ("lie", LieSplitting)

    def __new__(cls, scheme_type: str, scheme_class: type):
        obj = str.__new__(cls, scheme_type)
        obj._value_ = scheme_type
        obj.scheme_class = scheme_class
        return obj


def create_splitting_scheme(scheme_name: str | SplittingSchemeType) -> SplittingScheme:
    """
    Factory wrapper bridging scheme configuration identifiers to their instantiated classes.

    Parameters
    ----------
    scheme_name : str or :py:class:`~saetass.splitting.SplittingSchemeType`
        String identifier specifying the desired structural design.

    Returns
    -------
    SplittingScheme
        An untethered sub-class of :py:class:`~saetass.splitting.SplittingScheme`.

    Raises
    ------
    ValueError
        If an unrecognized splitting design is implicitly provided.
    """
    if isinstance(scheme_name, str):
        scheme_name = scheme_name.lower().strip()

    scheme_type = SplittingSchemeType(scheme_name)
    return scheme_type.scheme_class()


def _refine_t_grid(t_grid: np.ndarray, n_sub: int) -> np.ndarray:
    """
    Helper function interpolating a time-series by an integer partition multiplier.

    Parameters
    ----------
    t_grid : numpy.ndarray
        Array of macroscopic timesteps to process as bare floats, in canonical :py:data:`~saetass.units.TIME` units.
    n_sub : int
        Number of steps to forcefully insert equivalently between macro-intervals.

    Returns
    -------
    numpy.ndarray
        Dense temporal grid array.
    """
    su.require_bare(t_grid, "Splitting scheme t_grid")
    raw_t = np.asarray(t_grid, dtype=float)

    if n_sub > 1:
        t_grid_refined = []
        for j in range(len(raw_t) - 1):
            t_start = raw_t[j]
            t_end = raw_t[j + 1]
            t_grid_refined.extend(np.linspace(t_start, t_end, n_sub + 1)[:-1])
        t_grid_refined.append(raw_t[-1])
        return np.asarray(t_grid_refined, dtype=float)
    else:
        return raw_t
