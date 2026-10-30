"""
This module provides the :py:class:`~saetass.state.State` class, which serves as the single source of truth for the particle distribution and its associated metadata (grid, particle species, current time, operator-splitting stage and snapshot history) throughout a simulation.

The :py:class:`~saetass.state.State` object is passed by reference between operator subsolvers during an operator-splitting step; each subsolver reads and writes numerical values through the internal :py:meth:`~saetass.state.State._get_values` / :py:meth:`~saetass.state.State._update_values` API, ensuring modularity and numerical efficiency. Users interact with explicit distribution representations (:py:attr:`psi_p`, :py:attr:`f_ps`, :py:attr:`psi_E`).
"""

import logging
from functools import cached_property
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np

from . import units as su
from .grid import Grid
from .utils.energy_losses import Particle

logger = logging.getLogger(__name__)

CANONICAL_F_PS_UNIT = su.F_PS
CANONICAL_PSI_P_UNIT = su.PSI_P
CANONICAL_PSI_E_UNIT = su.PSI_E


class State:
    r"""
    Container and tracker for the particle distribution and simulation metadata.

    A :py:class:`State` instance wraps the numerical differential density array, :math:`\psi_p(t, r, p) = 4\pi p^2 f_\mathrm{ps}(t, r, p) = \frac{dn}{dp}`, together with the associated :py:class:`~saetass.grid.Grid`, particle species :py:class:`~saetass.utils.energy_losses.Particle`, current simulation time, time step, operator-splitting stage information and an optional snapshot history.

    The internal representation stores only the canonical momentum-space differential density, :math:`\psi_p`.
    However, intialization and updates can be performed using any of :math:`f_\mathrm{ps}`, :math:`\psi_p` or :math:`\psi_E`, as long as the grid dimensions are compatible with the provided array.
    All representations can also be retrieved using the :py:attr:`f_ps`, :py:attr:`psi_p` and :py:attr:`psi_E` properties, which act as thin wrappers around the internal representation.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        The associated spatial/momentum simulation grid.
    f_ps : astropy.units.Quantity, optional
        Phase-space distribution in momentum space :math:`f_\mathrm{ps}(r, p)`.
        Units compatible with :py:data:`~saetass.units.F_PS`.
    psi_p : astropy.units.Quantity, optional
        Differential density in momentum :math:`\psi_p(r, p) = \frac{dn}{dp} = 4\pi p^2 f_\mathrm{ps}`.
        Units compatible with :py:data:`~saetass.units.PSI_P`.
    psi_E : astropy.units.Quantity, optional
        Differential number density in kinetic energy :math:`\psi_E(r, E) = \frac{dn}{dE} = \frac{dp}{dE} \psi_p`.
        Units compatible with :py:data:`~saetass.units.PSI_E`.
    particle : Particle or str, optional
        The cosmic ray particle species. Default is ``Particle.PROTON``.
    t : astropy.units.Quantity, optional
        Initial simulation time. Units compatible with :py:data:`~saetass.units.TIME`. Default is ``0.0 * su.TIME``.
    dt : astropy.units.Quantity, optional
        Time step used in the last update. Units compatible with :py:data:`~saetass.units.TIME`. Default is ``0.0 * su.TIME``.
    stage : int, optional
        Current operator-splitting stage index. Default is ``0``.
    stage_name : str, optional
        Descriptive label for the current operator-splitting stage. Default is ``''``.
    history : list of dict, optional
        Pre-populated snapshot history. Default is an empty list.

    Attributes
    ----------
    ndim : ``{1, 2}``
        Dimensionality of the problem as inferred from the initial array:
        ``1`` for a 1D spatial or momentum problem,
        ``2`` for a full 2D spatial-momentum problem.
    """

    @u.quantity_input(
        f_ps=su.F_PS,
        psi_p=su.PSI_P,
        psi_E=su.PSI_E,
        t=su.TIME,
        dt=su.TIME,
    )
    def __init__(
        self,
        grid: Grid,
        f_ps: u.Quantity | None = None,
        psi_p: u.Quantity | None = None,
        psi_E: u.Quantity | None = None,
        particle: Particle | str = Particle.PROTON,
        t: u.Quantity = 0.0 * su.TIME,
        dt: u.Quantity = 0.0 * su.TIME,
        stage: int = 0,
        stage_name: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        if grid is None or not isinstance(grid, Grid):
            raise TypeError("State requires a valid Grid instance.")
        self.grid = grid

        self.particle = self._parse_particle(particle)

        rep_type, raw_data = self._resolve_input_representation(
            f_ps=f_ps, psi_p=psi_p, psi_E=psi_E
        )
        self.ndim, raw_arr_2d = self._validate_and_reshape_array(raw_data)
        self._validate_grid_compatibility(self.grid, raw_data, self.ndim, raw_arr_2d)

        # Store internal values as optimized, pure float ndarray in canonical units
        self._values: np.ndarray = np.ascontiguousarray(
            self._convert_to_canonical_psi_p(raw_arr_2d, rep_type), dtype=float
        )

        # Time is stored once, as canonical floats; ``t`` / ``dt`` are Quantity views.
        self._t: float = float(t.to_value(su.TIME))
        self._dt: float = float(dt.to_value(su.TIME))
        self.stage = int(stage)
        self.stage_name = str(stage_name)
        self.history = history if history is not None else []

    @property
    def t(self) -> u.Quantity:
        """Current simulation time. Units compatible with :py:data:`~saetass.units.TIME`."""
        return self._t * su.TIME

    @t.setter
    def t(self, value: u.Quantity) -> None:
        self._t = float(su.validate_quantity(value, su.TIME, "State time").value)

    @property
    def dt(self) -> u.Quantity:
        """Time step of the last update. Units compatible with :py:data:`~saetass.units.TIME`."""
        return self._dt * su.TIME

    @dt.setter
    def dt(self, value: u.Quantity) -> None:
        self._dt = float(su.validate_quantity(value, su.TIME, "State time step").value)

    @property
    def t_val(self) -> float:
        """Current simulation time as a bare float in canonical :py:data:`~saetass.units.TIME` units."""
        return self._t

    @staticmethod
    def _parse_particle(particle: Particle | str) -> Particle:
        # Particle is a StrEnum, so its members are handled as strings too.
        if not isinstance(particle, str):
            raise TypeError(
                f"particle must be a Particle or a str, got {type(particle).__name__}."
            )
        p_str = particle.lower()
        if p_str in ("proton", "hadronic", "p"):
            return Particle.PROTON
        if p_str in ("electron", "leptonic", "e"):
            return Particle.ELECTRON
        return Particle(p_str)

    @classmethod
    def _resolve_input_representation(
        cls,
        f_ps: u.Quantity | None = None,
        psi_p: u.Quantity | None = None,
        psi_E: u.Quantity | None = None,
    ) -> tuple[str, np.ndarray]:
        inputs = {
            "f_ps": f_ps,
            "psi_p": psi_p,
            "psi_E": psi_E,
        }
        provided = [k for k, v in inputs.items() if v is not None]

        if len(provided) == 0:
            raise ValueError(
                "State must be initialized with exactly one representation: "
                "'f_ps', 'psi_p', or 'psi_E'. None was provided."
            )
        if len(provided) > 1:
            raise ValueError(
                f"State representation parameters are mutually exclusive. Provided: {provided}"
            )

        rep_name = provided[0]
        rep_val = inputs[rep_name]
        return cls._validate_named_representation(rep_name, rep_val)

    @staticmethod
    def _validate_named_representation(
        rep_name: str, rep_val: u.Quantity
    ) -> tuple[str, np.ndarray]:
        canonical_units = {
            "f_ps": su.F_PS,
            "psi_p": su.PSI_P,
            "psi_E": su.PSI_E,
        }
        # Units were validated by quantity_input in the public entry points.
        raw_data = rep_val.to_value(canonical_units[rep_name])
        return rep_name, np.asarray(raw_data, dtype=float)

    @staticmethod
    def _validate_and_reshape_array(raw_data: Any) -> tuple[int, np.ndarray]:
        raw_arr = np.array(raw_data, dtype=float, copy=True)
        if raw_arr.ndim == 1:
            return 1, raw_arr.reshape((1, raw_arr.size))
        elif raw_arr.ndim == 2:
            return 2, raw_arr
        else:
            raise NotImplementedError("Distribution array must be 1D or 2D.")

    @staticmethod
    def _validate_grid_compatibility(
        grid: Grid, raw_data: Any, ndim: int, raw_arr_2d: np.ndarray
    ) -> None:
        """
        Validate that the provided distribution function array is compatible with the grid geometry.
        """
        raw_arr = np.asarray(raw_data)
        has_p = grid.p_centers is not None
        has_r = grid.r_centers is not None

        if has_p and has_r:
            expected_shape = (grid.num_cells_p, grid.num_cells_r)
            if raw_arr.ndim == 1:
                raise ValueError(
                    f"1D distribution array of shape {raw_arr.shape} is incompatible with 2D Grid shape {expected_shape}."
                )
            if raw_arr_2d.shape != expected_shape:
                raise ValueError(
                    f"Distribution array shape {raw_arr.shape} is incompatible with 2D Grid shape {expected_shape}."
                )
        elif has_r:
            expected_size = grid.num_cells_r
            if raw_arr.ndim == 1 and raw_arr.size != expected_size:
                raise ValueError(
                    f"Distribution array size {raw_arr.size} is incompatible with spatial Grid size {expected_size}."
                )
            elif raw_arr.ndim == 2 and raw_arr.shape != (1, expected_size):
                raise ValueError(
                    f"Distribution array shape {raw_arr.shape} is incompatible with 1D spatial Grid shape ({expected_size},)."
                )
        elif has_p:
            expected_size = grid.num_cells_p
            if raw_arr.ndim == 1 and raw_arr.size != expected_size:
                raise ValueError(
                    f"Distribution array size {raw_arr.size} is incompatible with momentum Grid size {expected_size}."
                )
            elif (
                raw_arr.ndim == 2
                and raw_arr.shape != (expected_size, 1)
                and raw_arr.shape != (1, expected_size)
            ):
                raise ValueError(
                    f"Distribution array shape {raw_arr.shape} is incompatible with 1D momentum Grid shape ({expected_size},)."
                )

    def _convert_to_canonical_psi_p(
        self, raw_arr_2d: np.ndarray, rep_type: str
    ) -> np.ndarray:
        if rep_type == "psi_p":
            return raw_arr_2d

        p_factor_dim = slice(None) if self.ndim == 1 else (slice(None), np.newaxis)

        if rep_type == "f_ps":
            # psi_p = 4 * pi * p^2 * f_ps
            four_pi_p2 = self.grid.four_pi_p2
            if four_pi_p2 is None:
                raise ValueError(
                    "Conversion between representations requires a Grid with momentum coordinates (p_centers)."
                )
            return raw_arr_2d * four_pi_p2.to_value(su.MOMENTUM**2)[p_factor_dim]

        # psi_E (representation names are validated by the callers):
        # psi_p = psi_E * dE/dp = psi_E * p / E_tot
        return raw_arr_2d * self.dE_dp[p_factor_dim]

    def _get_p_coords(self) -> np.ndarray:
        p = self.grid.p_centers_phys
        if p is None:
            raise ValueError(
                "Conversion between representations requires a Grid with momentum coordinates (p_centers)."
            )
        return p.to_value(su.MOMENTUM)

    def _process_representation(self, rep_name: str, rep_val: u.Quantity) -> np.ndarray:
        _, raw_data = self._validate_named_representation(rep_name, rep_val)
        ndim, raw_arr_2d = self._validate_and_reshape_array(raw_data)
        self._validate_grid_compatibility(self.grid, raw_data, ndim, raw_arr_2d)
        return self._convert_to_canonical_psi_p(raw_arr_2d, rep_name)

    def _update_metadata(
        self,
        dt: u.Quantity | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        if dt is not None:
            self.dt = dt
        if stage is not None:
            self.stage = int(stage)
        if stage_name is not None:
            self.stage_name = str(stage_name)

    # -------------------------------------------------------------------------
    # Internal Numerical Interface for Solvers & Kernels
    # -------------------------------------------------------------------------

    def _get_values(self) -> np.ndarray:
        r"""
        Internal numerical method: return the raw canonical :math:`\psi_p` array in its natural dimensionality.

        For 1D problems (``ndim == 1``) returns a 1D array of shape ``(n,)``;
        for 2D problems returns the full 2D array of shape ``(n_p, n_r)``.

        Returns
        -------
        numpy.ndarray
            The current numerical differential density array, in canonical :py:data:`~saetass.units.PSI_P` units.
        """
        if self.ndim == 1:
            return self._values[0]
        return self._values

    def _update_values(self, new_values: np.ndarray) -> None:
        """
        Internal numerical method: replace the internal canonical state array with ``new_values``.

        The new array must be shape-compatible with the current state.
        For 1D states a 1D input of length ``n_r`` is automatically promoted to shape ``(1, n_r)`` before storing.

        Parameters
        ----------
        new_values : numpy.ndarray
            New values, in canonical :py:data:`~saetass.units.PSI_P` units. Must have shape ``(n_p, n_r)`` or, for 1D states, ``(n_r,)``.

        Raises
        ------
        ValueError
            If ``new_values`` has an incompatible shape.
        """
        new_arr = np.asarray(new_values, dtype=float)
        if new_arr.shape != self._values.shape:
            if (
                self.ndim == 1
                and new_arr.ndim == 1
                and new_arr.size == self._values.size
            ):
                new_arr = new_arr.reshape((1, new_arr.size))
            else:
                raise ValueError(
                    f"new_values must have shape {self._values.shape}, got {new_arr.shape}"
                )
        self._values = np.ascontiguousarray(new_arr, dtype=float)

    # -------------------------------------------------------------------------
    # Properties (Dimensions & Grid)
    # -------------------------------------------------------------------------

    @property
    def n_p(self) -> int:
        """Number of momentum bins (rows of internal distribution array)."""
        return self._values.shape[0]

    @property
    def n_r(self) -> int:
        """Number of spatial bins (columns of internal distribution array)."""
        return self._values.shape[1]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the internal distribution array, ``(n_p, n_r)``."""
        return self._values.shape

    # -------------------------------------------------------------------------
    # Kinematic Cached Properties
    # -------------------------------------------------------------------------

    @cached_property
    def E_tot(self) -> u.Quantity:
        r"""
        Total energy :math:`E_\mathrm{tot} = \sqrt{p^2 c^2 + (m c^2)^2}` at momentum cell centers.

        Returns
        -------
        astropy.units.Quantity
            Array of total energies with shape ``(n_p,)``. Units compatible with :py:data:`~saetass.units.ENERGY`.
        """
        p_val = self._get_p_coords()
        p = p_val * su.MOMENTUM
        m_energy = (self.particle.mass * const.c**2).to(su.ENERGY)
        p_energy = (p * const.c).to(su.ENERGY)
        return np.sqrt(p_energy**2 + m_energy**2)

    @cached_property
    def E(self) -> u.Quantity:
        r"""
        Kinetic energy :math:`E = E_\mathrm{tot} - m c^2` at momentum cell centers.

        Returns
        -------
        astropy.units.Quantity
            Array of kinetic energies with shape ``(n_p,)``. Units compatible with :py:data:`~saetass.units.ENERGY`.
        """
        m_energy = (self.particle.mass * const.c**2).to(su.ENERGY)
        return self.E_tot - m_energy

    @cached_property
    def gamma(self) -> np.ndarray:
        r"""
        Lorentz factor :math:`\gamma = \frac{E_\mathrm{tot}}{m c^2}` at momentum cell centers.

        Returns
        -------
        numpy.ndarray
            Array of Lorentz factors with shape ``(n_p,)``.
        """
        m_energy = (self.particle.mass * const.c**2).to(su.ENERGY)
        return (self.E_tot / m_energy).to_value(u.dimensionless_unscaled)

    @cached_property
    def beta(self) -> np.ndarray:
        r"""
        Dimensionless velocity :math:`\beta = v/c = \frac{p c}{E_\mathrm{tot}}` at momentum cell centers.

        Returns
        -------
        numpy.ndarray
            Array of dimensionless velocities with shape ``(n_p,)``.
        """
        p = self._get_p_coords() * su.MOMENTUM
        p_energy = (p * const.c).to(su.ENERGY)
        return (p_energy / self.E_tot).to_value(u.dimensionless_unscaled)

    @cached_property
    def v(self) -> u.Quantity:
        r"""
        Particle velocity :math:`v = \beta c` at momentum cell centers.

        Returns
        -------
        astropy.units.Quantity
            Array of velocities with shape ``(n_p,)``. Units compatible with :py:data:`~saetass.units.VELOCITY`.
        """
        return (self.beta * const.c).to(su.VELOCITY)

    @cached_property
    def dp_dE(self) -> np.ndarray:
        r"""
        Jacobian conversion factor :math:`\frac{dp}{dE} = \frac{E_\mathrm{tot}}{p c^2}` relating kinetic energy and momentum differentials.

        Returns
        -------
        numpy.ndarray
            Array of :math:`dp/dE` with shape ``(n_p,)``, in :math:`(\mathrm{GeV}/c)\,\mathrm{GeV^{-1}}`.
        """
        p = self._get_p_coords() * su.MOMENTUM
        p_energy = (p * const.c).to(su.ENERGY)
        return (self.E_tot / p_energy).to_value(u.dimensionless_unscaled)

    @cached_property
    def dE_dp(self) -> np.ndarray:
        r"""
        Jacobian conversion factor :math:`\frac{dE}{dp} = \beta c = \frac{p c^2}{E_\mathrm{tot}}` relating kinetic energy and momentum differentials.

        Returns
        -------
        numpy.ndarray
            Array of :math:`dE/dp` with shape ``(n_p,)``, in :math:`\mathrm{GeV}\,(\mathrm{GeV}/c)^{-1}`.
        """
        return 1.0 / self.dp_dE

    # -------------------------------------------------------------------------
    # Explicit User-Facing Representation Properties & Setters
    # -------------------------------------------------------------------------

    @property
    def psi_p(self) -> u.Quantity:
        r"""Differential density in momentum space :math:`\psi_p(r, p) = \frac{dn}{dp}`."""
        return self.to_psi_p()

    @psi_p.setter
    def psi_p(self, value: u.Quantity) -> None:
        self.update_psi_p(value)

    @property
    def f_ps(self) -> u.Quantity:
        r"""Phase-space distribution function in momentum space :math:`f_\mathrm{ps}(r, p)`."""
        return self.to_f_ps()

    @f_ps.setter
    def f_ps(self, value: u.Quantity) -> None:
        self.update_f_ps(value)

    @property
    def psi_E(self) -> u.Quantity:
        r"""Differential number density in energy space :math:`\psi_E(r, E) = \frac{dn}{dE}`."""
        return self.to_psi_E()

    @psi_E.setter
    def psi_E(self, value: u.Quantity) -> None:
        self.update_psi_E(value)

    @property
    def dndE(self) -> u.Quantity:
        r"""Differential number density in energy space :math:`\frac{dn}{dE}` (alias for :py:attr:`psi_E`)."""
        return self.to_psi_E()

    @dndE.setter
    def dndE(self, value: u.Quantity) -> None:
        self.update_psi_E(value)

    # -------------------------------------------------------------------------
    # Explicit User-Facing Update Methods
    # -------------------------------------------------------------------------

    @u.quantity_input(psi_p=su.PSI_P, dt=su.TIME)
    def update_psi_p(
        self,
        psi_p: u.Quantity,
        dt: u.Quantity | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        r"""
        Update the state with a new momentum differential density :math:`\psi_p = \frac{dn}{dp}`.

        Parameters
        ----------
        psi_p : astropy.units.Quantity
            New momentum differential density array. Units compatible with :py:data:`~saetass.units.PSI_P`.
        dt : astropy.units.Quantity, optional
            Elapsed time step for this update. Units compatible with :py:data:`~saetass.units.TIME`.
        stage : int, optional
            Operator-splitting stage index.
        stage_name : str, optional
            Descriptive label for this stage.
        """
        psi_p_arr_2d = self._process_representation("psi_p", psi_p)
        self._update_values(psi_p_arr_2d)
        self._update_metadata(dt=dt, stage=stage, stage_name=stage_name)

    @u.quantity_input(f_ps=su.F_PS, dt=su.TIME)
    def update_f_ps(
        self,
        f_ps: u.Quantity,
        dt: u.Quantity | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        r"""
        Update the state with a new phase-space distribution :math:`f_\mathrm{ps}`.

        Parameters
        ----------
        f_ps : astropy.units.Quantity
            New phase-space distribution array. Units compatible with :py:data:`~saetass.units.F_PS`.
        dt : astropy.units.Quantity, optional
            Elapsed time step for this update. Units compatible with :py:data:`~saetass.units.TIME`.
        stage : int, optional
            Operator-splitting stage index.
        stage_name : str, optional
            Descriptive label for this stage.
        """
        psi_p_arr_2d = self._process_representation("f_ps", f_ps)
        self._update_values(psi_p_arr_2d)
        self._update_metadata(dt=dt, stage=stage, stage_name=stage_name)

    @u.quantity_input(psi_E=su.PSI_E, dt=su.TIME)
    def update_psi_E(
        self,
        psi_E: u.Quantity,
        dt: u.Quantity | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        r"""
        Update the state with a new energy differential density :math:`\psi_E = \frac{dn}{dE}`.

        Parameters
        ----------
        psi_E : astropy.units.Quantity
            New energy differential density array. Units compatible with :py:data:`~saetass.units.PSI_E`.
        dt : astropy.units.Quantity, optional
            Elapsed time step for this update. Units compatible with :py:data:`~saetass.units.TIME`.
        stage : int, optional
            Operator-splitting stage index.
        stage_name : str, optional
            Descriptive label for this stage.
        """
        psi_p_arr_2d = self._process_representation("psi_E", psi_E)
        self._update_values(psi_p_arr_2d)
        self._update_metadata(dt=dt, stage=stage, stage_name=stage_name)

    @u.quantity_input(
        f_ps=su.F_PS,
        psi_p=su.PSI_P,
        psi_E=su.PSI_E,
        dt=su.TIME,
    )
    def update(
        self,
        f_ps: u.Quantity | None = None,
        psi_p: u.Quantity | None = None,
        psi_E: u.Quantity | None = None,
        dt: u.Quantity | None = None,
        stage: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        """
        Update the state with an explicit distribution representation.

        Exactly one of ``f_ps``, ``psi_p``, or ``psi_E`` must be provided.

        Parameters
        ----------
        f_ps : astropy.units.Quantity, optional
            Phase-space distribution in momentum space. Units compatible with :py:data:`~saetass.units.F_PS`.
        psi_p : astropy.units.Quantity, optional
            Differential density in momentum space. Units compatible with :py:data:`~saetass.units.PSI_P`.
        psi_E : astropy.units.Quantity, optional
            Differential number density in kinetic energy. Units compatible with :py:data:`~saetass.units.PSI_E`.
        dt : astropy.units.Quantity, optional
            Elapsed time step for this update. Units compatible with :py:data:`~saetass.units.TIME`.
        stage : int, optional
            Operator-splitting stage index.
        stage_name : str, optional
            Descriptive label for this stage.
        """
        rep_type, raw_data = self._resolve_input_representation(
            f_ps=f_ps, psi_p=psi_p, psi_E=psi_E
        )
        ndim, raw_arr_2d = self._validate_and_reshape_array(raw_data)
        self._validate_grid_compatibility(self.grid, raw_data, ndim, raw_arr_2d)
        psi_p_arr_2d = self._convert_to_canonical_psi_p(raw_arr_2d, rep_type)
        self._update_values(psi_p_arr_2d)
        self._update_metadata(dt=dt, stage=stage, stage_name=stage_name)

    # -------------------------------------------------------------------------
    # Explicit User-Facing Conversion Methods
    # -------------------------------------------------------------------------

    def to_psi_p(self, unit: u.Unit | None = None) -> u.Quantity:
        r"""
        Return differential density in momentum space :math:`\psi_p(r, p) = \frac{dn}{dp} = 4\pi p^2 f_\mathrm{ps}`.

        Parameters
        ----------
        unit : astropy.units.Unit, optional
            Target unit for the output. If ``None``, the canonical :py:data:`~saetass.units.PSI_P` is used. Default is ``None``.

        Returns
        -------
        astropy.units.Quantity
            Differential density in momentum space. Units compatible with :py:data:`~saetass.units.PSI_P`.
        """
        vals = self._get_values()
        target_unit = unit if unit is not None else su.PSI_P
        return (vals * su.PSI_P).to(target_unit)

    def to_f_ps(self, unit: u.Unit | None = None) -> u.Quantity:
        r"""
        Convert to phase-space distribution function in momentum space :math:`f_\mathrm{ps}(r, p) = \frac{\psi_p(r, p)}{4\pi p^2}`.

        Parameters
        ----------
        unit : astropy.units.Unit, optional
            Target unit for the output. If ``None``, the canonical :py:data:`~saetass.units.F_PS` is used. Default is ``None``.

        Returns
        -------
        astropy.units.Quantity
            Phase-space distribution array in the specified units. Units compatible with :py:data:`~saetass.units.F_PS`.
        """
        four_pi_p2 = self.grid.four_pi_p2
        if four_pi_p2 is None:
            raise ValueError(
                "Conversion between representations requires a Grid with momentum coordinates (p_centers)."
            )
        four_pi_p2_val = four_pi_p2.to_value(su.MOMENTUM**2)
        p_factor_dim = slice(None) if self.ndim == 1 else (slice(None), np.newaxis)
        f_ps_arr = self._values / four_pi_p2_val[p_factor_dim]
        res = f_ps_arr[0] if self.ndim == 1 else f_ps_arr
        target_unit = unit if unit is not None else su.F_PS
        return (res * su.F_PS).to(target_unit)

    def to_phase_space(self, unit: u.Unit | None = None) -> u.Quantity:
        """Alias for :py:meth:`to_f_ps`."""
        return self.to_f_ps(unit=unit)

    def to_psi_E(self, unit: u.Unit | None = None) -> u.Quantity:
        r"""
        Convert to differential number density in energy space :math:`\psi_E(r, E) = \frac{dn}{dE} = \psi_p\,\frac{E_\mathrm{tot}}{p c^2}`.

        Parameters
        ----------
        unit : astropy.units.Unit, optional
            Target unit for the output. If ``None``, the canonical :py:data:`~saetass.units.PSI_E` is used. Default is ``None``.

        Returns
        -------
        astropy.units.Quantity
            Differential number density in kinetic energy. Units compatible with :py:data:`~saetass.units.PSI_E`.
        """
        p_factor_dim = slice(None) if self.ndim == 1 else (slice(None), np.newaxis)
        psi_E_arr = self._values * self.dp_dE[p_factor_dim]
        res = psi_E_arr[0] if self.ndim == 1 else psi_E_arr
        target_unit = unit if unit is not None else su.PSI_E
        return (res * su.PSI_E).to(target_unit)

    def to_dndE(self, unit: u.Unit | None = None) -> u.Quantity:
        """Alias for :py:meth:`to_psi_E`."""
        return self.to_psi_E(unit=unit)

    # -------------------------------------------------------------------------
    # History, Timestepping & Cloning
    # -------------------------------------------------------------------------

    def clone(self, copy_history: bool = False) -> "State":
        """
        Create a deep copy of the current state.

        Parameters
        ----------
        copy_history : bool, optional
            If ``True``, deep-copy the full snapshot history as well. Default is ``False``.

        Returns
        -------
        State
            A new :py:class:`State` with copied differential density and metadata.
            The ``history`` list of the clone is empty unless ``copy_history`` is ``True``.
        """
        new_state = State(
            psi_p=self._values.copy() * su.PSI_P,
            grid=self.grid,
            particle=self.particle,
            t=self.t,
            dt=self.dt,
            stage=int(self.stage),
            stage_name=str(self.stage_name),
        )
        if copy_history:
            new_state.history = [
                {
                    "t": self._canonical_time(snap["t"], "Snapshot time") * su.TIME,
                    "dt": self._canonical_time(snap["dt"], "Snapshot time step")
                    * su.TIME,
                    "stage": snap["stage"],
                    "stage_name": snap.get("stage_name", ""),
                    "values": snap["values"].copy(),
                }
                for snap in self.history
            ]
        return new_state

    def set_time(self, t: u.Quantity | float) -> None:
        """
        Set the simulation clock to an exact value.

        This method assigns ``t`` directly rather than accumulating increments, preventing floating-point drift when snapping to a canonical grid point.
        :py:attr:`dt` is updated to reflect the elapsed interval since the previous time.

        Parameters
        ----------
        t : astropy.units.Quantity or float
            Exact time value to assign. A float is interpreted in canonical :py:data:`~saetass.units.TIME` units.
        """
        # Hot path (called every global step by the splitting schemes): floats only.
        t_new = self._canonical_time(t, "Time")
        self._dt = t_new - self._t
        self._t = t_new

    def record_substep(self, stage_name: str | None = None) -> None:
        """
        Append a snapshot of the current state to the history.

        Each snapshot captures the current values of :py:attr:`t`, :py:attr:`dt`, :py:attr:`stage`, the ``stage_name`` (if provided) and a copy of the internal differential density array.
        Use :py:meth:`restore_substep` or :py:meth:`get_substep` to retrieve a saved snapshot later.

        Parameters
        ----------
        stage_name : str, optional
            Descriptive label to attach to this snapshot.
            If ``None``, the current :py:attr:`stage_name` attribute is used instead.
        """
        entry = {
            "t": self.t,
            "dt": self.dt,
            "stage": int(self.stage),
            "stage_name": (
                stage_name if stage_name is not None else str(self.stage_name)
            ),
            "values": self._values.copy(),
        }
        self.history.append(entry)

    @staticmethod
    def _canonical_time(value: u.Quantity | float, name: str) -> float:
        """Canonical float time from a Quantity or from a bare float already in canonical TIME units."""
        if isinstance(value, u.Quantity):
            return float(su.validate_quantity(value, su.TIME, name).value)
        return float(value)

    def restore_substep(self, identifier: int | str) -> "State":
        """
        Restore the state to a previously recorded snapshot.

        The snapshot to restore can be identified either by its numeric position in the history list or by its ``stage_name`` string.
        When identified by name, the *first* matching snapshot is used.

        Parameters
        ----------
        identifier : int or str
            Integer index into the :py:attr:`history` list, or a ``stage_name`` string to search for.

        Returns
        -------
        State
            The current instance (``self``) after restoration, for convenience.

        Raises
        ------
        ValueError
            If ``identifier`` is a string and no matching snapshot is found.
        IndexError
            If ``identifier`` is an integer that is out of range.
        """
        if isinstance(identifier, int):
            snap = self.history[identifier]
        else:
            matches = [h for h in self.history if h.get("stage_name") == identifier]
            if not matches:
                raise ValueError(f"No snapshot found with stage_name={identifier!r}")
            snap = matches[0]
        self._values = snap["values"].copy()
        self._t = self._canonical_time(snap["t"], "Snapshot time")
        self._dt = self._canonical_time(snap["dt"], "Snapshot time step")
        self.stage = int(snap["stage"])
        self.stage_name = str(snap.get("stage_name", ""))
        return self

    def get_substep(self, index: int) -> dict[str, Any]:
        """
        Retrieve a copy of a snapshot without modifying the current state.

        Parameters
        ----------
        index : int
            Index of the snapshot in :py:attr:`history`.

        Returns
        -------
        dict
            A copy of the snapshot dictionary with keys ``'t'``, ``'dt'``, ``'stage'``, ``'stage_name'`` and ``'values'`` (differential density ndarray copy).

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        snap = self.history[index]
        return {
            "t": self._canonical_time(snap["t"], "Snapshot time") * su.TIME,
            "dt": self._canonical_time(snap["dt"], "Snapshot time step") * su.TIME,
            "stage": int(snap["stage"]),
            "stage_name": str(snap.get("stage_name", "")),
            "values": snap["values"].copy(),
        }

    def clear_history(self) -> None:
        """Remove all recorded snapshots from :py:attr:`history`, leaving the current state intact."""
        self.history.clear()

    def step_stage(self, stage_name: str | None = None) -> None:
        """
        Increment the operator-splitting stage counter by one.

        Optionally updates :py:attr:`stage_name` to the given label.
        If no label is supplied, :py:attr:`stage_name` is reset to an empty string.

        Parameters
        ----------
        stage_name : str, optional
            Descriptive label for the new stage. If ``None``, :py:attr:`stage_name` is reset to ``''``. Default is ``None``.
        """
        self.stage += 1
        if stage_name is not None:
            self.stage_name = stage_name
        else:
            self.stage_name = ""

    def __repr__(self) -> str:
        particle_name = getattr(self.particle, "value", str(self.particle))
        species_name = getattr(self.particle, "species", "")
        spec_str = f" ({species_name})" if species_name else ""
        return (
            f"State(t={self._t:.3f} {su.TIME}, dt={self._dt:.3f} {su.TIME}, stage={self.stage}, "
            f"stage_name={self.stage_name!r}, particle={particle_name!r}{spec_str}, "
            f"shape={self._values.shape}, history_len={len(self.history)})"
        )
