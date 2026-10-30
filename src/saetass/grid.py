r"""
The grid module defines the :py:class:`~saetass.grid.Grid` class, which manages spatial, momentum and temporal discretization for simulations initialization.
The :py:class:`~saetass.grid.Grid` serves as the foundational data structure describing the discrete domain upon which the differential density function :math:`\psi_p(t, r, p)` evolves.
It is natively responsible for caching and providing access to the geometrical properties of the space and momentum domains, as well as the discrete time steps.

In the context of the greater solver pipeline, the instantiated :py:class:`~saetass.grid.Grid` object is shared globally between the parent orchestrator, :py:class:`~saetass.solver.Solver`, and the individual physics operators, i.e. any instance of :py:class:`~saetass.solver.SubSolver`.

Furthermore, :py:class:`~saetass.grid.Grid` provides ``@classmethods`` to instantiate uniform, non-uniform and logarithmically spaced grids.
"""

import logging
from functools import cached_property

import astropy.units as u
import numpy as np

from . import units as su
from .cli.output import plural

logger = logging.getLogger(__name__)


class Grid:
    r"""
    The :py:class:`~saetass.grid.Grid` class can be instantiated in several ways:

    - By providing the spatial and/or momentum faces grid coordinates.
    - By providing the spatial and/or momentum centers grid coordinates.
    - By providing the spatial and/or momentum faces and centers grid coordinates.
    - By instantiating any of the ``@classmethods`` that it provides.

    When centers or faces are not provided, the class will attempt to construct them from the provided faces or centers, respectively.
    Moreover, :py:class:`~saetass.grid.Grid` performs the coordinate mapping between physical and computational space for the case of the momentum grid in logarithmic scale (:math:`y = \log_{10}(p)`).

    Parameters
    ----------
    r_centers : astropy.units.Quantity, optional
        1D array containing the centroid coordinates of the spatial grid cells. Units compatible with :py:data:`~saetass.units.LENGTH`.
    r_faces : astropy.units.Quantity, optional
        1D array containing the boundary (interface) coordinates of the spatial grid cells. Length is one greater than ``r_centers``. Units compatible with :py:data:`~saetass.units.LENGTH`.
    p_centers : astropy.units.Quantity, optional
        1D array containing the centroid coordinates of the momentum grid cells. Units compatible with :py:data:`~saetass.units.MOMENTUM`.
    p_faces : astropy.units.Quantity, optional
        1D array containing the boundary (interface) coordinates of the momentum grid cells. Length is one greater than ``p_centers``. Units compatible with :py:data:`~saetass.units.MOMENTUM`.
    t_grid : astropy.units.Quantity, optional
        1D array defining the discrete macro-timesteps over which the simulation will globally advance. Units compatible with :py:data:`~saetass.units.TIME`.
    is_p_log : bool, optional
        Flag dictating whether numerical operations inside solvers should treat the momentum grid internally in logarithmic spacing :math:`\log_{10}(p)`. Default is ``True``.
    """

    @u.quantity_input(
        r_centers=su.LENGTH,
        r_faces=su.LENGTH,
        p_centers=su.MOMENTUM,
        p_faces=su.MOMENTUM,
        t_grid=su.TIME,
    )
    def __init__(
        self,
        r_centers: u.Quantity | None = None,
        r_faces: u.Quantity | None = None,
        p_centers: u.Quantity | None = None,
        p_faces: u.Quantity | None = None,
        t_grid: u.Quantity | None = None,
        is_p_log: bool = True,
    ):
        # Check that at least one grid type is provided
        if (
            r_faces is None
            and r_centers is None
            and p_faces is None
            and p_centers is None
        ):
            raise ValueError("At least one grid (spatial or momentum) must be provided")

        self.is_log_p = is_p_log

        # Initialize spatial grid if provided
        if r_faces is not None or r_centers is not None:
            self._init_spatial_grid(r_faces, r_centers)
        else:
            self.r_faces: u.Quantity | None = None
            self.r_centers: u.Quantity | None = None

        # Initialize momentum grid if provided
        if p_faces is not None or p_centers is not None:
            self._init_momentum_grid(p_faces, p_centers)
        else:
            self.p_faces: u.Quantity | None = None
            self.p_centers: u.Quantity | None = None

        # Temporal grid
        self.t_grid: u.Quantity | None = (
            t_grid.to(su.TIME) if t_grid is not None else None
        )
        if self.t_grid is not None and len(self.t_grid) > 1:
            self.dt: u.Quantity | None = np.diff(self.t_grid)
        else:
            self.dt = None

        if self.is_log_p and self.p_centers is not None:
            self._p_centers_phys = self.p_centers.copy()
            self._p_faces_phys = (
                self.p_faces.copy() if self.p_faces is not None else None
            )
            self.p_centers = self._p_to_y(self.p_centers)
            if self.p_faces is not None:
                self.p_faces = self._p_to_y(self.p_faces)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Created grid: %s", self.summary())

    def summary(self) -> str:
        """
        One-line description of the cells (the range of their centers) and time steps of the grid, in canonical units.

        Returns
        -------
        str
            For instance ``"71 r cells in [0, 10] pc, 400 log-spaced p cells in [1, 1e+06] GeV_c, 400 time steps in [0, 1] Myr"``.
        """
        parts = []
        if self.r_centers is not None:
            r = self.r_centers.to_value(su.LENGTH)
            parts.append(
                f"{plural(self.num_cells_r, 'r cell')} in [{r[0]:.4g}, {r[-1]:.4g}] "
                f"{su.LENGTH}"
            )
        if self.p_centers is not None:
            p = self.p_centers_phys.to_value(su.MOMENTUM)
            spacing = "log-spaced " if self.is_log_p else ""
            parts.append(
                f"{plural(self.num_cells_p, spacing + 'p cell')} "
                f"in [{p[0]:.4g}, {p[-1]:.4g}] "
                f"{su.MOMENTUM}"
            )
        if self.t_grid is not None:
            t = self.t_grid.to_value(su.TIME)
            parts.append(
                f"{plural(self.num_timesteps, 'time step')} "
                f"in [{t[0]:.4g}, {t[-1]:.4g}] {su.TIME}"
            )
        return ", ".join(parts)

    def _init_spatial_grid(
        self, r_faces: u.Quantity | None, r_centers: u.Quantity | None
    ):
        """Initialize the spatial grid from faces or centers, converting to canonical LENGTH."""
        if r_faces is not None:
            self.r_faces = r_faces.to(su.LENGTH)
            # Calculate cell centers as midpoints between faces
            self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        elif r_centers is not None:
            self.r_centers = r_centers.to(su.LENGTH)
            # Approximate face positions for non-uniform grid
            if len(self.r_centers) > 1:
                # For internal faces: midpoint between cell centers
                internal_faces = 0.5 * (self.r_centers[:-1] + self.r_centers[1:])
                # For boundary faces: extrapolate
                left_face = self.r_centers[0] - 0.5 * (
                    self.r_centers[1] - self.r_centers[0]
                )
                right_face = self.r_centers[-1] + 0.5 * (
                    self.r_centers[-1] - self.r_centers[-2]
                )
                self.r_faces = u.Quantity(
                    np.concatenate(
                        [
                            [left_face.to_value(su.LENGTH)],
                            internal_faces.to_value(su.LENGTH),
                            [right_face.to_value(su.LENGTH)],
                        ]
                    ),
                    su.LENGTH,
                )
            else:
                # Single cell case
                dr = 1.0 * su.LENGTH  # Default width for a single cell
                self.r_faces = u.Quantity(
                    [self.r_centers[0] - 0.5 * dr, self.r_centers[0] + 0.5 * dr],
                    su.LENGTH,
                )

    def _init_momentum_grid(
        self, p_faces: u.Quantity | None, p_centers: u.Quantity | None
    ):
        """Initialize the momentum grid from faces or centers, converting to canonical MOMENTUM."""
        if p_faces is not None:
            self.p_faces = p_faces.to(su.MOMENTUM)
            # Calculate cell centers as midpoints between faces
            self.p_centers = 0.5 * (self.p_faces[:-1] + self.p_faces[1:])
        elif p_centers is not None:
            self.p_centers = p_centers.to(su.MOMENTUM)
            # Approximate face positions for non-uniform grid
            if len(self.p_centers) > 1:
                # For internal faces: midpoint between cell centers
                internal_faces = 0.5 * (self.p_centers[:-1] + self.p_centers[1:])
                # For boundary faces: extrapolate
                left_face = self.p_centers[0] - 0.5 * (
                    self.p_centers[1] - self.p_centers[0]
                )
                right_face = self.p_centers[-1] + 0.5 * (
                    self.p_centers[-1] - self.p_centers[-2]
                )
                if left_face <= 0 * su.MOMENTUM:
                    raise ValueError("Momentum faces must be positive values.")
                self.p_faces = u.Quantity(
                    np.concatenate(
                        [
                            [left_face.to_value(su.MOMENTUM)],
                            internal_faces.to_value(su.MOMENTUM),
                            [right_face.to_value(su.MOMENTUM)],
                        ]
                    ),
                    su.MOMENTUM,
                )
            else:
                # Single cell case
                dp = 1.0 * su.MOMENTUM  # Default width for a single cell
                self.p_faces = u.Quantity(
                    [self.p_centers[0] - 0.5 * dp, self.p_centers[0] + 0.5 * dp],
                    su.MOMENTUM,
                )

    @cached_property
    def dr(self) -> u.Quantity | None:
        """
        Spatial cell widths.

        Returns
        -------
        astropy.units.Quantity or None
            Array of spatial cell widths, or ``None`` if the spatial grid is not initialized. Units compatible with :py:data:`~saetass.units.LENGTH`.
        """
        if self.r_faces is not None:
            return self.r_faces[1:] - self.r_faces[:-1]
        return None

    @cached_property
    def dp(self) -> u.Quantity | np.ndarray | None:
        r"""
        Momentum cell widths.

        Returns
        -------
        astropy.units.Quantity, numpy.ndarray or None
            Array of momentum cell widths, or ``None`` if the momentum grid is not initialized. For logarithmic grids, it is a bare array of widths in :math:`\log_{10}(p)`; otherwise, units compatible with :py:data:`~saetass.units.MOMENTUM`.
        """
        if self.p_faces is not None:
            return self.p_faces[1:] - self.p_faces[:-1]
        return None

    @property
    def p_centers_phys(self) -> u.Quantity | None:
        """Physical momentum coordinates at cell centers. Units compatible with :py:data:`~saetass.units.MOMENTUM`."""
        if getattr(self, "is_log_p", False) and hasattr(self, "_p_centers_phys"):
            return self._p_centers_phys
        return self.p_centers

    @property
    def p_faces_phys(self) -> u.Quantity | None:
        """Physical momentum coordinates at cell faces. Units compatible with :py:data:`~saetass.units.MOMENTUM`."""
        if getattr(self, "is_log_p", False) and hasattr(self, "_p_faces_phys"):
            return self._p_faces_phys
        return self.p_faces

    @cached_property
    def four_pi_p2(self) -> u.Quantity | None:
        r"""
        Factor :math:`4\pi p^2` evaluated at physical momentum cell centers :math:`p`.

        Returns
        -------
        astropy.units.Quantity or None
            Array of :math:`4\pi p^2`, or ``None`` if the momentum grid is not initialized. Units compatible with the square of :py:data:`~saetass.units.MOMENTUM`.
        """
        if self.p_centers is not None:
            p = self.p_centers_phys
            return 4.0 * np.pi * (p**2)
        return None

    @cached_property
    def volumes(self) -> u.Quantity | None:
        r"""
        Cell volumes based on spherical geometry.

        A cell centered at the origin, whose first face lies at :math:`-\Delta r / 2`, is the ball inside its outer face.

        Returns
        -------
        astropy.units.Quantity or None
            Array of cell volumes assuming spherical shells, or ``None`` if the spatial grid is not initialized. Units compatible with :py:data:`~saetass.units.VOLUME`.
        """
        if self.r_faces is not None:
            # Radii below the origin hold no volume
            faces = np.maximum(self.r_faces, 0.0 * su.LENGTH)
            return (4.0 * np.pi / 3.0) * (faces[1:] ** 3 - faces[:-1] ** 3)
        return None

    @cached_property
    def face_areas(self) -> u.Quantity | None:
        r"""
        Face areas based on spherical geometry.

        The first face of a cell centered at the origin, at :math:`-\Delta r / 2`, is the origin itself, with no area.

        Returns
        -------
        astropy.units.Quantity or None
            Array of face areas assuming spherical shells, or ``None`` if the spatial grid is not initialized. Units compatible with :py:data:`~saetass.units.AREA`.
        """
        if self.r_faces is not None:
            return 4.0 * np.pi * np.maximum(self.r_faces, 0.0 * su.LENGTH) ** 2
        return None

    @cached_property
    def num_timesteps(self) -> int:
        """
        Number of global timesteps.

        Returns
        -------
        int
            Number of recorded timesteps (length of temporal grid minus 1).

        Raises
        ------
        ValueError
            If the temporal grid is not properly defined with at least 2 points.
        """
        if self.t_grid is not None and len(self.t_grid) > 1:
            return len(self.t_grid) - 1
        else:
            raise ValueError("Temporal grid is not properly defined.")

    @cached_property
    def num_cells_r(self) -> int:
        """
        Number of spatial cells.

        Returns
        -------
        int
            Number of spatial cells in the grid.
        """
        if self.r_centers is not None:
            return self.r_centers.size
        return 0

    @cached_property
    def num_cells_p(self) -> int:
        """
        Number of momentum cells.

        Returns
        -------
        int
            Number of momentum cells in the grid.
        """
        if self.p_centers is not None:
            return self.p_centers.size
        return 0

    @cached_property
    def shape(self) -> tuple:
        """
        Shape of the grid.

        Returns
        -------
        tuple
            The shape of the grid as ``(n_p, n_r)`` if 2D, or a 1D tuple if only one dimension is defined.

        Raises
        ------
        ValueError
            If no grid dimensions are defined.
        """
        if self.p_centers is not None and self.r_centers is not None:
            return (self.num_cells_p, self.num_cells_r)
        elif self.r_centers is not None:
            return (self.num_cells_r,)
        elif self.p_centers is not None:
            return (self.num_cells_p,)
        else:
            raise ValueError("No grid dimensions are defined.")

    def _p_to_y(self, p: u.Quantity | np.ndarray) -> np.ndarray:
        r"""Convert momentum p to the logarithmic variable :math:`y = \log_{10}(p / \mathrm{GeV}\,c^{-1})`."""
        val = (
            p.to_value(su.MOMENTUM)
            if isinstance(p, u.Quantity)
            else np.asarray(p, dtype=float)
        )
        return np.log10(val)

    def _y_to_p(self, y: np.ndarray) -> u.Quantity:
        """Convert the logarithmic variable :math:`y` back to the momentum :math:`p`, as a Quantity."""
        return (10.0**y) * su.MOMENTUM

    def post_process_calculations(self):
        """
        Perform post-processing calculations after grid exit of :py:class:`~saetass.solver.Solver` pipeline.

        This function converts the logarithmic momentum grid tracking back into standard momentum physical
        values. It is intended to be called after the :py:class:`~saetass.solver.Solver` has completed its operations.
        """
        if self.is_log_p:
            self._p_centers_calc = self._y_to_p(self.p_centers)
            if self.p_faces is not None:
                self._p_faces_calc = self._y_to_p(self.p_faces)
            self.dp_calc = self.dp
            self.p_centers = self._p_centers_calc
            if self.p_faces is not None:
                self.p_faces = self._p_faces_calc
        else:
            logger.warning(
                "Post-processing calculations skipped as momentum grid is not logarithmic."
            )

    def is_compatible_array(self, array: np.ndarray) -> bool:
        """
        Check if the given array is compatible with this :py:class:`~saetass.grid.Grid`.

        Parameters
        ----------
        array : numpy.ndarray
            The array to check for shape compatibility.

        Returns
        -------
        bool
            ``True`` if the array shape matches the grid shape, ``False`` otherwise.
        """
        expected_shape = self.shape
        return array.shape == expected_shape

    @staticmethod
    @u.quantity_input(
        r_min=su.LENGTH,
        r_max=su.LENGTH,
        p_min=su.MOMENTUM,
        p_max=su.MOMENTUM,
        t_min=su.TIME,
        t_max=su.TIME,
    )
    def _validate_grid_params(
        r_min: u.Quantity | None = None,
        r_max: u.Quantity | None = None,
        num_r_cells: int | None = None,
        p_min: u.Quantity | None = None,
        p_max: u.Quantity | None = None,
        num_p_cells: int | None = None,
        t_min: u.Quantity | None = None,
        t_max: u.Quantity | None = None,
        num_timesteps: int | None = None,
        req_r_pos: bool = False,
        req_p_pos: bool = False,
    ) -> tuple[bool, bool, bool]:
        """
        Validate the grid generation boundaries.

        Parameters
        ----------
        r_min, r_max : astropy.units.Quantity, optional
            Spatial grid bounds. Units compatible with :py:data:`~saetass.units.LENGTH`.
        num_r_cells : int, optional
            Number of spatial cells.
        p_min, p_max : astropy.units.Quantity, optional
            Momentum grid bounds. Units compatible with :py:data:`~saetass.units.MOMENTUM`.
        num_p_cells : int, optional
            Number of momentum cells.
        t_min, t_max : astropy.units.Quantity, optional
            Temporal grid bounds. Units compatible with :py:data:`~saetass.units.TIME`.
        num_timesteps : int, optional
            Number of timesteps.
        req_r_pos : bool, optional
            Whether ``r_min`` must be strictly positive (for log spaces). Default is ``False``.
        req_p_pos : bool, optional
            Whether ``p_min`` must be strictly positive (for log spaces). Default is ``False``.

        Returns
        -------
        has_r, has_p, has_t : tuple of bool
            Boolean flags indicating which configurations are correctly fully specified.

        Raises
        ------
        ValueError
            If invalid parameter sets are given, such as min bounding over max,
            or required groupings of values are incompletely filled.
        """
        has_r = False
        if not (r_min is None and r_max is None and num_r_cells is None):
            if r_min is None or r_max is None or num_r_cells is None:
                raise ValueError(
                    "r_min, r_max, and num_r_cells must all be specified together."
                )
            if r_min < 0 * su.LENGTH:
                raise ValueError("r_min cannot be negative.")
            if req_r_pos and r_min <= 0 * su.LENGTH:
                raise ValueError("r_min must be strictly positive for this grid type.")
            if r_max <= r_min:
                raise ValueError("r_max must be strictly greater than r_min.")
            if num_r_cells <= 0:
                raise ValueError("num_r_cells must be a positive integer.")
            has_r = True

        has_p = False
        if not (p_min is None and p_max is None and num_p_cells is None):
            if p_min is None or p_max is None or num_p_cells is None:
                raise ValueError(
                    "p_min, p_max, and num_p_cells must all be specified together."
                )
            if p_min < 0 * su.MOMENTUM:
                raise ValueError("p_min cannot be negative.")
            if req_p_pos and p_min <= 0 * su.MOMENTUM:
                raise ValueError("p_min must be strictly positive for this grid type.")
            if p_max <= p_min:
                raise ValueError("p_max must be strictly greater than p_min.")
            if num_p_cells <= 0:
                raise ValueError("num_p_cells must be a positive integer.")
            has_p = True

        if not has_r and not has_p:
            raise ValueError(
                "At least one spatial (r) or momentum (p) grid must be fully specified."
            )

        has_t = False
        if not (t_min is None and t_max is None and num_timesteps is None):
            if t_min is None or t_max is None or num_timesteps is None:
                raise ValueError(
                    "t_min, t_max, and num_timesteps must all be specified together."
                )
            if t_max <= t_min:
                raise ValueError("t_max must be strictly greater than t_min.")
            if num_timesteps <= 0:
                raise ValueError("num_timesteps must be a positive integer.")
            has_t = True

        return has_r, has_p, has_t

    @classmethod
    @u.quantity_input(
        r_min=su.LENGTH,
        r_max=su.LENGTH,
        p_min=su.MOMENTUM,
        p_max=su.MOMENTUM,
        t_min=su.TIME,
        t_max=su.TIME,
    )
    def uniform(
        cls,
        r_min: u.Quantity | None = None,
        r_max: u.Quantity | None = None,
        num_r_cells: int | None = None,
        p_min: u.Quantity | None = None,
        p_max: u.Quantity | None = None,
        num_p_cells: int | None = None,
        t_min: u.Quantity | None = None,
        t_max: u.Quantity | None = None,
        num_timesteps: int | None = None,
    ):
        """
        Create a uniform grid linearly spaced.

        At least one space (:math:`r`) or momentum (:math:`p`) grid must be fully specified with
        its respective (min, max, cells) triplet.

        Parameters
        ----------
        r_min : astropy.units.Quantity, optional
            Minimum radius (must be >= 0). Units compatible with :py:data:`~saetass.units.LENGTH`.
        r_max : astropy.units.Quantity, optional
            Maximum radius. Units compatible with :py:data:`~saetass.units.LENGTH`.
        num_r_cells : int, optional
            Number of spatial cells.
        p_min : astropy.units.Quantity, optional
            Minimum momentum (must be >= 0). Units compatible with :py:data:`~saetass.units.MOMENTUM`.
        p_max : astropy.units.Quantity, optional
            Maximum momentum. Units compatible with :py:data:`~saetass.units.MOMENTUM`.
        num_p_cells : int, optional
            Number of momentum cells.
        t_min : astropy.units.Quantity, optional
            Minimum time. Units compatible with :py:data:`~saetass.units.TIME`.
        t_max : astropy.units.Quantity, optional
            Maximum time. Units compatible with :py:data:`~saetass.units.TIME`.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A uniform :py:class:`~saetass.grid.Grid` object linearly spaced.

        Raises
        ------
        ValueError
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, or incomplete parameter sets are given.
        """
        has_r, has_p, has_t = cls._validate_grid_params(
            r_min=r_min,
            r_max=r_max,
            num_r_cells=num_r_cells,
            p_min=p_min,
            p_max=p_max,
            num_p_cells=num_p_cells,
            t_min=t_min,
            t_max=t_max,
            num_timesteps=num_timesteps,
        )

        r_centers = (
            np.linspace(
                r_min.to_value(su.LENGTH), r_max.to_value(su.LENGTH), num_r_cells
            )
            * su.LENGTH
            if has_r
            else None
        )
        p_centers = (
            np.linspace(
                p_min.to_value(su.MOMENTUM), p_max.to_value(su.MOMENTUM), num_p_cells
            )
            * su.MOMENTUM
            if has_p
            else None
        )
        t_grid = (
            np.linspace(
                t_min.to_value(su.TIME), t_max.to_value(su.TIME), num_timesteps + 1
            )
            * su.TIME
            if has_t
            else None
        )

        return cls(
            r_centers=r_centers, p_centers=p_centers, t_grid=t_grid, is_p_log=False
        )

    @classmethod
    @u.quantity_input(
        r_min=su.LENGTH,
        r_max=su.LENGTH,
        cluster_center=su.LENGTH,
        cluster_width=su.LENGTH,
        t_min=su.TIME,
        t_max=su.TIME,
    )
    def non_uniform_clustering(
        cls,
        r_min: u.Quantity,
        r_max: u.Quantity,
        num_r_cells: int,
        cluster_center: u.Quantity,
        cluster_width: u.Quantity,
        cluster_strength: float = 0.9,
        t_min: u.Quantity | None = None,
        t_max: u.Quantity | None = None,
        num_timesteps: int | None = None,
    ):
        """
        Create a non-uniform grid with clustering around a specific spatial point.

        Requires full specification of the spatial parameters. Momentum grid construction
        is explicitly unused in this classmethod.

        Parameters
        ----------
        r_min : astropy.units.Quantity
            Minimum radius (must be >= 0). Units compatible with :py:data:`~saetass.units.LENGTH`.
        r_max : astropy.units.Quantity
            Maximum radius. Units compatible with :py:data:`~saetass.units.LENGTH`.
        num_r_cells : int
            Number of spatial cells.
        cluster_center : astropy.units.Quantity
            Center of the clustering region. Units compatible with :py:data:`~saetass.units.LENGTH`.
        cluster_width : astropy.units.Quantity
            Width of the clustering region. Units compatible with :py:data:`~saetass.units.LENGTH`.
        cluster_strength : float, optional
            Strength of clustering between 0 and 1. Default is ``0.9``.
        t_min : astropy.units.Quantity, optional
            Minimum time. Units compatible with :py:data:`~saetass.units.TIME`.
        t_max : astropy.units.Quantity, optional
            Maximum time. Units compatible with :py:data:`~saetass.units.TIME`.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A non-uniform :py:class:`~saetass.grid.Grid` object with cell clustering.

        Raises
        ------
        ValueError
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, or incomplete parameter sets are given.
        """
        has_r, _, has_t = cls._validate_grid_params(
            r_min=r_min,
            r_max=r_max,
            num_r_cells=num_r_cells,
            p_min=None,
            p_max=None,
            num_p_cells=None,
            t_min=t_min,
            t_max=t_max,
            num_timesteps=num_timesteps,
        )

        r_min_val = r_min.to_value(su.LENGTH)
        r_max_val = r_max.to_value(su.LENGTH)
        c_center_val = cluster_center.to_value(su.LENGTH)
        c_width_val = cluster_width.to_value(su.LENGTH)

        # Normalize to [0, 1]
        x_c = (c_center_val - r_min_val) / (r_max_val - r_min_val)
        width = c_width_val / (r_max_val - r_min_val)

        # Generate initial uniform grid in [0, 1]
        xi = np.linspace(0, 1, num_r_cells + 1)

        # Apply tanh clustering
        s = (xi - x_c) / (0.5 * width)
        xi = xi - cluster_strength * np.tanh(s) * (0.5 * width)

        # Ensure bounds and monotonicity
        xi = np.clip(xi, 0, 1)
        xi[0] = 0
        xi[-1] = 1
        xi = np.sort(xi)

        # Map back to original domain
        r_faces = (r_min_val + xi * (r_max_val - r_min_val)) * su.LENGTH

        t_grid = (
            np.linspace(
                t_min.to_value(su.TIME), t_max.to_value(su.TIME), num_timesteps + 1
            )
            * su.TIME
            if has_t
            else None
        )

        return cls(r_faces=r_faces, t_grid=t_grid, is_p_log=False)

    @classmethod
    @u.quantity_input(
        r_min=su.LENGTH,
        r_max=su.LENGTH,
        p_min=su.MOMENTUM,
        p_max=su.MOMENTUM,
        t_min=su.TIME,
        t_max=su.TIME,
    )
    def log_spaced(
        cls,
        r_min: u.Quantity | None = None,
        r_max: u.Quantity | None = None,
        num_r_cells: int | None = None,
        p_min: u.Quantity | None = None,
        p_max: u.Quantity | None = None,
        num_p_cells: int | None = None,
        t_min: u.Quantity | None = None,
        t_max: u.Quantity | None = None,
        num_timesteps: int | None = None,
    ):
        """
        Create a logarithmically spaced grid.

        At least one space (:math:`r`) or momentum (:math:`p`) grid must be fully specified with its respective (min, max, cells) triplet.

        Parameters
        ----------
        r_min : astropy.units.Quantity, optional
            Minimum radius (must be > 0 if provided). Units compatible with :py:data:`~saetass.units.LENGTH`.
        r_max : astropy.units.Quantity, optional
            Maximum radius. Units compatible with :py:data:`~saetass.units.LENGTH`.
        num_r_cells : int, optional
            Number of spatial cells.
        p_min : astropy.units.Quantity, optional
            Minimum momentum (must be > 0 if provided). Units compatible with :py:data:`~saetass.units.MOMENTUM`.
        p_max : astropy.units.Quantity, optional
            Maximum momentum. Units compatible with :py:data:`~saetass.units.MOMENTUM`.
        num_p_cells : int, optional
            Number of momentum cells.
        t_min : astropy.units.Quantity, optional
            Minimum time. Units compatible with :py:data:`~saetass.units.TIME`.
        t_max : astropy.units.Quantity, optional
            Maximum time. Units compatible with :py:data:`~saetass.units.TIME`.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A log-spaced :py:class:`~saetass.grid.Grid` object.

        Raises
        ------
        ValueError
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, 0 is supplied for min values, or incomplete parameter sets are given.
        """
        has_r, has_p, has_t = cls._validate_grid_params(
            r_min=r_min,
            r_max=r_max,
            num_r_cells=num_r_cells,
            p_min=p_min,
            p_max=p_max,
            num_p_cells=num_p_cells,
            t_min=t_min,
            t_max=t_max,
            num_timesteps=num_timesteps,
            req_r_pos=True,
            req_p_pos=True,
        )

        r_centers = (
            np.logspace(
                np.log10(r_min.to_value(su.LENGTH)),
                np.log10(r_max.to_value(su.LENGTH)),
                num_r_cells,
            )
            * su.LENGTH
            if has_r
            else None
        )
        p_centers = (
            np.logspace(
                np.log10(p_min.to_value(su.MOMENTUM)),
                np.log10(p_max.to_value(su.MOMENTUM)),
                num_p_cells,
            )
            * su.MOMENTUM
            if has_p
            else None
        )
        t_grid = (
            np.linspace(
                t_min.to_value(su.TIME), t_max.to_value(su.TIME), num_timesteps + 1
            )
            * su.TIME
            if has_t
            else None
        )

        return cls(
            r_centers=r_centers, p_centers=p_centers, t_grid=t_grid, is_p_log=True
        )

    def __str__(self) -> str:
        """String representation of the Grid."""
        info = ["Grid:"]

        if self.r_faces is not None:
            r_f = self.r_faces.to(su.LENGTH)
            info.append(
                f"  Spatial range: {r_f[0].value:.4e} to {r_f[-1].value:.4e} {r_f.unit}"
            )
            info.append(f"  Number of spatial cells: {len(self.r_centers)}")

        if self.p_faces is not None or hasattr(self, "_p_faces_phys"):
            p_f = self.p_faces_phys
            if p_f is not None:
                p_f_canon = p_f.to(su.MOMENTUM)
                info.append(
                    f"  Momentum range: {p_f_canon[0].value:.4e} to {p_f_canon[-1].value:.4e} {p_f_canon.unit}"
                )
            info.append(f"  Number of momentum cells: {len(self.p_centers)}")

        if self.t_grid is not None:
            t_g = self.t_grid.to(su.TIME)
            info.append(
                f"  Temporal range: {t_g[0].value:.4e} to {t_g[-1].value:.4e} {t_g.unit}"
            )
            info.append(f"  Number of timesteps: {len(self.t_grid) - 1}")

        return "\n".join(info)
