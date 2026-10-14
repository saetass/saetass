import logging
from types import MappingProxyType

import numpy as np
from numba import njit, prange

from .. import units as su
from ..diagnostics import CharacteristicScales
from ..grid import Grid
from ..solver import ParamSpec, SubSolver
from ..state import State

logger = logging.getLogger(__name__)


class DiffusionSolver(SubSolver):
    r"""
    Finite volume solver for spatial diffusion, inheriting from :py:class:`~saetass.solver.SubSolver`.

    Solves the spherical diffusion equation in conservative form,

    .. math::

        \frac{\partial \psi}{\partial t} = \frac{1}{r^2}\frac{\partial}{\partial r}\!\left(D(t,r,p)\,r^2\frac{\partial \psi}{\partial r}\right),

    using a fully implicit Crank-Nicolson scheme with a batched tridiagonal Thomas solver, so that all momentum slices are advanced simultaneously without any loop over :math:`p`.

    The scheme is cell-centred finite volume on a non-uniform radial grid. The flux at each internal face is computed via a harmonic-mean diffusion coefficient, ensuring continuity of the flux even for discontinuous :math:`D`,

    .. math::

        D_{i+1/2} = \frac{h_L + h_R}{h_L / D_i + h_R / D_{i+1}},

    where :math:`h_L` and :math:`h_R` are the distances from the face to the
    left and right cell centres respectively. At :math:`r = 0` a symmetry (zero-flux) condition is always enforced.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``r_centers`` and ``r_faces``.  The first centre must be at :math:`r = 0`.
        Optionally includes ``p_centers`` and ``p_faces`` for 2D problems.
    t_grid : numpy.ndarray
        Subproblem time grid.
        In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        D_values : astropy.units.Quantity or callable
            Diffusion coefficient at cell centres. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`. Shape must match the grid (``(nr,)`` for 1D or ``(np, nr)`` for 2D).  A callable must have signature ``D_values(t: Quantity) -> Quantity``.
        boundary_condition : ``{'dirichlet', 'neumann', 'outflow'}``, optional
            Outer boundary condition. Default is ``'dirichlet'``.
        psi_end : astropy.units.Quantity or callable, optional
            Differential density :math:`\psi` at the outer boundary for the Dirichlet condition. Units compatible with :py:data:`~saetass.units.PSI_P`. A callable must have signature ``psi_end(t: Quantity) -> Quantity``. Default is zero.
    """

    PARAM_SPECS = MappingProxyType(
        {
            "D_values": ParamSpec(su.DIFFUSION_COEFFICIENT, dynamic=True),
            "psi_end": ParamSpec(su.PSI_P, dynamic=True),
            "boundary_condition": ParamSpec(),
        }
    )

    def __init__(self, grid: Grid, t_grid: np.ndarray, params: dict, **kwargs) -> None:
        self._validate_unit_free_inputs(t_grid, params)
        self._unpack_grid(grid)
        self.t_grid = np.asarray(t_grid, dtype=float)
        if self.r_centers.ndim != 1 or self.r_centers.size < 2:
            raise ValueError("r_centers must be 1D with at least 2 points.")
        if abs(self.r_centers[0]) > 1e-12 * (self.r_centers[1] - self.r_centers[0]):
            raise ValueError("first r_center must be 0.")
        self.N = len(self.r_centers)
        self.h = np.asarray(self.grid.dr, dtype=float)
        self.V = (self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3) / 3.0
        self.d_centers = self.r_centers[1:] - self.r_centers[:-1]
        self.params = params or {}
        self._unpack_params()
        self._init_buffers()

    def _unpack_grid(self, grid):
        self.grid = grid
        if grid.r_centers is None:
            raise ValueError("Grid must include r_centers")
        self.r_centers = np.asarray(grid.r_centers, dtype=float)
        self.r_faces = np.asarray(grid.r_faces, dtype=float)
        if grid.p_centers is not None:
            self.p_centers = np.asarray(grid.p_centers, dtype=float)
            self.p_faces = np.asarray(grid.p_faces, dtype=float)
            logger.info(f"Found momentum grid with {len(self.p_centers)} points")
        else:
            self.p_centers = None
            self.p_faces = None
            logger.info("No momentum grid found, operating in 1D mode")

        self.n_p = 1 if self.p_centers is None else len(self.p_centers)

    def _unpack_params(self):
        expected = {"D_values"}
        if not expected.issubset(self.params):
            raise ValueError(f"Missing params: {expected - set(self.params.keys())}")

        D_input = self.params["D_values"]
        self.boundary_condition = self.params.get(
            "boundary_condition", "dirichlet"
        ).lower()

        if callable(D_input):
            self.is_D_dynamic = True

            def _get_d_dynamic(t):
                D = np.asarray(D_input(t), dtype=float)
                if D.ndim == 1:
                    if D.size != self.N:
                        raise ValueError(
                            "1D D_values must have the same size as r_cells"
                        )
                    return D[None, :]
                elif D.ndim == 2:
                    return D
                else:
                    raise ValueError("D_values must be 1D or 2D")

            self._get_D = _get_d_dynamic

            # Initial value for setup
            self.D_values = self._get_D(self.t_grid[0])

        else:
            self.is_D_dynamic = False
            D = np.asarray(D_input, dtype=float)
            if D.shape != self.grid.shape:
                raise ValueError(
                    f"D_values shape {D.shape} does not match grid shape {self.grid.shape}"
                )
            if D.ndim == 1:
                if D.size != self.N:
                    raise ValueError("1D D_values must have size of r_cells")
                self.D_values_static = D[None, :]
            elif D.ndim == 2:
                self.D_values_static = D
            else:
                raise ValueError("D_values must be 1D or 2D")

            self.D_values = self.D_values_static

        boundary_input = self.params.get("psi_end", 0.0)
        if callable(boundary_input):
            self.is_psi_end_dynamic = True

            def _get_psi_end_dynamic(t):
                b_val = boundary_input(t)
                return (
                    float(b_val)
                    if np.isscalar(b_val)
                    else np.asarray(b_val, dtype=float)
                )

            self._get_psi_end = _get_psi_end_dynamic
            self.psi_end = self._get_psi_end(self.t_grid[0])
        else:
            self.is_psi_end_dynamic = False
            self.psi_end = (
                float(boundary_input)
                if np.isscalar(boundary_input)
                else np.asarray(boundary_input, dtype=float)
            )

    def _init_buffers(self):
        # Buffers for batched Thomas solver
        n_p = self.n_p
        self._D_face = np.zeros((n_p, self.N + 1), dtype=float)
        self._G = np.zeros((n_p, self.N + 1), dtype=float)
        self._A_lower = np.zeros((n_p, self.N), dtype=float)
        self._A_diag = np.zeros((n_p, self.N), dtype=float)
        self._A_upper = np.zeros((n_p, self.N), dtype=float)
        self._B_lower = np.zeros((n_p, self.N), dtype=float)
        self._B_diag = np.zeros((n_p, self.N), dtype=float)
        self._B_upper = np.zeros((n_p, self.N), dtype=float)
        self._rhs = np.zeros((n_p, self.N), dtype=float)
        self._f_new = np.zeros((n_p, self.N), dtype=float)

        # Additional precomputations
        self.V_b = self.V[None, :]
        self.h_left = self.h[:-1].astype(float)
        self.h_right = self.h[1:].astype(float)
        self.hL_b = self.h_left[None, :]
        self.hR_b = self.h_right[None, :]

        # Precompute static conductances if D is not dynamic
        if getattr(self, "is_D_dynamic", False) is False:
            self._update_conductances(self.D_values_static)

    # ------------------- Public advance -------------------
    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the :py:class:`~saetass.state.State` by ``n_steps`` in :py:attr:`~saetass.solvers.diffusion_solver.DiffusionSolver.t_grid`.

        Uses a Crank-Nicolson scheme solved simultaneously via a batched Thomas (tridiagonal) solver.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        state : :py:class:`~saetass.state.State`
            Current simulation state. The differential density is updated in-place at the end of the call.
        """
        dt = self._dt * n_steps

        values_all = state._get_values()

        # If no momentum grid or state is 1D, use scalar path
        if (
            self.p_centers is None
            or getattr(state, "_values", None) is None
            or state._get_values().ndim == 1
        ):
            # Promote a 1D psi(r) into shape (1, N)
            values_all = state._get_values()

            if values_all.ndim == 1:
                # 1D case → promote
                if values_all.size != self.N:
                    raise ValueError("1D state values must have size N")
                values_all = values_all[None, :]  # shape = (1, N)

        if values_all.shape != (self.n_p, self.N):
            raise ValueError(
                f"State values shape {values_all.shape} does not match expected {(self.n_p, self.N)}"
            )

        # Dynamic update
        t_current = state.t_val
        if self.is_D_dynamic:
            D_values = self._get_D(t_current)
            self._update_conductances(D_values)

        if self.is_psi_end_dynamic:
            self.psi_end = self._get_psi_end(t_current)

        # Vectorized batched solve for all slices
        values_new_all = self._advance_all_slices_batched(dt, values_all)

        # Update state natively using the 2D array capability
        state._update_values(values_new_all)

    # ------------------- Core batched advance -------------------
    def _update_conductances(self, D_values: np.ndarray) -> None:
        """
        Recompute face conductances when D_values change.
        """
        self._compute_conductances(D_values, self._D_face, self._G)

    def _compute_conductances(
        self, D_values: np.ndarray, D_face: np.ndarray, G: np.ndarray
    ) -> None:
        """
        Compute face diffusion coefficients and conductances for ``D_values`` into the ``D_face`` and ``G`` buffers.
        """
        # 1) Compute D_face for all slices: shape (n_p, N+1)
        D_left = D_values[:, :-1]
        D_right = D_values[:, 1:]
        num = self.hL_b + self.hR_b
        den = self.hL_b / D_left + self.hR_b / D_right
        den = np.where(den == 0.0, np.finfo(float).tiny, den)
        D_face[:, 1:-1] = num / den
        D_face[:, 0] = D_values[:, 0]
        D_face[:, -1] = D_values[:, -1]

        # 2) Compute conductances G on faces
        rf2 = (self.r_faces[1:-1] ** 2)[None, :]
        G[:, 1:-1] = (rf2 * D_face[:, 1:-1]) / self.d_centers[None, :]
        G[:, 0] = 0.0

        if self.boundary_condition == "dirichlet":
            if self.N >= 2:
                denom_last = self.r_centers[-1] - self.r_centers[-2]
            else:
                denom_last = self.h[0] / 2.0
            G[:, -1] = (self.r_faces[-1] ** 2) * D_face[:, -1] / denom_last
        elif self.boundary_condition == "outflow":
            # Assume asymptotic behaviour f ~ 1/r -> df/dr = -f/r
            # At boundary, flux is roughly proportional to f itself
            # We fold the outflow into G[:, -1] multiplying f_N
            G[:, -1] = self.r_faces[-1] * D_face[:, -1]
        elif self.boundary_condition == "neumann":
            G[:, -1] = 0.0
        else:
            raise ValueError(f"Unknown boundary_condition: {self.boundary_condition}")

    def diffusion_coefficients(self, t: float) -> np.ndarray:
        """
        Diffusion coefficients at cell centers at time ``t``, without modifying the solver state.

        Parameters
        ----------
        t : float
            Time, as a bare float in canonical :py:data:`~saetass.units.TIME` units.

        Returns
        -------
        numpy.ndarray
            Diffusion coefficients with shape ``(n_p, N)``, in canonical :py:data:`~saetass.units.DIFFUSION_COEFFICIENT` units.
        """
        return self._get_D(t) if self.is_D_dynamic else self.D_values_static

    def characteristic_scales(self, t: float, dt: float) -> CharacteristicScales:
        r"""
        Fourier number of the Crank-Nicolson step and diffusion time across the domain.

        The Fourier number is :math:`\max_i \Delta t\,(G_{i-1/2} + G_{i+1/2}) / (2 V_i)`, the condition for the explicit half of the scheme to have non-negative coefficients.
        See :py:meth:`~saetass.solver.SubSolver.characteristic_scales`.
        """
        D_values = self.diffusion_coefficients(t)
        D_face, G = np.empty_like(self._D_face), np.empty_like(self._G)
        self._compute_conductances(D_values, D_face, G)
        coupling = dt * (G[:, :-1] + G[:, 1:]) / (2.0 * self.V_b)
        if self.boundary_condition == "dirichlet":
            coupling = coupling[:, :-1]  # the boundary cell is fixed, not integrated
        D_max = float(np.max(D_values))
        length = float(self.r_faces[-1] - self.r_faces[0])
        return CharacteristicScales(
            timescale=length**2 / D_max if D_max > 0.0 else np.inf,
            numbers={"fourier_number": float(np.max(coupling))},
        )

    def _build_matrices(self, dt: float) -> None:
        """
        Assemble A and B matrices given the current conductances and timestep dt.
        """
        a_left = self._G[:, :-1]
        a_right = self._G[:, 1:]
        self._A_diag[:] = 2.0 * self.V_b / dt + (a_left + a_right)
        self._B_diag[:] = 2.0 * self.V_b / dt - (a_left + a_right)

        self._A_lower.fill(0.0)
        self._B_lower.fill(0.0)
        self._A_upper.fill(0.0)
        self._B_upper.fill(0.0)

        self._A_lower[:, 1:] = -a_left[:, 1:]
        self._B_lower[:, 1:] = a_left[:, 1:]
        self._A_upper[:, :-1] = -a_right[:, :-1]
        self._B_upper[:, :-1] = a_right[:, :-1]

        if self.boundary_condition == "dirichlet":
            self._A_diag[:, -1] = 1.0
            self._A_lower[:, -1] = 0.0
            self._A_upper[:, -1] = 0.0
            self._B_diag[:, -1] = 0.0
            self._B_lower[:, -1] = 0.0
            self._B_upper[:, -1] = 0.0
        # If boundary_condition is 'neumann' or 'outflow', the finite volume updates perfectly track the conservative fluxes,
        # so keeping the standard diagonal works correctly without overwriting.

    def _advance_all_slices_batched(self, dt: float, f_all: np.ndarray) -> np.ndarray:
        """
        Vectorized Crank-Nicolson for all momentum slices at once.
        f_all: shape (n_p, N)
        Returns f_new_all shape (n_p, N)
        """
        # Matrices depend on dt and G. We rebuild them every step (dt could change).
        self._build_matrices(dt)

        # Build RHS: rhs = B @ f_current
        f = f_all
        self._rhs = self._B_diag * f
        self._rhs[:, 1:] += self._B_lower[:, 1:] * f[:, :-1]
        self._rhs[:, :-1] += self._B_upper[:, :-1] * f[:, 1:]

        if self.boundary_condition == "dirichlet":
            self._rhs[:, -1] = self.psi_end

        # Solve batched tridiagonal systems A * f_new = rhs
        f_new = _thomas_batched_numba(
            self._A_lower, self._A_diag, self._A_upper, self._rhs
        )

        return f_new


@njit(parallel=True, fastmath=True)
def _thomas_batched_numba(a_lower, a_diag, a_upper, rhs):
    n_p, N = a_diag.shape
    x = np.empty_like(rhs)
    eps = np.finfo(np.float64).eps

    # Work on local arrays per slice inside parallel loop
    for s in prange(n_p):
        # allocate small 1D temporaries per slice (fast)
        a = a_diag[s, :].copy()
        b = a_upper[s, :].copy()
        c = a_lower[s, :].copy()
        d = rhs[s, :].copy()

        # forward
        a0 = a[0]
        if a0 == 0.0:
            a0 = eps
        b[0] = b[0] / a0
        d[0] = d[0] / a0
        for i in range(1, N):
            denom = a[i] - c[i] * b[i - 1]
            if denom == 0.0:
                denom = eps
            if i < N - 1:
                b[i] = b[i] / denom
            d[i] = (d[i] - c[i] * d[i - 1]) / denom

        # back substitution
        x[s, N - 1] = d[N - 1]
        for i in range(N - 2, -1, -1):
            x[s, i] = d[i] - b[i] * x[s, i + 1]

    return x
