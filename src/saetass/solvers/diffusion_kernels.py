r"""
Numba kernels of :py:class:`~saetass.solvers.diffusion_solver.DiffusionSolver`.

The Crank-Nicolson step of every momentum is a tridiagonal system :math:`A f^{n+1} = B f^n`.
:py:func:`assemble_rows` builds the face coefficients, conductances and matrices of every
momentum and eliminates :math:`A` forward, storing its inverse pivots, which only has to be
redone when the time step or the coefficients change; :py:func:`solve_rows` then builds the
right-hand side and substitutes, for every step. Both advance blocks of :py:data:`BLOCK`
momenta together, and the blocks are distributed over threads by :py:mod:`saetass.parallel`,
so the results do not depend on the number of threads.
"""

import numpy as np
from numba import njit, prange

from ..parallel import twin_kernel

#: Serial costs per cell of :py:func:`assemble_rows` (assembly and forward elimination) and
#: of :py:func:`solve_rows`, in nanoseconds, to choose their threads.
ASSEMBLE_NS_PER_CELL = 12.0
SOLVE_NS_PER_CELL = 5.0

#: Codes of the outer boundary conditions passed to the kernels.
BOUNDARY_CODES = {"dirichlet": 0, "outflow": 1, "neumann": 2}

#: Pivot replacing an exactly zero one in the Thomas algorithm.
_EPS = np.finfo(np.float64).eps
#: Denominator replacing an exactly zero one in the harmonic mean of the face coefficients.
_TINY = np.finfo(np.float64).tiny
#: Momenta advanced together by the kernels: the substitutions of a single row form a
#: chain of dependent operations, and interleaving independent rows lets the processor
#: overlap them.
BLOCK = 8


@njit(cache=True)
def _factorize_block(a_lower, a_diag, a_upper, inv_pivots, upper, start, stop):
    """
    Forward elimination of the tridiagonal matrices of rows ``start:stop``: the inverses of their pivots and their normalized upper diagonals.

    Storing the inverses keeps divisions, which are slow and would lengthen the chain of dependent operations of every step, out of :py:func:`_solve_block`.
    """
    N = a_diag.shape[1]
    for s in range(start, stop):
        pivot = a_diag[s, 0]
        if pivot == 0.0:
            pivot = _EPS
        inv_pivots[s, 0] = 1.0 / pivot
        upper[s, 0] = a_upper[s, 0] / pivot
    for i in range(1, N):
        for s in range(start, stop):
            pivot = a_diag[s, i] - a_lower[s, i] * upper[s, i - 1]
            if pivot == 0.0:
                pivot = _EPS
            inv_pivots[s, i] = 1.0 / pivot
            upper[s, i] = a_upper[s, i] / pivot if i < N - 1 else a_upper[s, i]


@njit(cache=True)
def _assemble_row(
    D,
    h_left,
    h_right,
    rf2,
    d_centers,
    rf_last2,
    rf_last,
    denom_last,
    V,
    dt,
    boundary,
    D_face,
    G,
    a_lower,
    a_diag,
    a_upper,
    b_lower,
    b_diag,
    b_upper,
):
    """Conductances and Crank-Nicolson matrices of one momentum, with the arithmetic of :py:meth:`~saetass.solvers.diffusion_solver.DiffusionSolver._compute_conductances`."""
    N = D.shape[0]
    D_face[0] = D[0]
    D_face[N] = D[N - 1]
    G[0] = 0.0
    for k in range(1, N):
        den = h_left[k - 1] / D[k - 1] + h_right[k - 1] / D[k]
        if den == 0.0:
            den = _TINY
        D_face[k] = (h_left[k - 1] + h_right[k - 1]) / den
        G[k] = (rf2[k - 1] * D_face[k]) / d_centers[k - 1]
    if boundary == 0:  # dirichlet
        G[N] = rf_last2 * D_face[N] / denom_last
    elif boundary == 1:  # outflow
        G[N] = rf_last * D_face[N]
    else:
        G[N] = 0.0

    for i in range(N):
        two_volume = 2.0 * V[i] / dt
        coupling = G[i] + G[i + 1]
        a_diag[i] = two_volume + coupling
        b_diag[i] = two_volume - coupling
        a_lower[i] = -G[i] if i > 0 else 0.0
        b_lower[i] = G[i] if i > 0 else 0.0
        a_upper[i] = -G[i + 1] if i < N - 1 else 0.0
        b_upper[i] = G[i + 1] if i < N - 1 else 0.0
    if boundary == 0:
        # The boundary value comes from the right-hand side
        a_diag[N - 1] = 1.0
        a_lower[N - 1] = 0.0
        a_upper[N - 1] = 0.0
        b_diag[N - 1] = 0.0
        b_lower[N - 1] = 0.0
        b_upper[N - 1] = 0.0


@twin_kernel()
def assemble_rows(
    D,
    h_left,
    h_right,
    rf2,
    d_centers,
    rf_last2,
    rf_last,
    denom_last,
    V,
    dt,
    boundary,
    D_face,
    G,
    a_lower,
    a_diag,
    a_upper,
    b_lower,
    b_diag,
    b_upper,
    inv_pivots,
    upper,
):
    r"""
    Assemble the Crank-Nicolson matrices of every momentum and eliminate :math:`A` forward.

    The body of the parallel loop is a single call to a separately compiled block
    function, for the reasons given in :py:func:`~saetass.solvers.hyperbolic_kernels.advance_rows`.

    Parameters
    ----------
    D : numpy.ndarray
        Diffusion coefficients at the cell centers, shape ``(1, N)`` (shared by every momentum) or ``(n_p, N)``.
    h_left, h_right : numpy.ndarray
        Distances from each internal face to its left and right cell centers, length ``N - 1``.
    rf2, d_centers : numpy.ndarray
        Squared radii of the internal faces and distances between consecutive centers, length ``N - 1``.
    rf_last2, rf_last, denom_last : float
        Squared radius and radius of the outer face, and distance between the last two centers.
    V : numpy.ndarray
        Cell volumes divided by :math:`4\pi`, length ``N``.
    dt : float
        Time step.
    boundary : int
        Outer boundary condition, a value of :py:data:`BOUNDARY_CODES`.
    D_face, G : numpy.ndarray
        Face coefficients and conductances :math:`r_f^2 D_f / \Delta r`, shape ``(n_p, N + 1)``, written.
    a_lower, a_diag, a_upper, b_lower, b_diag, b_upper : numpy.ndarray
        Diagonals of :math:`A` and :math:`B`, shape ``(n_p, N)``, written.
    inv_pivots, upper : numpy.ndarray
        Inverse pivots and normalized upper diagonal of the forward elimination of :math:`A`, shape ``(n_p, N)``, written.
    """
    n_rows = a_diag.shape[0]
    for block in prange((n_rows + BLOCK - 1) // BLOCK):
        _assemble_block(
            D,
            h_left,
            h_right,
            rf2,
            d_centers,
            rf_last2,
            rf_last,
            denom_last,
            V,
            dt,
            boundary,
            D_face,
            G,
            a_lower,
            a_diag,
            a_upper,
            b_lower,
            b_diag,
            b_upper,
            inv_pivots,
            upper,
            block * BLOCK,
            min((block + 1) * BLOCK, n_rows),
        )


@njit(cache=True)
def _assemble_block(
    D,
    h_left,
    h_right,
    rf2,
    d_centers,
    rf_last2,
    rf_last,
    denom_last,
    V,
    dt,
    boundary,
    D_face,
    G,
    a_lower,
    a_diag,
    a_upper,
    b_lower,
    b_diag,
    b_upper,
    inv_pivots,
    upper,
    start,
    stop,
):
    """Assemble the matrices of rows ``start:stop``, then factorize them together."""
    n_D = D.shape[0]
    for s in range(start, stop):
        _assemble_row(
            D[0] if n_D == 1 else D[s],
            h_left,
            h_right,
            rf2,
            d_centers,
            rf_last2,
            rf_last,
            denom_last,
            V,
            dt,
            boundary,
            D_face[s],
            G[s],
            a_lower[s],
            a_diag[s],
            a_upper[s],
            b_lower[s],
            b_diag[s],
            b_upper[s],
        )
    _factorize_block(a_lower, a_diag, a_upper, inv_pivots, upper, start, stop)


@njit(cache=True)
def _solve_block(
    f,
    b_lower,
    b_diag,
    b_upper,
    a_lower,
    inv_pivots,
    upper,
    psi_end,
    dirichlet,
    out,
    start,
    stop,
):
    """Solve ``A x = B f`` for rows ``start:stop`` from the elimination of ``A``, into ``out``."""
    N = f.shape[1]
    n_psi_end = psi_end.shape[0]
    # Forward substitution of the right-hand side B f, kept in out
    for i in range(N):
        for s in range(start, stop):
            rhs = b_diag[s, i] * f[s, i]
            if i > 0:
                rhs += b_lower[s, i] * f[s, i - 1]
            if i < N - 1:
                rhs += b_upper[s, i] * f[s, i + 1]
            if dirichlet and i == N - 1:
                rhs = psi_end[0] if n_psi_end == 1 else psi_end[s]
            if i == 0:
                out[s, 0] = rhs * inv_pivots[s, 0]
            else:
                out[s, i] = (rhs - a_lower[s, i] * out[s, i - 1]) * inv_pivots[s, i]
    # Back substitution
    for i in range(N - 2, -1, -1):
        for s in range(start, stop):
            out[s, i] = out[s, i] - upper[s, i] * out[s, i + 1]


@twin_kernel()
def solve_rows(
    f, b_lower, b_diag, b_upper, a_lower, inv_pivots, upper, psi_end, dirichlet, out
):
    """
    Crank-Nicolson step of every momentum: solve :math:`A x = B f` with the elimination of :py:func:`assemble_rows`.

    Parameters
    ----------
    f : numpy.ndarray
        Distribution before the step, shape ``(n_p, N)``.
    b_lower, b_diag, b_upper, a_lower : numpy.ndarray
        Diagonals of :math:`B` and lower diagonal of :math:`A`, shape ``(n_p, N)``.
    inv_pivots, upper : numpy.ndarray
        Forward elimination of :math:`A`, from :py:func:`assemble_rows`.
    psi_end : numpy.ndarray
        Value of the last cell under a Dirichlet condition, of length 1 or ``n_p``.
    dirichlet : bool
        Whether the outer boundary condition is Dirichlet.
    out : numpy.ndarray
        Distribution after the step, shape ``(n_p, N)``, written.
    """
    n_rows = f.shape[0]
    for block in prange((n_rows + BLOCK - 1) // BLOCK):
        _solve_block(
            f,
            b_lower,
            b_diag,
            b_upper,
            a_lower,
            inv_pivots,
            upper,
            psi_end,
            dirichlet,
            out,
            block * BLOCK,
            min((block + 1) * BLOCK, n_rows),
        )
