r"""
Fused Numba kernels of :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

Each kernel advances every row of the conservative variable :math:`U` (shape ``(M, N)``, the active axis last) through a sequence of CFL sub-cycles in a single pass per sub-cycle: limited slopes, reconstruction, MUSCL-Hancock predictor, upwind and boundary fluxes, and the finite-volume update, without temporary arrays.
Rows are independent, so they are distributed over threads by :py:mod:`saetass.parallel` and the result does not depend on the number of threads.

The loops over cells have no divisions and no branches on the data, which would keep the compiler from vectorizing them: they multiply by reciprocal spacings computed once (:py:func:`reciprocal_spacings`), and the limiters are written without branches.
The limiters compare signs instead of testing products such as ``a * b > 0``: a product squares the magnitude of the slopes, so it underflows to zero (silently dropping the scheme to first order) or overflows for slopes beyond ~1e+-154, which would make the kernels depend on the units in which :math:`U` is expressed.
"""

import numpy as np
from numba import njit, prange

from ..parallel import twin_kernel

#: Codes of the slope limiters passed to the kernels.
LIMITERS = {"minmod": 0, "vanleer": 1, "mc": 2}

#: Left boundary of the active axis: no flux (the origin of a spherical grid).
LEFT_ZERO_FLUX = 0
#: Left boundary of the active axis: upwind flux with a linearly extrapolated velocity.
LEFT_EXTRAPOLATED = 1

#: Serial cost per cell and sub-cycle of the kernels, in nanoseconds, to choose their threads.
NS_PER_CELL = {1: 1.5, 2: 6.0}

#: The caller converts between the distribution and the conservative variable :math:`U`.
TRANSFORM_NONE = 0
#: Spherical advection: :math:`U = r^2 \psi`, with the origin cell extrapolated back.
TRANSFORM_SPHERICAL = 1
#: Momentum losses: :math:`U = p\,\psi`.
TRANSFORM_MOMENTUM = 2


def reciprocal_spacings(c, dx, r2=None):
    r"""
    Reciprocal spacings of the centers ``c`` and cells ``dx`` used by :py:func:`advance_rows`, shape ``(4, N)``.

    Rows: :math:`1/(c_i - c_{i-1})` (from ``i = 1``), :math:`1/(c_{i+1} - c_{i-1})` (for ``1 <= i <= N - 2``), :math:`1/\Delta x_i` and :math:`1/r_i^2` from ``r2`` (zero where :math:`r = 0`, and zeros without ``r2``). Unused entries are zero.
    """
    N = c.shape[0]
    inv = np.zeros((4, N))
    inv[0, 1:] = 1.0 / (c[1:] - c[:-1])
    inv[1, 1 : N - 1] = 1.0 / (c[2:] - c[:-2])
    inv[2] = 1.0 / dx
    if r2 is not None:
        nonzero = r2 != 0.0
        inv[3, nonzero] = 1.0 / r2[nonzero]
    return inv


@njit(inline="always", error_model="numpy")
def _minmod(a, b, c):
    r"""
    Minmod of three slopes, without branches: :math:`s \max(\min(|a|, s b, s c), 0)` with :math:`s` the sign of ``a``.

    When the three slopes share a sign, :math:`s b = |b|` and :math:`s c = |c|` exactly, so the result is :math:`s \min(|a|, |b|, |c|)`; otherwise the minimum is not positive and the result is zero.
    Branches on the data would keep the compiler from vectorizing the loops over cells.
    """
    s = np.copysign(1.0, a)
    return s * max(min(abs(a), s * b, s * c), 0.0)


@njit(inline="always", error_model="numpy")
def _vanleer(a, b):
    """
    Van Leer slope, the harmonic mean ``2ab / (a + b)`` of same-sign slopes and zero otherwise, without branches.

    It is evaluated as ``2 sign(a) |a| (|b| / (|a| + |b|))``, so that no intermediate value squares the slopes; ``q`` is ``|b|`` when ``b`` has the sign of ``a`` and zero otherwise.
    """
    s = np.copysign(1.0, a)
    abs_a = abs(a)
    q = max(s * b, 0.0)
    total = abs_a + q
    # Both slopes vanish: any positive denominator gives the zero slope
    total = total if total > 0.0 else 1.0
    return s * (2.0 * abs_a * (q / total))


@njit(inline="always", error_model="numpy")
def _limited_edge(d_one, d_two, limiter):
    """Slope of a boundary cell from its one-cell and two-cell differences."""
    if limiter == 0:
        return _minmod(d_one, d_two, d_one)
    if limiter == 1:
        return _vanleer(d_one, d_two)
    return _minmod(2.0 * d_one, 2.0 * d_two, d_two)


@njit(inline="always", error_model="numpy")
def _slopes(u, inv_left, inv_center, limiter, out):
    """
    Limited slopes of ``u`` on non-uniform centers :math:`c_i`, into ``out``.

    ``inv_left[i]`` is :math:`1 / (c_i - c_{i-1})` and ``inv_center[i]`` is :math:`1 / (c_{i+1} - c_{i-1})`: multiplying by them instead of dividing keeps divisions, which are slow and serialize the loop, out of every cell.
    """
    N = u.shape[0]
    if N <= 2:
        for i in range(N):
            out[i] = 0.0
        return
    # One loop per limiter, so that the loops over cells have no branches
    if limiter == 0:
        for i in range(1, N - 1):
            d_left = (u[i] - u[i - 1]) * inv_left[i]
            d_right = (u[i + 1] - u[i]) * inv_left[i + 1]
            d_center = (u[i + 1] - u[i - 1]) * inv_center[i]
            out[i] = _minmod(d_left, d_right, d_center)
    elif limiter == 1:
        for i in range(1, N - 1):
            d_left = (u[i] - u[i - 1]) * inv_left[i]
            d_right = (u[i + 1] - u[i]) * inv_left[i + 1]
            out[i] = _vanleer(d_left, d_right)
    else:
        for i in range(1, N - 1):
            d_left = (u[i] - u[i - 1]) * inv_left[i]
            d_right = (u[i + 1] - u[i]) * inv_left[i + 1]
            d_center = (u[i + 1] - u[i - 1]) * inv_center[i]
            out[i] = _minmod(2.0 * d_left, 2.0 * d_right, d_center)
    d_one = (u[1] - u[0]) * inv_left[1]
    d_two = (u[2] - u[0]) * inv_center[1]
    out[0] = _limited_edge(d_one, d_two, limiter)
    d_one = (u[N - 1] - u[N - 2]) * inv_left[N - 1]
    d_two = (u[N - 1] - u[N - 3]) * inv_center[N - 2]
    out[N - 1] = _limited_edge(d_one, d_two, limiter)


@njit(inline="always", error_model="numpy")
def _spherical_slopes(
    u, inv_left, inv_center, r, r2, inv_r2, limiter, f, slopes_f, out
):
    r"""
    Slopes of :math:`U = r^2 f` from the limited slopes of :math:`f`.

    Limiting the nearly flat :math:`f` instead of the quadratic :math:`U` avoids clipping the slopes; they map back as :math:`dU/dr = 2 r f + r^2 df/dr`.
    The origin cell, where :math:`r = 0`, takes :math:`f` linearly extrapolated from its neighbours.
    """
    N = u.shape[0]
    for i in range(N):
        f[i] = u[i] * inv_r2[i]  # zero where r = 0
    # Grids of fewer than 3 cells cannot extrapolate (and fail when inverting U)
    if r2[0] == 0.0 and N >= 3:
        f[0] = 2.0 * f[1] - f[2]
    _slopes(f, inv_left, inv_center, limiter, slopes_f)
    for i in range(N):
        out[i] = 2.0 * r[i] * f[i] + r2[i] * slopes_f[i]


@njit(inline="always", error_model="numpy")
def _first_order_fluxes(u, vc, vf, inflow, F):
    N = u.shape[0]
    F[0] = 0.0
    for k in range(N - 1):
        F[k + 1] = vf[k] * u[k] if vf[k] >= 0.0 else vf[k] * u[k + 1]
    F[N] = u[N - 1] * (vc[N - 1] if vc[N - 1] >= 0.0 else inflow)


@njit(inline="always", error_model="numpy")
def _second_order_fluxes(
    u,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    inflow,
    dt,
    limiter,
    left_mode,
    spherical,
    w,
):
    """MUSCL-Hancock fluxes at the ``N + 1`` faces, with the reciprocal spacings ``inv`` and the work arrays ``w``."""
    N = u.shape[0]
    inv_left, inv_center, inv_r2 = inv[0], inv[1], inv[3]
    slopes_u, slopes_F, F_centers, f_work, slopes_work, F = (
        w[0],
        w[1],
        w[2],
        w[3],
        w[4],
        w[5],
    )
    for i in range(N):
        F_centers[i] = vc[i] * u[i]
    if spherical:
        _spherical_slopes(
            u,
            inv_left,
            inv_center,
            r,
            r2,
            inv_r2,
            limiter,
            f_work,
            slopes_work,
            slopes_u,
        )
        _spherical_slopes(
            F_centers,
            inv_left,
            inv_center,
            r,
            r2,
            inv_r2,
            limiter,
            f_work,
            slopes_work,
            slopes_F,
        )
    else:
        _slopes(u, inv_left, inv_center, limiter, slopes_u)
        _slopes(F_centers, inv_left, inv_center, limiter, slopes_F)

    half_dt = 0.5 * dt
    # Internal face k + 1: upwind of the states predicted to half the step
    for k in range(N - 1):
        if vf[k] >= 0.0:
            F[k + 1] = vf[k] * ((u[k] + slopes_u[k] * dx_R[k]) - half_dt * slopes_F[k])
        else:
            F[k + 1] = vf[k] * (
                (u[k + 1] - slopes_u[k + 1] * dx_L[k + 1]) - half_dt * slopes_F[k + 1]
            )

    if left_mode == LEFT_ZERO_FLUX:
        F[0] = 0.0
    else:
        slope = (vc[1] - vc[0]) * inv_left[1]
        v_in = vc[0] - slope * (c[0] - f[0])
        if v_in >= 0.0:
            F[0] = v_in * 0.0
        else:
            F[0] = v_in * ((u[0] - slopes_u[0] * dx_L[0]) - half_dt * slopes_F[0])

    slope = (vc[N - 1] - vc[N - 2]) * inv_left[N - 1]
    v_out = vc[N - 1] + slope * (f[N] - c[N - 1])
    if v_out >= 0.0:
        F[N] = v_out * (
            (u[N - 1] + slopes_u[N - 1] * dx_R[N - 1]) - half_dt * slopes_F[N - 1]
        )
    else:
        F[N] = v_out * inflow


@njit(inline="always", error_model="numpy")
def _sub_cycle(
    u,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    inflow,
    dt,
    order,
    limiter,
    left_mode,
    spherical,
    work,
):
    """Advance ``u`` by one CFL sub-cycle of length ``dt``."""
    N = u.shape[0]
    F = work[5]
    if order == 1:
        _first_order_fluxes(u, vc, vf, inflow, F)
    else:
        _second_order_fluxes(
            u,
            vc,
            vf,
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            inflow,
            dt,
            limiter,
            left_mode,
            spherical,
            work,
        )
    inv_dx = inv[2]
    for i in range(N):
        u[i] = u[i] - (dt * inv_dx[i]) * (F[i + 1] - F[i])


@njit(cache=True, error_model="numpy")
def _advance_row(
    u,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    work,
):
    """Advance one row ``u`` in place through the sub-cycles of length ``steps``, with the work arrays ``work``."""
    for dt in steps:
        _sub_cycle(
            u,
            vc,
            vf,
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            inflow,
            dt,
            order,
            limiter,
            left_mode,
            spherical,
            work,
        )


@njit(inline="always", error_model="numpy")
def _load(values_in, scale, U):
    r"""Conservative variable of one row, :math:`U = \psi \times` ``scale``."""
    for i in range(U.shape[0]):
        U[i] = values_in[i] * scale[i]


@njit(inline="always", error_model="numpy")
def _store(U, scale, inv_scale, transform, values_out):
    r"""
    Distribution of one row, :math:`\psi = U \times` ``inv_scale``, with the origin cell of spherical advection (where ``inv_scale`` is zero) extrapolated linearly, and the positivity floor :math:`\psi \geq 0`.
    """
    N = U.shape[0]
    for i in range(N):
        values_out[i] = U[i] * inv_scale[i]
    if transform == TRANSFORM_SPHERICAL and scale[0] == 0.0 and N >= 3:
        values_out[0] = 2.0 * values_out[1] - values_out[2]
    # Positivity floor, which keeps NaN as NumPy's maximum does
    for i in range(N):
        value = values_out[i]
        if not (value >= 0.0 or value != value):
            values_out[i] = 0.0


@njit(cache=True, error_model="numpy")
def _process_row(
    values_in,
    U,
    values_out,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
    work,
):
    r"""
    Advance one row: load :math:`U` from the distribution, run the sub-cycles, store the distribution back.

    The conversions are :math:`U = \psi\,r^2` and :math:`\psi = U\,(1/r^2)` (zero where :math:`r = 0`, then the origin cell extrapolated linearly), or :math:`U = p\,\psi` and :math:`\psi = U\,(1/p)`, followed by the positivity floor :math:`\psi \geq 0`.
    """
    if load:
        _load(values_in, scale, U)
    _advance_row(
        U,
        vc,
        vf,
        c,
        f,
        inv,
        dx_L,
        dx_R,
        r,
        r2,
        inflow,
        steps,
        order,
        limiter,
        left_mode,
        spherical,
        work,
    )
    if store:
        _store(U, scale, inv_scale, transform, values_out)


@njit(cache=True, error_model="numpy")
def _process_rows(
    start,
    stop,
    values_in,
    U,
    values_out,
    V_centers,
    V_faces,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
):
    """Advance rows ``start:stop``, sharing one set of work arrays."""
    N = U.shape[1]
    work = (
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N + 1),
    )
    n_inflow = inflow.shape[0]
    for j in range(start, stop):
        _process_row(
            values_in[j],
            U[j],
            values_out[j],
            V_centers[j],
            V_faces[j],
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            scale,
            inv_scale,
            inflow[0] if n_inflow == 1 else inflow[j],
            steps,
            order,
            limiter,
            left_mode,
            spherical,
            transform,
            load,
            store,
            work,
        )


@twin_kernel(error_model="numpy")
def advance_rows(
    values_in,
    U,
    values_out,
    V_centers,
    V_faces,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
    n_blocks,
):
    r"""
    Advance every row of ``U`` in place through the sub-cycles of length ``steps``.

    With ``load``, rows of ``U`` are first computed from the rows of the distribution ``values_in``; with ``store``, the distribution is written back to the rows of ``values_out``. Both may be strided views (the rows of a transposed array), so the caller never copies the state.

    Rows are advanced in ``n_blocks`` contiguous blocks, each with its own work arrays, so
    that short rows do not pay an allocation each. The body of the parallel loop is a
    single call to the serial block kernel: Numba's parallel transformation of a larger
    body changed its results (and raced between threads), while a called function is
    compiled on its own.

    Parameters
    ----------
    values_in, values_out : numpy.ndarray
        Rows of the distribution read with ``load`` and written with ``store``, shape ``(M, N)``.
    U : numpy.ndarray
        Conservative variable, shape ``(M, N)``, updated in place.
    V_centers, V_faces : numpy.ndarray
        Generalized velocities at the cell centers ``(M, N)`` and internal faces ``(M, N - 1)``.
    c, f, dx_L, dx_R : numpy.ndarray
        Centers, faces and center-to-face distances of the active axis.
    inv : numpy.ndarray
        Reciprocal spacings, shape ``(4, N)``: :math:`1/(c_i - c_{i-1})`, :math:`1/(c_{i+1} - c_{i-1})`, :math:`1/\Delta x_i` and :math:`1/r_i^2` (zero where :math:`r = 0`, or unused); see :py:func:`reciprocal_spacings`.
    r, r2 : numpy.ndarray
        Radii and squared radii of the centers when ``spherical`` (otherwise unused).
    scale, inv_scale : numpy.ndarray
        :math:`U / \psi` and its inverse (zero where it vanishes) for ``transform``.
    inflow : numpy.ndarray
        :math:`U` entering through the last face, of length 1 or ``M``.
    steps : numpy.ndarray
        Time steps of the successive sub-cycles.
    order, limiter, left_mode : int
        Order of the scheme (1 or 2), limiter code (:py:data:`LIMITERS`) and left boundary.
    spherical : bool
        Whether slopes are limited on :math:`f = U / r^2` (spherical advection).
    transform : int
        Conversion between the distribution and :math:`U`, one of the ``TRANSFORM_*`` codes.
    load, store : bool
        Whether to compute ``U`` from ``values_in`` before the sub-cycles and to write ``values_out`` after them.
    n_blocks : int
        Number of blocks of rows: 1 for a serial call, a few per thread otherwise.
    """
    M = U.shape[0]
    for block in prange(n_blocks):
        _process_rows(
            block * M // n_blocks,
            (block + 1) * M // n_blocks,
            values_in,
            U,
            values_out,
            V_centers,
            V_faces,
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            scale,
            inv_scale,
            inflow,
            steps,
            order,
            limiter,
            left_mode,
            spherical,
            transform,
            load,
            store,
        )


# ---------------------------------------------------------------------------
# Velocities that combine components (Separable parameters)
# ---------------------------------------------------------------------------


@njit(inline="always", error_model="numpy")
def _combine(factors, components, out):
    """``out = sum_k factors[k] * components[k]``, accumulated in the order of the components."""
    for i in range(out.shape[0]):
        out[i] = factors[0] * components[0, i]
    for k in range(1, factors.shape[0]):
        for i in range(out.shape[0]):
            out[i] += factors[k] * components[k, i]


@njit(cache=True, error_model="numpy")
def _advance_row_combined(
    u,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    work,
    factors,
):
    r"""
    :py:func:`_advance_row` with velocities that combine components for every sub-cycle.

    ``vc`` and ``vf`` hold the components at the centers ``(K, N)`` and faces ``(K, N - 1)``; the velocities of sub-cycle ``s`` are :math:`\sum_k` ``factors[s, k]`` times component ``k``.
    """
    vc_step, vf_step = work[6], work[7]
    for s in range(steps.shape[0]):
        _combine(factors[s], vc, vc_step)
        _combine(factors[s], vf, vf_step)
        _sub_cycle(
            u,
            vc_step,
            vf_step,
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            inflow,
            steps[s],
            order,
            limiter,
            left_mode,
            spherical,
            work,
        )


@njit(cache=True, error_model="numpy")
def _process_row_combined(
    values_in,
    U,
    values_out,
    vc,
    vf,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
    work,
    factors,
):
    r"""
    :py:func:`_process_row` with velocities that combine components (see :py:func:`_advance_row_combined`).
    """
    if load:
        _load(values_in, scale, U)
    _advance_row_combined(
        U,
        vc,
        vf,
        c,
        f,
        inv,
        dx_L,
        dx_R,
        r,
        r2,
        inflow,
        steps,
        order,
        limiter,
        left_mode,
        spherical,
        work,
        factors,
    )
    if store:
        _store(U, scale, inv_scale, transform, values_out)


@njit(cache=True, error_model="numpy")
def _process_rows_combined(
    start,
    stop,
    values_in,
    U,
    values_out,
    V_centers,
    V_faces,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
    factors,
):
    """:py:func:`_process_rows` with velocities that combine components ``(K, M, N)``."""
    N = U.shape[1]
    work = (
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N),
        np.empty(N + 1),
        np.empty(N),
        np.empty(max(N - 1, 0)),
    )
    n_inflow = inflow.shape[0]
    for j in range(start, stop):
        _process_row_combined(
            values_in[j],
            U[j],
            values_out[j],
            V_centers[:, j],
            V_faces[:, j],
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            scale,
            inv_scale,
            inflow[0] if n_inflow == 1 else inflow[j],
            steps,
            order,
            limiter,
            left_mode,
            spherical,
            transform,
            load,
            store,
            work,
            factors,
        )


@twin_kernel(error_model="numpy")
def advance_rows_combined(
    values_in,
    U,
    values_out,
    V_centers,
    V_faces,
    c,
    f,
    inv,
    dx_L,
    dx_R,
    r,
    r2,
    scale,
    inv_scale,
    inflow,
    steps,
    order,
    limiter,
    left_mode,
    spherical,
    transform,
    load,
    store,
    factors,
    n_blocks,
):
    r"""
    :py:func:`advance_rows` with velocities that combine components for every sub-cycle.

    ``V_centers`` and ``V_faces`` hold ``K`` components, shapes ``(K, M, N)`` and ``(K, M, N - 1)``, and ``factors`` (``(len(steps), K)``) their factors: the velocities of sub-cycle ``s`` are :math:`\sum_k` ``factors[s, k]`` times component ``k``. This is how :py:class:`~saetass.parameters.Separable` parameters advance every sub-cycle of a step in one call.
    """
    M = U.shape[0]
    for block in prange(n_blocks):
        _process_rows_combined(
            block * M // n_blocks,
            (block + 1) * M // n_blocks,
            values_in,
            U,
            values_out,
            V_centers,
            V_faces,
            c,
            f,
            inv,
            dx_L,
            dx_R,
            r,
            r2,
            scale,
            inv_scale,
            inflow,
            steps,
            order,
            limiter,
            left_mode,
            spherical,
            transform,
            load,
            store,
            factors,
        )


@njit(cache=True, error_model="numpy")
def faces_and_max(V, dist_left, dist_right, denom, out):
    """
    Velocities at the internal faces of the rows ``V`` (``M, N``) into ``out`` (``M, N - 1``), and their largest magnitude.

    The interpolation is that of :py:meth:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver._face_generalized_velocity_interpolated`, with the same arithmetic.
    """
    M, N = V.shape
    largest = 0.0
    for m in range(M):
        for k in range(N - 1):
            value = (V[m, k] * dist_right[k] + V[m, k + 1] * dist_left[k]) / denom[k]
            out[m, k] = value
            largest = max(largest, abs(value))
    return largest


@njit(cache=True, error_model="numpy")
def separable_largest_speed(factors, V_faces, V_last):
    r"""
    Largest magnitude of :math:`\sum_k` ``factors[k]`` times the components ``V_faces`` (``K, M, N - 1``) and ``V_last`` (``K, L``).

    The velocities of the CFL condition of a separable parameter: those of the internal faces and of the last cells along the active axis.
    """
    K = factors.shape[0]
    largest = 0.0
    for m in range(V_faces.shape[1]):
        for i in range(V_faces.shape[2]):
            value = factors[0] * V_faces[0, m, i]
            for k in range(1, K):
                value += factors[k] * V_faces[k, m, i]
            largest = max(largest, abs(value))
    for i in range(V_last.shape[1]):
        value = factors[0] * V_last[0, i]
        for k in range(1, K):
            value += factors[k] * V_last[k, i]
        largest = max(largest, abs(value))
    return largest
