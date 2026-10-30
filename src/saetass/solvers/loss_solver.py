import logging
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import units as su
from ..grid import Grid
from ..parameters import CanonicalSeparable
from ..solver import ParamSpec
from . import hyperbolic_kernels as kernels
from .hyperbolic_solver import HyperbolicSolver

logger = logging.getLogger(__name__)


def velocity_divergence(grid: Grid, v_centers: np.ndarray) -> np.ndarray:
    r"""
    Divergence of a radial flow, :math:`\nabla \cdot v = r^{-2}\, \partial_r (r^2 v)`, averaged over each radial cell of a :py:class:`~saetass.grid.Grid`.

    By the divergence theorem, the average over cell :math:`i` is the net flow through its faces divided by its volume,

    .. math::

        \langle \nabla \cdot v \rangle_i = \frac{A_{i+1/2}\, v_{i+1/2} - A_{i-1/2}\, v_{i-1/2}}{V_i},

    with the face areas :py:attr:`~saetass.grid.Grid.face_areas` and cell volumes :py:attr:`~saetass.grid.Grid.volumes` of the grid.
    The velocities at the faces are those of the advection operator: interpolated linearly between the two neighbouring centers at internal faces, and extrapolated linearly from the two closest centers at the boundary faces.
    A cell centered at the origin is the ball inside its outer face, with no flow through its center.
    Across a shock, the averages of the cells next to it add up to the jump of :math:`r^2 v`, whatever the resolution.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        Grid with a radial axis of at least 2 cells.
    v_centers : numpy.ndarray
        Radial velocities at the cell centers, with the radial axis last, in canonical :py:data:`~saetass.units.VELOCITY` units.

    Returns
    -------
    numpy.ndarray
        Divergence averaged over each cell, with the shape of ``v_centers``, in inverse canonical :py:data:`~saetass.units.TIME` units.
    """
    return _divergence(_divergence_geometry(grid), v_centers)


def _divergence_geometry(grid: Grid) -> tuple[np.ndarray, ...]:
    """Bare canonical centers, faces, face areas and inverse volumes of the radial cells, computed once per operator."""
    if grid.r_centers is None or grid.r_centers.size < 2:
        raise ValueError("The divergence of a velocity needs at least 2 radial cells.")
    centers = grid.r_centers.to_value(su.LENGTH)
    faces = grid.r_faces.to_value(su.LENGTH)
    areas = grid.face_areas.to_value(su.AREA)
    volumes = grid.volumes.to_value(su.VOLUME)
    return centers, faces, areas, 1.0 / volumes


def _divergence(geometry: tuple[np.ndarray, ...], v_centers: np.ndarray) -> np.ndarray:
    """:py:func:`velocity_divergence` with the geometry of :py:func:`_divergence_geometry`."""
    centers, faces, areas, inv_volumes = geometry
    v_centers = np.asarray(v_centers, dtype=float)
    v_faces = np.empty(v_centers.shape[:-1] + (faces.size,))
    # Internal faces: the distance-weighted interpolation of the advection operator
    dist_left = faces[1:-1] - centers[:-1]
    dist_right = centers[1:] - faces[1:-1]
    v_faces[..., 1:-1] = (
        v_centers[..., :-1] * dist_right + v_centers[..., 1:] * dist_left
    ) / (dist_left + dist_right)
    # Boundary faces: linear extrapolation from the two closest centers
    slope = (v_centers[..., -1] - v_centers[..., -2]) / (centers[-1] - centers[-2])
    v_faces[..., -1] = v_centers[..., -1] + slope * (faces[-1] - centers[-1])
    slope = (v_centers[..., 1] - v_centers[..., 0]) / (centers[1] - centers[0])
    v_faces[..., 0] = v_centers[..., 0] - slope * (centers[0] - faces[0])

    flow = areas * v_faces
    return (flow[..., 1:] - flow[..., :-1]) * inv_volumes


def _add_velocities(first: Any, second: Any) -> Any:
    """Sum of two generalized velocities, each static, a callable of time or a :py:class:`~saetass.parameters.CanonicalSeparable`."""
    if first is None or second is None:
        return second if first is None else first
    if not callable(first) and not callable(second):
        return first + second
    # Keep a separable sum separable, so that the kernels still combine it in one call
    if isinstance(first, CanonicalSeparable) or isinstance(second, CanonicalSeparable):
        parts = [first, second]
        if all(
            isinstance(part, CanonicalSeparable) or not callable(part) for part in parts
        ):
            profiles, factors = [], []
            for part in parts:
                if isinstance(part, CanonicalSeparable):
                    profiles += part.profiles
                    factors += part.factors
                else:
                    profiles.append(part)
                    factors.append(_unit_factor)
            shape = np.broadcast_shapes(*(np.shape(profile) for profile in profiles))
            return CanonicalSeparable(
                [np.broadcast_to(profile, shape).copy() for profile in profiles],
                factors,
            )

    def total(t: float) -> np.ndarray:
        return _value_at(first, t) + _value_at(second, t)

    return total


def _unit_factor(t: float) -> float:
    return 1.0


def _value_at(velocity: Any, t: float) -> np.ndarray:
    return velocity(t) if callable(velocity) else velocity


class LossSolver(HyperbolicSolver):
    r"""
    Finite volume solver for energy losses in momentum space, inheriting from :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

    Solves the momentum-loss equation in conservative form,

    .. math::

        \frac{\partial \psi}{\partial t} + \frac{\partial}{\partial p}\bigl(\dot{p}(t,p)\,\psi\bigr) = 0,

    where :math:`\dot{p} = dp/dt` is the (signed) momentum loss rate, negative for losses. The solver uses the conservative variable :math:`U = p \psi` and, also, :math:`V(t,y) = \frac{\dot{p}}{p\ln(10)}` and :math:`y = \log_{10}(p)`.
    The finite volume update is delegated to the base class across the momentum (p) axis.

    With ``adiabatic_losses=True``, the momentum change of the particles in a flow of velocity :math:`v(t, r)`,

    .. math::

        \dot{p}_\mathrm{ad} = -\frac{p}{3}\, \nabla \cdot v,

    is added to :math:`\dot{p}`: a loss where the flow expands and a gain where it is compressed. Together with :py:class:`~saetass.solvers.advection_solver.AdvectionSolver`, it completes the transport equation of :cite:ct:`Parker1965`.
    The divergence is averaged over each radial cell by :py:func:`velocity_divergence`. At a shock, it concentrates the velocity jump in the cells next to it, which together with diffusion produces diffusive shock acceleration (:cite:ct:`Drury1983`), provided that the diffusion length :math:`D / v` spans several radial cells.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``p_centers`` and ``p_faces``; optionally ``r_centers`` and ``r_faces`` for 2D problems.
    t_grid : numpy.ndarray
        Subproblem time grid. In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        P_dot : astropy.units.Quantity or callable, optional
            Momentum loss rate, :math:`\dot{p}`, at cell centers, without the adiabatic losses. A callable must have signature ``P_dot(t: Quantity) -> Quantity``. A :py:class:`~saetass.parameters.Separable` is also accepted. Required unless ``adiabatic_losses`` is ``True``. Units compatible with :py:data:`~saetass.units.MOMENTUM_LOSS_RATE`.
        limiter : ``{'minmod', 'vanleer', 'mc'}``, optional
            Slope limiter of the second-order scheme. Default is ``'minmod'`` for order 2; it must not be given for order 1, which does not limit slopes.
        cfl : float, optional
            CFL number for the adaptive sub-step calculation. Default is ``0.5``.
        inflow_value_psi : astropy.units.Quantity, optional
            Differential density :math:`\psi` at the high-momentum boundary, used as an inflow condition when :math:`\dot{p} > 0` (i.e. momentum gain). Units compatible with :py:data:`~saetass.units.PSI_P`. If neither ``inflow_value_psi`` nor ``inflow_value_U`` is given, nothing enters. Default is zero.
        inflow_value_U : astropy.units.Quantity, optional
            Same inflow condition given directly for the conservative variable :math:`U = p \psi`. Mutually exclusive with ``inflow_value_psi``. Units compatible with :py:data:`~saetass.units.MOMENTUM` times :py:data:`~saetass.units.PSI_P`. Default is zero.
        order : ``{1, 2}``, optional
            Order of the numerical scheme: 1 for first-order upwind, 2 for second-order MUSCL-Hancock. Default is ``2``.
        adiabatic_losses : bool, optional
            Whether to add the adiabatic momentum change :math:`-(p/3)\, \nabla \cdot v` of the flow ``v_centers_physical`` to ``P_dot``. Requires a radial axis. Default is ``False``.
        v_centers_physical : astropy.units.Quantity or callable, optional
            Velocity of the flow at the radial cell centers for the adiabatic losses, of shape ``(n_r,)`` or ``(n_p, n_r)``. A callable must have signature ``v_centers_physical(t: Quantity) -> Quantity``. A :py:class:`~saetass.parameters.Separable` is also accepted. Only used with ``adiabatic_losses=True``. Units compatible with :py:data:`~saetass.units.VELOCITY`. Default is the ``v_centers`` of the advection operator, which :py:class:`~saetass.solver.Solver` passes when the problem includes advection.
    """

    PARAM_SPECS = MappingProxyType(
        {
            **HyperbolicSolver.PARAM_SPECS,
            "P_dot": ParamSpec(su.MOMENTUM_LOSS_RATE, dynamic=True),
            "inflow_value_psi": ParamSpec(su.PSI_P),
            "inflow_value_U": ParamSpec(su.MOMENTUM * su.PSI_P),
            "adiabatic_losses": ParamSpec(),
            "v_centers_physical": ParamSpec(su.VELOCITY, dynamic=True),
        }
    )

    _KERNEL_TRANSFORM = kernels.TRANSFORM_MOMENTUM

    @classmethod
    def convert_params(cls, params: dict[str, Any], grid: Grid) -> dict[str, Any]:
        """
        Convert the parameters as :py:meth:`SubSolver.convert_params <saetass.solver.SubSolver.convert_params>` does.

        ``P_dot`` is required unless ``adiabatic_losses`` is ``True``.
        """
        converted = super().convert_params(params, grid)
        if "P_dot" not in params and not params.get("adiabatic_losses", False):
            raise ValueError(
                f"Missing required {cls.__name__} parameter(s) ['P_dot']: "
                "give P_dot, adiabatic_losses=True or both."
            )
        return converted

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initialize the loss solver."""
        self._validate_unit_free_inputs(t_grid, params)
        if grid.p_centers_phys is None:
            raise ValueError("LossSolver requires a Grid with a momentum axis.")
        # Physical momenta and their reciprocals are cached once: the conservative
        # transforms and the generalized velocity run every (sub)step.
        self.p_centers_phys = grid.p_centers_phys.to_value(su.MOMENTUM)
        if np.any(self.p_centers_phys <= 0.0):
            raise ValueError(
                "LossSolver requires strictly positive momentum cell centers."
            )
        self._inv_p = 1.0 / self.p_centers_phys
        self._inv_p_ln10 = self._inv_p / np.log(10.0)

        # Convert momentum loss parameters to general hyperbolic solver format
        loss_params = params.copy()

        # Set momentum axis as the main axis for losses
        loss_params["axis"] = 0

        adiabatic = loss_params.pop("adiabatic_losses", False)
        v_flow = loss_params.pop("v_centers_physical", None)
        if v_flow is not None and not adiabatic:
            raise ValueError(
                "v_centers_physical is only used with adiabatic_losses=True."
            )
        P_dot_input = loss_params.pop("P_dot", None)
        if P_dot_input is None and not adiabatic:
            raise ValueError("LossSolver needs P_dot, adiabatic_losses=True or both.")

        # Rename loss-specific parameters to match the base class
        V_losses = None
        if P_dot_input is not None:
            if isinstance(P_dot_input, CanonicalSeparable):
                # The generalized velocity is linear in P_dot: transform each profile
                V_losses = P_dot_input.map(self._generalized_velocity)
            elif callable(P_dot_input):

                def dynamic_V_centers(t):
                    return self._generalized_velocity(P_dot_input(t))

                V_losses = dynamic_V_centers
            else:
                V_losses = self._generalized_velocity(P_dot_input)

        V_adiabatic = None
        if adiabatic:
            if grid.r_centers is None:
                raise ValueError("Adiabatic losses require a Grid with a radial axis.")
            if v_flow is None:
                raise ValueError(
                    "Adiabatic losses need the velocity of the flow, v_centers_physical."
                )
            self._divergence_geometry = _divergence_geometry(grid)
            self._shape = (self.p_centers_phys.size, grid.r_centers.size)
            if isinstance(v_flow, CanonicalSeparable):
                V_adiabatic = v_flow.map(self._adiabatic_velocity)
            elif callable(v_flow):

                def dynamic_V_adiabatic(t):
                    return self._adiabatic_velocity(v_flow(t))

                V_adiabatic = dynamic_V_adiabatic
            else:
                V_adiabatic = self._adiabatic_velocity(v_flow)

            # Momentum-only rates (n_p,) apply at every radius
            V_losses = self._as_grid_shape(V_losses)
        loss_params["V_centers"] = _add_velocities(V_losses, V_adiabatic)

        if "inflow_value_psi" in loss_params:
            if "inflow_value_U" in loss_params:
                raise ValueError(
                    "Provide either inflow_value_psi or inflow_value_U, not both."
                )
            loss_params["inflow_value_U"] = self._generalized_variable(
                loss_params.pop("inflow_value_psi"), grid
            )[-1]
        loss_params.setdefault("inflow_value_U", 0.0)

        # Initialize the base class
        super().__init__(grid, t_grid, loss_params, **kwargs)

    def _kernel_scales(self) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`U / \psi = p` and its cached inverse."""
        return self.p_centers_phys, self._inv_p

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the primitive differential density to the conservative variable.
        """
        return self.p_centers_phys * f

    def _inverse_generalized_variable(self, U: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the conservative variable back to the primitive differential density.
        """
        inv_p = self._inv_p  # p > 0 is guaranteed at construction

        # 1D CASE
        if U.ndim == 1:
            if U.shape[0] != inv_p.shape[0]:
                raise ValueError(f"Shape mismatch: U {U.shape}, p {inv_p.shape}")
            return U * inv_p

        # 2D CASE: U is (n_r, n_p), broadcast along the last (momentum) axis
        elif U.ndim == 2:
            if U.shape[1] != inv_p.shape[0]:
                raise ValueError(
                    f"Expected U.shape[1] == p.shape[0], got U {U.shape}, p {inv_p.shape}"
                )
            return U * inv_p

        else:
            raise ValueError(
                "inverse_generalized_variable only supports 1D or 2D arrays."
            )

    def loss_rates(self, t: float) -> np.ndarray:
        r"""
        Relative momentum loss rates :math:`|\dot{p}| / p` at cell centers at time ``t``, without modifying the solver state.

        Parameters
        ----------
        t : float
            Time, as a bare float in canonical :py:data:`~saetass.units.TIME` units.

        Returns
        -------
        numpy.ndarray
            Loss rates with the shape of the :py:class:`~saetass.state.State` differential density, in inverse canonical :py:data:`~saetass.units.TIME` units.
        """
        return np.log(10.0) * np.abs(self.velocities(t)).T

    def _characteristic_timescale(self, V_centers: np.ndarray) -> float:
        r"""Shortest loss timescale :math:`\min p / |\dot{p}| = 1 / (\ln 10 \max |V|)`."""
        V_max = float(np.max(np.abs(V_centers)))
        return 1.0 / (np.log(10.0) * V_max) if V_max > 0.0 else np.inf

    def _adiabatic_velocity(self, v_flow: np.ndarray) -> np.ndarray:
        r"""
        Generalized velocity of the adiabatic losses, :math:`\dot{p}_\mathrm{ad} / (p \ln 10) = -\nabla \cdot v / (3 \ln 10)`, of shape ``(n_p, n_r)``.

        It does not depend on the momentum: on the logarithmic momentum axis, adiabatic changes shift every momentum by the same amount.
        """
        divergence = _divergence(self._divergence_geometry, v_flow)
        V = divergence * (-1.0 / (3.0 * np.log(10.0)))
        return np.broadcast_to(V, self._shape).copy()

    def _as_grid_shape(self, V: Any) -> Any:
        """Generalized velocities of shape ``(n_p, n_r)``, those of shape ``(n_p,)`` repeated at every radius."""
        if V is None:
            return V
        shape = self._shape
        if isinstance(V, CanonicalSeparable):
            return V.map(lambda profile: self._broadcast(profile, shape))
        if callable(V):
            return lambda t: self._broadcast(V(t), shape)
        return self._broadcast(V, shape)

    @staticmethod
    def _broadcast(V: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            V = V[:, np.newaxis]
        return np.broadcast_to(V, shape).copy()

    def _generalized_velocity(self, P_dot: np.ndarray) -> np.ndarray:
        r"""
        Convert the physical momentum loss rate to the generalized velocity :math:`\dot{p} / (p \ln 10)` used by the base-class finite-volume update.

        ``P_dot`` has shape ``(n_p,)`` or ``(n_p, n_r)``; a single broadcast multiply avoids per-call temporaries.
        """
        P_dot = np.asarray(P_dot, dtype=float)
        inv = self._inv_p_ln10 if P_dot.ndim == 1 else self._inv_p_ln10[:, None]
        return P_dot * inv
