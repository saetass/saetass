import logging
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import units as su
from ..grid import Grid
from ..solver import ParamSpec
from . import hyperbolic_kernels as kernels
from .hyperbolic_solver import HyperbolicSolver

logger = logging.getLogger(__name__)


class LossSolver(HyperbolicSolver):
    r"""
    Finite volume solver for energy losses in momentum space, inheriting from :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

    Solves the momentum-loss equation in conservative form,

    .. math::

        \frac{\partial \psi}{\partial t} + \frac{\partial}{\partial p}\bigl(\dot{p}(t,p)\,\psi\bigr) = 0,

    where :math:`\dot{p} = dp/dt \leq 0` is the (signed) momentum loss rate. The solver uses the conservative variable :math:`U = p \psi` and, also, :math:`V(t,y) = \frac{\dot{p}}{p\ln(10)}` and :math:`y = \log_{10}(p)`.
    The finite volume update is delegated to the base class across the momentum (p) axis.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``p_centers`` and ``p_faces``; optionally ``r_centers`` and ``r_faces`` for 2D problems.
    t_grid : numpy.ndarray
        Subproblem time grid. In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        P_dot : astropy.units.Quantity or callable
            Momentum loss rate, :math:`\dot{p}`, at cell centers. Units compatible with :py:data:`~saetass.units.MOMENTUM_LOSS_RATE`. A callable must have signature ``P_dot(t: Quantity) -> Quantity``.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL number for the adaptive sub-step calculation.
        inflow_value_psi : astropy.units.Quantity, optional
            Differential density :math:`\psi` at the high-momentum boundary, used as an inflow condition when :math:`\dot{p} > 0` (i.e. momentum gain). Units compatible with :py:data:`~saetass.units.PSI_P`.
        inflow_value_U : astropy.units.Quantity, optional
            Same inflow condition given directly for the conservative variable :math:`U = p \psi`. Mutually exclusive with ``inflow_value_psi``. Units compatible with :py:data:`~saetass.units.MOMENTUM` times :py:data:`~saetass.units.PSI_P`.
        order : ``{1, 2}``
            Order of the numerical scheme.
        adiabatic_losses : bool, optional
            Reserved for adiabatic losses, which are not implemented yet: ``True`` raises :py:class:`NotImplementedError`. Default is ``False``.
        v_centers_physical : astropy.units.Quantity, optional
            Reserved for the advection velocity of adiabatic losses; it is currently ignored. Units compatible with :py:data:`~saetass.units.VELOCITY`.
    """

    PARAM_SPECS = MappingProxyType(
        {
            **HyperbolicSolver.PARAM_SPECS,
            "P_dot": ParamSpec(su.MOMENTUM_LOSS_RATE, dynamic=True),
            "inflow_value_psi": ParamSpec(su.PSI_P),
            "inflow_value_U": ParamSpec(su.MOMENTUM * su.PSI_P),
            "adiabatic_losses": ParamSpec(),
            "v_centers_physical": ParamSpec(su.VELOCITY),
        }
    )

    _KERNEL_TRANSFORM = kernels.TRANSFORM_MOMENTUM

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

        loss_params.pop("v_centers_physical", None)
        if loss_params.pop("adiabatic_losses", False):
            raise NotImplementedError(
                "Adiabatic losses are not implemented yet: set adiabatic_losses=False."
            )

        # Rename loss-specific parameters to match the base class
        if "P_dot" in loss_params:
            P_dot_input = loss_params.pop("P_dot")
            if callable(P_dot_input):

                def dynamic_V_centers(t):
                    return self._generalized_velocity(P_dot_input(t))

                loss_params["V_centers"] = dynamic_V_centers
            else:
                loss_params["V_centers"] = self._generalized_velocity(P_dot_input)

        if "inflow_value_psi" in loss_params:
            if "inflow_value_U" in loss_params:
                raise ValueError(
                    "Provide either inflow_value_psi or inflow_value_U, not both."
                )
            loss_params["inflow_value_U"] = self._generalized_variable(
                loss_params.pop("inflow_value_psi"), grid
            )[-1]

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

    def _generalized_velocity(self, P_dot: np.ndarray) -> np.ndarray:
        r"""
        Convert the physical momentum loss rate to the generalized velocity :math:`\dot{p} / (p \ln 10)` used by the base-class finite-volume update.

        ``P_dot`` has shape ``(n_p,)`` or ``(n_p, n_r)``; a single broadcast multiply avoids per-call temporaries.
        """
        P_dot = np.asarray(P_dot, dtype=float)
        inv = self._inv_p_ln10 if P_dot.ndim == 1 else self._inv_p_ln10[:, None]
        return P_dot * inv
