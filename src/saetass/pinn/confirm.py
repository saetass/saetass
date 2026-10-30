"""
Simulations of the wind bubble with :py:class:`~saetass.solver.Solver`, to confirm the results of a surrogate.
"""

import astropy.units as u
import numpy as np

from .. import units as su
from ..grid import Grid
from ..solver import Solver
from ..state import Particle, State
from ..utils.bubble_profiles import BubbleProfileCalculator


def simulate(
    r: u.Quantity,
    energy: u.Quantity,
    t: u.Quantity,
    n_steps: int,
    L_wind: u.Quantity,
    M_dot: u.Quantity,
    rho_0: u.Quantity,
    t_b: u.Quantity,
    E_max: u.Quantity,
    eta_B: float = 0.1,
    eta_inj: float = 0.1,
    s: float = 4.0,
    p_inj: u.Quantity = 1.0 * su.MOMENTUM,
    diffusion_model: str = "kolmogorov",
    **kwargs,
) -> u.Quantity:
    r"""
    Run the wind bubble with :py:class:`~saetass.solver.Solver`, with the setup of the example of the multi-energy diffusion.

    Parameters
    ----------
    r : astropy.units.Quantity
        Cell centers, of which the first must be zero and the last sets the outer boundary. Units compatible with :py:data:`~saetass.units.LENGTH`.
    energy : astropy.units.Quantity
        A single kinetic energy or momentum. Units compatible with :py:data:`~saetass.units.ENERGY` or :py:data:`~saetass.units.MOMENTUM`.
    t : astropy.units.Quantity
        Final time. Units compatible with :py:data:`~saetass.units.TIME`.
    n_steps : int
        Number of points of the time grid.
    L_wind, M_dot, rho_0, t_b, E_max, eta_B, eta_inj, s, p_inj, diffusion_model
        The arguments of :py:func:`~saetass.pinn.wind_bubble.physical_bubble`.
    **kwargs
        Other parameters of :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`.

    Returns
    -------
    astropy.units.Quantity
        :math:`\psi` at ``r`` at the final time. Units compatible with :py:data:`~saetass.units.PSI_P`.
    """
    calculator = BubbleProfileCalculator(
        r_grid=r,
        model="Morlino21",
        L_wind=L_wind,
        M_dot=M_dot,
        rho_0=rho_0,
        t_b=t_b,
        **kwargs,
    )
    setup = calculator.get_all_profiles(
        energy,
        eta_B=eta_B,
        eta_inj=eta_inj,
        E_max=E_max,
        s=s,
        p_inj=p_inj,
        diffusion_model=diffusion_model,
    )
    grid = Grid(
        r_centers=r, t_grid=np.linspace(0.0, t.to_value(su.TIME), n_steps) * su.TIME
    )
    state = State(
        psi_p=np.zeros(r.size) * su.PSI_P, grid=grid, particle=Particle.PROTON
    )
    solver = Solver(
        state=state,
        problem_type="advection-source-diffusion",
        operator_params={
            "advection": {
                "v_centers": setup["v_field"],
                "cfl": 0.8,
                "inflow_value_U": 0.0 * su.AREA * su.PSI_P,
            },
            "diffusion": {"D_values": setup["D_values"], "psi_end": 0.0 * su.PSI_P},
            "source": {"source": setup["Q"]},
        },
        quiet=True,
    )
    return solver.run().psi_p
