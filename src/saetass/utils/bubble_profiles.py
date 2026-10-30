"""
This module provides functions and classes to compute the spatial profiles of plasma and magnetic field parameters inside a stellar wind bubble.
The models available are based on the theoretical frameworks by

- :cite:ct:`Weaver1977`
- :cite:ct:`Morlino2021`, which adds the magnetic field, diffusion coefficient and cosmic-ray source of the bubble

The swept-up shell can be uniform or, as in :cite:ct:`Peron2025`, fragmented by instabilities, with its mass penetrating into the hot bubble.
The shell only changes the gas density, not the dynamics of the bubble, so both shells can be combined with either model.

The module provides an extensible framework using :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator` to easily extract density, velocity, magnetic field or transport parameters to be used in SAETASS simulations.
"""

import logging
import math
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import quad

from .. import units as su
from ..grid import Grid
from .energy_losses import Particle


class BubbleModel(StrEnum):
    """Auxiliary class for correct bubble model handling.

    .. note::
        Currently, the supported models are: ``"Weaver77"`` and ``"Morlino21"``.

    Parameters
    ----------
    model_type : str
        The model type (e.g., "Weaver77" or "Morlino21").
    """

    WEAVER77 = "Weaver77"
    MORLINO21 = "Morlino21"

    def __new__(cls, model_type: str):
        obj = str.__new__(cls, model_type)
        obj._value_ = model_type
        return obj


logger = logging.getLogger(__name__)

#: Coefficients :math:`(a_1, a_2, a_3, a_4)` of the fit to the cutoff of the spectrum at the
#: termination shock of :cite:ct:`Morlino2021`, from Table 1 of :cite:ct:`Menchiari2024`.
CUTOFF_COEFFICIENTS = MappingProxyType(
    {
        "kolmogorov": (10.0, 0.308653, 22.0241, 0.43112),
        "kraichnan": (5.0, 0.448549, 12.52, 0.642666),
        "bohm": (8.94, 1.29597, 5.31019, 1.13245),
    }
)


class BubbleProfileCalculator:
    r"""
    Main class to compute stellar wind bubble spatial profiles.

    This class is initialized with the radial grid and the corresponding model.
    It manages the calculation of macroscopic kinematic properties depending on the selected model.
    Built-in models may require additional parameters to compute specific profiles.

    Parameters
    ----------
    r_grid : astropy.units.Quantity
        Radial grid for the profiles computation. Units compatible with :py:data:`~saetass.units.LENGTH`.
    model : BubbleModel or str
        The bubble description model (e.g., "Weaver77" or "Morlino21").
    particle : :py:class:`~saetass.utils.energy_losses.Particle` or str, optional
        Cosmic-ray species, one of the members of :py:class:`~saetass.utils.energy_losses.Particle`, whose mass and charge set the momentum, velocity and Larmor radius of the particles. Default is ``"proton"``.
    **kwargs :
        Specific physical parameters required by the selected model. For 'Weaver77' and
        'Morlino21', the required kwargs are:

        - ``L_wind`` : astropy.units.Quantity. Wind mechanical luminosity. Units compatible with :py:data:`~saetass.units.LUMINOSITY`.
        - ``M_dot`` : astropy.units.Quantity. Mass loss rate. Units compatible with :py:data:`~saetass.units.MASS_LOSS_RATE`.
        - ``rho_0`` : astropy.units.Quantity. Ambient mass density. Units compatible with :py:data:`~saetass.units.MASS_DENSITY`.
        - ``t_b`` : astropy.units.Quantity. Bubble age. Units compatible with :py:data:`~saetass.units.TIME`.

        Other optional kwargs:

        - ``R_c`` : astropy.units.Quantity. Core radius. Units compatible with :py:data:`~saetass.units.LENGTH`. Default is ``0 pc``.
        - ``eta_m`` : float. Fraction of the wind luminosity that drives the bubble expansion, in :math:`(0, 1]`, accounting for radiative losses. It replaces :math:`L_w` by :math:`\eta_m L_w` in the bubble and termination shock radii, while the wind velocity keeps using :math:`L_w`. Default is ``1``.
        - ``shell`` : str. Density profile of the swept-up shell, ``"uniform"`` or ``"fragmented"`` (see :py:meth:`compute_density_profile`). Default is ``"uniform"``.
        - ``M_shell`` : astropy.units.Quantity. Mass of the swept-up shell. Units compatible with :py:data:`~saetass.units.MASS`. Default is the swept-up mass, :math:`(4\pi/3) R_b^3 \rho_0`.
        - ``delta_shell`` : float. Thickness of the shell as a fraction of :math:`R_b`, in :math:`(0, 1)`: the contact discontinuity is at :math:`R_{cd} = (1 - \delta_\mathrm{shell}) R_b`. Default is ``0.05``.
        - ``shell_width`` : astropy.units.Quantity. Width of the Gaussian density profile with which the fragmented shell penetrates into the hot bubble, required if ``shell="fragmented"``. Units compatible with :py:data:`~saetass.units.LENGTH`.
        - ``shell_penetration`` : astropy.units.Quantity. Depth to which the fragmented shell penetrates into the hot bubble, required if ``shell="fragmented"``. Units compatible with :py:data:`~saetass.units.LENGTH`.
    """

    #: Canonical units of the physical model parameters accepted as keyword arguments.
    _KWARG_UNITS = MappingProxyType(
        {
            "L_wind": su.LUMINOSITY,
            "M_dot": su.MASS_LOSS_RATE,
            "rho_0": su.MASS_DENSITY,
            "t_b": su.TIME,
            "R_c": su.LENGTH,
            "M_shell": su.MASS,
            "shell_width": su.LENGTH,
            "shell_penetration": su.LENGTH,
        }
    )

    @u.quantity_input(r_grid=su.LENGTH)
    def __init__(
        self,
        r_grid: u.Quantity,
        model: BubbleModel | str = "Morlino21",
        particle: Particle | str = Particle.PROTON,
        **kwargs,
    ):
        self.r_grid = r_grid.to(su.LENGTH)
        if isinstance(particle, str):
            try:
                particle = Particle(particle.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown particle {particle!r}; choose one of {[p.value for p in Particle]}."
                ) from None
        elif not isinstance(particle, Particle):
            raise TypeError(
                f"particle must be a Particle or a str, got {type(particle).__name__}."
            )
        self.particle = particle

        if isinstance(model, str):
            for m in BubbleModel:
                if m.value.lower() == model.lower():
                    model = m
                    break
        self.model = BubbleModel(model)

        for key, unit in self._KWARG_UNITS.items():
            if key in kwargs:
                su.validate_quantity(
                    kwargs[key], unit, f"BubbleProfileCalculator argument '{key}'"
                )
        eta_m = kwargs.get("eta_m", 1.0)
        if (
            isinstance(eta_m, u.Quantity)
            or not np.isscalar(eta_m)
            or not 0.0 < eta_m <= 1.0
        ):
            raise ValueError(
                f"BubbleProfileCalculator argument 'eta_m' must be a number in (0, 1], got {eta_m!r}."
            )
        if kwargs.get("shell", "uniform") not in ("uniform", "fragmented"):
            raise ValueError(
                f"BubbleProfileCalculator argument 'shell' must be 'uniform' or 'fragmented', got {kwargs['shell']!r}."
            )
        if kwargs.get("shell") == "fragmented":
            for key in ("shell_width", "shell_penetration"):
                if key not in kwargs:
                    raise ValueError(
                        f"BubbleProfileCalculator requires '{key}' for a fragmented shell."
                    )
        delta_shell = kwargs.get("delta_shell", 0.05)
        if (
            isinstance(delta_shell, u.Quantity)
            or not np.isscalar(delta_shell)
            or not 0.0 < delta_shell < 1.0
        ):
            raise ValueError(
                f"BubbleProfileCalculator argument 'delta_shell' must be a number in (0, 1), got {delta_shell!r}."
            )
        self.kwargs = kwargs

        self.R_TS = None
        self.R_b = None
        self.v_w = None
        self.rho_w = None
        self.R_cd = None
        self.masks = {}

        self._compute_kinematics()

    def _compute_kinematics(self):
        """Dispatch the kinematics computation based on the model."""
        # All models share the bubble radii of the intermediate stage
        self._compute_Weaver77_kinematics()

    def _compute_Weaver77_kinematics(self):
        """
        Compute the radii of the bubble in its intermediate stage, common to all models.
        Expected kwargs: L_wind, M_dot, rho_0, t_b
        Optional kwargs: R_c, eta_m, and the shell parameters
        """
        required = ["L_wind", "M_dot", "rho_0", "t_b"]
        for key in required:
            if key not in self.kwargs:
                raise ValueError(
                    f"Model '{self.model}' requires parameter '{key}' in kwargs."
                )

        L_wind = self.kwargs["L_wind"]
        M_dot = self.kwargs["M_dot"]
        rho_0 = self.kwargs["rho_0"]
        t_b = self.kwargs["t_b"]

        eta_m = self.kwargs.get("eta_m", 1.0)

        # The wind velocity uses the full luminosity; only the fraction eta_m drives the bubble
        self.v_w = np.sqrt(2 * L_wind / M_dot).to(u.km / u.s)

        # Weaver et al. (1977), intermediate stage: R_b = (125 / (154 pi))^(1/5) (L t^3 / rho_0)^(1/5)
        self.R_b = (
            (125 / (154 * np.pi)) ** (1 / 5)
            * (eta_m * L_wind / rho_0 * t_b**3) ** (1 / 5)
        ).to(u.pc)

        # Termination shock from the balance of the wind ram pressure with the bubble pressure
        self.R_TS = (
            np.sqrt(((3850 * np.pi) ** (2 / 5) / (28 * np.pi)))
            * (eta_m * L_wind) ** -0.2
            * (M_dot**0.5 * self.v_w**0.5)
            * rho_0 ** (-3 / 10)
            * t_b ** (2 / 5)
        ).to(u.pc)

        # Density of the cold wind just upstream of the termination shock
        self.rho_w = (M_dot / (4 * math.pi * (self.R_TS**2) * self.v_w)).to(
            u.g / u.cm**3
        )
        # Contact discontinuity between the hot bubble and the swept-up shell
        self.R_cd = (1.0 - self.kwargs.get("delta_shell", 0.05)) * self.R_b

        r_val = self.r_grid.to(u.pc).value
        # If R_c is provided, use it, else 0
        R_c = self.kwargs.get("R_c", 0 * u.pc)
        R_c_val = R_c.to(u.pc).value

        R_TS_val, R_b_val = self.R_TS.value, self.R_b.value
        r_bubble = (r_val >= R_TS_val) & (r_val <= R_b_val)
        # Every cell that overlaps the shell [R_cd, R_b] or, if fragmented, its penetration
        # into the hot bubble
        r_shell = self._overlap_volumes(self._shell_inner_radius(), self.R_b) > 0.0
        self.masks = {
            "r_core": r_val < R_c_val,
            "r_wind": (r_val >= R_c_val) & (r_val < R_TS_val),
            "r_hot": r_bubble & ~r_shell,
            "r_shell": r_shell,
            "r_bubble": r_bubble,
            "r_ISM": r_val > R_b_val,
        }
        # Wind also includes the core region
        self.masks["r_wind_overall"] = r_val < self.R_TS.value
        logger.debug("Weaver77 kinematics computed")

    def _shell_inner_radius(self) -> u.Quantity:
        """Inner radius of the gas of the shell: R_cd, or R_cd minus the penetration if fragmented."""
        if self.kwargs.get("shell", "uniform") == "fragmented":
            return self.R_cd - self.kwargs["shell_penetration"]
        return self.R_cd

    def compute_density_profile(self) -> u.Quantity:
        r"""
        Compute the gas density profile.

        The regions of the bubble have the following densities:

        - **Core** (:math:`r < R_c`): constant, equal to the wind density at :math:`R_c`.
        - **Cold wind** (:math:`R_c \leq r < R_{TS}`): :math:`n_w(r) = \dot{M} / (4 \pi r^2 v_w m_p)`.
        - **Hot bubble** (:math:`R_{TS} \leq r < R_{cd}`): uniform, filled by the mass evaporated from the swept-up shell by thermal conduction, :math:`n_b = \dot{M}_s t_b / (m_p V_b)`, with :math:`V_b = (4\pi/3)(R_{cd}^3 - R_{TS}^3)` and the evaporation rate given by

          .. math::

              \dot{M}_s \simeq 2 \times 10^{-4} \left(\frac{\eta_m L_w}{10^{37}\,\mathrm{erg\,s^{-1}}}\right)^{27/35} \left(\frac{\rho_0}{10\, m_p\,\mathrm{cm^{-3}}}\right)^{-2/35} \left(\frac{t_b}{1\,\mathrm{Myr}}\right)^{6/35} M_\odot\,\mathrm{yr^{-1}}.

        - **Swept-up shell**, with the mass ``M_shell`` (by default all the ambient gas swept up, :math:`(4\pi/3) R_b^3 \rho_0`), at the uniform density :math:`n_s` between the contact discontinuity, :math:`R_{cd} = (1 - \delta_\mathrm{shell}) R_b`, and :math:`R_b`:

          - ``shell="uniform"``: all the mass is in :math:`[R_{cd}, R_b]`.
          - ``shell="fragmented"``: instabilities fragment the shell and part of its mass penetrates into the hot bubble down to :math:`R_{cd} - \ell`, with the profile :math:`n_s \exp[-((R_{cd} - r)/w)^2]` added to the hot-bubble density (:cite:ct:`ElBadry2019`; :cite:ct:`Peron2025`). The density is continuous at :math:`R_{cd}` and peaks in the shell.

          Cells that the shell only partly covers average its density with that of the hot bubble or the ambient medium, weighted by volume, so the shell mass is exact for any grid.

        - **Ambient medium** (:math:`r > R_b`): :math:`n_0 = \rho_0 / m_p`.

        Returns
        -------
        astropy.units.Quantity
            Gas number density profile. Units compatible with :py:data:`~saetass.units.NUMBER_DENSITY`.
        """
        L_wind = self.kwargs["L_wind"]
        M_dot = self.kwargs["M_dot"]
        rho_0 = self.kwargs["rho_0"]
        t_b = self.kwargs["t_b"]
        R_c = self.kwargs.get("R_c", 0 * u.pc)
        eta_m = self.kwargs.get("eta_m", 1.0)
        cm3 = u.cm**-3

        n0 = (rho_0 / const.m_p).to_value(cm3)

        n_c = 0.0
        if R_c.to_value(u.pc) > 0:
            n_c = (M_dot / (4 * np.pi * R_c**2 * self.v_w) / const.m_p).to_value(cm3)

        # Avoid divide-by-zero for r=0
        r_safe = self.r_grid.to(u.pc).copy()
        r_safe[r_safe == 0 * u.pc] = 1e-10 * u.pc
        n_w = (M_dot / (4 * np.pi * r_safe**2 * self.v_w) / const.m_p).to_value(cm3)
        if R_c.to_value(u.pc) == 0 and len(n_w) > 1:
            n_w[0] = n_w[1]

        # Hot bubble: mass evaporated from the shell over the age of the bubble
        M_dot_s = (
            2e-4
            * (eta_m * L_wind / (1e37 * u.erg / u.s)).to_value(u.one) ** (27 / 35)
            * (rho_0 / (10 * const.m_p / u.cm**3)).to_value(u.one) ** (-2 / 35)
            * t_b.to_value(u.Myr) ** (6 / 35)
            * u.M_sun
            / u.yr
        )
        V_hot = 4 / 3 * np.pi * (self.R_cd**3 - self.R_TS**3)
        n_b = (M_dot_s * t_b / (V_hot * const.m_p)).to_value(cm3)

        # Swept-up shell
        shell = self.masks["r_shell"]
        M_shell = self.kwargs.get("M_shell", 4 / 3 * np.pi * self.R_b**3 * rho_0)
        volumes = Grid(r_centers=self.r_grid).volumes
        # Shell mass of each cell, in units of n_s m_p: the volume of the cell within the
        # shell plus, if fragmented, that within the penetration depth weighted by the
        # Gaussian profile at the middle of that part
        weights = self._overlap_volumes(self.R_cd, self.R_b)[shell]
        r_in = self._shell_inner_radius()
        if r_in < self.R_cd:
            width = self.kwargs["shell_width"]
            faces = Grid(r_centers=self.r_grid).r_faces
            middle = (
                0.5
                * (
                    np.clip(faces[:-1], r_in, self.R_cd)
                    + np.clip(faces[1:], r_in, self.R_cd)
                )[shell]
            )
            weights = weights + self._overlap_volumes(r_in, self.R_cd)[shell] * np.exp(
                -(((self.R_cd - middle) / width).to_value(u.one) ** 2)
            )
        n_s = (M_shell / (np.sum(weights) * const.m_p)).to_value(cm3)
        # On top of the hot bubble below R_cd and the ambient medium beyond R_b
        below = self._overlap_volumes(0.0 * u.pc, self.R_cd)[shell]
        beyond = self._overlap_volumes(self.R_b, np.inf * u.pc)[shell]
        n_shell = (
            (n_s * weights + n_b * below + n0 * beyond) / volumes[shell]
        ).to_value(u.one)

        n_profile = np.zeros(self.r_grid.shape)
        n_profile[self.masks["r_core"]] = n_c
        n_profile[self.masks["r_wind"]] = n_w[self.masks["r_wind"]]
        n_profile[self.masks["r_hot"]] = n_b
        n_profile[self.masks["r_ISM"]] = n0
        if np.any(shell):
            n_profile[shell] = n_shell

        return n_profile * cm3

    def _overlap_volumes(self, r_in: u.Quantity, r_out: u.Quantity) -> u.Quantity:
        """Volume of each finite-volume cell that lies between the radii ``r_in`` and ``r_out``."""
        faces = Grid(r_centers=self.r_grid).r_faces
        lower = np.clip(faces[:-1], r_in, r_out)
        upper = np.clip(faces[1:], r_in, r_out)
        return 4 / 3 * np.pi * (upper**3 - lower**3)

    @u.quantity_input(T_w=u.K, T_ISM=u.K, T_bubble=u.K, T_shell=u.K, T_core=u.K)
    def compute_temperature_profile(
        self,
        T_w: u.Quantity = 2e2 * u.K,
        T_ISM: u.Quantity = 10 * u.K,
        T_bubble: u.Quantity = 1e6 * u.K,
        T_shell: u.Quantity = 100 * u.K,
        T_core: u.Quantity = 1e4 * u.K,
    ) -> u.Quantity:
        r"""
        Compute the gas temperature profile.

        Parameters
        ----------
        T_w : astropy.units.Quantity, optional
            Temperature in the wind region. Units compatible with :math:`\mathrm{K}`.
        T_ISM : astropy.units.Quantity, optional
            Temperature in the interstellar medium. Units compatible with :math:`\mathrm{K}`.
        T_bubble : astropy.units.Quantity, optional
            Temperature in the hot bubble. Units compatible with :math:`\mathrm{K}`.
        T_shell : astropy.units.Quantity, optional
            Temperature in the shell. Units compatible with :math:`\mathrm{K}`.
        T_core : astropy.units.Quantity, optional
            Temperature in the core. Units compatible with :math:`\mathrm{K}`.

        Returns
        -------
        astropy.units.Quantity
            Temperature profile. Units compatible with :math:`\mathrm{K}`.
        """
        T_profile = np.zeros_like(self.r_grid.value)

        T_profile[self.masks["r_core"]] = T_core.to(u.K).value
        T_profile[self.masks["r_wind"]] = T_w.to(u.K).value
        T_profile[self.masks["r_hot"]] = T_bubble.to(u.K).value
        T_profile[self.masks["r_ISM"]] = T_ISM.to(u.K).value
        T_profile[self.masks["r_shell"]] = T_shell.to(u.K).value

        return T_profile * u.K

    def compute_velocity_profile(self) -> u.Quantity:
        """
        Compute the plasma advection velocity profile.

        Returns
        -------
        astropy.units.Quantity
            Advection velocity profile. Units compatible with :py:data:`~saetass.units.VELOCITY`.
        """
        v_field = np.zeros_like(self.r_grid.value)
        v_field[self.masks["r_wind_overall"]] = self.v_w.to(u.km / u.s).value

        # v inside bubble = v_w / 4 * (R_TS / r)^2
        r_b_mask = self.masks["r_bubble"]
        r_bubble_pc = self.r_grid[r_b_mask].to(u.pc).value

        if len(r_bubble_pc) > 0:
            v_field[r_b_mask] = (
                self.v_w.to(u.km / u.s).value
                / 4
                * (self.R_TS.to(u.pc).value / r_bubble_pc) ** 2
            )

        return v_field * (u.km / u.s)

    def compute_magnetic_field_profile(self, eta_B: float = 0.1) -> u.Quantity:
        r"""
        Compute the turbulent magnetic field profile.

        In the Morlino21 model a fraction :math:`\eta_B` of the kinetic energy flux of the wind is converted into magnetic turbulence, :math:`r^2 v_w \delta B^2 = \eta_B \dot{M} v_w^2 / 2`, so that

        - **Cold wind** (:math:`R_c \leq r < R_{TS}`): :math:`\delta B(r) = \sqrt{\eta_B \dot{M} v_w / 2} / r`.
        - **Core** (:math:`r < R_c`): constant, equal to its value at :math:`R_c`, as the density, since the wind only develops beyond the core. Without a core the field diverges as :math:`1/r` down to the first cell.
        - **Hot bubble** (:math:`R_{TS} \leq r \leq R_b`): uniform, compressed by the shock to :math:`\sqrt{11}` times its value at :math:`R_{TS}`.
        - **Ambient medium** (:math:`r > R_b`): zero.

        Parameters
        ----------
        eta_B : float, optional
            Fraction of the kinetic energy of the wind converted into magnetic turbulence. Default is ``0.1``.

        Returns
        -------
        astropy.units.Quantity
            Magnetic field profile. Units compatible with :py:data:`~saetass.units.MAGNETIC_FIELD`.
        """
        delta_B = np.zeros_like(self.r_grid.value) * u.G

        if self.model == BubbleModel.MORLINO21:
            M_dot = self.kwargs["M_dot"]
            R_c = self.kwargs.get("R_c", 0 * u.pc).to_value(u.cm)

            r_win = self.masks["r_wind_overall"]
            r_bub = self.masks["r_bubble"]

            # The field is constant inside the core, and finite at r = 0 without it
            r_w_cm = np.maximum(self.r_grid[r_win].to_value(u.cm), R_c)
            if len(r_w_cm) > 0 and r_w_cm[0] == 0:
                r_w_cm[0] = r_w_cm[1] if len(r_w_cm) > 1 else 1e-10

            val = np.sqrt(
                0.5 * eta_B * M_dot.to(u.g / u.s).value * self.v_w.to(u.cm / u.s).value
            )

            delta_B[r_win] = (1 / r_w_cm * val) * u.G

            delta_B[r_bub] = (np.sqrt(11) / self.R_TS.to(u.cm).value * val) * u.G

            return delta_B
        else:
            raise NotImplementedError(
                f"Magnetic field profile not natively supported for {self.model}."
            )

    def _momentum(self, p_or_E_k: u.Quantity) -> u.Quantity:
        """Momentum of the particle, given either its momentum or its kinetic energy."""
        if p_or_E_k.unit.is_equivalent(su.MOMENTUM):
            return p_or_E_k.to(su.MOMENTUM)
        mc2 = self.particle.mass * const.c**2
        return (np.sqrt(p_or_E_k**2 + 2 * p_or_E_k * mc2) / const.c).to(su.MOMENTUM)

    def _kinetic_energy(self, p: u.Quantity) -> u.Quantity:
        """Kinetic energy of the particle with momentum ``p``."""
        mc2 = self.particle.mass * const.c**2
        return (np.sqrt((p * const.c) ** 2 + mc2**2) - mc2).to(su.ENERGY)

    def _get_larmor_radius(self, p: u.Quantity, B_field: u.Quantity) -> u.Quantity:
        """Larmor radius, from the rigidity ``p c / (|Z| e)`` of the particle."""
        charge = abs(self.particle.charge_number) * const.e.si
        r_L = np.zeros(np.broadcast_shapes(p.shape, B_field.shape)) * u.pc
        mask = B_field.value > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r_L_raw = p / (charge * B_field)
        r_L[..., mask] = r_L_raw[..., mask].to(u.pc)
        return r_L

    def _get_particle_velocity(self, p: u.Quantity) -> u.Quantity:
        """Velocity of the particle with momentum ``p``."""
        mc2 = self.particle.mass * const.c**2
        return p * const.c**2 / np.sqrt((p * const.c) ** 2 + mc2**2)

    @u.quantity_input(
        p_or_E_k=[su.MOMENTUM, su.ENERGY],
        r_Inj=su.LENGTH,
        D_ISM=su.DIFFUSION_COEFFICIENT,
    )
    def compute_diffusion_profile(
        self,
        p_or_E_k: u.Quantity,
        r_Inj: u.Quantity = 1.0 * u.pc,
        D_ISM: u.Quantity | None = None,
        diffusion_model: str = "kolmogorov",
        eta_B: float = 0.1,
    ) -> u.Quantity:
        r"""
        Compute diffusion coefficient profile.

        Parameters
        ----------
        p_or_E_k : astropy.units.Quantity
            Particle momenta, scalar or 1D, or their kinetic energies. Units compatible with :py:data:`~saetass.units.MOMENTUM` or :py:data:`~saetass.units.ENERGY`.
        r_Inj : astropy.units.Quantity, optional
            Injection scale for turbulence. Units compatible with :py:data:`~saetass.units.LENGTH`. Default is ``1.0 * u.pc``.
        D_ISM : astropy.units.Quantity, optional
            Diffusion coefficient in the ISM. If ``None``, :math:`3 \times 10^{28} (E_k / \mathrm{GeV})^{1/3}\ \mathrm{cm^2\,s^{-1}}` is used. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`. Default is ``None``.
        diffusion_model : str, optional
            Diffusion model inside the bubble, ``'kolmogorov'``, ``'kraichnan'`` or ``'bohm'``. Default is ``'kolmogorov'``.
        eta_B : float, optional
            Magnetic field efficiency parameter to compute the magnetic field if needed. Default is ``0.1``.

        Returns
        -------
        astropy.units.Quantity
            Diffusion coefficient profile with shape ``(len(p_or_E_k), len(r_grid))``, or ``(len(r_grid),)`` for a scalar ``p_or_E_k``. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`.
        """
        if self.model == BubbleModel.MORLINO21:
            is_scalar = p_or_E_k.isscalar
            p = self._momentum(np.atleast_1d(np.squeeze(p_or_E_k)))[:, np.newaxis]

            B_field = self.compute_magnetic_field_profile(eta_B=eta_B)
            r_L = self._get_larmor_radius(p, B_field)
            v_p = self._get_particle_velocity(p)

            match diffusion_model.lower():
                case "bohm":
                    D_values = 1 / 3 * v_p * r_L
                case "kraichnan":
                    D_values = 1 / 3 * v_p * r_L ** (1 / 2) * r_Inj ** (1 / 2)
                case "kolmogorov":
                    D_values = 1 / 3 * v_p * r_L ** (1 / 3) * r_Inj ** (2 / 3)
                case _:
                    raise ValueError(
                        "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
                    )

            if D_ISM is None:
                E_k = self._kinetic_energy(p)
                D_ISM = (3e28 * (E_k / (1 * u.GeV)) ** (1 / 3)) * u.cm**2 / u.s

            D_values[..., self.masks["r_ISM"]] = D_ISM
            out = D_values.to(u.cm**2 / u.s)
            return np.squeeze(out) if is_scalar else out
        else:
            raise NotImplementedError(
                f"Diffusion profile not implemented natively for {self.model}."
            )

    @u.quantity_input(
        p_or_E_k=[su.MOMENTUM, su.ENERGY],
        E_max=[su.ENERGY, su.MOMENTUM],
        p_inj=su.MOMENTUM,
    )
    def compute_source_term(
        self,
        p_or_E_k: u.Quantity,
        E_max: u.Quantity,
        eta_inj: float = 0.1,
        s: float = 4.0,
        p_inj: u.Quantity = 1.0 * su.MOMENTUM,
        diffusion_model: str = "kolmogorov",
    ) -> u.Quantity:
        r"""
        Compute the cosmic-ray source term injected at the termination shock.

        Particles are injected in the cell that contains :math:`R_{TS}` with the spectrum of :cite:ct:`Morlino2021` at the termination shock: a power law in momentum, :math:`f(p) \propto p^{-s}` in phase space, starting at :math:`p_\mathrm{inj}`, with a cutoff :math:`e^{-\Gamma(p)}` set by the escape of particles upstream and downstream of the shock.
        Its shape depends on the turbulence and is given by the fit of :cite:ct:`Menchiari2024` (see :py:data:`~saetass.utils.bubble_profiles.CUTOFF_COEFFICIENTS`),

        .. math::

            e^{-\Gamma(p)} \simeq \left[1 + a_1 \left(\frac{p}{p_\mathrm{max}}\right)^{a_2}\right] e^{-a_3 (p / p_\mathrm{max})^{a_4}},

        where :math:`p_\mathrm{max}` is the momentum at which the upstream diffusion length equals the termination shock radius, :math:`D_w(p_\mathrm{max}) = v_w R_{TS}`.
        Since SAETASS evolves :math:`\psi_p = 4 \pi p^2 f`, the source term per unit momentum is

        .. math::

            Q(p) = A\, \left(\frac{p}{\mathrm{GeV}/c}\right)^{2 - s} e^{-\Gamma(p)}, \qquad p \geq p_\mathrm{inj},

        and zero below :math:`p_\mathrm{inj}`.
        The amplitude :math:`A` makes the power injected into cosmic rays, :math:`V_{TS} \int_{p_\mathrm{inj}}^\infty Q(p)\, E_k(p)\, dp`, equal to :math:`\eta_\mathrm{inj} L_w`, where :math:`V_{TS}` is the volume of the injection cell.
        The normalization is therefore independent of the energies at which the source term is evaluated.

        Parameters
        ----------
        p_or_E_k : astropy.units.Quantity
            Particle momenta at which the source term is evaluated, scalar or 1D, or their kinetic energies. Units compatible with :py:data:`~saetass.units.MOMENTUM` or :py:data:`~saetass.units.ENERGY`.
        E_max : astropy.units.Quantity
            Maximum kinetic energy, whose momentum is the :math:`p_\mathrm{max}` of the cutoff, or directly :math:`p_\mathrm{max}`. Units compatible with :py:data:`~saetass.units.ENERGY` or :py:data:`~saetass.units.MOMENTUM`.
        eta_inj : float, optional
            Fraction of the wind luminosity injected into cosmic rays, in :math:`(0, 1]`. Default is ``0.1``.
        s : float, optional
            Index of the injected phase-space spectrum, :math:`f(p) \propto p^{-s}`; diffusive shock acceleration at a strong shock gives :math:`s = 4`. Default is ``4.0``.
        p_inj : astropy.units.Quantity, optional
            Injection momentum, the lowest momentum of the injected spectrum. Units compatible with :py:data:`~saetass.units.MOMENTUM`. Default is ``1 GeV/c``.
        diffusion_model : str, optional
            Turbulence that sets the shape of the cutoff, ``'kolmogorov'``, ``'kraichnan'`` or ``'bohm'``. Default is ``'kolmogorov'``.

        Returns
        -------
        astropy.units.Quantity
            Source term with shape ``(len(p_or_E_k), len(r_grid))``, or ``(len(r_grid),)`` for a scalar ``p_or_E_k``. Units compatible with :py:data:`~saetass.units.SOURCE_PSI_P`.

        Raises
        ------
        ValueError
            If :math:`R_{TS}` lies outside the radial grid, ``eta_inj`` is not in :math:`(0, 1]`, ``p_inj`` is not positive or ``diffusion_model`` is unknown.
        """
        if diffusion_model.lower() not in CUTOFF_COEFFICIENTS:
            raise ValueError(
                "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
            )
        a1, a2, a3, a4 = CUTOFF_COEFFICIENTS[diffusion_model.lower()]
        if not 0.0 < eta_inj <= 1.0:
            raise ValueError(f"eta_inj must be in (0, 1], got {eta_inj!r}.")
        if not p_inj > 0.0 * su.MOMENTUM:
            raise ValueError(f"p_inj must be positive, got {p_inj!r}.")

        # Injection cell: the finite-volume cell of the solver grid that contains R_TS
        cells = Grid(r_centers=self.r_grid)
        faces = cells.r_faces
        if not faces[0] <= self.R_TS < faces[-1]:
            raise ValueError(
                f"The termination shock, at {self.R_TS.to(u.pc):.3g}, lies outside the radial grid."
            )
        i_TS = (
            int(
                np.searchsorted(
                    faces.to_value(su.LENGTH),
                    self.R_TS.to_value(su.LENGTH),
                    side="right",
                )
            )
            - 1
        )
        V_TS = cells.volumes[i_TS]

        # Energy carried by the normalized spectrum, int x^(2-s) e^(-Gamma) E_k(x) dx with
        # x = p / (GeV/c), integrated in log x up to well beyond the cutoff
        mc2 = (self.particle.mass * const.c**2).to_value(u.GeV)
        x_max = self._momentum(E_max).to_value(su.MOMENTUM)
        x_inj = p_inj.to_value(su.MOMENTUM)

        def cutoff(x):
            z = x / x_max
            return (1.0 + a1 * z**a2) * np.exp(-a3 * z**a4)

        def integrand(y):
            x = np.exp(y)
            return x ** (3.0 - s) * cutoff(x) * (np.sqrt(x**2 + mc2**2) - mc2)

        y_end = np.log(max(x_max, x_inj) * 1e4)
        energy, _ = quad(integrand, np.log(x_inj), y_end, limit=500)
        A = eta_inj * self.kwargs["L_wind"] / (V_TS * energy * u.GeV * su.MOMENTUM)

        is_scalar = p_or_E_k.isscalar
        x = self._momentum(np.atleast_1d(p_or_E_k)).to_value(su.MOMENTUM)
        spectrum = np.where(x >= x_inj, x ** (2.0 - s) * cutoff(x), 0.0)

        Q = np.zeros((x.size, self.r_grid.size))
        Q[:, i_TS] = spectrum
        Q = Q * A.to(su.SOURCE_PSI_P)
        return Q[0] if is_scalar else Q

    @u.quantity_input(D_values=su.DIFFUSION_COEFFICIENT)
    def compute_analytical_CR_profile(
        self, D_values: u.Quantity, f_gal: float = 1.0, f_TS: float = 1.0
    ) -> np.ndarray:
        """
        Calculate the analytical steady-state cosmic ray profile
        following the theoretical Morlino21 model equations.

        Parameters
        ----------
        D_values : astropy.units.Quantity
            Diffusion coefficient profile. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`.
        f_gal : float, optional
            Galactic background level. Default is ``1.0``.
        f_TS : float, optional
            Termination shock level. Default is ``1.0``.

        Returns
        -------
        np.ndarray
            Analytical steady-state CR spatial profile.
        """
        is_1d = D_values.ndim == 1
        if is_1d:
            D_values = D_values[np.newaxis, :]

        r_val = self.r_grid.to(u.pc).value
        r_bubble = self.masks["r_bubble"]
        r_ISM = self.masks["r_ISM"]
        r_wind = self.masks["r_wind_overall"]

        v_b = self.v_w.to(u.pc / u.Myr) / 4

        # D inside bubble at mid-point
        r_bubble_len = np.sum(r_bubble)
        if r_bubble_len > 0:
            mid_idx = r_bubble_len // 2
            D_b = D_values[..., r_bubble][..., mid_idx].to(u.pc**2 / u.Myr)
        else:
            D_b = D_values[..., 0].to(u.pc**2 / u.Myr)  # Fallback

        r_ism_len = np.sum(r_ISM)
        if r_ism_len > 0:
            D_out = D_values[..., r_ISM][..., 0].to(u.pc**2 / u.Myr)
        else:
            D_out = D_values[..., -1].to(u.pc**2 / u.Myr)

        R_TS_pc = self.R_TS.to(u.pc).value
        R_b_pc = self.R_b.to(u.pc).value

        D_b_col = D_b[..., np.newaxis]
        D_out_col = D_out[..., np.newaxis]

        # α(r,p)
        alpha = (v_b * self.R_TS / D_b_col) * (
            1.0 - R_TS_pc / np.where(r_val[r_bubble] == 0, 1e-10, r_val[r_bubble])
        )
        alpha = alpha.decompose().value

        alpha_b = (v_b * self.R_TS / D_b_col) * (1.0 - R_TS_pc / R_b_pc)
        alpha_b = alpha_b.decompose().value

        # β(p)
        beta = (D_out_col * self.R_b) / (v_b * self.R_TS**2)
        beta = beta.decompose().value

        EXP_MAX = 700.0
        EXP_MIN = -700.0
        alpha_clip = np.clip(alpha, EXP_MIN, EXP_MAX)
        alpha_b_clip = np.clip(alpha_b, EXP_MIN, EXP_MAX)

        # Normal case
        denominator = 1.0 + beta * (np.exp(alpha_b_clip) - 1.0)
        numerator = (
            np.exp(alpha_clip) + beta * (np.exp(alpha_b_clip) - np.exp(alpha_clip))
        ) + f_gal / f_TS * beta * (np.exp(alpha_clip) - 1.0)
        f_b_normal = numerator / denominator

        f_b_RB_normal = (
            np.exp(alpha_b_clip) + f_gal / f_TS * beta * (np.exp(alpha_b_clip) - 1.0)
        ) / denominator

        # Extreme case
        f_b_extreme = 1 + (1 - beta) / beta * np.exp(alpha - alpha_b)
        f_b_RB_extreme = 1 / beta * np.ones_like(alpha_b)

        extreme_mask = (alpha > EXP_MAX) | (alpha_b > EXP_MAX)
        extreme_mask_RB = alpha_b > EXP_MAX

        f_b_over_ts = np.where(extreme_mask, f_b_extreme, f_b_normal)
        f_b_over_ts_RB = np.where(extreme_mask_RB, f_b_RB_extreme, f_b_RB_normal)

        # f_out(r,p) / f_TS
        f_out_over_ts = f_b_over_ts_RB * (
            R_b_pc / np.where(r_val[r_ISM] == 0, 1e-10, r_val[r_ISM])
        ) + f_gal / f_TS * (
            1.0 - R_TS_pc / np.where(r_val[r_ISM] == 0, 1e-10, r_val[r_ISM])
        )

        f_b = f_b_over_ts
        f_out = f_out_over_ts

        # Compute f_w(r,p) inside wind zone
        f_w = np.zeros_like(r_val[r_wind])
        r_wind_pc = r_val[r_wind]
        D_w = D_values[..., r_wind].to(u.pc**2 / u.Myr).value

        if len(r_wind_pc) > 1:
            integrand = self.v_w.to(u.pc / u.Myr).value / np.where(D_w == 0, 1e-10, D_w)
            # Use cumulative_trapezoid instead of cumtrapz
            I_r = cumtrapz(integrand, r_wind_pc, axis=-1, initial=0.0)
            I_r = I_r[..., -1:] - I_r
            f_w = f_TS * np.exp(-I_r)
        else:
            f_w = f_TS * np.ones_like(np.broadcast_to(r_wind_pc, D_w.shape))

        out = np.concatenate([f_w, f_b, f_out], axis=-1)
        return np.squeeze(out) if is_1d else out

    @u.quantity_input(p_or_E_k=[su.MOMENTUM, su.ENERGY])
    def get_all_profiles(self, p_or_E_k: u.Quantity, **kwargs) -> dict[str, Any]:
        """
        Compute all profiles and return them in a dictionary.

        Spatial profiles (``n_gas``, ``T_gas``, ``v_field`` and ``B_field``) have shape ``(len(r_grid),)``.
        Profiles that depend on momentum (``D_values`` and ``Q``) have shape ``(len(p_or_E_k), len(r_grid))``, or ``(len(r_grid),)`` for a scalar ``p_or_E_k``.

        Parameters
        ----------
        p_or_E_k : astropy.units.Quantity
            Particle momenta, scalar or 1D, or their kinetic energies. Units compatible with :py:data:`~saetass.units.MOMENTUM` or :py:data:`~saetass.units.ENERGY`.
        **kwargs
            Parameters of the Morlino21 profiles: ``eta_B`` and ``diffusion_model`` (see :py:meth:`compute_diffusion_profile`), and ``E_max`` (required), ``eta_inj``, ``s`` and ``p_inj`` (see :py:meth:`compute_source_term`).

        Returns
        -------
        dict of str to astropy.units.Quantity
            Dictionary containing all relevant profiles and parameters.

        Raises
        ------
        ValueError
            If ``E_max`` is missing for the Morlino21 model.
        """
        res = {
            "r_grid": self.r_grid,
            "R_TS": self.R_TS,
            "R_b": self.R_b,
            "R_cd": self.R_cd,
            "v_w": self.v_w,
            "rho_w": self.rho_w,
            "masks": self.masks,
            "n_gas": self.compute_density_profile(),
            "T_gas": self.compute_temperature_profile(),
            "v_field": self.compute_velocity_profile(),
        }

        if self.model == BubbleModel.MORLINO21:
            eta_B = kwargs.get("eta_B", 0.1)
            diffusion_model = kwargs.get("diffusion_model", "kolmogorov")
            if "E_max" not in kwargs:
                raise ValueError(
                    "get_all_profiles requires 'E_max' to compute the source term of the Morlino21 model."
                )
            source_params = {
                key: kwargs[key]
                for key in ("E_max", "eta_inj", "s", "p_inj")
                if key in kwargs
            }

            res["B_field"] = self.compute_magnetic_field_profile(eta_B=eta_B)
            res["D_values"] = self.compute_diffusion_profile(
                p_or_E_k, diffusion_model=diffusion_model, eta_B=eta_B
            )
            res["Q"] = self.compute_source_term(
                p_or_E_k, diffusion_model=diffusion_model, **source_params
            )

        return res
