r"""
Dimensionless form of the transport of cosmic rays in a wind bubble, without energy losses.

The wind bubble of :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`, with the Morlino21 profiles and advection, diffusion and a source at the termination shock, has three zones:

- **Wind** (:math:`r < R_{TS}`): velocity :math:`v_w` and :math:`D(r) = D_{TS}\, (r / R_{TS})^\delta`.
- **Bubble** (:math:`R_{TS} \leq r \leq R_b`): velocity :math:`v_b\, (R_{TS} / r)^2`, with :math:`v_b = v_w / 4`, and uniform :math:`D_b`.
- **Interstellar medium** (:math:`r > R_b`): no velocity and uniform :math:`D_\mathrm{ISM}`, up to the Dirichlet boundary at :math:`r_\mathrm{max}`.

Without energy losses each momentum evolves independently, and only through the diffusion coefficients and the injection rate :math:`S(p)`.
In units of :math:`R_{TS}` for lengths, :math:`R_{TS} / v_b` for times and :math:`S / (4 \pi v_b R_{TS}^2)` for :math:`\psi`, the problem depends on four numbers and on the turbulence, through :math:`\delta`, which also fixes :math:`D_{TS} / D_b = 11^{\delta / 2}`:

.. math::

    \mathrm{Pe} = \frac{v_b R_{TS}}{D_b}, \qquad
    \lambda = \frac{D_\mathrm{ISM}}{D_b}, \qquad
    a = \frac{R_b}{R_{TS}}, \qquad
    b = \frac{r_\mathrm{max}}{R_b}.

The steady state of this problem has a closed form, :py:func:`steady_state`, which is what the surrogates factor out.
"""

from dataclasses import dataclass
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import quad

from .. import units as su
from ..utils.bubble_profiles import CUTOFF_COEFFICIENTS, BubbleProfileCalculator

#: Exponent of the radial dependence of the diffusion coefficient in the wind, for each turbulence.
DELTA = {"kolmogorov": 1.0 / 3.0, "kraichnan": 0.5, "bohm": 1.0}


@dataclass(frozen=True)
class DimensionlessBubble:
    r"""
    Dimensionless wind bubble, as defined in :py:mod:`saetass.pinn.wind_bubble`.

    Each field is a float or an array, all of them broadcastable to each other.

    Parameters
    ----------
    peclet : float or numpy.ndarray
        Péclet number of the bubble, :math:`\mathrm{Pe}`.
    diffusion_ratio : float or numpy.ndarray
        Ratio of the diffusion coefficients of the interstellar medium and of the bubble, :math:`\lambda`.
    bubble_ratio : float or numpy.ndarray
        Radius of the bubble in units of that of the termination shock, :math:`a`.
    domain_ratio : float or numpy.ndarray
        Outer radius in units of that of the bubble, :math:`b`.
    delta : float, optional
        Exponent of the diffusion coefficient in the wind, :math:`\delta`. Default is ``1 / 3``.
    """

    peclet: float | np.ndarray
    diffusion_ratio: float | np.ndarray
    bubble_ratio: float | np.ndarray
    domain_ratio: float | np.ndarray
    delta: float = DELTA["kolmogorov"]

    @property
    def wind_diffusion(self) -> float | np.ndarray:
        r"""Dimensionless diffusion coefficient in the wind at the shock, :math:`11^{\delta / 2} / \mathrm{Pe}`."""
        return 11.0 ** (self.delta / 2.0) / self.peclet

    @property
    def outer_radius(self) -> float | np.ndarray:
        """Dimensionless outer radius of the domain, :math:`a b`."""
        return self.bubble_ratio * self.domain_ratio


@dataclass(frozen=True)
class PhysicalBubble:
    r"""
    Physical scales of a wind bubble for a set of momenta, as bare floats in canonical units.

    Parameters
    ----------
    R_TS : float
        Radius of the termination shock, in canonical :py:data:`~saetass.units.LENGTH` units.
    R_b : float
        Radius of the bubble, in canonical :py:data:`~saetass.units.LENGTH` units.
    r_max : float
        Outer radius of the domain, in canonical :py:data:`~saetass.units.LENGTH` units.
    v_w : float
        Velocity of the wind, in canonical :py:data:`~saetass.units.VELOCITY` units.
    D_b : numpy.ndarray
        Diffusion coefficient in the bubble for each momentum, in canonical :py:data:`~saetass.units.DIFFUSION_COEFFICIENT` units.
    D_ISM : numpy.ndarray
        Diffusion coefficient in the interstellar medium for each momentum, in canonical :py:data:`~saetass.units.DIFFUSION_COEFFICIENT` units.
    rate : numpy.ndarray
        Particles injected per unit time and momentum at each momentum, :math:`S(p)`, in units of :py:data:`~saetass.units.PSI_P` times :py:data:`~saetass.units.VOLUME` over :py:data:`~saetass.units.TIME`.
    delta : float
        Exponent of the diffusion coefficient in the wind.
    """

    R_TS: float
    R_b: float
    r_max: float
    v_w: float
    D_b: np.ndarray
    D_ISM: np.ndarray
    rate: np.ndarray
    delta: float

    @property
    def v_b(self) -> float:
        """Velocity just downstream of the termination shock."""
        return self.v_w / 4.0

    @property
    def dimensionless(self) -> DimensionlessBubble:
        """The dimensionless numbers of this bubble, one set per momentum."""
        return DimensionlessBubble(
            peclet=self.v_b * self.R_TS / self.D_b,
            diffusion_ratio=self.D_ISM / self.D_b,
            bubble_ratio=self.R_b / self.R_TS,
            domain_ratio=self.r_max / self.R_b,
            delta=self.delta,
        )

    @property
    def time_unit(self) -> float:
        """Time unit of the dimensionless problem, :math:`R_{TS} / v_b`."""
        return self.R_TS / self.v_b

    @property
    def psi_unit(self) -> np.ndarray:
        r"""Unit of :math:`\psi` of the dimensionless problem at each momentum, :math:`S / (4 \pi v_b R_{TS}^2)`."""
        return self.rate / (4.0 * np.pi * self.v_b * self.R_TS**2)


def physical_bubble(
    momenta: u.Quantity,
    L_wind: u.Quantity,
    M_dot: u.Quantity,
    rho_0: u.Quantity,
    t_b: u.Quantity,
    r_max: u.Quantity,
    E_max: u.Quantity,
    eta_B: float = 0.1,
    eta_inj: float = 0.1,
    s: float = 4.0,
    p_inj: u.Quantity = 1.0 * su.MOMENTUM,
    diffusion_model: str = "kolmogorov",
    **kwargs,
) -> PhysicalBubble:
    r"""
    Physical scales of a bubble, computed by :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator` so that they are those of a simulation.

    Parameters
    ----------
    momenta : astropy.units.Quantity
        Particle momenta or kinetic energies, scalar or 1D. Units compatible with :py:data:`~saetass.units.MOMENTUM` or :py:data:`~saetass.units.ENERGY`.
    L_wind : astropy.units.Quantity
        Luminosity of the wind. Units compatible with :math:`\mathrm{erg\,s^{-1}}`.
    M_dot : astropy.units.Quantity
        Mass-loss rate of the wind. Units compatible with :math:`M_\odot\,\mathrm{yr^{-1}}`.
    rho_0 : astropy.units.Quantity
        Density of the ambient medium. Units compatible with :math:`\mathrm{g\,cm^{-3}}`.
    t_b : astropy.units.Quantity
        Age of the bubble, which sets its radii. Units compatible with :py:data:`~saetass.units.TIME`.
    r_max : astropy.units.Quantity
        Outer radius of the domain. Units compatible with :py:data:`~saetass.units.LENGTH`.
    E_max : astropy.units.Quantity
        Maximum kinetic energy or momentum of the injected spectrum. Units compatible with :py:data:`~saetass.units.ENERGY` or :py:data:`~saetass.units.MOMENTUM`.
    eta_B : float, optional
        Fraction of the kinetic energy of the wind converted into magnetic turbulence. Default is ``0.1``.
    eta_inj : float, optional
        Fraction of the luminosity of the wind injected into cosmic rays. Default is ``0.1``.
    s : float, optional
        Index of the injected phase-space spectrum. Default is ``4.0``.
    p_inj : astropy.units.Quantity, optional
        Injection momentum. Units compatible with :py:data:`~saetass.units.MOMENTUM`. Default is ``1 GeV/c``.
    diffusion_model : str, optional
        Turbulence in the wind and the bubble, ``'kolmogorov'``, ``'kraichnan'`` or ``'bohm'``. Default is ``'kolmogorov'``.
    **kwargs
        Other parameters of :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`.

    Returns
    -------
    PhysicalBubble
        The scales of the bubble, with one value per momentum where they depend on it.

    Raises
    ------
    ValueError
        If the turbulence is unknown, if ``r_max`` lies inside the bubble or if ``R_c`` is not zero.
    """
    model = diffusion_model.lower()
    if model not in DELTA:
        raise ValueError(
            "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
        )
    if kwargs.get("R_c", 0.0 * u.pc).to_value(u.pc) != 0.0:
        raise ValueError(
            "The dimensionless bubble has no core: its wind extends down to r = 0, so R_c must be zero."
        )
    arguments = dict(
        model="Morlino21", L_wind=L_wind, M_dot=M_dot, rho_0=rho_0, t_b=t_b, **kwargs
    )
    probe = BubbleProfileCalculator(r_grid=np.array([0.0, 1.0]) * u.pc, **arguments)
    R_TS = probe.R_TS.to_value(su.LENGTH)
    R_b = probe.R_b.to_value(su.LENGTH)
    outer = r_max.to_value(su.LENGTH)
    if outer <= R_b:
        raise ValueError(
            f"r_max must lie beyond the bubble, of radius {R_b:.4g} {su.LENGTH}, got {outer:.4g} {su.LENGTH}."
        )

    # One radius per zone: the coefficients are uniform in the bubble and in the medium
    radii = np.array([0.5 * R_TS, 0.5 * (R_TS + R_b), 0.5 * (R_b + outer)])
    calculator = BubbleProfileCalculator(r_grid=radii * su.LENGTH, **arguments)
    p = np.atleast_1d(momenta)
    D = calculator.compute_diffusion_profile(
        p, diffusion_model=model, eta_B=eta_B
    ).to_value(su.DIFFUSION_COEFFICIENT)
    Q = calculator.compute_source_term(
        p, E_max=E_max, eta_inj=eta_inj, s=s, p_inj=p_inj, diffusion_model=model
    ).to_value(su.SOURCE_PSI_P)
    # The source fills one cell: its rate, Q times the volume, does not depend on the cell
    from ..grid import Grid

    volumes = Grid(r_centers=radii * su.LENGTH).volumes.to_value(su.VOLUME)
    rate = np.sum(Q * volumes, axis=-1)
    return PhysicalBubble(
        R_TS=R_TS,
        R_b=R_b,
        r_max=outer,
        v_w=probe.v_w.to_value(su.VELOCITY),
        D_b=D[..., 1],
        D_ISM=D[..., 2],
        rate=rate,
        delta=DELTA[model],
    )


def exp_of(value: Any) -> Any:
    """Exponential of a number, an array or a tensor."""
    return value.exp() if hasattr(value, "exp") else np.exp(value)


@dataclass(frozen=True)
class BatchedScales:
    r"""
    Dimensionless numbers and units of many bubbles and momenta, as arrays or tensors.

    Parameters
    ----------
    peclet, diffusion_ratio, bubble_ratio, domain_ratio : numpy.ndarray or torch.Tensor
        The fields of :py:class:`DimensionlessBubble`.
    R_TS : numpy.ndarray or torch.Tensor
        Radius of the termination shock, in canonical :py:data:`~saetass.units.LENGTH` units.
    time_unit : numpy.ndarray or torch.Tensor
        Time unit, in canonical :py:data:`~saetass.units.TIME` units.
    psi_unit : numpy.ndarray or torch.Tensor
        Unit of :math:`\psi`, in canonical :py:data:`~saetass.units.PSI_P` units.
    """

    peclet: Any
    diffusion_ratio: Any
    bubble_ratio: Any
    domain_ratio: Any
    R_TS: Any
    time_unit: Any
    psi_unit: Any


class FastScales:
    r"""
    The scales of :py:func:`physical_bubble` for arrays of parameters, without Astropy.

    :py:func:`physical_bubble` is the definition of the scales, through :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`, but it takes milliseconds per bubble. This class repeats the same formulas with plain arithmetic, for NumPy arrays or PyTorch tensors, which also makes them differentiable. It is meant for parameter searches, and the tests check it against :py:func:`physical_bubble`.

    Parameters
    ----------
    E_max : astropy.units.Quantity
        Maximum kinetic energy or momentum of the injected spectrum. Units compatible with :py:data:`~saetass.units.ENERGY` or :py:data:`~saetass.units.MOMENTUM`.
    s : float, optional
        Index of the injected phase-space spectrum. Default is ``4.0``.
    p_inj : astropy.units.Quantity, optional
        Injection momentum. Units compatible with :py:data:`~saetass.units.MOMENTUM`. Default is ``1 GeV/c``.
    diffusion_model : str, optional
        Turbulence, ``'kolmogorov'``, ``'kraichnan'`` or ``'bohm'``. Default is ``'kolmogorov'``.
    """

    def __init__(
        self,
        E_max: u.Quantity,
        s: float = 4.0,
        p_inj: u.Quantity = 1.0 * su.MOMENTUM,
        diffusion_model: str = "kolmogorov",
    ) -> None:
        model = diffusion_model.lower()
        if model not in DELTA:
            raise ValueError(
                "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
            )
        self.delta = DELTA[model]
        self.s = s
        self.coefficients = CUTOFF_COEFFICIENTS[model]
        self.mc2 = (const.m_p * const.c**2).to_value(u.GeV)
        self.p_inj = p_inj.to_value(su.MOMENTUM)
        if E_max.unit.is_equivalent(su.MOMENTUM):
            self.p_max = E_max.to_value(su.MOMENTUM)
        else:
            E = E_max.to_value(u.GeV)
            self.p_max = np.sqrt(E**2 + 2.0 * E * self.mc2)
        # Energy of the normalized spectrum, as in compute_source_term
        y_end = np.log(max(self.p_max, self.p_inj) * 1e4)
        self.energy, _ = quad(
            lambda y: (
                np.exp(y) ** (3.0 - s)
                * self._cutoff(np.exp(y))
                * (np.sqrt(np.exp(y) ** 2 + self.mc2**2) - self.mc2)
            ),
            np.log(self.p_inj),
            y_end,
            limit=500,
        )
        self.seconds_per_myr = (1.0 * u.Myr).to_value(u.s)
        self.cm_per_pc = (1.0 * u.pc).to_value(u.cm)
        self.gram_per_msun_per_yr = (const.M_sun / u.yr).to_value(u.g / u.s)
        self.gev_per_erg = (1.0 * u.erg).to_value(u.GeV)
        # Larmor radius in pc of 1 GeV/c in 1 G
        self.larmor = ((su.MOMENTUM / (const.e.si * u.G)).to(u.pc)).value
        self.speed_of_light = const.c.to_value(u.cm / u.s)

    def _cutoff(self, x: Any) -> Any:
        a1, a2, a3, a4 = self.coefficients
        z = x / self.p_max
        exponent = -a3 * z**a4
        factor = exp_of(exponent)
        return (1.0 + a1 * z**a2) * factor

    def __call__(
        self,
        p: Any,
        L_wind: Any,
        M_dot: Any,
        rho_0: Any,
        t_b: Any,
        r_max: Any,
        eta_B: Any = 0.1,
        eta_inj: Any = 0.1,
        r_inj: float = 1.0,
    ) -> BatchedScales:
        r"""
        Scales of bubbles given by broadcastable arrays or tensors, which must be in double precision: the physical inputs overflow single precision.

        Parameters
        ----------
        p : numpy.ndarray or torch.Tensor
            Momenta, in :py:data:`~saetass.units.MOMENTUM` units.
        L_wind : numpy.ndarray or torch.Tensor
            Luminosity of the wind, in :math:`\mathrm{erg\,s^{-1}}`.
        M_dot : numpy.ndarray or torch.Tensor
            Mass-loss rate of the wind, in :math:`M_\odot\,\mathrm{yr^{-1}}`.
        rho_0 : numpy.ndarray or torch.Tensor
            Density of the ambient medium, in :math:`\mathrm{g\,cm^{-3}}`.
        t_b : numpy.ndarray or torch.Tensor
            Age of the bubble, in :py:data:`~saetass.units.TIME` units.
        r_max : numpy.ndarray or torch.Tensor
            Outer radius of the domain, in :py:data:`~saetass.units.LENGTH` units.
        eta_B : numpy.ndarray or torch.Tensor, optional
            Fraction of the kinetic energy of the wind in magnetic turbulence. Default is ``0.1``.
        eta_inj : numpy.ndarray or torch.Tensor, optional
            Fraction of the luminosity of the wind injected into cosmic rays. Default is ``0.1``.
        r_inj : float, optional
            Injection scale of the turbulence, in :py:data:`~saetass.units.LENGTH` units. Default is ``1.0``.

        Returns
        -------
        BatchedScales
            The scales, broadcast against each other.
        """
        M = M_dot * self.gram_per_msun_per_yr
        t = t_b * self.seconds_per_myr
        v_w = (2.0 * L_wind / M) ** 0.5
        R_b = (125.0 / (154.0 * np.pi)) ** 0.2 * (L_wind / rho_0 * t**3) ** 0.2
        R_TS = (
            ((3850.0 * np.pi) ** 0.4 / (28.0 * np.pi)) ** 0.5
            * L_wind**-0.2
            * (M * v_w) ** 0.5
            * rho_0**-0.3
            * t**0.4
        )
        B_b = 11.0**0.5 / R_TS * (0.5 * eta_B * M * v_w) ** 0.5

        energy = (p**2 + self.mc2**2) ** 0.5
        speed = p / energy * self.speed_of_light
        larmor = self.larmor * p / B_b
        D_b = speed / 3.0 * larmor**self.delta * r_inj ** (1.0 - self.delta)
        D_b = D_b * self.cm_per_pc
        D_ism = 3e28 * (energy - self.mc2) ** (1.0 / 3.0)

        v_b = 0.25 * v_w
        peclet = v_b * R_TS / D_b
        spectrum = p ** (2.0 - self.s) * self._cutoff(p)
        # Particles per unit time and momentum, over 4 pi v_b R_TS^2, in canonical units
        rate = (
            eta_inj * L_wind * self.gev_per_erg * self.seconds_per_myr / self.energy
        ) * spectrum
        v_canonical = v_b * self.seconds_per_myr / self.cm_per_pc
        R_canonical = R_TS / self.cm_per_pc
        return BatchedScales(
            peclet=peclet,
            diffusion_ratio=D_ism / D_b,
            bubble_ratio=R_b / R_TS,
            domain_ratio=r_max * self.cm_per_pc / R_b,
            R_TS=R_canonical,
            time_unit=R_canonical / v_canonical,
            psi_unit=rate / (4.0 * np.pi * v_canonical * R_canonical**2),
        )


def wind_exponent(x: np.ndarray, bubble: DimensionlessBubble) -> np.ndarray:
    r"""
    Exponent of the steady state in the wind, :math:`\int_x^1 v / D\, dx'` with :math:`v = 4` and :math:`D = D_{TS}\, x'^\delta`.

    Parameters
    ----------
    x : numpy.ndarray
        Dimensionless radii, in :math:`[0, 1]`.
    bubble : DimensionlessBubble
        Dimensionless bubble.

    Returns
    -------
    numpy.ndarray
        The exponent, which vanishes at the shock.
    """
    delta = bubble.delta
    scale = 4.0 / bubble.wind_diffusion
    if delta == 1.0:
        return -scale * np.log(x)
    return scale * (1.0 - x ** (1.0 - delta)) / (1.0 - delta)


def steady_state(x: np.ndarray, bubble: DimensionlessBubble) -> np.ndarray:
    r"""
    Dimensionless steady state of the wind bubble.

    Beyond the shock the total flux is that of the source, which gives

    .. math::

        \psi = \begin{cases}
            \psi_{TS}\, e^{-\int_x^1 4 / D\, dx'} & x < 1, \\
            1 + C\, e^{\mathrm{Pe}\, (1 - 1 / x)} & 1 \leq x \leq a, \\
            (\mathrm{Pe} / \lambda)\, (1 / x - 1 / (a b)) & x > a,
        \end{cases}

    with :math:`\psi_{TS} = 1 + C` and :math:`C` given by the continuity at :math:`x = a`.

    Parameters
    ----------
    x : numpy.ndarray
        Dimensionless radii, in :math:`[0, a b]`.
    bubble : DimensionlessBubble
        Dimensionless bubble.

    Returns
    -------
    numpy.ndarray
        Dimensionless :math:`\psi` at ``x``.
    """
    x = np.asarray(x, dtype=float)
    Pe, ratio = bubble.peclet, bubble.diffusion_ratio
    a, outer = bubble.bubble_ratio, bubble.outer_radius
    edge = (Pe / ratio) * (1.0 / a - 1.0 / outer)
    # C e^alpha, written with exponents that stay small: alpha(x) - alpha(a) <= 0
    level = (edge - 1.0) * np.exp(Pe * (1.0 / a - 1.0))
    psi_ts = 1.0 + level
    wind = psi_ts * np.exp(-wind_exponent(np.minimum(x, 1.0), bubble))
    safe = np.maximum(x, 1.0)
    bubble_zone = 1.0 + (edge - 1.0) * np.exp(Pe * (1.0 / a - 1.0 / safe))
    outer_zone = (Pe / ratio) * (1.0 / np.maximum(x, 1e-300) - 1.0 / outer)
    return np.where(x < 1.0, wind, np.where(x <= a, bubble_zone, outer_zone))


def log_steady_state(x: np.ndarray, bubble: DimensionlessBubble) -> np.ndarray:
    r"""
    Logarithm of :py:func:`steady_state`, which stays finite where the steady state underflows in the wind.

    Parameters
    ----------
    x : numpy.ndarray
        Dimensionless radii, in :math:`[0, a b)`.
    bubble : DimensionlessBubble
        Dimensionless bubble.

    Returns
    -------
    numpy.ndarray
        :math:`\ln \psi_\mathrm{ss}` at ``x``.
    """
    x = np.asarray(x, dtype=float)
    shock = steady_state(np.ones_like(x), bubble)
    wind = np.log(shock) - wind_exponent(np.minimum(x, 1.0), bubble)
    with np.errstate(divide="ignore"):
        return np.where(x < 1.0, wind, np.log(steady_state(x, bubble)))


def torch_log_steady_state(
    x: Any, peclet: Any, ratio: Any, a: Any, outer: Any, delta: float
) -> Any:
    r"""
    :py:func:`log_steady_state` for tensors, differentiable and with the fields as arguments.

    Parameters
    ----------
    x : torch.Tensor
        Dimensionless radii, in :math:`[0, a b)`.
    peclet, ratio, a, outer : torch.Tensor
        :math:`\mathrm{Pe}`, :math:`\lambda`, :math:`a` and :math:`a b`.
    delta : float
        Exponent of the diffusion coefficient in the wind, below one.

    Returns
    -------
    torch.Tensor
        :math:`\ln \psi_\mathrm{ss}` at ``x``.
    """
    edge = (peclet / ratio) * (1.0 / a - 1.0 / outer)
    level = (edge - 1.0) * (peclet * (1.0 / a - 1.0)).exp()
    scale = 4.0 * peclet / 11.0 ** (delta / 2.0)
    # Each branch only sees radii of its zone, so that the gradients of the others stay finite
    in_wind = x.clamp(min=1e-6, max=1.0)
    wind = (1.0 + level).log() - scale * (1.0 - in_wind ** (1.0 - delta)) / (
        1.0 - delta
    )
    in_bubble = x.clamp(min=1.0)
    in_bubble = in_bubble.where(in_bubble <= a, a)
    bubble = (1.0 + (edge - 1.0) * (peclet * (1.0 / a - 1.0 / in_bubble)).exp()).log()
    beyond = x.where(x > a, a)
    medium = ((peclet / ratio) * (1.0 / beyond - 1.0 / outer)).log()
    return wind.where(x < 1.0, bubble.where(x <= a, medium))
