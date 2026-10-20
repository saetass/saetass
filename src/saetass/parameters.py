r"""
Time-dependent parameters with a separable time dependence.

A time-dependent parameter can be given to :py:class:`~saetass.solver.Solver` as any callable of time, but most time dependences in practice are sums of fixed profiles scaled by functions of time:

.. math::

    X(t, r, p) = \sum_k f_k(t)\, X_k(r, p).

Synchrotron losses scale as :math:`B(t)^2`, inverse Compton losses with the energy density of the photon field, bremsstrahlung and ionisation with the gas density, Bohm diffusion as :math:`1/B(t)`, and the velocity of a wind with its amplitude.
:py:class:`Separable` declares that structure.
Any operator accepts it wherever it accepts a callable, and the advection and loss operators use it to advance all the CFL sub-cycles of a step in a single compiled call: per sub-cycle only the scalar factors :math:`f_k(t)` are evaluated in Python, instead of the whole parameter array.

Separable parameters are optional: a plain callable always works and gives the same result, only more slowly when the operator needs many sub-cycles per step.
The simplest way to build one is to write the formula, multiplying each profile by a :py:class:`TimeFactor`:

>>> B_squared = TimeFactor(lambda t: (field(t) / B0) ** 2)  # doctest: +SKIP
>>> P_dot = P_dot_synchrotron * B_squared + P_dot_other  # doctest: +SKIP

:py:meth:`~saetass.utils.energy_losses.EnergyLossCalculator.get_momentum_loss_rate` builds the separable loss rate itself when given the time dependence of each loss mechanism.
"""

from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

import astropy.units as u
import numpy as np

from . import units as su

__all__ = ["Separable", "TimeFactor"]


class Separable:
    r"""
    A parameter :math:`\sum_k f_k(t)\, X_k` made of fixed profiles :math:`X_k` scaled by functions of time :math:`f_k(t)`.

    Parameters
    ----------
    profiles : sequence of astropy.units.Quantity
        The profiles :math:`X_k`, with units compatible with those of the parameter they are given for. Their shapes must broadcast to a common shape, the shape of the parameter.
    factors : sequence of callable or float
        The factors :math:`f_k`, one per profile: callables of the time (an :py:class:`~astropy.units.Quantity`) returning a dimensionless number or Quantity, or constant numbers.

    Notes
    -----
    Separable parameters support the arithmetic of their formula: they can be added to each other and to Quantities (profiles with a constant factor of one), subtracted, negated and multiplied by numbers.
    A Quantity multiplied by a :py:class:`TimeFactor` is a Separable with a single profile.

    Examples
    --------
    Synchrotron losses that follow a magnetic field :math:`B(t)`, on top of constant losses:

    >>> P_dot = Separable(
    ...     [P_dot_synchrotron, P_dot_other],
    ...     [lambda t: (field(t) / B0) ** 2, 1.0],
    ... )  # doctest: +SKIP

    The same parameter written as its formula:

    >>> P_dot = (
    ...     P_dot_synchrotron * TimeFactor(lambda t: (field(t) / B0) ** 2) + P_dot_other
    ... )  # doctest: +SKIP

    Calling it returns the parameter at a time, so it is also a valid time-dependent parameter for every operator:

    >>> P_dot(0.1 * u.Myr)  # doctest: +SKIP
    """

    # Let Quantities and arrays defer their operators with a Separable to it
    __array_ufunc__ = None

    def __init__(
        self,
        profiles: Sequence[u.Quantity],
        factors: Sequence[Callable[[u.Quantity], Any] | float],
    ) -> None:
        profiles, factors = list(profiles), list(factors)
        if not profiles:
            raise ValueError("Separable needs at least one profile.")
        if len(factors) != len(profiles):
            raise ValueError(
                f"Separable needs one factor per profile, got {len(factors)} factors "
                f"for {len(profiles)} profiles."
            )
        for profile in profiles:
            if not isinstance(profile, u.Quantity):
                raise TypeError(
                    f"Separable profiles must be astropy Quantities, got {type(profile).__name__}."
                )
        unit = profiles[0].unit
        for profile in profiles[1:]:
            if not profile.unit.is_equivalent(unit):
                raise u.UnitsError(
                    f"Separable profiles must have compatible units, got {unit} and {profile.unit}."
                )
        shapes = [np.shape(profile) for profile in profiles]
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError:
            raise ValueError(
                f"Separable profiles must broadcast to a common shape, got {shapes}."
            ) from None
        profiles = [np.broadcast_to(profile, shape, subok=True) for profile in profiles]
        for factor in factors:
            if not (callable(factor) or isinstance(factor, Real)):
                raise TypeError(
                    f"Separable factors must be callables or numbers, got {type(factor).__name__}."
                )
        self.profiles = profiles
        self.factors = factors

    def __call__(self, *args: Any) -> u.Quantity:
        """
        The parameter at a time, the last argument (other arguments, such as the coordinates given to sources, are ignored).
        """
        t = args[-1]
        total = _factor_value(self.factors[0], t) * self.profiles[0]
        for factor, profile in zip(self.factors[1:], self.profiles[1:]):
            total = total + _factor_value(factor, t) * profile
        return total

    def canonical(self, unit: u.UnitBase, name: str) -> "CanonicalSeparable":
        """
        The same parameter in bare canonical floats, for the numerical kernels.

        Parameters
        ----------
        unit : astropy.units.UnitBase
            Canonical unit of the parameter.
        name : str
            Description of the parameter, used in error messages.
        """
        profiles = [
            np.asarray(su.validate_quantity(profile, unit, f"{name} profile").value)
            for profile in self.profiles
        ]
        factors = [_canonical_factor(factor, name) for factor in self.factors]
        return CanonicalSeparable(profiles, factors)

    def __add__(self, other: Any) -> "Separable":
        other = _as_separable(other)
        if other is NotImplemented:
            return NotImplemented
        return Separable(self.profiles + other.profiles, self.factors + other.factors)

    def __radd__(self, other: Any) -> "Separable":
        other = _as_separable(other)
        if other is NotImplemented:
            return NotImplemented
        return other + self

    def __neg__(self) -> "Separable":
        return -1.0 * self

    def __sub__(self, other: Any) -> "Separable":
        other = _as_separable(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: Any) -> "Separable":
        other = _as_separable(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, number: Any) -> "Separable":
        if not isinstance(number, Real):
            return NotImplemented
        return Separable([number * profile for profile in self.profiles], self.factors)

    __rmul__ = __mul__


class TimeFactor:
    r"""
    A function of time that scales a profile, to write a :py:class:`Separable` parameter as its formula.

    Multiplying a Quantity by a ``TimeFactor`` gives a :py:class:`Separable` with that single profile, and Separables add to each other and to Quantities.

    Parameters
    ----------
    function : callable
        Function of the time (an :py:class:`~astropy.units.Quantity`) returning a dimensionless number or Quantity.

    Examples
    --------
    Losses with a synchrotron term that follows a magnetic field :math:`B(t)` and an inverse Compton term that follows the luminosity of a star:

    >>> P_dot = (
    ...     P_dot_synchrotron * TimeFactor(lambda t: (field(t) / B0) ** 2)
    ...     + P_dot_inverse_compton * TimeFactor(lambda t: luminosity(t) / L0)
    ...     + P_dot_gas
    ... )  # doctest: +SKIP
    """

    __array_ufunc__ = None

    def __init__(self, function: Callable[[u.Quantity], Any]) -> None:
        if not callable(function):
            raise TypeError(
                f"TimeFactor needs a callable of time, got {type(function).__name__}."
            )
        self.function = function

    def __call__(self, t: u.Quantity) -> Any:
        return self.function(t)

    def __mul__(self, profile: Any) -> Separable:
        if not isinstance(profile, u.Quantity):
            return NotImplemented
        return Separable([profile], [self.function])

    __rmul__ = __mul__


def _as_separable(value: Any) -> "Separable":
    """A Separable, or a Quantity as a Separable with a constant factor of one."""
    if isinstance(value, Separable):
        return value
    if isinstance(value, u.Quantity):
        return Separable([value], [1.0])
    return NotImplemented


class CanonicalSeparable:
    """
    A :py:class:`Separable` in bare canonical floats: profiles as arrays and factors as functions of the bare time.

    Calling it with a bare time returns the bare parameter, as the callables of every other time-dependent parameter do.
    """

    def __init__(
        self, profiles: list[np.ndarray], factors: list[Callable[[float], float]]
    ) -> None:
        self.profiles = profiles
        self.factors = factors

    def factor_values(self, t: float) -> np.ndarray:
        """The factors at the bare time ``t``, as an array."""
        return np.array([factor(t) for factor in self.factors])

    def __call__(self, t: float) -> np.ndarray:
        values = self.factor_values(t)
        total = values[0] * self.profiles[0]
        for value, profile in zip(values[1:], self.profiles[1:]):
            total = total + value * profile
        return total

    def map(
        self, transform: Callable[[np.ndarray], np.ndarray]
    ) -> "CanonicalSeparable":
        """The same factors with each profile transformed by a linear ``transform``."""
        return CanonicalSeparable(
            [transform(profile) for profile in self.profiles], self.factors
        )


def _factor_value(factor: Callable[[u.Quantity], Any] | float, t: u.Quantity) -> float:
    if not callable(factor):
        return float(factor)
    value = factor(t)
    if isinstance(value, u.Quantity):
        return float(value.to_value(u.dimensionless_unscaled))
    return float(value)


def _canonical_factor(
    factor: Callable[[u.Quantity], Any] | float, name: str
) -> Callable[[float], float]:
    """A factor as a function of the bare time in canonical units, returning a float."""
    if not callable(factor):
        constant = float(factor)
        return lambda t: constant

    def numeric(t: float) -> float:
        value = factor(t * su.TIME)
        if isinstance(value, u.Quantity):
            return float(
                su.validate_quantity(
                    value, u.dimensionless_unscaled, f"{name} factor"
                ).value
            )
        return float(value)

    return numeric
