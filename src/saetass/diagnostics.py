r"""
The diagnostics module analyzes the characteristic scales of a configured simulation to detect, before running it, time steps that compromise the accuracy of its numerical schemes.

Each operator reports its characteristic timescale and the dimensionless numbers that control the behaviour of its numerical scheme, evaluated for the time step it integrates in each call:

- **Advection and losses** report their Courant number :math:`\max |V| \Delta t / \Delta x` and the number of CFL sub-cycles performed per call.
  These operators are sub-cycled automatically, so their Courant number measures cost rather than stability.
  Their timescales are the domain crossing time :math:`L / \max|v|` and the shortest loss time :math:`\min p / |\dot{p}|`, respectively.
- **Diffusion** reports its Fourier number, generalized to the non-uniform spherical grid as :math:`\max_i \Delta t\,(G_{i-1/2} + G_{i+1/2}) / (2 V_i)`, with face conductances :math:`G` and cell volumes :math:`V`, which reduces to :math:`D \Delta t / \Delta r^2` on uniform Cartesian grids.
  Crank-Nicolson is unconditionally stable and is guaranteed to preserve positivity when this number is at most one.
  Larger values mainly cost accuracy on features resolved by few cells, while smooth solutions tolerate values of hundreds, which are common in astrophysical setups, so this number is reported but not warned about.
  Its timescale is the diffusion time across the domain, :math:`L^2 / \max D`.
- **Time-dependent sources** report their variation timescale :math:`\tau_Q = \max |Q| / \max |\partial_t Q|` and the relative change of the source over one step, :math:`\Delta t / \tau_Q`.
  The magnitude of a source does not define a timescale: the transport equation is linear, so scaling :math:`Q` only scales the part of the solution it produces, while each step injects :math:`Q` evaluated at its start, which is accurate only if :math:`Q` changes little within the step.
  Static sources report no scales.
- **With several operators**, the splitting ratio :math:`\Delta t / \tau_\mathrm{min}` compares the global time step to the shortest operator timescale, which controls the operator-splitting and time-integration errors.
  A warning is logged when it exceeds :py:data:`WARNING_SPLITTING_RATIO`, with the number of timesteps needed to bring it below.
- **Pairs of operators** report cell numbers comparing their rates at the grid scale, as the largest value over the cells where both act:

  - the cell Péclet number :math:`|v| \Delta r / D`, with advection and diffusion, tells which of them dominates;
  - the cell Damköhler numbers :math:`(\Delta r / |v|)\,|\dot{p}| / p` and :math:`(\Delta r^2 / D)\,|\dot{p}| / p`, with losses and advection or diffusion, compare the time to cross a cell with the loss time.
    Above one, particles lose their energy before leaving the cell where they are, so their spatial profile at those momenta is not resolved by the grid.
    Momenta with fast losses are often meant to be confined near their sources, so these numbers are reported but not warned about.

Dimensionless numbers are invariant under changes of units, so they characterize the discrete problem itself, unlike the magnitudes of the values handled by the solver.
Since parameters may depend on time, every quantity is evaluated at several times of the simulation and the worst case is reported.

.. code-block:: python

    diagnostics = solver.diagnostics()  # logs a warning for each problematic time step
    print(diagnostics)
    diagnostics["diffusion"].numbers["fourier_number"]

Warnings are logged at the ``WARNING`` level, on the console of SAETASS (see :py:mod:`saetass.cli.output`), and are always available in :py:attr:`SimulationDiagnostics.warnings`.

--------------
"""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from . import units as su

if TYPE_CHECKING:
    from .solver import Solver

logger = logging.getLogger(__name__)

#: Splitting ratio above which a warning is logged. Splitting and time-integration errors
#: grow with this ratio, reaching a fraction of it for first-order steps, so this value
#: keeps them at the level of a few percent.
WARNING_SPLITTING_RATIO = 0.1


@dataclass(frozen=True)
class CharacteristicScales:
    """
    Characteristic scales of an operator, as reported by :py:meth:`~saetass.solver.SubSolver.characteristic_scales`.

    Parameters
    ----------
    timescale : float or None, optional
        Characteristic timescale of the operator as a bare float, in canonical :py:data:`~saetass.units.TIME` units, or ``None`` if the operator has none. Default is ``None``.
    numbers : dict of str to float, optional
        Dimensionless numbers of the operator for the given time step. Default is an empty dictionary.
    """

    timescale: float | None = None
    numbers: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorDiagnostics:
    """
    Diagnostics of a single operator, as the worst case over the sampled times.

    Parameters
    ----------
    name : str
        Name of the operator, e.g. ``"diffusion"``.
    time_step : astropy.units.Quantity
        Time step integrated by the operator in each call. Units compatible with :py:data:`~saetass.units.TIME`.
    timescale : astropy.units.Quantity or None
        Shortest characteristic timescale of the operator, or ``None`` if it has none. Units compatible with :py:data:`~saetass.units.TIME`.
    numbers : dict of str to float
        Largest value of each dimensionless number of the operator.
    """

    name: str
    time_step: u.Quantity
    timescale: u.Quantity | None
    numbers: Mapping[str, float]


@dataclass(frozen=True)
class SimulationDiagnostics:
    """
    Diagnostics of a configured simulation, returned by :py:func:`diagnose`.

    Operators can be looked up by name, e.g. ``diagnostics["advection"]``.

    Parameters
    ----------
    time_step : astropy.units.Quantity
        Global (macro) time step of the simulation. Units compatible with :py:data:`~saetass.units.TIME`.
    sampled_times : astropy.units.Quantity
        Times at which the time-dependent parameters were evaluated. Units compatible with :py:data:`~saetass.units.TIME`.
    operators : tuple of OperatorDiagnostics
        Diagnostics of each operator, in the order of the problem type.
    numbers : dict of str to float
        Dimensionless numbers involving several operators: ``"splitting_ratio"``, ``"cell_peclet"``, ``"cell_damkohler_advection"`` and ``"cell_damkohler_diffusion"``, when applicable.
    warnings : tuple of str
        Descriptions of the problematic time steps found, with a suggested number of timesteps.
    """

    time_step: u.Quantity
    sampled_times: u.Quantity
    operators: tuple[OperatorDiagnostics, ...]
    numbers: Mapping[str, float]
    warnings: tuple[str, ...]

    def __getitem__(self, name: str) -> OperatorDiagnostics:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise KeyError(f"No diagnostics for operator '{name}'.")

    def __str__(self) -> str:
        return self.report()

    def report(self, include_warnings: bool = True) -> str:
        """
        Text report of the diagnostics: one line per operator and the numbers involving several of them.

        Parameters
        ----------
        include_warnings : bool, optional
            Whether to end the report with the :py:attr:`warnings`. Default is ``True``.

        Returns
        -------
        str
            The report, as printed by ``print(diagnostics)``.
        """
        lines = [
            f"SAETASS diagnostics: time step {self.time_step:.4g}, "
            f"worst case over {len(self.sampled_times)} sampled times"
        ]
        for op in self.operators:
            timescale = f"{op.timescale:.4g}" if op.timescale is not None else "-"
            numbers = ", ".join(f"{k} = {v:.4g}" for k, v in op.numbers.items())
            lines.append(
                f"  {op.name:<10} step {op.time_step:.4g}, timescale {timescale}"
                + (f", {numbers}" if numbers else "")
            )
        if self.numbers:
            lines.append(
                "  " + ", ".join(f"{k} = {v:.4g}" for k, v in self.numbers.items())
            )
        if include_warnings:
            lines += [f"  WARNING: {message}" for message in self.warnings]
        return "\n".join(lines)


def diagnose(solver: "Solver", n_samples: int = 3) -> SimulationDiagnostics:
    """
    Analyze the characteristic scales of a configured simulation.

    Time-dependent parameters are evaluated at ``n_samples`` times evenly spread over the time grid, and the worst case is reported.
    Usually called through :py:meth:`~saetass.solver.Solver.diagnostics`.

    Parameters
    ----------
    solver : :py:class:`~saetass.solver.Solver`
        The configured solver, before or during its run.
    n_samples : int, optional
        Number of times at which the parameters are evaluated. Default is ``3`` (start, middle and end).

    Returns
    -------
    SimulationDiagnostics
        The diagnostics of the simulation.

    Raises
    ------
    ValueError
        If ``n_samples`` is not a positive integer.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    t_grid = solver.grid.t_grid.to_value(su.TIME)
    num_timesteps = len(t_grid) - 1
    times = t_grid[
        np.unique(np.linspace(0, num_timesteps, n_samples).round().astype(int))
    ]
    dt_global = float(np.max(np.diff(t_grid)))

    operators = []
    for op, subsolver in zip(solver.operator_list, solver.operator_subsolvers):
        # Each call integrates all the substeps of the operator at once
        dt = solver.substeps_per_op[op] * float(np.diff(subsolver.t_grid)[0])
        samples = [subsolver.characteristic_scales(t, dt) for t in times]
        timescales = [s.timescale for s in samples if s.timescale is not None]
        numbers = {
            key: max(s.numbers[key] for s in samples) for key in samples[0].numbers
        }
        operators.append(
            OperatorDiagnostics(
                name=op.value,
                time_step=dt * su.TIME,
                timescale=min(timescales) * su.TIME if timescales else None,
                numbers=MappingProxyType(numbers),
            )
        )

    numbers = {}
    messages = []

    timed = [
        op for op in operators if op.timescale is not None and np.isfinite(op.timescale)
    ]
    if len(operators) > 1 and timed:
        fastest = min(timed, key=lambda op: op.timescale)
        ratio = dt_global / fastest.timescale.to_value(su.TIME)
        numbers["splitting_ratio"] = ratio
        if ratio > WARNING_SPLITTING_RATIO:
            messages.append(
                f"The time step of {dt_global * su.TIME:.4g} is {ratio:.4g} times the shortest "
                f"operator timescale ({fastest.name}, {fastest.timescale:.4g}): splitting and "
                f"time-integration errors may be significant. Use at least "
                f"{math.ceil(num_timesteps * ratio / WARNING_SPLITTING_RATIO)} timesteps "
                f"(currently {num_timesteps})."
            )

    subsolvers = {
        op.value: sub
        for op, sub in zip(solver.operator_list, solver.operator_subsolvers)
    }
    numbers.update(_cell_numbers(subsolvers, solver.grid, times))

    return SimulationDiagnostics(
        time_step=dt_global * su.TIME,
        sampled_times=times * su.TIME,
        operators=tuple(operators),
        numbers=MappingProxyType(numbers),
        warnings=tuple(messages),
    )


def _cell_numbers(subsolvers: dict, grid, times: np.ndarray) -> dict[str, float]:
    """Largest cell numbers of the pairs of operators present, over the sampled times."""
    advection = subsolvers.get("advection")
    diffusion = subsolvers.get("diffusion")
    loss = subsolvers.get("loss")
    pairs = {}
    if advection is not None and diffusion is not None:
        # Advection over diffusion across a cell: |v| dr / D
        pairs["cell_peclet"] = lambda t, dr: _cell_ratio(
            np.abs(advection.velocities(t)) * dr, diffusion.diffusion_coefficients(t)
        )
    if advection is not None and loss is not None:
        # Cell crossing time over loss time: (dr / |v|) (|p_dot| / p)
        pairs["cell_damkohler_advection"] = lambda t, dr: _cell_ratio(
            loss.loss_rates(t) * dr, np.abs(advection.velocities(t))
        )
    if diffusion is not None and loss is not None:
        # Cell diffusion time over loss time: (dr^2 / D) (|p_dot| / p)
        pairs["cell_damkohler_diffusion"] = lambda t, dr: _cell_ratio(
            loss.loss_rates(t) * dr**2, diffusion.diffusion_coefficients(t)
        )
    if not pairs:
        return {}
    dr = grid.dr.to_value(su.LENGTH)
    return {
        name: max(cell_number(t, dr) for t in times)
        for name, cell_number in pairs.items()
    }


def _cell_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Largest ratio over the cells where both processes act, or zero if there are none."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    active = (numerator > 0.0) & (denominator > 0.0)
    return float(np.max(numerator[active] / denominator[active], initial=0.0))


def warn(diagnostics: SimulationDiagnostics) -> None:
    """Log a warning for each problem found in ``diagnostics``."""
    for message in diagnostics.warnings:
        logger.warning(message)
