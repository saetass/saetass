"""
Switch for the terminal output of SAETASS: the banner and the live progress bar that every :py:class:`~saetass.solver.Solver` prints with `Rich <https://rich.readthedocs.io>`_.

They can be turned off, in order of precedence:

1. for one simulation, with ``Solver(..., quiet=True)``;
2. for the whole process, with :py:func:`set_quiet`;
3. through the environment variable ``SAETASS_QUIET`` (``1``, ``true``, ``yes`` or ``on`` turn it off, and ``0``, ``false``, ``no`` or ``off`` turn it on).

Without any of them, they are shown in the main process and hidden in child processes, such as the workers of :py:mod:`multiprocessing` or :py:func:`saetass.sweep.run_sweep`: the live progress bars of simulations running side by side would overwrite each other on the same terminal.
Silencing them is also useful in terminals and log files that do not render Rich well.
"""

import multiprocessing
import os

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

_quiet: bool | None = None


def set_quiet(quiet: bool | None) -> None:
    """
    Turn the banner and progress bar of every :py:class:`~saetass.solver.Solver` of this process off or on.

    Parameters
    ----------
    quiet : bool or None
        ``True`` to hide them, ``False`` to show them, or ``None`` to go back to ``SAETASS_QUIET`` or, without it, the default (shown only in the main process).
    """
    global _quiet
    _quiet = None if quiet is None else bool(quiet)


def is_quiet() -> bool:
    """
    Whether a :py:class:`~saetass.solver.Solver` created now hides its banner and progress bar, unless told otherwise.

    Returns
    -------
    bool
        The value set by :py:func:`set_quiet`, else that of ``SAETASS_QUIET``, else whether this is a child process.

    Raises
    ------
    ValueError
        If ``SAETASS_QUIET`` is set to a value that is not recognized.
    """
    if _quiet is not None:
        return _quiet
    value = os.environ.get("SAETASS_QUIET", "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    if value:
        raise ValueError(
            f"SAETASS_QUIET must be one of {sorted(_TRUE | _FALSE)}, got {value!r}."
        )
    return multiprocessing.parent_process() is not None
