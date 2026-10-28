"""
Terminal output of SAETASS: its log messages, and the banner and live progress bar that every :py:class:`~saetass.solver.Solver` prints, all rendered with `Rich <https://rich.readthedocs.io>`_ on the same console, so that messages appear above a running progress bar instead of breaking it.
Rich adapts the output to the terminal: without one (a file, a pipe, a CI log) it neither animates nor colours, and it also honours ``NO_COLOR`` and ``TERM=dumb``, non-UTF-8 encodings and the legacy Windows console.

Log messages
------------

Every module logs to a child of the ``saetass`` logger, which has its own Rich handler and does not pass messages on to the root logger.
Its level, ``INFO`` by default, is set with :py:func:`set_log_level` or the environment variable ``SAETASS_LOG_LEVEL`` (a level name, such as ``DEBUG`` or ``WARNING``, or number), read when SAETASS is imported.
``INFO`` reports what each simulation does (operators, grid, time steps, duration), ``DEBUG`` adds the details of every operator, the threads of the kernels and the sub-cycles of each step.
To handle the messages with your own logging configuration instead, call ``set_rich_logging(False)``.

.. tip::

    For more detail, for instance to follow the CFL sub-cycles of a slow simulation, or for warnings only:

    .. code-block:: python

        from saetass.cli import set_log_level

        set_log_level("DEBUG")  # or "WARNING"

    or ``export SAETASS_LOG_LEVEL=DEBUG`` in the shell, before starting Python.

Banner and progress bar
-----------------------

They can be turned off, in order of precedence:

1. for one simulation, with ``Solver(..., quiet=True)``;
2. for the whole process, with :py:func:`set_quiet`;
3. through the environment variable ``SAETASS_QUIET`` (``1``, ``true``, ``yes`` or ``on`` turn it off, and ``0``, ``false``, ``no`` or ``off`` turn it on).

Without any of them, they are shown in the main process and hidden in child processes, such as the workers of :py:mod:`multiprocessing` or :py:func:`saetass.sweep.run_sweep`: the live progress bars of simulations running side by side would overwrite each other on the same terminal.
Log messages are not affected.

.. note::

    Some terminals, notebook front ends and log files do not render the banner and the progress bar well. If they look garbled, hide them:

    .. code-block:: python

        solver = Solver(state, "diffusion", operator_params, quiet=True)  # one simulation

        from saetass.cli import set_quiet

        set_quiet(True)  # every simulation of this process

    or ``export SAETASS_QUIET=1`` in the shell, for every simulation started from it.
"""

import logging
import multiprocessing
import os

from rich.logging import RichHandler

from .progress import console

#: Name of the logger of SAETASS, the parent of the logger of every module.
LOGGER_NAME = "saetass"

#: Level of the log messages when ``SAETASS_LOG_LEVEL`` is not set.
DEFAULT_LOG_LEVEL = logging.INFO

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

_quiet: bool | None = None


def plural(count: int, noun: str, plural_noun: str | None = None) -> str:
    """
    ``count`` followed by ``noun`` in singular or plural, for messages.

    Parameters
    ----------
    count : int
        Number of things.
    noun : str
        Singular noun.
    plural_noun : str, optional
        Plural noun. Default is ``None``, meaning ``noun`` followed by ``s``.

    Returns
    -------
    str
        For instance ``"1 step"`` or ``"3 steps"``.
    """
    return f"{count} {noun if count == 1 else plural_noun or noun + 's'}"


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


def _parse_log_level(level: int | str, name: str) -> int:
    """Level number of a level name or number."""
    if isinstance(level, str):
        text = level.strip()
        level = int(text) if text.isdigit() else logging.getLevelName(text.upper())
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError(
            f"{name} must be a logging level such as 'DEBUG', 'INFO' or 'WARNING', "
            f"or its number."
        )
    return level


def set_log_level(level: int | str) -> None:
    """
    Set the level of the log messages of SAETASS.

    Parameters
    ----------
    level : int or str
        A level of :py:mod:`logging`, as a number (``logging.DEBUG``) or a name (``"DEBUG"``).

    Raises
    ------
    ValueError
        If ``level`` is not a logging level.
    """
    logging.getLogger(LOGGER_NAME).setLevel(_parse_log_level(level, "level"))


_handler = RichHandler(
    console=console, show_path=False, markup=False, rich_tracebacks=False
)


def set_rich_logging(enabled: bool) -> None:
    """
    Render the log messages of SAETASS with its Rich console, or hand them to the root logger.

    Parameters
    ----------
    enabled : bool
        ``True`` (the default when SAETASS is imported) to print them with the Rich handler of SAETASS, ``False`` to pass them on to the root logger, for your own logging configuration (for instance :py:func:`logging.basicConfig`) to handle them.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if enabled:
        if _handler not in logger.handlers:
            logger.addHandler(_handler)
        logger.propagate = False
    else:
        logger.removeHandler(_handler)
        logger.propagate = True


set_rich_logging(True)
set_log_level(os.environ.get("SAETASS_LOG_LEVEL", "").strip() or DEFAULT_LOG_LEVEL)
