"""
CLI Module for SAETASS.

Contains tools for displaying a progress bar and the SAETASS banner, and for
setting up the log messages and turning the output off (see :py:mod:`saetass.cli.output`).
"""

from .banner import print_banner
from .output import is_quiet, set_log_level, set_quiet, set_rich_logging
from .progress import create_progress_bar

__all__ = [
    "print_banner",
    "create_progress_bar",
    "is_quiet",
    "set_quiet",
    "set_log_level",
    "set_rich_logging",
]
