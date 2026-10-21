"""
CLI Module for SAETASS.

Contains tools for displaying a progress bar, integrating rich logging,
printing the SAETASS banner and turning both off (see :py:mod:`saetass.cli.output`).
"""

from .banner import print_banner
from .output import is_quiet, set_quiet
from .progress import create_progress_bar, setup_rich_logging

__all__ = [
    "print_banner",
    "setup_rich_logging",
    "create_progress_bar",
    "is_quiet",
    "set_quiet",
]
