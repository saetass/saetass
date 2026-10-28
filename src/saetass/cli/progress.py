import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

from .palette import (
    SAETASS_BLUE,
    SAETASS_DARK,
    SAETASS_GREY,
    SAETASS_ORANGE,
    SAETASS_YELLOW,
)

# SAETASS Theme
custom_theme = Theme(
    {
        "info": SAETASS_GREY,
        "warning": SAETASS_YELLOW,
        "danger": SAETASS_ORANGE,
        "success": SAETASS_BLUE,
        "saetass_yellow": SAETASS_YELLOW,
        "saetass_orange": SAETASS_ORANGE,
        "saetass_blue": SAETASS_BLUE,
        "saetass_grey": SAETASS_GREY,
        "saetass_dark": SAETASS_DARK,
    }
)

width = None if sys.stdout.isatty() else 120
console = Console(theme=custom_theme, width=width)


class ProgressBarSingleton:
    """
    Singleton wrapper around rich.progress.Progress to ensure only one Live display is active.
    It tracks reference counts via start() and stop() to manage the lifecycle cleanly across multiple usages.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._ref_count = 0
            cls._instance._progress = None
        return cls._instance

    def _ensure_progress(self):
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots", style=SAETASS_YELLOW),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    bar_width=None,
                    complete_style=SAETASS_BLUE,
                    finished_style=SAETASS_BLUE,
                    pulse_style=SAETASS_BLUE,
                ),
                MofNCompleteColumn(),
                TextColumn("[dim]metrics:[/] {task.fields[metrics]}"),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            )

    def start(self):
        self._ensure_progress()
        if self._ref_count == 0:
            self._progress.start()
        self._ref_count += 1

    def stop(self):
        if self._ref_count > 0:
            self._ref_count -= 1
        if self._ref_count == 0 and self._progress is not None:
            self._progress.stop()
            self._progress = None

    def add_task(self, *args, **kwargs):
        self._ensure_progress()
        return self._progress.add_task(*args, **kwargs)

    def update(self, *args, **kwargs):
        self._ensure_progress()
        return self._progress.update(*args, **kwargs)


def create_progress_bar():
    """
    Returns the singleton instance of the Progress bar manager for SAETASS solvers.

    Returns
    -------
    ProgressBarSingleton
        A singleton acting as a rich.progress.Progress proxy.
    """
    return ProgressBarSingleton()
