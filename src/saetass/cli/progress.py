import sys
import time

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

console = Console(theme=custom_theme)
if not (console.is_jupyter or sys.stdout.isatty()):
    console.width = 120  # a file, a pipe, a CI log

#: Seconds between updates of the progress bar in notebooks.
NOTEBOOK_REFRESH_INTERVAL = 0.1


class NotebookProgress(Progress):
    """
    Progress bar for notebooks, drawn in an output that is updated in place.

    Rich draws its bar in a widget, which static pages (GitHub, the documentation) do not show and which is not refreshed when the bar stops. This output is saved with the notebook, in its final state.
    """

    _display = None
    _next_refresh = 0.0

    def start(self) -> None:
        """Display the bar."""
        from IPython.display import display

        self._display = display(self, display_id=True)

    def stop(self) -> None:
        """Leave the bar in its final state."""
        self.refresh()
        self._display = None

    def refresh(self) -> None:
        """Redraw the bar, if it is displayed."""
        if self._display is not None:
            self._next_refresh = time.monotonic() + NOTEBOOK_REFRESH_INTERVAL
            self._display.update(self)

    def update(self, *args, **kwargs) -> None:
        """Update a task, redrawing the bar every :py:data:`NOTEBOOK_REFRESH_INTERVAL` seconds."""
        super().update(*args, **kwargs)
        if time.monotonic() >= self._next_refresh:
            self.refresh()


class ProgressBarSingleton:
    """
    Singleton wrapper around rich.progress.Progress (:py:class:`NotebookProgress` in notebooks) to ensure only one Live display is active.
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
            progress_class = NotebookProgress if console.is_jupyter else Progress
            self._progress = progress_class(
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
