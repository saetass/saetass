import logging

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from .palette import SAETASS_ORANGE, SAETASS_YELLOW
from .progress import console as default_console

logger = logging.getLogger(__name__)

DEFAULT_BANNER = r"""
          :::::::::::::::           
      :::::::::::::::::::::::       
    :::::::::++++++++++::::::::     
   ::::::::++++++++++++++::::::::   
  :::::::++++++++++++++++++:::::::  
 ::::::::++++++++++++++++++:::::::: 
 :::::::++++++++++++++++++++::::::: 
 :::::::+++++++++++++++++++:::::::: 
 ::::::::++++++++++++++++++:::::::: 
  ::::::::::++++++++++++++          
   :::::::::::::::++++++            
     :::::::::::::::::::::          
        ::::::::::::::::::::::      
            ++:::::::::::::::::::   
          ++++++++++++::::::::::::  
        ++++++++++++++++++::::::::: 
::::::::++++++++++++++++++++::::::::
:::::::++++++++++++++++++++++:::::::
:::::::++++++++++++++++++++++:::::::
 :::::::++++++++++++++++++++::::::::
 ::::::::++++++++++++++++++:::::::: 
   :::::::++++++++++++++++::::::::  
    :::::::::++++++++++:::::::::    
       :::::::::::::::::::::::      
           :::::::::::::::          
"""


#: Title of the banner: the name of SAETASS and, after the colon, what it stands for.
DEFAULT_TITLE = "SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry"


def _banner(console: Console, text: str, title: str) -> RenderableType:
    """
    The banner that fits ``console``.

    - With room for the art and the whole title on one line: the art in a panel titled ``title``.
    - With room for the art only: the art in a panel titled with the name before the colon of ``title``, and the rest of the title wrapped below the art.
    - Otherwise, and whenever the output is not an interactive terminal or notebook (a file, a pipe, a CI log), where a large banner only clutters the output: ``title`` alone, as a single line that wraps if needed.

    The height of the terminal does not matter: the banner is printed once per process and scrolls up as any other output.

    Rich already adapts each layout to the terminal: ASCII borders where the encoding is not UTF-8, the border style of the legacy Windows console, and fewer colours (or none, with ``NO_COLOR`` or ``TERM=dumb``) where the terminal supports fewer.
    """
    art = text.strip("\n").splitlines()
    art_width = max((len(line) for line in art), default=0)
    name, _, description = title.partition(": ")
    interactive = console.is_terminal or console.is_jupyter
    # Two border columns and two padding columns
    if not (interactive and console.width >= art_width + 4):
        return Text.assemble((name, f"bold {SAETASS_YELLOW}"), (title[len(name) :], ""))

    rich_text = Text("\n".join(art), no_wrap=True)
    rich_text.highlight_regex(r":", SAETASS_ORANGE)
    rich_text.highlight_regex(r"\+", SAETASS_YELLOW)
    if console.width >= len(title) + 6 or not description:
        content, panel_title = Align.center(rich_text), title
    else:
        content = Group(
            Align.center(rich_text),
            Text(""),
            Text(description, justify="center"),
        )
        panel_title = name
    return Align.center(
        Panel(
            content,
            title=f"[bold {SAETASS_YELLOW}]{panel_title}[/]",
            expand=False,
            border_style=SAETASS_ORANGE,
        )
    )


class BannerManager:
    """
    Singleton manager for printing the SAETASS ASCII banner.
    Ensures the banner is printed at most once per Python process lifecycle,
    avoiding clutter when multiple Solver instances are created sequentially.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BannerManager, cls).__new__(cls)
            cls._instance._printed = False
        return cls._instance

    def print_once(
        self,
        console: Console = None,
        text: str = DEFAULT_BANNER,
        title: str = DEFAULT_TITLE,
    ):
        """
        Prints the banner that fits the console if it hasn't been printed yet by this singleton.
        """
        if self._printed:
            return

        if console is None:
            console = default_console

        console.print(_banner(console, text, title))
        self._printed = True

    def reset(self):
        """Allow resetting the singleton state, useful for testing."""
        self._printed = False


def print_banner(
    console: Console = None,
    text: str = DEFAULT_BANNER,
    title: str = DEFAULT_TITLE,
):
    """
    Print the SAETASS ASCII banner, in the layout that fits the console.
    It delegates to a Singleton to ensure it's only printed once per runtime.

    The art is shown only in interactive terminals and notebooks wide enough for it; otherwise the banner is the title alone, on a single line.

    Parameters
    ----------
    console : rich.console.Console, optional
        A rich Console instance to use for printing.
    text : str, optional
        The ASCII art/text to print.
    title : str, optional
        The string to place at the top of the panel border. When the console is too narrow for it, the part before the colon stays there and the rest goes below the art.
    """
    BannerManager().print_once(console=console, text=text, title=title)
