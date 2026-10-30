import matplotlib as mpl
import matplotlib.pyplot as plt

SAETASS_YELLOW = "#F3AC4B"
SAETASS_ORANGE = "#F3664B"
SAETASS_BLUE = "#4D7EA8"


def get_quantitative_style(step_idx=None, total_steps=None):
    """Style for quantitative metrics (convergence, errors, particle counts)."""
    if step_idx is None or total_steps is None or total_steps <= 1:
        return {"color": SAETASS_ORANGE, "linestyle": "-", "marker": "o", "alpha": 1.0}

    # Calculate opacity between 0.4 and 1.0 for temporal evolutions
    alpha_min, alpha_max = 0.4, 1.0
    normalized_step = step_idx / (total_steps - 1)
    alpha = alpha_min + (alpha_max - alpha_min) * normalized_step

    return {"color": SAETASS_ORANGE, "linestyle": "-", "marker": "o", "alpha": alpha}


def get_analytical_style():
    """Style for analytical solutions."""
    return {"color": SAETASS_YELLOW, "linestyle": "--"}


def get_numerical_style(
    is_initial=False, is_final=False, step_idx=None, total_steps=None
):
    """
    Style for numerical solutions.
    - initial: dashed, low opacity
    - final: solid, high opacity
    - intermediate: solid, increasing opacity
    """
    if is_initial:
        return {"color": SAETASS_BLUE, "linestyle": "--", "alpha": 0.4}

    if is_final or step_idx is None or total_steps is None or total_steps <= 1:
        return {"color": SAETASS_BLUE, "linestyle": "-", "alpha": 1.0}

    # Calculate opacity between 0.2 and 1.0 for temporal evolutions
    alpha_min, alpha_max = 0.2, 1.0
    normalized_step = step_idx / (total_steps - 1)
    alpha = alpha_min + (alpha_max - alpha_min) * normalized_step

    return {"color": SAETASS_BLUE, "linestyle": "-", "alpha": alpha}


def add_time_colorbar(fig, ax, t_min, t_max, cmap_name="Blues"):
    """
    Adds a colorbar to the figure to represent the time evolution.
    Matches the SAETASS Blue gradient used in get_numerical_style.
    """
    # Create an invisible mappable object for the colorbar
    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)

    # We create a custom colormap that matches the alpha blending from get_numerical_style
    # get_numerical_style goes from alpha=0.2 to alpha=1.0 using SAETASS_BLUE
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    color_rgba = to_rgba(SAETASS_BLUE)
    color_low_alpha = (color_rgba[0], color_rgba[1], color_rgba[2], 0.2)
    color_high_alpha = (color_rgba[0], color_rgba[1], color_rgba[2], 1.0)

    saetass_blues_cmap = LinearSegmentedColormap.from_list(
        "saetass_blues", [color_low_alpha, color_high_alpha]
    )

    sm = mpl.cm.ScalarMappable(cmap=saetass_blues_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time: $t$ (a. u.)")
    return cbar


def apply_plot_style():
    """
    Apply unified plot style settings for validation cases.
    """
    # Use LaTeX for matplotlib text rendering when available
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            # increased font sizes for publication-quality figures
            "font.size": 12,
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 15,
            "lines.linewidth": 3,
            # Automatically adjust subplot parameters to give specified padding
            # and prevent axes labels from being cut off
            "figure.autolayout": True,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )
