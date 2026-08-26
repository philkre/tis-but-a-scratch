"""
Shared plot styling.

Palette is the validated default from the dataviz reference: the categorical
trio clears colorblind separation on all pairs (worst deutan dE 9.2), and the
blue ramp is an ordinal scale with monotone lightness and visible step gaps.
Don't substitute colors ad hoc -- re-run the validator if the set changes.
"""

import os

import matplotlib.pyplot as plt

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text, axis labels, reference lines
GRID = "#d8d7d2"       # recessive grid and spines

# categorical: fixed order, never cycled
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CATEGORICAL = [BLUE, ORANGE, AQUA]

# ordinal ramp, light -> dark, for ordered series such as epochs
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

DPI = 300


def style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive axes, muted labels. Text wears text colors, never series colors."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(alpha=0.25, linewidth=0.7, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)


def legend(ax, **kwargs):
    """Legend with muted text -- identity comes from the mark, not colored words."""
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", 8.5)
    leg = ax.legend(**kwargs)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return leg


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
