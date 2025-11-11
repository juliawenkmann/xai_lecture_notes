"""Global Matplotlib style: modern look with blue & orange palette."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2563EB"    # blue-600
ORANGE = "#F59E0B"  # amber-500
GRID = "#E5E7EB"    # gray-200
FG = "#111827"      # gray-900
BG = "#FFFFFF"      # white

FIGSIZE = (7.0, 4.2)
DPI = 180
FONTSIZE = 11
LINEWIDTH = 2.0

def apply_style():
    mpl.rcParams.update({
        # geometry
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        # typography
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE + 1,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE - 1,
        "ytick.labelsize": FONTSIZE - 1,
        # colors
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": GRID,
        "text.color": FG,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        # grid and spines
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # lines & legend
        "lines.linewidth": LINEWIDTH,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE]),
    })

def blue_orange_cmap(name: str = "blue_orange"):
    return LinearSegmentedColormap.from_list(name, [BLUE, ORANGE], N=256)
