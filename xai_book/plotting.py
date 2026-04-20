from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BOOK_ORANGE_BASE = "#D4742C"
BOOK_BLUE_BASE = "#1A73E8"
CHAPTER_GRAY_BASE = "#5A5A5A"
BOOK_LATEX_FONT_FAMILY = "Computer Modern Roman"
PLOTLY_FONT_FAMILY = "Latin Modern Roman, Computer Modern Roman, serif"
LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{amssymb}"


def _blend_with_white(color: str, amount: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {color!r}")

    channels = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    blended = [round(channel + (255 - channel) * amount) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in blended)


# Softer textbook-style variants of the original book colors.
BOOK_ORANGE = _blend_with_white(BOOK_ORANGE_BASE, 0.35)
BOOK_BLUE = _blend_with_white(BOOK_BLUE_BASE, 0.35)
CHAPTER_GRAY = _blend_with_white(CHAPTER_GRAY_BASE, 0.20)

SOFT_BLUE = _blend_with_white(BOOK_BLUE_BASE, 0.58)
SOFT_ORANGE = _blend_with_white(BOOK_ORANGE_BASE, 0.58)
SOFT_SAGE = "#B7D0BE"
SOFT_TAUPE = "#C7B3A2"

BLUE = BOOK_BLUE
ORANGE = BOOK_ORANGE
GRID = "#E8EDF3"
FOREGROUND = CHAPTER_GRAY
BACKGROUND = "#FCFCFB"
TRANSPARENT_BACKGROUND = "none"
PLOTLY_TRANSPARENT_BACKGROUND = "rgba(0,0,0,0)"
ANNOTATION_EDGE = "#D3DCE8"

BOOK_EXPORT_WIDTH = 6.8
BOOK_BASE_HEIGHT = 4.0
BOOK_ADDITIONAL_ROW_HEIGHT = 2.1
DEFAULT_DPI = 180
DEFAULT_FONT_SIZE = 11
DEFAULT_LINE_WIDTH = 2.0
NOTEBOOK_FONT_SIZE = 12
NOTEBOOK_TITLE_SIZE = 15
NOTEBOOK_LABEL_SIZE = 12
NOTEBOOK_TICK_SIZE = 11
NOTEBOOK_LEGEND_SIZE = 11
NOTEBOOK_ANNOTATION_SIZE = 11

LATEX_TEXT_REPLACEMENTS = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def latex_is_available() -> bool:
    return shutil.which("latex") is not None


def latex_escape(text: Any) -> str:
    escaped = str(text)
    for source, target in LATEX_TEXT_REPLACEMENTS:
        escaped = escaped.replace(source, target)
    return escaped


def panel_figure_size(
    n_rows: int = 1,
    n_cols: int = 1,
    *,
    extra_width: float = 0.0,
    extra_height: float = 0.0,
) -> tuple[float, float]:
    width = BOOK_EXPORT_WIDTH + extra_width
    height = BOOK_BASE_HEIGHT + max(0, n_rows - 1) * BOOK_ADDITIONAL_ROW_HEIGHT + extra_height
    if n_cols > 1:
        height += 0.15 * (n_cols - 1)
    return (width, height)


FIGURE_SIZE_TOKENS = {
    "single": panel_figure_size(),
    "wide": panel_figure_size(extra_height=0.2),
    "square": (BOOK_EXPORT_WIDTH * 0.84, BOOK_EXPORT_WIDTH * 0.84),
    "two_panel": panel_figure_size(n_cols=2),
    "three_panel": panel_figure_size(n_cols=3, extra_height=-0.1),
    "grid_2x2": panel_figure_size(n_rows=2, n_cols=2, extra_height=0.55),
    "breakdown": panel_figure_size(extra_height=0.35),
    "heatmap": panel_figure_size(extra_height=0.8),
    "tree": panel_figure_size(extra_height=1.8),
}

DEFAULT_FIGSIZE = FIGURE_SIZE_TOKENS["single"]
NOTEBOOK_WIDE_FIGSIZE = FIGURE_SIZE_TOKENS["wide"]
NOTEBOOK_SQUARE_FIGSIZE = FIGURE_SIZE_TOKENS["square"]


def figure_size(
    size: str = "single",
    *,
    n_rows: int | None = None,
    n_cols: int | None = None,
    extra_width: float = 0.0,
    extra_height: float = 0.0,
) -> tuple[float, float]:
    if n_rows is not None or n_cols is not None or extra_width or extra_height:
        return panel_figure_size(
            n_rows=n_rows or 1,
            n_cols=n_cols or 1,
            extra_width=extra_width,
            extra_height=extra_height,
        )
    if size not in FIGURE_SIZE_TOKENS:
        raise KeyError(f"Unknown figure size token {size!r}")
    return FIGURE_SIZE_TOKENS[size]


def book_palette(*, include_gray: bool = True) -> list[str]:
    palette = [BLUE, ORANGE]
    if include_gray:
        palette.append(CHAPTER_GRAY)
    return palette


def extended_book_palette() -> list[str]:
    return [BLUE, ORANGE, CHAPTER_GRAY, SOFT_BLUE, SOFT_SAGE, SOFT_TAUPE]


def _build_rc_params(
    *,
    figsize: tuple[float, float],
    font_size: float,
    title_size: float,
    label_size: float,
    tick_size: float,
    line_width: float,
    grid_axis: str,
    background: str,
) -> dict[str, Any]:
    use_latex = latex_is_available()
    return {
        "figure.figsize": figsize,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.facecolor": background,
        "savefig.edgecolor": background,
        "font.family": "serif",
        "font.serif": [BOOK_LATEX_FONT_FAMILY, "Latin Modern Roman", "CMU Serif"],
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "axes.titleweight": "semibold",
        "axes.titlecolor": FOREGROUND,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": tick_size,
        "legend.title_fontsize": label_size,
        "axes.facecolor": background,
        "figure.facecolor": background,
        "axes.edgecolor": ANNOTATION_EDGE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": FOREGROUND,
        "text.color": FOREGROUND,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "xtick.color": FOREGROUND,
        "ytick.color": FOREGROUND,
        "axes.grid": True,
        "axes.grid.axis": grid_axis,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.95,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": line_width,
        "patch.edgecolor": background,
        "patch.force_edgecolor": False,
        "legend.frameon": False,
        "legend.labelcolor": FOREGROUND,
        "legend.borderaxespad": 0.6,
        "axes.prop_cycle": mpl.cycler(color=book_palette(include_gray=False)),
        "mathtext.fontset": "cm",
        "text.usetex": use_latex,
        "text.latex.preamble": LATEX_PREAMBLE if use_latex else "",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def apply_plot_style(
    *,
    size: str | None = None,
    figsize: tuple[float, float] | None = None,
    font_size: float | None = None,
    title_size: float | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
    line_width: float | None = None,
    grid_axis: str = "both",
    background: str | None = None,
    use_seaborn: bool = True,
) -> None:
    figsize = figsize or figure_size(size or "single")
    font_size = font_size or DEFAULT_FONT_SIZE
    title_size = title_size or (font_size + 1)
    label_size = label_size or font_size
    tick_size = tick_size or (font_size - 1)
    line_width = line_width or DEFAULT_LINE_WIDTH
    background = TRANSPARENT_BACKGROUND if background is None else background

    rc_params = _build_rc_params(
        figsize=figsize,
        font_size=font_size,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        line_width=line_width,
        grid_axis=grid_axis,
        background=background,
    )

    if use_seaborn:
        try:
            import seaborn as sns
        except Exception:
            sns = None
        if sns is not None:
            sns.set_theme(
                style="whitegrid",
                context="notebook",
                palette=book_palette(include_gray=False),
                rc=rc_params,
            )
            return

    mpl.rcParams.update(rc_params)


def apply_notebook_style(
    *,
    size: str | None = None,
    figsize: tuple[float, float] | None = None,
    grid_axis: str = "y",
    background: str | None = None,
    use_seaborn: bool = True,
) -> None:
    apply_plot_style(
        figsize=figsize or figure_size(size or "wide"),
        font_size=NOTEBOOK_FONT_SIZE,
        title_size=NOTEBOOK_TITLE_SIZE,
        label_size=NOTEBOOK_LABEL_SIZE,
        tick_size=NOTEBOOK_TICK_SIZE,
        line_width=2.2,
        grid_axis=grid_axis,
        background=background,
        use_seaborn=use_seaborn,
    )


def book_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    size: str = "single",
    style: str = "plot",
    figsize: tuple[float, float] | None = None,
    background: str | None = None,
    grid_axis: str = "both",
    constrained_layout: bool = True,
    extra_width: float = 0.0,
    extra_height: float = 0.0,
    **kwargs,
):
    resolved_figsize = figsize or figure_size(
        size,
        n_rows=nrows if size == "grid" else None,
        n_cols=ncols if size == "grid" else None,
        extra_width=extra_width,
        extra_height=extra_height,
    )
    if style == "notebook":
        apply_notebook_style(figsize=resolved_figsize, grid_axis=grid_axis, background=background)
    else:
        apply_plot_style(figsize=resolved_figsize, grid_axis=grid_axis, background=background)
    return plt.subplots(nrows, ncols, figsize=resolved_figsize, constrained_layout=constrained_layout, **kwargs)


def plotly_figure_size(
    size: str = "square",
    *,
    n_rows: int | None = None,
    n_cols: int | None = None,
    extra_width: float = 0.0,
    extra_height: float = 0.0,
    pixels_per_inch: int = 100,
) -> tuple[int, int]:
    width, height = figure_size(
        size,
        n_rows=n_rows,
        n_cols=n_cols,
        extra_width=extra_width,
        extra_height=extra_height,
    )
    return (round(width * pixels_per_inch), round(height * pixels_per_inch))


def apply_plotly_style(
    fig,
    *,
    size: str = "square",
    n_rows: int | None = None,
    n_cols: int | None = None,
    extra_width: float = 0.0,
    extra_height: float = 0.0,
    background: str | None = None,
) -> None:
    width, height = plotly_figure_size(
        size,
        n_rows=n_rows,
        n_cols=n_cols,
        extra_width=extra_width,
        extra_height=extra_height,
    )
    bg = PLOTLY_TRANSPARENT_BACKGROUND if background is None else background
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family=PLOTLY_FONT_FAMILY, size=NOTEBOOK_TICK_SIZE, color=FOREGROUND),
        title_font=dict(size=NOTEBOOK_TITLE_SIZE, color=FOREGROUND),
        legend=dict(font=dict(size=NOTEBOOK_LEGEND_SIZE, color=FOREGROUND)),
        margin=dict(l=56, r=28, t=76, b=56),
    )


def blue_orange_cmap(name: str = "blue_orange") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [BLUE, ORANGE], N=256)


def white_to_color_cmap(color: str, name: str | None = None) -> LinearSegmentedColormap:
    cmap_name = name or f"white_to_{color.lstrip('#').lower()}"
    return LinearSegmentedColormap.from_list(cmap_name, [BACKGROUND, color], N=256)


def save_figure(fig, path: Path | str, dpi: int = 300, tight_layout: bool = True) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_tight_layout = tight_layout and hasattr(fig, "tight_layout")
    if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
        use_tight_layout = False
    if use_tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    facecolor = fig.get_facecolor()
    transparent = isinstance(facecolor, tuple) and len(facecolor) == 4 and facecolor[-1] == 0.0
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    return out_path


def display_and_save_figure(fig, path: Path | str, dpi: int = 300, tight_layout: bool = True) -> Path:
    from IPython.display import display

    display(fig)
    return save_figure(fig, path, dpi=dpi, tight_layout=tight_layout)


def save_plotly_html(fig, path: Path | str, *, include_plotlyjs: str = "cdn") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs=include_plotlyjs,
        include_mathjax="cdn",
        full_html=True,
        auto_open=False,
    )
    return out_path
