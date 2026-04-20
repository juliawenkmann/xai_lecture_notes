from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "out"
FIGURES_DIR = OUT_DIR
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_dir(output_dir: Path | str | None = None) -> Path:
    if output_dir is None:
        return ensure_dir(FIGURES_DIR)
    return ensure_dir(Path(output_dir))


def chapter_figure_dir(chapter: str) -> Path:
    return ensure_dir(FIGURES_DIR / chapter)


def chapter_figure_path(
    chapter: str,
    filename: str,
    suffix: str = ".png",
    output_dir: Path | str | None = None,
) -> Path:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    base_dir = resolve_output_dir(output_dir)
    if output_dir is None:
        base_dir = base_dir / chapter
    return ensure_dir(base_dir) / f"{filename}{suffix}"
