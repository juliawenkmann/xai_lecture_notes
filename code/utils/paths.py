from pathlib import Path

# Determine project root as parent of this file's parent (code/)
CODE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = CODE_DIR.parent

FIGURES_DIR = ROOT_DIR / "figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

def chapter_dir(chapter: str, base: Path = FIGURES_DIR) -> Path:
    d = base / chapter
    d.mkdir(parents=True, exist_ok=True)
    return d

def figures_chapter_dir(chapter: str) -> Path:
    """Return (and create if needed) the chapter directory under figures/."""
    return chapter_dir(chapter, FIGURES_DIR)
