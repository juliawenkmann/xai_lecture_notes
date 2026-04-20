from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
XAI_BOOK_DIR = ROOT_DIR / "xai_book"
LEGACY_TOKENS = (
    "#1ABC9C",
    "#8E44AD",
    "#EA3356",
    "#2563EB",
    "#F59E0B",
    "#FB923C",
    "#FFAB00",
    "bookTurquoise",
    "bookPurple",
    "bookGold",
)
RAW_BOOK_HEX_TOKENS = (
    "#1A73E8",
    "#D4742C",
    "#5A5A5A",
)


class ColorPaletteTests(unittest.TestCase):
    def test_shared_plotting_palette_uses_book_colors(self):
        from xai_book import plotting

        self.assertEqual(plotting.BOOK_ORANGE_BASE, "#D4742C")
        self.assertEqual(plotting.BOOK_BLUE_BASE, "#1A73E8")
        self.assertEqual(plotting.CHAPTER_GRAY_BASE, "#5A5A5A")
        self.assertEqual(plotting.BOOK_ORANGE, "#E3A576")
        self.assertEqual(plotting.BOOK_BLUE, "#6AA4F0")
        self.assertEqual(plotting.CHAPTER_GRAY, "#7B7B7B")
        self.assertEqual(plotting.ORANGE, plotting.BOOK_ORANGE)
        self.assertEqual(plotting.BLUE, plotting.BOOK_BLUE)
        self.assertEqual(plotting.FOREGROUND, plotting.CHAPTER_GRAY)
        self.assertEqual(plotting.book_palette(include_gray=False), [plotting.BOOK_BLUE, plotting.BOOK_ORANGE])
        self.assertEqual(
            plotting.book_palette(),
            [plotting.BOOK_BLUE, plotting.BOOK_ORANGE, plotting.CHAPTER_GRAY],
        )
        self.assertEqual(
            plotting.extended_book_palette(),
            [
                plotting.BOOK_BLUE,
                plotting.BOOK_ORANGE,
                plotting.CHAPTER_GRAY,
                plotting.SOFT_BLUE,
                plotting.SOFT_SAGE,
                plotting.SOFT_TAUPE,
            ],
        )

    def test_notebook_source_does_not_use_legacy_palette_tokens(self):
        offenders: list[str] = []

        for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                source = "".join(cell.get("source", []))
                for token in LEGACY_TOKENS:
                    if token in source:
                        offenders.append(f"{path.relative_to(ROOT_DIR)}#cell-{index} -> {token}")

        self.assertEqual(offenders, [])

    def test_notebook_source_uses_shared_palette_not_raw_book_hex(self):
        offenders: list[str] = []

        for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                source = "".join(cell.get("source", []))
                for token in RAW_BOOK_HEX_TOKENS:
                    if token in source:
                        offenders.append(f"{path.relative_to(ROOT_DIR)}#cell-{index} -> {token}")

        self.assertEqual(offenders, [])

    def test_python_modules_do_not_use_legacy_palette_tokens(self):
        offenders: list[str] = []

        for base_dir in (XAI_BOOK_DIR, NOTEBOOKS_DIR):
            for path in sorted(base_dir.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                for token in LEGACY_TOKENS:
                    if token in source:
                        offenders.append(f"{path.relative_to(ROOT_DIR)} -> {token}")

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
