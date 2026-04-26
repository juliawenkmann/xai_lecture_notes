from __future__ import annotations

import json
import re
import tempfile
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
                plotting.SOFT_ORANGE,
                plotting.SOFT_GRAY,
            ],
        )

    def test_saved_matplotlib_figures_use_standard_book_canvas(self):
        from PIL import Image

        from xai_book import plotting
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            pdf_fig, pdf_axis = plt.subplots()
            pdf_axis.plot([0, 1], [0, 1])
            pdf_path = plotting.save_figure(pdf_fig, tmp_path / "figure.pdf")
            pdf_bytes = pdf_path.read_bytes()
            match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_bytes)
            self.assertIsNotNone(match)
            width_points, height_points = (float(match.group(1)), float(match.group(2)))
            self.assertAlmostEqual(width_points / 72, plotting.BOOK_CANVAS_SIZE[0], places=2)
            self.assertAlmostEqual(height_points / 72, plotting.BOOK_CANVAS_SIZE[1], places=2)

            png_fig, png_axis = plt.subplots()
            png_axis.plot([0, 1], [1, 0])
            png_path = plotting.save_figure(png_fig, tmp_path / "figure.png")
            with Image.open(png_path) as image:
                self.assertEqual(
                    image.size,
                    tuple(round(length * 300) for length in plotting.BOOK_CANVAS_SIZE),
                )

    def test_shared_figure_size_tokens_resolve_to_book_canvas(self):
        from xai_book import plotting
        import matplotlib as mpl

        for token in plotting.FIGURE_SIZE_TOKENS:
            self.assertEqual(plotting.figure_size(token), plotting.BOOK_CANVAS_SIZE)
        self.assertEqual(
            plotting.figure_size("grid", n_rows=3, n_cols=2, extra_height=1.0),
            plotting.BOOK_CANVAS_SIZE,
        )
        plotting.apply_plot_style(figsize=(12.0, 8.0), use_seaborn=False)
        self.assertEqual(tuple(mpl.rcParams["figure.figsize"]), plotting.BOOK_CANVAS_SIZE)

        fig, _ = plotting.book_subplots(figsize=(12.0, 8.0))
        self.assertEqual(tuple(fig.get_size_inches()), plotting.BOOK_CANVAS_SIZE)
        plotting.plt.close(fig)

    def test_plotly_style_uses_shared_book_palette_and_canvas(self):
        from xai_book import plotting

        class DummyFigure:
            def __init__(self):
                self.layout = {}

            def update_layout(self, **kwargs):
                self.layout.update(kwargs)

        fig = DummyFigure()
        plotting.apply_plotly_style(fig)

        self.assertEqual(fig.layout["colorway"], plotting.extended_book_palette())
        self.assertEqual(fig.layout["width"], round(plotting.BOOK_CANVAS_SIZE[0] * 100))
        self.assertEqual(fig.layout["height"], round(plotting.BOOK_CANVAS_SIZE[1] * 100))

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
