from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xai_book.chapters.concept_based import run_broden_dataset_figure
from xai_book.concept_based import broden_texture_dir


class ConceptBasedSmokeTests(unittest.TestCase):
    def test_broden_texture_dir_exists(self):
        self.assertTrue(broden_texture_dir().exists())

    def test_broden_runner_saves_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("xai_book.paths.FIGURES_DIR", Path(tmp_dir)):
                output_path = run_broden_dataset_figure(chapter="smoke", output_name="broden_random", seed=42)
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.suffix, ".pdf")


if __name__ == "__main__":
    unittest.main()
