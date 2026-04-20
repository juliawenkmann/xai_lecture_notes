from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xai_book.chapters.interpretable_models import run_mdi_and_mda_figures


class InterpretableModelsSmokeTests(unittest.TestCase):
    def test_mdi_and_mda_runner_saves_expected_figures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("xai_book.paths.FIGURES_DIR", Path(tmp_dir)):
                outputs = run_mdi_and_mda_figures(chapter="smoke")

                expected_keys = {
                    "decision_tree",
                    "mdi_from_scratch",
                    "mda_bar",
                    "mda_visualization",
                    "mdi_vs_mda",
                }
                self.assertEqual(set(outputs), expected_keys)
                for path in outputs.values():
                    self.assertTrue(path.exists())
                    self.assertEqual(path.suffix, ".pdf")


if __name__ == "__main__":
    unittest.main()
