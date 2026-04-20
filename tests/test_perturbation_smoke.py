from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xai_book.chapters import perturbation


class PerturbationSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("lime"), "lime is not installed")
    def test_synthetic_lime_runner_saves_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("xai_book.paths.FIGURES_DIR", Path(tmp_dir)):
                output_path = perturbation.run_synthetic_lime_plot(
                    chapter="smoke",
                    output_name="lime",
                    sample_index=0,
                )
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.suffix, ".png")

    @unittest.skipUnless(importlib.util.find_spec("shap"), "shap is not installed")
    def test_synthetic_shap_runner_saves_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch("xai_book.paths.FIGURES_DIR", Path(tmp_dir)):
                output_path = perturbation.run_synthetic_shap_plot(
                    chapter="smoke",
                    output_name="shap",
                    explain_samples=8,
                    max_background=20,
                    max_display=6,
                )
                self.assertTrue(output_path.exists())
                self.assertEqual(output_path.suffix, ".png")
