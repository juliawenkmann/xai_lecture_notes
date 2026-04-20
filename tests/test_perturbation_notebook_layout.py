from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CHAPTER_DIR = ROOT_DIR / "notebooks/03_perturbation_based"


class PerturbationNotebookLayoutTests(unittest.TestCase):
    def test_old_root_runner_notebooks_are_gone(self):
        for filename in [
            "01_lime.ipynb",
            "02_shap.ipynb",
            "02a_lime.ipynb",
            "02b_shap.ipynb",
            "02c_shap_timeseries.ipynb",
        ]:
            self.assertFalse((CHAPTER_DIR / filename).exists(), filename)

    def test_topic_folder_runner_notebooks_exist(self):
        expected = [
            CHAPTER_DIR / "lime_for_tabular/lime_synthetic.ipynb",
            CHAPTER_DIR / "lime_for_tabular/lime_tabular.ipynb",
            CHAPTER_DIR / "shap/shap_synthetic.ipynb",
            CHAPTER_DIR / "shap/shap_tabular.ipynb",
            CHAPTER_DIR / "shap/shap_timeseries.ipynb",
        ]
        for path in expected:
            self.assertTrue(path.exists(), path)

    def test_root_out_directory_is_gone(self):
        self.assertFalse((CHAPTER_DIR / "out").exists())


if __name__ == "__main__":
    unittest.main()
