from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class NotebookDataLayoutTests(unittest.TestCase):
    def test_helper_data_has_been_moved_into_chapter_data_directories(self):
        self.assertFalse((ROOT_DIR / "notebooks/01_introduction/dataset").exists())
        self.assertFalse((ROOT_DIR / "notebooks/01_introduction/model_checkpoints").exists())
        self.assertTrue((ROOT_DIR / "notebooks/01_introduction/data/train").exists())
        self.assertTrue((ROOT_DIR / "notebooks/01_introduction/data/test").exists())
        self.assertTrue((ROOT_DIR / "notebooks/01_introduction/data/model_checkpoints").exists())
        self.assertTrue((ROOT_DIR / "notebooks/01_introduction/data/mnist.npz").exists())
        self.assertTrue((ROOT_DIR / "notebooks/01_introduction/mnist.ipynb").exists())
        self.assertTrue((ROOT_DIR / "notebooks/02_interpretable_models/concept_bottleneck.ipynb").exists())
        self.assertTrue((ROOT_DIR / "notebooks/02_interpretable_models/data/cub/README.md").exists())

        self.assertFalse(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/ECG200_TRAIN.txt").exists()
        )
        self.assertFalse(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/ECG200_TEST.txt").exists()
        )
        self.assertFalse(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/inception_time_ecg200.pth").exists()
        )
        self.assertTrue(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/data/ECG200_TRAIN.txt").exists()
        )
        self.assertTrue(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/data/ECG200_TEST.txt").exists()
        )
        self.assertTrue(
            (ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/data/inception_time_ecg200.pth").exists()
        )

        self.assertFalse((ROOT_DIR / "notebooks/03_perturbation_based/shap/kjb.png").exists())
        self.assertTrue((ROOT_DIR / "notebooks/03_perturbation_based/shap/data/kjb.png").exists())

        self.assertFalse((ROOT_DIR / "notebooks/05_concept_based/broden1_224").exists())
        self.assertTrue((ROOT_DIR / "notebooks/05_concept_based/data/broden1_224").exists())

    def test_notebooks_and_helpers_use_local_data_directories(self):
        offenders: list[str] = []
        checks = {
            ROOT_DIR / "notebooks/01_introduction/ribeiro2016_husky_wolf_example.ipynb": [
                "drive.mount('/content/drive')",
                "folder_path =",
                "ImageFolder(folder_path + '/dataset1'",
                "Image.open(folder_path + '/test/",
                "torch.save(model.state_dict(), folder_path + '/biasedModel1.pt')",
            ],
            ROOT_DIR / "notebooks/01_introduction/mnist.ipynb": [
                ".keras/datasets/mnist.npz",
                "keras.datasets.mnist",
                "fetch_openml(",
            ],
            ROOT_DIR
            / "notebooks/03_perturbation_based/lime_for_time_series/interpretable_features_for_time_series.ipynb": [
                "ECG200Dataset('ECG200_TRAIN.txt')",
                "ECG200Dataset('ECG200_TEST.txt')",
                "torch.save(model.state_dict(), 'inception_time_ecg200.pth')",
            ],
            ROOT_DIR / "notebooks/03_perturbation_based/lime_for_time_series/lime_for_time_series.ipynb": [
                "ECG200Dataset('ECG200_TRAIN.txt')",
                "ECG200Dataset('ECG200_TEST.txt')",
                "torch.load('inception_time_ecg200.pth'",
            ],
            ROOT_DIR / "xai_book/concept_based.py": [
                'concept_notebook_dir() / "broden1_224"',
            ],
        }

        for path, patterns in checks.items():
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                haystack = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                )
            else:
                haystack = path.read_text(encoding="utf-8")

            for pattern in patterns:
                if pattern in haystack:
                    offenders.append(f"{path.relative_to(ROOT_DIR)} -> {pattern}")

        self.assertEqual(offenders, [])

    def test_concept_bottleneck_notebook_uses_local_data_and_out(self):
        notebook = json.loads(
            (ROOT_DIR / "notebooks/02_interpretable_models/concept_bottleneck.ipynb").read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

        self.assertIn('DATA_DIR = Path("data")', code)
        self.assertIn('OUT_DIR = Path("out")', code)
        self.assertIn('data_dir=DATA_DIR', code)


if __name__ == "__main__":
    unittest.main()
