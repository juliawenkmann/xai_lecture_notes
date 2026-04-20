from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from xai_book.concept_bottleneck import (
    CBMVariantResult,
    CBM_VARIANT_LABELS,
    SUPPORTED_CBM_VARIANTS,
    _variant_label,
    compute_concept_metrics,
    compute_model_metrics,
    cub_attribute_labels_path,
    cub_attribute_names_path,
    cub_data_paths,
    cub_dataset_available,
    cub_missing_paths,
    cub_raw_dir_candidates,
    prepare_cub_dataset,
    ensure_cub_dataset_available,
    plot_cbm_concept_quality,
    plot_cbm_model_comparison,
    plot_cbm_prediction_explanation,
)


class ConceptBottleneckHelperTests(unittest.TestCase):
    def test_cub_paths_follow_chapter_local_data_layout(self):
        base_dir = Path("notebooks/02_interpretable_models/data")
        paths = cub_data_paths(base_dir)

        self.assertEqual(paths.raw_dir, base_dir / "cub" / "raw" / "CUB_200_2011")
        self.assertEqual(paths.cache_dir, base_dir / "cub" / "cache")

    def test_cub_paths_accept_explicit_raw_dir(self):
        base_dir = Path("notebooks/02_interpretable_models/data")
        explicit = Path("/tmp/example/CUB_200_2011")

        paths = cub_data_paths(base_dir, raw_dir=explicit)

        self.assertEqual(paths.raw_dir, explicit)

    def test_candidate_paths_include_short_and_nested_layouts(self):
        base_dir = Path("notebooks/02_interpretable_models/data")
        candidates = cub_raw_dir_candidates(base_dir)

        self.assertIn(base_dir / "cub" / "raw" / "CUB_200_2011", candidates)
        self.assertIn(base_dir / "cub" / "CUB_200_2011", candidates)
        self.assertIn(base_dir / "CUB_200_2011", candidates)

    def test_missing_dataset_error_mentions_expected_location(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = cub_data_paths(Path(tmp_dir))
            with self.assertRaisesRegex(FileNotFoundError, "raw_dir"):
                ensure_cub_dataset_available(paths)

    def test_dataset_available_detects_images_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            raw_dir = base_dir / "cub" / "raw" / "CUB_200_2011"
            raw_dir.mkdir(parents=True)
            (raw_dir / "images.txt").write_text("1 001.jpg\n", encoding="utf-8")

            self.assertFalse(cub_dataset_available(cub_data_paths(base_dir)))
            self.assertTrue(cub_missing_paths(raw_dir))

    def test_dataset_available_requires_attribute_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            raw_dir = base_dir / "cub" / "raw" / "CUB_200_2011"
            (raw_dir / "attributes").mkdir(parents=True)
            (raw_dir / "images").mkdir()
            (raw_dir / "images.txt").write_text("1 001.jpg\n", encoding="utf-8")
            (raw_dir / "image_class_labels.txt").write_text("1 1\n", encoding="utf-8")
            (raw_dir / "train_test_split.txt").write_text("1 1\n", encoding="utf-8")
            (raw_dir / "classes.txt").write_text("1 class\n", encoding="utf-8")
            (raw_dir / "attributes" / "attributes.txt").write_text("1 has_bill_shape::curved_(up_or_down)\n", encoding="utf-8")
            (raw_dir / "attributes" / "image_attribute_labels.txt").write_text("1 1 1 4 0.0\n", encoding="utf-8")

            self.assertTrue(cub_dataset_available(cub_data_paths(base_dir)))

    def test_attribute_names_can_live_next_to_cub_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            raw_dir = base_dir / "cub" / "raw" / "CUB_200_2011"
            (raw_dir / "attributes").mkdir(parents=True)
            (raw_dir.parent / "attributes.txt").write_text("1 has_wing_color::blue\n", encoding="utf-8")
            (raw_dir / "attributes" / "image_attribute_labels.txt").write_text("1 1 1 4 0.0\n", encoding="utf-8")

            self.assertEqual(cub_attribute_names_path(raw_dir), raw_dir.parent / "attributes.txt")
            self.assertEqual(
                cub_attribute_labels_path(raw_dir),
                raw_dir / "attributes" / "image_attribute_labels.txt",
            )

    def test_prepare_cub_dataset_uses_downloader_when_enabled(self):
        base_dir = Path("notebooks/02_interpretable_models/data")
        expected_paths = cub_data_paths(base_dir)

        with mock.patch("xai_book.concept_bottleneck.cub_dataset_available", return_value=False):
            with mock.patch(
                "xai_book.concept_bottleneck.download_cub_dataset",
                return_value=expected_paths,
            ) as download_mock:
                resolved = prepare_cub_dataset(base_dir, download_if_missing=True)

        self.assertEqual(resolved, expected_paths)
        download_mock.assert_called_once_with(base_dir, raw_dir=None, download_url=None)

    def test_metric_helpers_support_multiple_cbm_variants(self):
        y_test = np.array([0, 1, 2], dtype=np.int64)
        concept_truth = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        variant_results = {
            "oracle_concepts": CBMVariantResult(
                variant="oracle_concepts",
                label=CBM_VARIANT_LABELS["oracle_concepts"],
                class_probabilities=np.array(
                    [[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.02, 0.08, 0.90]],
                    dtype=np.float32,
                ),
                class_predictions=np.array([0, 1, 2], dtype=np.int64),
                class_labels=np.array([0, 1, 2], dtype=np.int64),
                concept_probabilities=concept_truth,
                concept_truth=concept_truth,
            ),
            "sequential_cbm": CBMVariantResult(
                variant="sequential_cbm",
                label=CBM_VARIANT_LABELS["sequential_cbm"],
                class_probabilities=np.array(
                    [[0.90, 0.08, 0.02], [0.10, 0.82, 0.08], [0.04, 0.11, 0.85]],
                    dtype=np.float32,
                ),
                class_predictions=np.array([0, 1, 2], dtype=np.int64),
                class_labels=np.array([0, 1, 2], dtype=np.int64),
                concept_probabilities=np.array(
                    [[0.88, 0.12], [0.16, 0.82], [0.77, 0.81]],
                    dtype=np.float32,
                ),
                concept_truth=concept_truth,
            ),
            "joint_cbm": CBMVariantResult(
                variant="joint_cbm",
                label=CBM_VARIANT_LABELS["joint_cbm"],
                class_probabilities=np.array(
                    [[0.84, 0.12, 0.04], [0.08, 0.88, 0.04], [0.05, 0.13, 0.82]],
                    dtype=np.float32,
                ),
                class_predictions=np.array([0, 1, 2], dtype=np.int64),
                class_labels=np.array([0, 1, 2], dtype=np.int64),
                concept_probabilities=np.array(
                    [[0.79, 0.21], [0.19, 0.81], [0.73, 0.84]],
                    dtype=np.float32,
                ),
                concept_truth=concept_truth,
            ),
            "feature_baseline": CBMVariantResult(
                variant="feature_baseline",
                label=CBM_VARIANT_LABELS["feature_baseline"],
                class_probabilities=np.array(
                    [[0.92, 0.05, 0.03], [0.07, 0.89, 0.04], [0.04, 0.14, 0.82]],
                    dtype=np.float32,
                ),
                class_predictions=np.array([0, 1, 2], dtype=np.int64),
                class_labels=np.array([0, 1, 2], dtype=np.int64),
            ),
        }

        model_metrics = compute_model_metrics(y_test=y_test, variant_results=variant_results)
        concept_metrics = compute_concept_metrics(
            concept_names=["wing color", "bill shape"],
            variant_results=variant_results,
        )

        self.assertTupleEqual(SUPPORTED_CBM_VARIANTS, tuple(CBM_VARIANT_LABELS))
        self.assertEqual(model_metrics["variant"].tolist(), list(variant_results))
        self.assertEqual(set(concept_metrics["variant"]), {"sequential_cbm", "joint_cbm"})
        self.assertEqual(len(concept_metrics), 4)
        self.assertEqual(sorted(concept_metrics["concept"].unique().tolist()), ["bill shape", "wing color"])

    def test_plot_helpers_return_figures(self):
        metrics_frame = pd.DataFrame(
            {
                "model": ["Concept bottleneck", "Feature baseline"],
                "accuracy": [0.61, 0.68],
                "top_3_accuracy": [0.83, 0.86],
            }
        )
        concept_metrics = pd.DataFrame(
            {
                "concept": ["wing color: black", "bill shape: hooked", "breast pattern: spotted"],
                "test_prevalence": [0.34, 0.41, 0.27],
                "balanced_accuracy": [0.71, 0.66, 0.63],
                "average_precision": [0.79, 0.74, 0.69],
            }
        )
        explanation = pd.DataFrame(
            {
                "concept": ["wing color: black", "bill shape: hooked", "breast pattern: spotted"],
                "contribution": [0.82, 0.47, -0.18],
                "predicted_probability": [0.92, 0.74, 0.21],
                "ground_truth": [1, 1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.jpg"
            image = np.full((48, 48, 3), fill_value=220, dtype=np.uint8)
            image[:, :, 2] = 255
            Image.fromarray(image).save(image_path)
            sample_row = pd.Series(
                {
                    "image_file": image_path,
                    "predicted_class_name": "Black footed Albatross",
                    "class_name": "Laysan Albatross",
                }
            )

            figures = [
                plot_cbm_model_comparison(metrics_frame),
                plot_cbm_concept_quality(concept_metrics, top_n=3),
                plot_cbm_prediction_explanation(sample_row, explanation, top_n=3),
            ]

            for fig in figures:
                self.assertGreaterEqual(len(fig.axes), 1)
                plt.close(fig)

    def test_direct_probe_alias_maps_to_feature_baseline_label(self):
        self.assertEqual(_variant_label("feature_baseline"), "Feature baseline")
        self.assertEqual(_variant_label("direct_probe"), "Feature baseline")


if __name__ == "__main__":
    unittest.main()
