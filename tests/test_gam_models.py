from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from xai_book.gam_models import (
    GAM_FIGURE_FILENAMES,
    GAM_TABLE_FILENAMES,
    ensure_gam_demo_dataset,
    run_gam_regression_analysis,
)


class GamModelTests(unittest.TestCase):
    def test_demo_dataset_is_written_to_data_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "gam_synthetic_regression.csv"
            frame = ensure_gam_demo_dataset(data_path, n_samples=32, random_state=11)

            self.assertTrue(data_path.exists())
            self.assertEqual(list(frame.columns), ["temperature_c", "marketing_spend_k", "discount_pct", "demand_index"])
            self.assertEqual(len(frame), 32)

    def test_gam_analysis_improves_on_linear_baseline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "gam_synthetic_regression.csv"
            analysis = run_gam_regression_analysis(data_path=data_path, test_size=0.25, random_state=7)

            metrics = analysis["metrics"].set_index("model")
            self.assertLess(
                metrics.loc["Additive spline model", "RMSE"],
                metrics.loc["Linear baseline", "RMSE"],
            )
            self.assertEqual(set(analysis["figures"]), set(GAM_FIGURE_FILENAMES))
            self.assertIn("data_overview", analysis["figures"])
            self.assertEqual(set(GAM_TABLE_FILENAMES), {"metrics", "prediction_breakdown"})

            predicted = float(analysis["breakdown_frame"]["predicted"].iloc[0])
            contribution_sum = float(analysis["breakdown"].loc["Prediction"])
            self.assertAlmostEqual(predicted, contribution_sum, places=6)

            for fig in analysis["figures"].values():
                self.assertGreaterEqual(len(fig.axes), 1)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
