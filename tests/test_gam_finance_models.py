from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from xai_book.gam_finance_models import (
    DEFAULT_FINANCE_GAM_PEER,
    DEFAULT_FINANCE_GAM_TARGET,
    FINANCE_GAM_FIGURE_FILENAMES,
    FINANCE_GAM_MODEL_LABELS,
    FINANCE_GAM_TABLE_FILENAMES,
    build_finance_gam_dataset,
    run_finance_gam_analysis,
)


class FinanceGamModelTests(unittest.TestCase):
    def test_finance_gam_dataset_has_expected_columns(self):
        data_path = Path("notebooks/02_interpretable_models/data/finance_var_sample_stocks.csv")
        prepared = build_finance_gam_dataset(data_path=data_path)

        self.assertEqual(prepared["target"], DEFAULT_FINANCE_GAM_TARGET)
        self.assertEqual(prepared["peer"], DEFAULT_FINANCE_GAM_PEER)
        self.assertIn("target_return", prepared["dataset"].columns)
        self.assertIn("target_price", prepared["dataset"].columns)
        self.assertEqual(len(prepared["feature_columns"]), 4)

    def test_finance_gam_analysis_produces_figures_and_strong_baselines(self):
        data_path = Path("notebooks/02_interpretable_models/data/finance_var_sample_stocks.csv")
        analysis = run_finance_gam_analysis(data_path=data_path)

        metrics = analysis["metrics"].set_index("model")
        self.assertLess(
            metrics.loc[FINANCE_GAM_MODEL_LABELS["gam"], "price_MAPE_pct"],
            metrics.loc[FINANCE_GAM_MODEL_LABELS["flat"], "price_MAPE_pct"],
        )
        self.assertLess(
            metrics.loc[FINANCE_GAM_MODEL_LABELS["gam"], "return_RMSE_bp"],
            metrics.loc[FINANCE_GAM_MODEL_LABELS["var"], "return_RMSE_bp"],
        )
        self.assertEqual(set(analysis["figures"]), set(FINANCE_GAM_FIGURE_FILENAMES))
        self.assertEqual(set(FINANCE_GAM_TABLE_FILENAMES), {"metrics", "prediction_breakdown", "validation_search"})
        self.assertIn("validation_price_MAPE_pct", analysis["search"].columns)

        predicted = float(analysis["breakdown_frame"]["predicted_return"].iloc[0])
        contribution_sum = float(analysis["breakdown"].loc["Prediction"])
        self.assertAlmostEqual(predicted, contribution_sum, places=6)

        for fig in analysis["figures"].values():
            self.assertGreaterEqual(len(fig.axes), 1)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
