from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class VarNotebookLayoutTests(unittest.TestCase):
    def test_var_assets_live_at_chapter_root(self):
        self.assertFalse((ROOT_DIR / "notebooks/02_interpretable_models/VAR").exists())
        self.assertTrue(
            (ROOT_DIR / "notebooks/02_interpretable_models/data/finance_var_sample_stocks.csv").exists()
        )
        self.assertTrue((ROOT_DIR / "notebooks/02_interpretable_models/out").exists())

    def test_var_notebook_exists_at_chapter_root(self):
        self.assertTrue((ROOT_DIR / "notebooks/02_interpretable_models/var_autoregression_finance.ipynb").exists())

    def test_redundant_executed_var_notebook_is_gone(self):
        self.assertFalse(
            (ROOT_DIR / "notebooks/02_interpretable_models/var_autoregression_finance_executed.ipynb").exists()
        )

    def test_var_notebook_uses_chapter_data_and_out_directories(self):
        notebook = json.loads(
            (ROOT_DIR / "notebooks/02_interpretable_models/var_autoregression_finance.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertLessEqual(len(notebook.get("cells", [])), 10)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        self.assertIn('DATA_DIR = Path("data")', code)
        self.assertIn('OUT_DIR = Path("out")', code)
        self.assertIn("run_var_finance_analysis", code)
        self.assertNotIn('Path("VAR")', code)
        self.assertNotIn("run_var_finance_demo", code)

    def test_var_timeseries_indices_keep_frequency(self):
        from xai_book.var_models import compute_log_returns, load_finance_var_prices, split_train_test

        prices = load_finance_var_prices(
            ROOT_DIR / "notebooks/02_interpretable_models/data/finance_var_sample_stocks.csv"
        )
        returns = compute_log_returns(prices)
        train, test = split_train_test(returns, n_test=12)

        self.assertIsNotNone(prices.index.freqstr)
        self.assertEqual(prices.index.freqstr, returns.index.freqstr)
        self.assertEqual(returns.index.freqstr, train.index.freqstr)
        self.assertEqual(train.index.freqstr, test.index.freqstr)

    def test_var_candidate_comparison_returns_summary_and_analyses(self):
        from xai_book.var_models import (
            DEFAULT_LECTURE_VAR_LABEL,
            LECTURE_VAR_CANDIDATES,
            compare_var_candidate_models,
        )

        frame, analyses = compare_var_candidate_models(
            data_path=ROOT_DIR / "notebooks/02_interpretable_models/data/finance_var_sample_stocks.csv",
            candidates=LECTURE_VAR_CANDIDATES[:2],
            n_test=12,
            validation_size=8,
            maxlags=3,
        )

        self.assertIn(DEFAULT_LECTURE_VAR_LABEL, analyses)
        self.assertIn("test_var_mape_pct", frame.columns)
        self.assertIn("validation_mape_pct", frame.columns)
        self.assertTrue((frame["selected_lag"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
