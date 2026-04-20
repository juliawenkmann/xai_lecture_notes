from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CHAPTER_DIR = ROOT_DIR / "notebooks/02_interpretable_models"


def notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


class InterpretableModelPlotExportTests(unittest.TestCase):
    def test_linreg_notebook_displays_and_saves_figures(self):
        code = notebook_code(CHAPTER_DIR / "linreg_ridge_lasso.ipynb")
        self.assertIn('NOTEBOOK_SLUG = "linreg_ridge_lasso"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn("linreg_ridge_lasso_interactive_latest.pdf", code)
        self.assertIn("linreg_ridge_closed_form.pdf", code)

    def test_logistic_notebook_displays_and_saves_figures(self):
        code = notebook_code(CHAPTER_DIR / "logistic_regression.ipynb")
        self.assertIn('NOTEBOOK_SLUG = "logistic_regression"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn("logistic_regression_interactive_latest.pdf", code)
        self.assertIn("logistic_regression_from_scratch.pdf", code)

    def test_impurity_widget_is_saved_to_out(self):
        code = notebook_code(CHAPTER_DIR / "interactive_impurity_split.ipynb")
        self.assertIn('NOTEBOOK_SLUG = "interactive_impurity_split"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("apply_plotly_style", code)
        self.assertIn("plotly_figure_size", code)
        self.assertIn("save_plotly_html", code)
        self.assertIn("interactive_impurity_split_latest.html", code)
        self.assertIn("display(fig, slider, out)", code)

    def test_mdi_mda_notebook_displays_saved_figures(self):
        code = notebook_code(CHAPTER_DIR / "mdi_and_mda.ipynb")
        self.assertIn('NOTEBOOK_SLUG = "mdi_and_mda"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("build_mdi_and_mda_figures", code)
        self.assertIn("display_and_save_figure", code)

    def test_var_notebook_uses_shared_display_and_save_helper(self):
        code = notebook_code(CHAPTER_DIR / "var_autoregression_finance.ipynb")
        self.assertIn('NOTEBOOK_SLUG = "var_autoregression_finance"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn("show_and_save(", code)
        self.assertIn("compare_var_candidate_models", code)
        self.assertIn("plot_forecast_summary", code)
        self.assertIn("var_candidate_comparison.pdf", code)
        self.assertIn("var_forecast_summary.pdf", code)

    def test_gam_notebook_uses_local_data_and_out(self):
        code = notebook_code(CHAPTER_DIR / "generalized_additive_models.ipynb")
        self.assertIn('DATA_DIR = Path("data")', code)
        self.assertIn('NOTEBOOK_SLUG = "generalized_additive_models"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("GAM_FIGURE_FILENAMES", code)
        self.assertIn("GAM_TABLE_FILENAMES", code)
        self.assertIn("run_gam_regression_analysis", code)
        self.assertIn("ensure_gam_demo_dataset", code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn('show_and_save("data_overview")', code)
        self.assertIn('show_and_save("model_comparison")', code)
        self.assertIn('show_and_save("feature_effects")', code)
        self.assertIn('show_and_save("prediction_breakdown")', code)

    def test_finance_gam_notebook_uses_var_data_and_out(self):
        code = notebook_code(CHAPTER_DIR / "gam_finance_forecasting.ipynb")
        self.assertIn('DATA_DIR = Path("data")', code)
        self.assertIn('NOTEBOOK_SLUG = "gam_finance_forecasting"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("run_finance_gam_analysis", code)
        self.assertIn("FINANCE_GAM_FIGURE_FILENAMES", code)
        self.assertIn("FINANCE_GAM_TABLE_FILENAMES", code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn("finance_var_sample_stocks.csv", code)
        self.assertIn('show_and_save("model_comparison")', code)
        self.assertIn('show_and_save("forecast")', code)
        self.assertIn('show_and_save("forecast_zoom")', code)
        self.assertIn('show_and_save("feature_effects")', code)
        self.assertIn('show_and_save("prediction_breakdown")', code)
        self.assertIn('show_and_save("var_prediction_breakdown")', code)

    def test_concept_bottleneck_notebook_uses_local_data_and_out(self):
        code = notebook_code(CHAPTER_DIR / "concept_bottleneck.ipynb")
        self.assertIn('DATA_DIR = Path("data")', code)
        self.assertIn('NOTEBOOK_SLUG = "concept_bottleneck"', code)
        self.assertIn('OUT_DIR = Path("out") / NOTEBOOK_SLUG', code)
        self.assertIn("build_cub_concept_bottleneck_demo", code)
        self.assertIn('CONFIG["raw_dir"]', code)
        self.assertIn('CONFIG["variants"]', code)
        self.assertIn('CONFIG["primary_variant"]', code)
        self.assertIn('CONFIG["joint_epochs"]', code)
        self.assertIn("SUPPORTED_CBM_VARIANTS", code)
        self.assertIn("CBM_VARIANT_LABELS", code)
        self.assertIn("cub_data_paths", code)
        self.assertIn("CBM_FIGURE_FILENAMES", code)
        self.assertIn("display_and_save_figure", code)
        self.assertIn('results["figures"]', code)

    def test_shared_plotting_module_centralizes_latex_and_size_tokens(self):
        code = (ROOT_DIR / "xai_book" / "plotting.py").read_text(encoding="utf-8")
        self.assertIn("LATEX_PREAMBLE", code)
        self.assertIn("text.usetex", code)
        self.assertIn("FIGURE_SIZE_TOKENS", code)
        self.assertIn("book_subplots", code)
        self.assertIn("include_mathjax", code)

    def test_shared_plotting_modules_do_not_hardcode_numeric_figsizes(self):
        for path in (ROOT_DIR / "xai_book").glob("*.py"):
            if path.name == "plotting.py":
                continue
            code = path.read_text(encoding="utf-8")
            self.assertNotRegex(code, r"figsize=\(\s*[0-9]")


if __name__ == "__main__":
    unittest.main()
