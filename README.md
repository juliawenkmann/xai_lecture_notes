# Clean XAI Codebase (LIME & SHAP)

This is a minimal, clean structure with short notebooks that *only* call functions.
Plots are written to `figures/<chapter>/` and notebooks live in `notebooks/<chapter>/`.

## Layout
```text
figures/
  02_perturbation_based/            # saved images from notebooks
notebooks/
  02_perturbation_based/
    02a_lime.ipynb
    02b_shap.ipynb
code/
  data/                             # datasets
  explainers/                       # LIME & SHAP wrappers
  models/                           # tiny model zoo
  plots/                            # orchestration to generate figures
  utils/                            # io, paths, seeding
```

## Quickstart
1. Create an environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebooks in `notebooks/02_perturbation_based/` and run all cells.
   - `02a_lime.ipynb` -> saves a LIME feature-importance plot.
   - `02b_shap.ipynb` -> saves a KernelSHAP summary plot.

## Notes
- Dataset: `sklearn.datasets.load_breast_cancer` (binary classification).
- The SHAP notebook uses **KernelSHAP** (perturbation-based) for broad model support.
- Folders are created on demand; outputs go to `figures/02_perturbation_based/`.
