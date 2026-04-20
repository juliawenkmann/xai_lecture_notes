# XAI Lecture Notes

The repo uses one shared Python package, `xai_book`. The cleaned notebooks under `notebooks/` stay as thin runners that import shared helpers instead of duplicating library code.

## Layout
```text
xai_book/
  attention_based.py               # BERT attention / BertViz helpers
  chapters/                         # chapter-level figure runners
  concept_based.py                  # Broden + sensitivity-score helpers
  concept_bottleneck.py             # compact CUB concept-bottleneck helpers
  datasets.py                       # small reusable dataset loaders
  gradient_based.py                 # CAM / Grad-CAM helpers
  interpretable_models.py           # MDI / MDA helpers
  models.py                         # tiny model helpers
  perturbation.py                   # LIME/SHAP plotting functions
  plotting.py                       # style + save helpers
  paths.py                          # project and output paths
notebooks/
  01_introduction/                 # standalone intro notebooks plus local data/
  02_interpretable_models/          # chapter notebooks plus local data/ and out/
  03_perturbation_based/            # topic folders with local notebooks, data/, and out/
  04_gradient_based/                # thin notebooks that call xai_book
  05_concept_based/                 # thin notebooks that call xai_book
  06_attention_based/               # thin notebooks that call xai_book
out/
  ...                               # package-default fallback output location
```

## Quickstart
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Optional but recommended:
   ```bash
   pip install -e .
   ```
3. Open a notebook in `notebooks/02_interpretable_models/` through `notebooks/06_attention_based/` and run it top to bottom.

## Notes
- `xai_book` is the source of truth for shared plotting, dataset, model, perturbation, interpretable-model, gradient-based, concept-based, and attention-based logic.
- The cleaned notebooks from `02_interpretable_models` through `06_attention_based` use a shared bootstrap pattern, import from `xai_book`, and save into local `out/` folders.
- Notebook-local helper datasets, checkpoints, and other static assets live under each chapter's `data/` folder.
- The interpretable-models chapter now includes a compact CUB concept-bottleneck notebook; place the raw `CUB_200_2011` release under `notebooks/02_interpretable_models/data/cub/raw/`.
- The perturbation chapter now uses topic folders only, so LIME and SHAP runners live next to their related supplementary notebooks instead of being duplicated at chapter root.
- Supplementary notebooks and bonus material remain under `notebooks/` as standalone examples where appropriate.

## Remaining Work
- Continue the same extraction pattern for the introduction, evaluation, deep-dream, and bonus material if you want the whole repo normalized.
- Keep notebooks limited to parameter setup and function calls.
