from __future__ import annotations

from ..concept_based import (
    compute_sensitivity_scores,
    plot_broden_mosaic,
    plot_sensitivity_scores,
    sample_broden_images,
)
from ..paths import chapter_figure_path
from ..plotting import save_figure


def run_broden_dataset_figure(
    *,
    chapter: str = "05_concept_based",
    output_name: str = "broden_random",
    seed: int = 42,
    output_dir=None,
):
    samples = sample_broden_images(seed=seed)
    figure = plot_broden_mosaic(samples)
    return save_figure(
        figure,
        chapter_figure_path(chapter, output_name, suffix=".pdf", output_dir=output_dir),
    )


def run_sensitivity_scores_figure(
    *,
    chapter: str = "05_concept_based",
    output_name: str = "sensitivity_scores",
    output_dir=None,
):
    scores = compute_sensitivity_scores()
    figure = plot_sensitivity_scores(scores)
    output_path = save_figure(
        figure,
        chapter_figure_path(chapter, output_name, suffix=".pdf", output_dir=output_dir),
    )
    return output_path, scores
