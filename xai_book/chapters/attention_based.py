from __future__ import annotations

from ..attention_based import plot_self_attention_connections, save_attention_heatmaps_pdf
from ..paths import chapter_figure_path
from ..plotting import save_figure


def run_attention_figures(
    *,
    chapter: str = "06_attention_based",
    output_dir=None,
):
    heatmap_path = chapter_figure_path(chapter, "bert_attention_maps", suffix=".pdf", output_dir=output_dir)
    save_attention_heatmaps_pdf("I love mathematics and AI", heatmap_path)

    figure = plot_self_attention_connections("I love mathematics and AI. Math is beautiful")
    connector_path = chapter_figure_path(
        chapter,
        "self_attention_bookstyle_colored",
        suffix=".pdf",
        output_dir=output_dir,
    )
    save_figure(figure, connector_path, tight_layout=True)

    return {
        "bert_attention_maps": heatmap_path,
        "self_attention_bookstyle_colored": connector_path,
    }
