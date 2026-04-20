from __future__ import annotations

import numpy as np

from ..gradient_based import (
    compute_cam_and_gradcam,
    load_demo_image,
    load_resnet18_with_preprocessing,
    plot_cam_gradcam_comparison,
    plot_top_channel_overlays,
    preprocess_image,
)
from ..paths import chapter_figure_path
from ..plotting import save_figure


def run_gradcam_figures(
    *,
    chapter: str = "04_gradient_based",
    image_path=None,
    seed: int = 42,
    top_k: int = 5,
    output_dir=None,
):
    model, preprocess, device = load_resnet18_with_preprocessing()
    image = load_demo_image(image_path)
    input_tensor = preprocess_image(image, preprocess, device)

    predicted = compute_cam_and_gradcam(model, input_tensor)
    comparison_path = save_figure(
        plot_cam_gradcam_comparison(image, predicted["cam"], predicted["gradcam"]),
        chapter_figure_path(chapter, "gradcam_cam_comparison", suffix=".pdf", output_dir=output_dir),
    )

    top_class_figure, top_channels = plot_top_channel_overlays(
        image,
        predicted["activations"],
        predicted["gradients"],
        title=f"Top channels for predicted class {predicted['class_index']}",
        top_k=top_k,
    )
    top_class_path = save_figure(
        top_class_figure,
        chapter_figure_path(chapter, "top_class_channels", suffix=".pdf", output_dir=output_dir),
    )

    rng = np.random.default_rng(seed)
    random_class = int(rng.integers(model.fc.out_features))
    if random_class == predicted["class_index"]:
        random_class = (random_class + 1) % model.fc.out_features

    random_result = compute_cam_and_gradcam(model, input_tensor, class_index=random_class)
    random_figure, random_channels = plot_top_channel_overlays(
        image,
        random_result["activations"],
        random_result["gradients"],
        title=f"Top channels for comparison class {random_class}",
        top_k=top_k,
    )
    random_class_path = save_figure(
        random_figure,
        chapter_figure_path(chapter, "random_class_channels", suffix=".pdf", output_dir=output_dir),
    )

    return {
        "gradcam_cam_comparison": comparison_path,
        "top_class_channels": top_class_path,
        "random_class_channels": random_class_path,
        "predicted_class_index": predicted["class_index"],
        "random_class_index": random_class,
        "top_channels": top_channels,
        "random_channels": random_channels,
    }
