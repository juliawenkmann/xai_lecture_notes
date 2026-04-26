from __future__ import annotations

import random
from pathlib import Path

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from matplotlib.lines import Line2D

from .paths import ROOT_DIR
from .plotting import BLUE, CHAPTER_GRAY, ORANGE, book_subplots


def concept_notebook_dir() -> Path:
    return ROOT_DIR / "notebooks" / "05_concept_based"


def broden_texture_dir() -> Path:
    return concept_data_dir() / "broden1_224" / "images" / "dtd"


def concept_data_dir() -> Path:
    return concept_notebook_dir() / "data"


def _resnet50_modules():
    import torch
    import torchvision.transforms as transforms
    from torchvision import models

    try:
        from torchvision.models import ResNet50_Weights
    except ImportError:
        ResNet50_Weights = None

    return torch, transforms, models, ResNet50_Weights


def load_resnet50_with_preprocessing(device: str | None = None):
    torch, transforms, models, ResNet50_Weights = _resnet50_modules()
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if ResNet50_Weights is not None:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights).to(device_obj)
        preprocess = weights.transforms()
    else:
        model = models.resnet50(pretrained=True).to(device_obj)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    model.eval()
    return model, preprocess, device_obj


def sample_broden_images(
    *,
    images_dir: Path | None = None,
    concepts_to_show: int = 3,
    samples_per_concept: int = 5,
    seed: int = 42,
):
    images_dir = images_dir or broden_texture_dir()
    rng = random.Random(seed)

    files = [
        path.name
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and "_color" not in path.stem
    ]
    concepts = sorted({filename.split("_")[0] for filename in files})
    selected_concepts = rng.sample(concepts, min(concepts_to_show, len(concepts)))

    samples = {}
    for concept in selected_concepts:
        concept_files = [filename for filename in files if filename.startswith(f"{concept}_")]
        samples[concept] = rng.sample(concept_files, min(samples_per_concept, len(concept_files)))
    return samples


def plot_broden_mosaic(
    samples: dict[str, list[str]],
    *,
    images_dir: Path | None = None,
):
    images_dir = images_dir or broden_texture_dir()
    n_rows = len(samples)
    n_cols = max(len(paths) for paths in samples.values())

    fig, axes = book_subplots(n_rows, n_cols, size="grid", extra_height=-0.6)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_index, (concept, image_names) in enumerate(samples.items()):
        for col_index in range(n_cols):
            axis = axes[row_index, col_index]
            axis.axis("off")
            if col_index >= len(image_names):
                continue
            image = Image.open(images_dir / image_names[col_index]).convert("RGB")
            axis.imshow(image)
            if col_index == 0:
                axis.set_ylabel(concept.capitalize(), fontsize=10)

    for separator_index in range(1, n_rows):
        y = 1 - separator_index / n_rows
        fig.add_artist(Line2D([0, 1], [y, y], transform=fig.transFigure, color=CHAPTER_GRAY, linewidth=1))
    return fig


def _load_image_batch(image_dir: Path, preprocess, device, limit: int = 20):
    _, _, _, _ = _resnet50_modules()
    images = []
    for image_path in sorted(image_dir.glob("*.jpg"))[:limit]:
        image = Image.open(image_path).convert("RGB")
        images.append(preprocess(image))
    if not images:
        raise FileNotFoundError(f"No JPG images found in {image_dir}")
    torch, _, _, _ = _resnet50_modules()
    return torch.stack(images).to(device)


def compute_concept_normal_vector(
    model,
    concept_batch,
    reference_batch,
):
    torch, _, _, _ = _resnet50_modules()
    activations = {}

    def hook(_module, _inputs, outputs):
        activations["value"] = outputs.detach()

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(torch.cat((concept_batch, reference_batch), dim=0))
    handle.remove()

    n_concept = concept_batch.shape[0]
    features = activations["value"].reshape(n_concept * 2, -1).cpu().numpy()
    labels = np.concatenate((np.ones(n_concept), np.zeros(reference_batch.shape[0])))

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(features, labels)

    normal_vector = classifier.coef_[0]
    normal_vector = normal_vector - normal_vector.mean()
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector


def compute_image_sensitivity_score(
    model,
    preprocess,
    image_path: Path,
    normal_vector: np.ndarray,
    device,
):
    torch, _, _, _ = _resnet50_modules()
    captured = {}

    def hook(_module, _inputs, outputs):
        captured["avgpool"] = outputs
        outputs.retain_grad()

    handle = model.avgpool.register_forward_hook(hook)
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    model.zero_grad()
    logits = model(input_tensor)
    class_index = int(logits.argmax(dim=1).item())
    logits[0, class_index].backward()
    handle.remove()

    gradient_vector = captured["avgpool"].grad.reshape(-1).detach().cpu().numpy()
    gradient_vector = gradient_vector / np.linalg.norm(gradient_vector)
    return float(np.dot(normal_vector, gradient_vector))


def compute_sensitivity_scores(
    *,
    concept_dir: Path | None = None,
    reference_dir: Path | None = None,
    target_images: dict[str, Path] | None = None,
):
    concept_dir = concept_dir or (concept_data_dir() / "striped")
    reference_dir = reference_dir or (concept_data_dir() / "imagenet_random_images")
    target_images = target_images or {
        "Zebra": concept_data_dir() / "zebra.jpg",
        "Tiger": concept_data_dir() / "tiger.jpg",
        "Dog": concept_data_dir() / "dog.jpg",
    }

    model, preprocess, device = load_resnet50_with_preprocessing()
    concept_batch = _load_image_batch(concept_dir, preprocess, device)
    reference_batch = _load_image_batch(reference_dir, preprocess, device)
    normal_vector = compute_concept_normal_vector(model, concept_batch, reference_batch)

    scores = {
        label: compute_image_sensitivity_score(model, preprocess, image_path, normal_vector, device)
        for label, image_path in target_images.items()
    }
    return scores


def plot_sensitivity_scores(scores: dict[str, float]):
    colors = {
        "Zebra": BLUE,
        "Tiger": ORANGE,
        "Dog": CHAPTER_GRAY,
    }

    fig, axis = book_subplots(size="single")
    labels = list(scores.keys())
    values = [scores[label] for label in labels]
    bars = axis.bar(labels, values, color=[colors.get(label, ORANGE) for label in labels])
    axis.set_ylim(-1, 1)
    axis.set_ylabel("Cosine similarity")
    axis.set_title("Image sensitivity scores")
    axis.axhline(0, color=CHAPTER_GRAY, linewidth=1)

    for bar, value in zip(bars, values):
        offset = 0.03 if value >= 0 else -0.05
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    return fig
