from __future__ import annotations

from pathlib import Path

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .paths import ROOT_DIR
from .plotting import book_subplots


def default_gradcam_image_path() -> Path:
    return ROOT_DIR / "notebooks" / "04_gradient_based" / "data" / "left-dog.jpg"


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.models import resnet18

    try:
        from torchvision.models import ResNet18_Weights
    except ImportError:
        ResNet18_Weights = None

    return torch, nn, F, transforms, resnet18, ResNet18_Weights


def resolve_device(device: str | None = None):
    torch, _, _, _, _, _ = _torch_modules()
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_resnet18_with_preprocessing(device: str | None = None):
    torch, nn, _, transforms, resnet18, ResNet18_Weights = _torch_modules()
    device_obj = resolve_device(device)

    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights).to(device_obj)
        preprocess = weights.transforms()
    else:
        model = resnet18(pretrained=True).to(device_obj)
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
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model, preprocess, device_obj


def load_demo_image(image_path: str | Path | None = None) -> Image.Image:
    path = Path(image_path) if image_path is not None else default_gradcam_image_path()
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image, preprocess, device) -> object:
    return preprocess(image).unsqueeze(0).to(device)


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    values = values - values.min()
    max_value = values.max()
    if max_value > 0:
        values = values / max_value
    return values


def resize_heatmap(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray((normalize_heatmap(values) * 255).astype(np.uint8))
    return np.asarray(resized.resize(size, Image.BILINEAR), dtype=float) / 255.0


def compute_cam_and_gradcam(
    model,
    input_tensor,
    *,
    class_index: int | None = None,
):
    torch, _, F, _, _, _ = _torch_modules()

    activations = {}
    gradients = {}
    target_layer = model.layer4[-1]

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()
        output.register_hook(lambda grad: gradients.setdefault("value", grad.detach()))

    handle = target_layer.register_forward_hook(forward_hook)
    logits = model(input_tensor)
    if class_index is None:
        class_index = int(logits.argmax(dim=1).item())

    model.zero_grad()
    logits[0, class_index].backward()
    handle.remove()

    feature_maps = activations["value"][0]
    gradient_maps = gradients["value"][0]

    cam = torch.einsum("c,chw->hw", model.fc.weight[class_index].detach(), feature_maps)
    grad_weights = gradient_maps.mean(dim=(1, 2))
    gradcam = torch.relu((grad_weights[:, None, None] * feature_maps).sum(dim=0))

    return {
        "class_index": class_index,
        "activations": feature_maps.detach().cpu().numpy(),
        "gradients": gradient_maps.detach().cpu().numpy(),
        "cam": cam.detach().cpu().numpy(),
        "gradcam": gradcam.detach().cpu().numpy(),
    }


def plot_cam_gradcam_comparison(
    image: Image.Image,
    cam_heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray,
):
    fig, axes = book_subplots(1, 3, size="three_panel")
    resized_cam = resize_heatmap(cam_heatmap, image.size)
    resized_gradcam = resize_heatmap(gradcam_heatmap, image.size)

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(resized_cam, cmap="jet", alpha=0.75)
    axes[1].set_title("CAM")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(resized_gradcam, cmap="jet", alpha=0.75)
    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")
    return fig


def plot_top_channel_overlays(
    image: Image.Image,
    activations: np.ndarray,
    gradients: np.ndarray,
    *,
    title: str,
    top_k: int = 5,
):
    channel_weights = np.maximum(gradients.mean(axis=(1, 2)), 0.0)
    top_indices = np.argsort(channel_weights)[-top_k:][::-1]

    fig, axes = book_subplots(1, top_k, size="grid", extra_height=-0.5)
    if top_k == 1:
        axes = [axes]

    for axis, channel_index in zip(axes, top_indices):
        channel_map = np.maximum(activations[channel_index], 0.0)
        overlay = resize_heatmap(channel_map, image.size)
        axis.imshow(image)
        axis.imshow(overlay, cmap="jet", alpha=0.7)
        axis.set_title(f"Ch {channel_index}\n{channel_weights[channel_index]:.3f}")
        axis.axis("off")

    fig.suptitle(title, y=1.02)
    return fig, top_indices.tolist()
