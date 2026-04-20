from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tarfile
import textwrap
from typing import Any
from urllib.request import Request, urlopen

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .plotting import BLUE, CHAPTER_GRAY, ORANGE, book_subplots, latex_escape


CBM_FIGURE_FILENAMES = {
    "model_comparison": "concept_bottleneck_model_comparison",
    "concept_quality": "concept_bottleneck_concept_quality",
    "prediction_explanation": "concept_bottleneck_prediction_explanation",
}

CBM_TABLE_FILENAMES = {
    "model_metrics": "concept_bottleneck_model_metrics",
    "concept_metrics": "concept_bottleneck_concept_metrics",
    "explanation": "concept_bottleneck_explanation",
}

CBM_VARIANT_LABELS = {
    "oracle_concepts": "Oracle concepts",
    "sequential_cbm": "Sequential CBM",
    "joint_cbm": "Joint CBM",
    "feature_baseline": "Feature baseline",
}
CBM_VARIANT_ALIASES = {
    "direct_probe": "feature_baseline",
}
SUPPORTED_CBM_VARIANTS = tuple(CBM_VARIANT_LABELS)
EXPLAINABLE_CBM_VARIANTS = {"sequential_cbm", "joint_cbm"}

CUB_DOWNLOAD_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
CUB_REQUIRED_RELATIVE_PATHS = (
    Path("images.txt"),
    Path("image_class_labels.txt"),
    Path("train_test_split.txt"),
    Path("classes.txt"),
    Path("images"),
)


@dataclass(frozen=True)
class CubPaths:
    raw_dir: Path
    cache_dir: Path


@dataclass
class CBMVariantResult:
    variant: str
    label: str
    class_probabilities: np.ndarray
    class_predictions: np.ndarray
    class_labels: np.ndarray
    concept_probabilities: np.ndarray | None = None
    concept_truth: np.ndarray | None = None
    label_model: Any | None = None
    joint_model: Any | None = None


def _normalize_cub_raw_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.name == "CUB_200_2011":
        return path
    nested = path / "CUB_200_2011"
    if nested.exists():
        return nested
    return path


def cub_raw_dir_candidates(data_dir: Path | str, *, raw_dir: Path | str | None = None) -> list[Path]:
    data_dir = Path(data_dir)
    base_dir = Path(data_dir) / "cub"
    candidates: list[Path] = []
    if raw_dir is not None:
        candidates.append(_normalize_cub_raw_dir(Path(raw_dir)))
    candidates.extend(
        [
            base_dir / "raw" / "CUB_200_2011",
            base_dir / "CUB_200_2011",
            data_dir / "CUB_200_2011",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
    return unique_candidates


def cub_data_paths(data_dir: Path | str, *, raw_dir: Path | str | None = None) -> CubPaths:
    data_dir = Path(data_dir)
    base_dir = data_dir / "cub"
    if raw_dir is not None:
        return CubPaths(
            raw_dir=_normalize_cub_raw_dir(Path(raw_dir)),
            cache_dir=base_dir / "cache",
        )

    candidates = cub_raw_dir_candidates(data_dir, raw_dir=None)
    existing = next((candidate for candidate in candidates if not cub_missing_paths(candidate)), candidates[0])
    return CubPaths(raw_dir=existing, cache_dir=base_dir / "cache")


def _resolve_optional_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def cub_attribute_names_path(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "attributes" / "attributes.txt",
        raw_dir / "attributes.txt",
        raw_dir.parent / "attributes.txt",
    ]
    path = _resolve_optional_path(candidates)
    return path or candidates[0]


def cub_attribute_labels_path(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "attributes" / "image_attribute_labels.txt",
        raw_dir / "image_attribute_labels.txt",
        raw_dir.parent / "image_attribute_labels.txt",
    ]
    path = _resolve_optional_path(candidates)
    return path or candidates[0]


def cub_missing_paths(raw_dir: Path) -> list[Path]:
    missing = [raw_dir / relative_path for relative_path in CUB_REQUIRED_RELATIVE_PATHS if not (raw_dir / relative_path).exists()]
    if not cub_attribute_names_path(raw_dir).exists():
        missing.append(cub_attribute_names_path(raw_dir))
    if not cub_attribute_labels_path(raw_dir).exists():
        missing.append(cub_attribute_labels_path(raw_dir))
    return missing


def cub_dataset_available(paths: CubPaths) -> bool:
    return not cub_missing_paths(paths.raw_dir)


def ensure_cub_dataset_available(paths: CubPaths) -> None:
    missing = cub_missing_paths(paths.raw_dir)
    if missing:
        searched = "\n".join(f"- {candidate}" for candidate in cub_raw_dir_candidates(paths.cache_dir.parent.parent))
        missing_display = "\n".join(f"- {path}" for path in missing[:6])
        hint = (
            f"Expected the raw CUB dataset under {paths.raw_dir}. "
            "Place the official CUB_200_2011 download into data/cub/raw/ or pass CONFIG['raw_dir'] to the notebook.\n"
            f"Missing required files:\n{missing_display}\n"
            f"Searched:\n{searched}"
        )
        raise FileNotFoundError(hint)


def _file_md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError(f"Refusing to extract unexpected path: {member.name}")
        archive.extractall(destination)


def download_cub_dataset(
    data_dir: Path | str,
    *,
    raw_dir: Path | str | None = None,
    download_url: str | None = None,
) -> CubPaths:
    paths = cub_data_paths(data_dir, raw_dir=raw_dir)
    if cub_dataset_available(paths):
        return paths

    download_root = paths.raw_dir.parent
    archive_path = download_root / "CUB_200_2011.tgz"
    download_root.mkdir(parents=True, exist_ok=True)
    url = download_url or CUB_DOWNLOAD_URL

    print(f"Downloading CUB from {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    checksum = _file_md5(archive_path)
    if checksum != CUB_ARCHIVE_MD5:
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"CUB archive checksum mismatch: expected {CUB_ARCHIVE_MD5}, got {checksum}")

    print(f"Extracting {archive_path.name} into {download_root}")
    _safe_extract_tar(archive_path, download_root)
    archive_path.unlink(missing_ok=True)

    paths = cub_data_paths(data_dir, raw_dir=raw_dir)
    ensure_cub_dataset_available(paths)
    return paths


def prepare_cub_dataset(
    data_dir: Path | str,
    *,
    raw_dir: Path | str | None = None,
    download_if_missing: bool = True,
    download_url: str | None = None,
) -> CubPaths:
    paths = cub_data_paths(data_dir, raw_dir=raw_dir)
    if cub_dataset_available(paths):
        return paths
    if download_if_missing:
        return download_cub_dataset(data_dir, raw_dir=raw_dir, download_url=download_url)
    ensure_cub_dataset_available(paths)
    return paths


def _clean_class_name(name: str) -> str:
    if "." in name:
        name = name.split(".", 1)[1]
    return name.replace("_", " ")


def _clean_attribute_name(name: str) -> str:
    return (
        name.replace("has_", "")
        .replace("::", ": ")
        .replace("_", " ")
        .replace("(", "")
        .replace(")", "")
    )


def load_cub_metadata(raw_dir: Path) -> pd.DataFrame:
    images = pd.read_csv(
        raw_dir / "images.txt",
        sep=" ",
        names=["image_id", "image_path"],
    )
    labels = pd.read_csv(
        raw_dir / "image_class_labels.txt",
        sep=" ",
        names=["image_id", "class_id"],
    )
    splits = pd.read_csv(
        raw_dir / "train_test_split.txt",
        sep=" ",
        names=["image_id", "is_train"],
    )
    classes = pd.read_csv(
        raw_dir / "classes.txt",
        sep=" ",
        names=["class_id", "class_name"],
    )
    classes["class_name"] = classes["class_name"].map(_clean_class_name)

    metadata = images.merge(labels, on="image_id").merge(splits, on="image_id").merge(classes, on="class_id")
    metadata["image_file"] = metadata["image_path"].map(lambda rel_path: raw_dir / "images" / rel_path)
    return metadata.sort_values("image_id").reset_index(drop=True)


def load_cub_attribute_matrix(raw_dir: Path) -> pd.DataFrame:
    attribute_names = pd.read_csv(
        cub_attribute_names_path(raw_dir),
        sep=" ",
        names=["attribute_id", "attribute_name"],
    )
    attribute_names["attribute_name"] = attribute_names["attribute_name"].map(_clean_attribute_name)

    labels = pd.read_csv(
        cub_attribute_labels_path(raw_dir),
        sep=" ",
        names=["image_id", "attribute_id", "is_present", "certainty_id", "annotation_time"],
        usecols=["image_id", "attribute_id", "is_present"],
    )

    matrix = labels.pivot(index="image_id", columns="attribute_id", values="is_present").sort_index()
    matrix = matrix.fillna(0).astype(np.float32)
    name_map = dict(zip(attribute_names["attribute_id"], attribute_names["attribute_name"]))
    matrix.columns = [name_map[attribute_id] for attribute_id in matrix.columns]
    return matrix


def select_cub_class_slice(metadata: pd.DataFrame, *, n_classes: int = 20) -> pd.DataFrame:
    class_frame = metadata[["class_id", "class_name"]].drop_duplicates().sort_values("class_id").reset_index(drop=True)
    if n_classes >= len(class_frame):
        return metadata.copy()

    positions = np.linspace(0, len(class_frame) - 1, num=n_classes, dtype=int)
    selected_ids = class_frame.iloc[np.unique(positions)]["class_id"].to_list()
    subset = metadata.loc[metadata["class_id"].isin(selected_ids)].copy()
    return subset.sort_values("image_id").reset_index(drop=True)


def select_concept_columns(
    train_concepts: pd.DataFrame,
    *,
    n_concepts: int = 12,
    min_prevalence: float = 0.15,
    max_prevalence: float = 0.85,
) -> list[str]:
    prevalence = train_concepts.mean(axis=0)
    candidates = prevalence[(prevalence >= min_prevalence) & (prevalence <= max_prevalence)]
    if candidates.empty:
        candidates = prevalence[(prevalence > 0.0) & (prevalence < 1.0)]

    if candidates.empty:
        raise ValueError("Could not find any usable concept columns in the selected CUB split.")

    balance_score = 0.5 - (candidates - 0.5).abs()
    selected = balance_score.sort_values(ascending=False).head(n_concepts).index.to_list()
    return selected


def _load_resnet18_feature_extractor(device: str | None = None):
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18

    try:
        from torchvision.models import ResNet18_Weights
    except ImportError:
        ResNet18_Weights = None

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weight_name = "untrained"

    if ResNet18_Weights is not None:
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            preprocess = weights.transforms()
            weight_name = getattr(weights, "name", "DEFAULT")
        except Exception:
            model = resnet18(weights=None)
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
    else:
        try:
            model = resnet18(weights=None)
        except TypeError:
            model = resnet18()
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

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device_obj)
    feature_extractor.eval()
    return torch, feature_extractor, preprocess, device_obj, weight_name


def extract_cub_features(
    metadata: pd.DataFrame,
    *,
    cache_path: Path,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[np.ndarray, str]:
    image_ids = metadata["image_id"].to_numpy(dtype=np.int64)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["image_ids"], image_ids):
            return cached["features"], str(cached["backbone"])

    torch, feature_extractor, preprocess, device_obj, weight_name = _load_resnet18_feature_extractor(device=device)

    features: list[np.ndarray] = []
    for start in range(0, len(metadata), batch_size):
        batch_frame = metadata.iloc[start : start + batch_size]
        batch_tensors = []
        for image_path in batch_frame["image_file"]:
            image = Image.open(image_path).convert("RGB")
            batch_tensors.append(preprocess(image))

        with torch.no_grad():
            batch = torch.stack(batch_tensors).to(device_obj)
            batch_features = feature_extractor(batch).flatten(1).cpu().numpy().astype(np.float32)
        features.append(batch_features)

    stacked = np.concatenate(features, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_ids=image_ids,
        features=stacked,
        backbone=np.asarray(weight_name),
    )
    return stacked, weight_name


def _make_binary_concept_model(*, random_state: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def _make_multiclass_model(*, random_state: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
    )


def _canonicalize_variant(variant: str) -> str:
    return CBM_VARIANT_ALIASES.get(variant, variant)


def _variant_label(variant: str) -> str:
    variant = _canonicalize_variant(variant)
    if variant not in CBM_VARIANT_LABELS:
        raise ValueError(f"Unknown CBM variant: {variant}")
    return CBM_VARIANT_LABELS[variant]


def fit_concept_models(
    train_features: np.ndarray,
    train_concepts: np.ndarray,
    *,
    random_state: int,
) -> list[Any]:
    models: list[Any] = []
    for index in range(train_concepts.shape[1]):
        model = _make_binary_concept_model(random_state=random_state + index)
        model.fit(train_features, train_concepts[:, index])
        models.append(model)
    return models


def predict_concept_probabilities(models: list[Any], features: np.ndarray) -> np.ndarray:
    probabilities = []
    for model in models:
        estimator = model.named_steps["logisticregression"]
        positive_index = int(np.where(estimator.classes_ == 1)[0][0])
        probabilities.append(model.predict_proba(features)[:, positive_index])
    return np.column_stack(probabilities)


def fit_oracle_concept_variant(
    train_concepts: np.ndarray,
    test_concepts: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> CBMVariantResult:
    model = _make_multiclass_model(random_state=random_state)
    model.fit(train_concepts, y_train)
    class_probabilities = model.predict_proba(test_concepts)
    class_predictions = model.predict(test_concepts)
    return CBMVariantResult(
        variant="oracle_concepts",
        label=_variant_label("oracle_concepts"),
        class_probabilities=class_probabilities,
        class_predictions=class_predictions,
        class_labels=model.named_steps["logisticregression"].classes_,
        concept_probabilities=test_concepts,
        concept_truth=test_concepts,
        label_model=model,
    )


def fit_sequential_cbm_variant(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_concepts: np.ndarray,
    test_concepts: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> CBMVariantResult:
    concept_models = fit_concept_models(train_features, train_concepts, random_state=random_state)
    concept_train_proba = predict_concept_probabilities(concept_models, train_features)
    concept_test_proba = predict_concept_probabilities(concept_models, test_features)

    label_model = _make_multiclass_model(random_state=random_state)
    label_model.fit(concept_train_proba, y_train)
    class_probabilities = label_model.predict_proba(concept_test_proba)
    class_predictions = label_model.predict(concept_test_proba)
    return CBMVariantResult(
        variant="sequential_cbm",
        label=_variant_label("sequential_cbm"),
        class_probabilities=class_probabilities,
        class_predictions=class_predictions,
        class_labels=label_model.named_steps["logisticregression"].classes_,
        concept_probabilities=concept_test_proba,
        concept_truth=test_concepts,
        label_model=label_model,
    )


def fit_feature_baseline_variant(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> CBMVariantResult:
    model = _make_multiclass_model(random_state=random_state)
    model.fit(train_features, y_train)
    class_probabilities = model.predict_proba(test_features)
    class_predictions = model.predict(test_features)
    return CBMVariantResult(
        variant="feature_baseline",
        label=_variant_label("feature_baseline"),
        class_probabilities=class_probabilities,
        class_predictions=class_predictions,
        class_labels=model.named_steps["logisticregression"].classes_,
        label_model=model,
    )


def fit_direct_probe_variant(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
) -> CBMVariantResult:
    return fit_feature_baseline_variant(
        train_features,
        test_features,
        y_train,
        random_state=random_state,
    )


def fit_joint_cbm_variant(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_concepts: np.ndarray,
    test_concepts: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    epochs: int = 200,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    concept_loss_weight: float = 1.0,
) -> CBMVariantResult:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    feature_scaler = StandardScaler()
    train_scaled = feature_scaler.fit_transform(train_features).astype(np.float32)
    test_scaled = feature_scaler.transform(test_features).astype(np.float32)

    class_labels = np.unique(y_train)
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y_train_index = np.asarray([class_to_index[label] for label in y_train], dtype=np.int64)

    class JointCBM(nn.Module):
        def __init__(self, input_dim: int, n_concepts: int, n_classes: int):
            super().__init__()
            self.concept_head = nn.Linear(input_dim, n_concepts)
            self.label_head = nn.Linear(n_concepts, n_classes)

        def forward(self, features):
            concept_logits = self.concept_head(features)
            concept_probabilities = torch.sigmoid(concept_logits)
            class_logits = self.label_head(concept_probabilities)
            return concept_logits, concept_probabilities, class_logits

    torch.manual_seed(random_state)
    model = JointCBM(train_scaled.shape[1], train_concepts.shape[1], len(class_labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    concept_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(train_scaled),
        torch.from_numpy(train_concepts.astype(np.float32)),
        torch.from_numpy(y_train_index),
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_features, batch_concepts, batch_labels in loader:
            optimizer.zero_grad()
            concept_logits, _concept_probabilities, class_logits = model(batch_features)
            loss = class_loss_fn(class_logits, batch_labels)
            loss = loss + concept_loss_weight * concept_loss_fn(concept_logits, batch_concepts)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _concept_logits, train_concept_probabilities, _train_class_logits = model(torch.from_numpy(train_scaled))
        _concept_logits, test_concept_probabilities, test_class_logits = model(torch.from_numpy(test_scaled))

    class_probabilities = torch.softmax(test_class_logits, dim=1).cpu().numpy()
    class_predictions = class_labels[class_probabilities.argmax(axis=1)]
    return CBMVariantResult(
        variant="joint_cbm",
        label=_variant_label("joint_cbm"),
        class_probabilities=class_probabilities,
        class_predictions=class_predictions,
        class_labels=class_labels,
        concept_probabilities=test_concept_probabilities.cpu().numpy(),
        concept_truth=test_concepts,
        joint_model=model,
    )


def compute_model_metrics(
    *,
    y_test: np.ndarray,
    variant_results: dict[str, CBMVariantResult],
) -> pd.DataFrame:
    top_k = min(3, len(np.unique(y_test)))
    rows = []
    for result in variant_results.values():
        rows.append(
            {
                "variant": result.variant,
                "model": result.label,
                "accuracy": accuracy_score(y_test, result.class_predictions),
                "top_3_accuracy": top_k_accuracy_score(
                    y_test,
                    result.class_probabilities,
                    labels=result.class_labels,
                    k=top_k,
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_concept_metrics(
    *,
    concept_names: list[str],
    variant_results: dict[str, CBMVariantResult],
) -> pd.DataFrame:
    rows = []
    for result in variant_results.values():
        if result.variant not in EXPLAINABLE_CBM_VARIANTS:
            continue
        if result.concept_probabilities is None or result.concept_truth is None:
            continue

        for index, concept_name in enumerate(concept_names):
            truth = result.concept_truth[:, index]
            probability = result.concept_probabilities[:, index]
            positive_rate = float(truth.mean())
            average_precision = np.nan
            balanced_accuracy = np.nan
            if 0.0 < positive_rate < 1.0:
                average_precision = average_precision_score(truth, probability)
                balanced_accuracy = balanced_accuracy_score(truth, probability >= 0.5)

            rows.append(
                {
                    "variant": result.variant,
                    "model": result.label,
                    "concept": concept_name,
                    "test_prevalence": positive_rate,
                    "balanced_accuracy": balanced_accuracy,
                    "average_precision": average_precision,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["model", "average_precision", "balanced_accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_prediction_explanation(
    *,
    variant_result: CBMVariantResult,
    concept_names: list[str],
    test_metadata: pd.DataFrame,
    y_test: np.ndarray,
    variant_label: str,
) -> tuple[pd.Series, pd.DataFrame]:
    if variant_result.concept_probabilities is None or variant_result.concept_truth is None:
        raise ValueError(f"Variant {variant_result.variant} does not expose concept predictions for explanation.")

    confidence = variant_result.class_probabilities.max(axis=1)
    candidate_indices = np.flatnonzero(variant_result.class_predictions == y_test)
    if len(candidate_indices) == 0:
        sample_index = int(confidence.argmax())
    else:
        sample_index = int(candidate_indices[np.argmax(confidence[candidate_indices])])

    predicted_class = int(variant_result.class_predictions[sample_index])
    if variant_result.variant in {"oracle_concepts", "sequential_cbm"}:
        scaler = variant_result.label_model.named_steps["standardscaler"]
        classifier = variant_result.label_model.named_steps["logisticregression"]
        scaled_concepts = scaler.transform(variant_result.concept_probabilities[[sample_index]])[0]
        class_position = int(np.where(classifier.classes_ == predicted_class)[0][0])
        contributions = scaled_concepts * classifier.coef_[class_position]
    elif variant_result.variant == "joint_cbm":
        classifier = variant_result.joint_model.label_head
        class_position = int(np.where(variant_result.class_labels == predicted_class)[0][0])
        classifier_weights = classifier.weight.detach().cpu().numpy()[class_position]
        concept_values = variant_result.concept_probabilities[sample_index]
        contributions = concept_values * classifier_weights
    else:
        raise ValueError(f"Variant {variant_result.variant} is not explainable through concepts.")

    explanation = pd.DataFrame(
        {
            "variant": variant_result.variant,
            "model": variant_label,
            "concept": concept_names,
            "contribution": contributions,
            "predicted_probability": variant_result.concept_probabilities[sample_index],
            "ground_truth": variant_result.concept_truth[sample_index].astype(int),
        }
    ).sort_values("contribution", ascending=False)

    sample_row = test_metadata.iloc[sample_index].copy()
    sample_row["variant"] = variant_result.variant
    sample_row["model"] = variant_label
    sample_row["predicted_class_id"] = predicted_class
    sample_row["predicted_class_name"] = test_metadata.loc[
        test_metadata["class_id"] == predicted_class,
        "class_name",
    ].iloc[0]
    sample_row["prediction_confidence"] = float(confidence[sample_index])
    return sample_row, explanation.reset_index(drop=True)


def _wrap_label(text: str, *, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def plot_cbm_model_comparison(metrics_frame: pd.DataFrame, *, title: str = "Concept bottleneck variants"):
    fig, axis = book_subplots(size="wide", style="notebook", grid_axis="y")

    x = np.arange(len(metrics_frame))
    width = 0.34
    accuracy_bars = axis.bar(
        x - width / 2,
        metrics_frame["accuracy"],
        width=width,
        color=BLUE,
        label="Accuracy",
    )
    top3_bars = axis.bar(
        x + width / 2,
        metrics_frame["top_3_accuracy"],
        width=width,
        color=ORANGE,
        label="Top-3 accuracy",
    )

    tick_labels = [_wrap_label(label, width=16) for label in metrics_frame["model"]]
    axis.set_xticks(x, tick_labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.35,
    )

    for bars in (accuracy_bars, top3_bars):
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.02, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    return fig


def plot_cbm_concept_quality(concept_metrics: pd.DataFrame, *, top_n: int = 10, title: str = "Concept prediction quality"):
    frame = concept_metrics.head(top_n).sort_values("average_precision", ascending=True)

    fig, axis = book_subplots(size="wide", style="notebook", grid_axis="x")
    y_positions = np.arange(len(frame))
    tick_labels = [_wrap_label(latex_escape(label), width=26) for label in frame["concept"]]

    axis.barh(y_positions, frame["average_precision"], color=BLUE, alpha=0.95, label="Average precision")
    axis.scatter(
        frame["test_prevalence"],
        y_positions,
        color=ORANGE,
        s=42,
        zorder=3,
        label="Prevalence",
    )
    axis.set_yticks(y_positions, tick_labels)
    axis.set_xlim(0.0, 1.02)
    axis.set_xlabel("Score")
    axis.set_title(title)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.35,
    )
    return fig


def plot_cbm_prediction_explanation(
    sample_row: pd.Series,
    explanation_frame: pd.DataFrame,
    *,
    top_n: int = 8,
    title: str = "Top concept contributions",
):
    top_frame = explanation_frame.head(top_n).sort_values("contribution", ascending=True)

    fig, axes = book_subplots(1, 2, size="two_panel", style="notebook", grid_axis="x", gridspec_kw={"width_ratios": [1.0, 1.55]})

    image = Image.open(sample_row["image_file"]).convert("RGB")
    axes[0].imshow(image)
    axes[0].set_axis_off()
    predicted_label = _wrap_label(latex_escape(sample_row["predicted_class_name"]), width=18)
    true_label = _wrap_label(latex_escape(sample_row["class_name"]), width=18)
    axes[0].set_title(
        f"Predicted: {predicted_label}\nTrue: {true_label}",
        loc="left",
        fontsize=11,
    )

    colors = [ORANGE if value >= 0 else BLUE for value in top_frame["contribution"]]
    tick_labels = [_wrap_label(latex_escape(label), width=24) for label in top_frame["concept"]]
    bars = axes[1].barh(tick_labels, top_frame["contribution"], color=colors)
    axes[1].axvline(0.0, color=CHAPTER_GRAY, linewidth=1.1)
    axes[1].set_xlabel("Contribution to predicted logit")
    axes[1].set_title(title)
    contribution_values = top_frame["contribution"].to_numpy()
    left_limit = min(0.0, contribution_values.min()) - 0.35
    right_limit = max(0.0, contribution_values.max()) + 0.55
    axes[1].set_xlim(left_limit, right_limit)

    for bar, value in zip(bars, top_frame["contribution"]):
        x_position = value + (0.05 if value >= 0 else -0.05)
        axes[1].text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )
    return fig


def run_cbm_variants(
    *,
    variants: list[str],
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_concepts: np.ndarray,
    test_concepts: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    joint_epochs: int,
    joint_learning_rate: float,
    joint_weight_decay: float,
    joint_batch_size: int,
    concept_loss_weight: float,
) -> dict[str, CBMVariantResult]:
    variants = [_canonicalize_variant(variant) for variant in variants]
    unknown = [variant for variant in variants if variant not in SUPPORTED_CBM_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported CBM variants: {unknown}")

    results: dict[str, CBMVariantResult] = {}
    for variant in variants:
        if variant == "oracle_concepts":
            results[variant] = fit_oracle_concept_variant(
                train_concepts,
                test_concepts,
                y_train,
                random_state=random_state,
            )
        elif variant == "sequential_cbm":
            results[variant] = fit_sequential_cbm_variant(
                train_features,
                test_features,
                train_concepts,
                test_concepts,
                y_train,
                random_state=random_state,
            )
        elif variant == "joint_cbm":
            results[variant] = fit_joint_cbm_variant(
                train_features,
                test_features,
                train_concepts,
                test_concepts,
                y_train,
                random_state=random_state,
                epochs=joint_epochs,
                learning_rate=joint_learning_rate,
                weight_decay=joint_weight_decay,
                batch_size=joint_batch_size,
                concept_loss_weight=concept_loss_weight,
            )
        elif variant == "feature_baseline":
            results[variant] = fit_feature_baseline_variant(
                train_features,
                test_features,
                y_train,
                random_state=random_state,
            )
    return results


def build_cub_concept_bottleneck_demo(
    *,
    data_dir: Path | str,
    raw_dir: Path | str | None = None,
    download_if_missing: bool = True,
    download_url: str | None = None,
    variants: list[str] | None = None,
    primary_variant: str = "joint_cbm",
    n_classes: int = 20,
    n_concepts: int = 12,
    batch_size: int = 32,
    random_state: int = 7,
    device: str | None = None,
    joint_epochs: int = 200,
    joint_learning_rate: float = 5e-3,
    joint_weight_decay: float = 1e-4,
    joint_batch_size: int = 256,
    concept_loss_weight: float = 1.0,
) -> dict[str, Any]:
    variants = list(dict.fromkeys(_canonicalize_variant(variant) for variant in (variants or SUPPORTED_CBM_VARIANTS)))
    primary_variant = _canonicalize_variant(primary_variant)
    paths = prepare_cub_dataset(
        data_dir,
        raw_dir=raw_dir,
        download_if_missing=download_if_missing,
        download_url=download_url,
    )

    metadata = load_cub_metadata(paths.raw_dir)
    metadata = select_cub_class_slice(metadata, n_classes=n_classes)
    attribute_matrix = load_cub_attribute_matrix(paths.raw_dir)
    attribute_matrix = attribute_matrix.loc[metadata["image_id"]]
    attribute_matrix.index = metadata["image_id"].to_numpy()

    train_metadata = metadata.loc[metadata["is_train"] == 1].reset_index(drop=True)
    test_metadata = metadata.loc[metadata["is_train"] == 0].reset_index(drop=True)

    train_concepts_frame = attribute_matrix.loc[train_metadata["image_id"]]
    selected_concepts = select_concept_columns(train_concepts_frame, n_concepts=n_concepts)

    train_concepts = train_concepts_frame[selected_concepts].to_numpy(dtype=np.float32)
    test_concepts = attribute_matrix.loc[test_metadata["image_id"], selected_concepts].to_numpy(dtype=np.float32)
    y_train = train_metadata["class_id"].to_numpy(dtype=np.int64)
    y_test = test_metadata["class_id"].to_numpy(dtype=np.int64)

    class_ids = sorted(metadata["class_id"].unique().tolist())
    class_key = f"{len(class_ids)}cls_{class_ids[0]}_{class_ids[-1]}"
    train_features, backbone_name = extract_cub_features(
        train_metadata,
        cache_path=paths.cache_dir / f"cub_resnet18_{class_key}_train.npz",
        batch_size=batch_size,
        device=device,
    )
    test_features, _ = extract_cub_features(
        test_metadata,
        cache_path=paths.cache_dir / f"cub_resnet18_{class_key}_test.npz",
        batch_size=batch_size,
        device=device,
    )

    variant_results = run_cbm_variants(
        variants=variants,
        train_features=train_features,
        test_features=test_features,
        train_concepts=train_concepts,
        test_concepts=test_concepts,
        y_train=y_train,
        random_state=random_state,
        joint_epochs=joint_epochs,
        joint_learning_rate=joint_learning_rate,
        joint_weight_decay=joint_weight_decay,
        joint_batch_size=joint_batch_size,
        concept_loss_weight=concept_loss_weight,
    )
    if primary_variant not in variant_results:
        raise ValueError(f"primary_variant must be one of {list(variant_results)}")
    if primary_variant not in EXPLAINABLE_CBM_VARIANTS:
        raise ValueError("primary_variant must be one of the explainable CBM variants.")

    model_metrics = compute_model_metrics(
        y_test=y_test,
        variant_results=variant_results,
    )
    concept_metrics = compute_concept_metrics(
        concept_names=selected_concepts,
        variant_results=variant_results,
    )
    primary_result = variant_results[primary_variant]
    primary_label = _variant_label(primary_variant)
    sample_row, explanation = build_prediction_explanation(
        variant_result=primary_result,
        concept_names=selected_concepts,
        test_metadata=test_metadata,
        y_test=y_test,
        variant_label=primary_label,
    )
    primary_concept_metrics = concept_metrics.loc[concept_metrics["variant"] == primary_variant].drop(
        columns=["variant", "model"]
    )

    figures = {
        "model_comparison": plot_cbm_model_comparison(model_metrics),
        "concept_quality": plot_cbm_concept_quality(
            primary_concept_metrics,
            title=f"Concept prediction quality ({primary_label})",
        ),
        "prediction_explanation": plot_cbm_prediction_explanation(
            sample_row,
            explanation,
            title=f"Top concept contributions ({primary_label})",
        ),
    }

    summary = {
        "backbone": backbone_name,
        "selected_classes": len(class_ids),
        "selected_concepts": len(selected_concepts),
        "train_images": len(train_metadata),
        "test_images": len(test_metadata),
        "variants": variants,
        "variant_labels": [_variant_label(variant) for variant in variants],
        "primary_variant": primary_variant,
        "primary_variant_label": primary_label,
        "class_names": metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")["class_name"]
        .tolist(),
    }

    return {
        "summary": summary,
        "tables": {
            "model_metrics": model_metrics,
            "concept_metrics": concept_metrics,
            "explanation": explanation,
        },
        "figures": figures,
    }
