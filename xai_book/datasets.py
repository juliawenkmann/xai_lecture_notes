from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split


def load_synthetic_classification(
    n_samples: int = 600,
    n_features: int = 10,
    n_informative: int = 6,
    n_redundant: int = 2,
    n_classes: int = 2,
    class_sep: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    class_names = [f"class_{index}" for index in range(n_classes)]
    stratify = y if n_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, y_train, X_test, y_test, feature_names, class_names


def load_tabular_dataset(
    name: str = "breast_cancer",
    test_size: float = 0.2,
    random_state: int = 42,
):
    dataset_name = (name or "").lower()
    if dataset_name in {"breast_cancer", "cancer"}:
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    elif dataset_name == "iris":
        data = load_iris(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    elif dataset_name == "wine":
        data = load_wine(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    else:
        raise ValueError(f"Unknown tabular dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names


def load_image_dataset(name: str = "digits"):
    dataset_name = (name or "").lower()
    if dataset_name in {"digits", "sklearn_digits"}:
        data = load_digits()
        images = data.images
        y = data.target
        class_names = [str(label) for label in np.unique(y)]
        return images, y, images.shape[1:], class_names

    if dataset_name == "skimage_demo":
        from skimage import data as skimage_data

        images = {
            "astronaut": skimage_data.astronaut(),
            "coffee": skimage_data.coffee(),
            "camera": skimage_data.camera(),
        }
        return images, None, None, None

    raise ValueError(f"Unknown image dataset: {name}")


def load_mnist_dataset(
    *,
    train_size: int | None = 6000,
    test_size: int | None = 1500,
    random_state: int = 42,
    normalize: bool = True,
):
    candidates = [
        Path(__file__).resolve().parent.parent / "notebooks/01_introduction/data/mnist.npz",
        Path.home() / ".keras/datasets/mnist.npz",
        Path.home() / ".cache/keras/datasets/mnist.npz",
    ]

    mnist_path = next((path for path in candidates if path.exists()), None)
    if mnist_path is None:
        raise RuntimeError(
            "MNIST could not be found locally. Place `mnist.npz` under "
            "`notebooks/01_introduction/data/` or `~/.keras/datasets/`."
        )

    with np.load(mnist_path, allow_pickle=False) as data:
        train_images = data["x_train"].astype(np.float32)
        train_labels = data["y_train"].astype(np.int64)
        test_images = data["x_test"].astype(np.float32)
        test_labels = data["y_test"].astype(np.int64)

    if normalize:
        train_images /= 255.0
        test_images /= 255.0

    if train_size is not None and train_size < len(train_images):
        train_images, _, train_labels, _ = train_test_split(
            train_images,
            train_labels,
            train_size=train_size,
            random_state=random_state,
            stratify=train_labels,
        )

    if test_size is not None and test_size < len(test_images):
        test_images, _, test_labels, _ = train_test_split(
            test_images,
            test_labels,
            train_size=test_size,
            random_state=random_state,
            stratify=test_labels,
        )

    X_train = train_images.reshape(len(train_images), -1)
    X_test = test_images.reshape(len(test_images), -1)
    class_names = [str(label) for label in range(10)]
    return X_train, X_test, train_labels, test_labels, train_images, test_images, class_names


def load_timeseries_dataset(name: str = "sunspots") -> pd.Series:
    dataset_name = (name or "").lower()
    try:
        import statsmodels.api as sm
    except Exception as exc:
        raise RuntimeError("statsmodels is required for time-series datasets.") from exc

    if dataset_name == "sunspots":
        frame = sm.datasets.sunspots.load_pandas().data
        return pd.Series(
            frame["SUNACTIVITY"].values,
            index=pd.Index(frame["YEAR"].values, name="year"),
            name="sunspots",
        )

    if dataset_name == "co2":
        frame = sm.datasets.co2.load_pandas().data
        series = frame["co2"].asfreq("MS").interpolate()
        series.name = "co2"
        return series

    raise ValueError(f"Unknown time-series dataset: {name}")


def make_lagged_frame(series: pd.Series, n_lags: int = 12):
    X_rows = []
    y_rows = []
    for index in range(n_lags, len(series)):
        X_rows.append(series.values[index - n_lags : index])
        y_rows.append(series.values[index])

    columns = [f"lag_{offset}" for offset in range(n_lags, 0, -1)]
    X = pd.DataFrame(np.asarray(X_rows), columns=columns)
    y = pd.Series(np.asarray(y_rows), name="target").reset_index(drop=True)
    return X, y
