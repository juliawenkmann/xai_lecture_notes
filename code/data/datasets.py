from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, make_classification

def _to_frame(X, feature_names=None):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=feature_names)

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
    """Return a simple synthetic binary classification dataset ready for demos."""
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
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    class_names = [f"class_{i}" for i in range(n_classes)]
    stratify = y if n_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, y_train, X_test, y_test, feature_names, class_names

# ---------- Tabular datasets ----------
def load_tabular(name: str = "breast_cancer", test_size: float = 0.2, random_state: int = 42):
    name = (name or "").lower()
    if name in {"breast_cancer", "cancer"}:
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    elif name in {"iris"}:
        data = load_iris(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    elif name in {"wine"}:
        data = load_wine(as_frame=True)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        class_names = list(map(str, data.target_names))
    else:
        raise ValueError(f"Unknown tabular dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names

# ---------- Image datasets ----------
def load_images(name: str = "digits"):
    name = (name or "").lower()
    if name in {"digits", "sklearn_digits"}:
        ds = load_digits()
        # X: (n_samples, 64), reshape to (8,8)
        images = ds.images  # already shaped (n, 8, 8)
        y = ds.target
        class_names = [str(i) for i in np.unique(y)]
        return images, y, images.shape[1:], class_names
    elif name in {"skimage_demo"}:
        # Return a set of demo images (no labels)
        from skimage import data as skdata
        imgs = {
            "astronaut": skdata.astronaut(),
            "coffee": skdata.coffee(),
            "camera": skdata.camera(),
        }
        return imgs, None, None, None
    else:
        raise ValueError(f"Unknown image dataset: {name}")

# ---------- Time-series datasets ----------
def load_timeseries(name: str = "sunspots"):
    name = (name or "").lower()
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise RuntimeError("statsmodels is required for time-series datasets.") from e

    if name in {"sunspots"}:
        df = sm.datasets.sunspots.load_pandas().data
        # df columns: YEAR, SUNACTIVITY
        ts = pd.Series(df['SUNACTIVITY'].values, index=pd.Index(df['YEAR'].values, name='year'), name='sunspots')
        return ts
    elif name in {"co2"}:
        # monthly atmospheric CO2 concentrations
        df = sm.datasets.co2.load_pandas().data
        ts = df['co2'].asfreq('MS').interpolate()
        ts.name = "co2"
        return ts
    else:
        raise ValueError(f"Unknown time-series dataset: {name}")

def make_lagged(ts: pd.Series, n_lags: int = 12):
    """Create supervised tabular data from a univariate series with lag features."""
    X, y = [], []
    for t in range(n_lags, len(ts)):
        X.append(ts.values[t-n_lags:t])
        y.append(ts.values[t])
    X = np.asarray(X)
    y = np.asarray(y)
    cols = [f"lag_{i}" for i in range(n_lags, 0, -1)]
    X_df = pd.DataFrame(X, columns=cols)
    y_ser = pd.Series(y, name="target").reset_index(drop=True)
    return X_df, y_ser
