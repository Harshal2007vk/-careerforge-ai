"""
Preprocessing for CareerForge-AI ML pipeline.
Handles numeric, categorical, and multi-label columns with missing values.
"""

import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Multi-label: parse "A; B; C" or "A, B" into list of labels
try:
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:
    MultiLabelBinarizer = None  # type: ignore

from ml_config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MULTILABEL_FEATURES,
    ARTIFACTS_DIR,
    MAX_CARDINALITY_ONEHOT,
)


# ---------------------------------------------------------------------------
# Multi-label parser
# ---------------------------------------------------------------------------
def parse_multilabel(series: pd.Series, sep: str = ";") -> List[List[str]]:
    """Parse semicolon- or comma-separated strings into lists of stripped labels."""
    out = []
    for v in series:
        if pd.isna(v) or (isinstance(v, str) and not v.strip()):
            out.append([])
            continue
        s = str(v).strip()
        # Support both ; and , as separators
        parts = re.split(r"[;,]", s)
        out.append([p.strip() for p in parts if p.strip()])
    return out


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps MultiLabelBinarizer for use in ColumnTransformer.
    Expects column with semicolon-separated values.
    """

    def __init__(self, sep: str = ";", max_labels: Optional[int] = None):
        self.sep = sep
        self.max_labels = max_labels  # cap number of labels to avoid explosion
        self._mlb = None
        self._columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MultiLabelBinarizerTransformer":
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        parsed = parse_multilabel(X.iloc[:, 0], sep=self.sep)
        self._mlb = MultiLabelBinarizer(sparse_output=False)
        self._mlb.fit(parsed)
        self._columns = [f"ml_{c}" for c in self._mlb.classes_]
        if self.max_labels and len(self._columns) > self.max_labels:
            # Keep top freq classes only (simplified: keep first max_labels)
            self._columns = self._columns[: self.max_labels]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        parsed = parse_multilabel(X.iloc[:, 0], sep=self.sep)
        out = self._mlb.transform(parsed)
        if self.max_labels and out.shape[1] > self.max_labels:
            out = out[:, : self.max_labels]
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.max_labels and len(self._columns) > self.max_labels:
            return np.array(self._columns[: self.max_labels])
        return np.array(self._columns)


class CategoricalReducer(BaseEstimator, TransformerMixin):
    """
    Reduces high-cardinality categoricals: map rare categories to 'Other'.
    """

    def __init__(self, max_cardinality: int = MAX_CARDINALITY_ONEHOT):
        self.max_cardinality = max_cardinality
        self._value_counts: Optional[dict] = None
        self._feature_name: str = "x0"

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalReducer":
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if hasattr(X, "columns") and len(X.columns):
            self._feature_name = str(X.columns[0])
        col = X.iloc[:, 0].astype(str).fillna("__NA__")
        vc = col.value_counts()
        top = vc.head(self.max_cardinality).index.tolist()
        self._value_counts = {c: c for c in top}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        col = X.iloc[:, 0].astype(str).fillna("__NA__")
        return col.map(lambda x: self._value_counts.get(x, "Other")).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array([self._feature_name])


# ---------------------------------------------------------------------------
# Build full preprocessor
# ---------------------------------------------------------------------------
def _get_numeric_features(df: pd.DataFrame) -> List[str]:
    return [c for c in NUMERIC_FEATURES if c in df.columns]


def _get_categorical_features(df: pd.DataFrame) -> List[str]:
    return [c for c in CATEGORICAL_FEATURES if c in df.columns]


def _get_multilabel_features(df: pd.DataFrame) -> List[str]:
    return [c for c in MULTILABEL_FEATURES if c in df.columns]


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
    - Numeric: impute median, then scale
    - Categorical: reduce cardinality, one-hot encode
    - Multi-label: parse and binarize

    Returns (fitted ColumnTransformer, list of output feature names, list of input column names).
    """
    num_cols = _get_numeric_features(df)
    cat_cols = _get_categorical_features(df)
    multi_cols = _get_multilabel_features(df)
    input_columns = num_cols + cat_cols + multi_cols

    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    for col in cat_cols:
        pipe = Pipeline([
            ("reduce", CategoricalReducer(max_cardinality=MAX_CARDINALITY_ONEHOT)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append((f"cat_{col}", pipe, [col]))

    if MultiLabelBinarizer is not None:
        for col in multi_cols:
            mlb = MultiLabelBinarizerTransformer(sep=";", max_labels=100)
            transformers.append((f"ml_{col}", mlb, [col]))

    ct = ColumnTransformer(
        transformers,
        remainder="drop",
        n_jobs=-1,
        verbose=0,
    )

    ct.fit(df)
    feature_names = _get_ct_feature_names(ct)
    return ct, feature_names, input_columns


def _get_ct_feature_names(ct: ColumnTransformer) -> List[str]:
    """Extract feature names from fitted ColumnTransformer (sklearn >= 1.0)."""
    if hasattr(ct, "get_feature_names_out"):
        return ct.get_feature_names_out().tolist()
    names = []
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols).tolist())
    return names


# ---------------------------------------------------------------------------
# Main preprocessor class (fit on train, transform train/test)
# ---------------------------------------------------------------------------
class CareerForgePreprocessor:
    """
    Single entry point for preprocessing. Fit on training data, then transform
    train/test or inference data. Persists encoder state, feature names, and input columns.
    """

    def __init__(self):
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.input_columns_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "CareerForgePreprocessor":
        self.column_transformer_, self.feature_names_, self.input_columns_ = build_preprocessor(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.column_transformer_ is None:
            raise RuntimeError("Preprocessor not fitted.")
        return self.column_transformer_.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or ARTIFACTS_DIR / "preprocessor.joblib"
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "column_transformer": self.column_transformer_,
                "feature_names": self.feature_names_,
                "input_columns": self.input_columns_,
            },
            path,
        )
        # Also save feature names as JSON for non-Python consumers
        fn_path = path.parent / "feature_names.json"
        with open(fn_path, "w") as f:
            json.dump(self.feature_names_, f, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CareerForgePreprocessor":
        import joblib
        path = path or ARTIFACTS_DIR / "preprocessor.joblib"
        data = joblib.load(path)
        obj = cls()
        obj.column_transformer_ = data["column_transformer"]
        obj.feature_names_ = data["feature_names"]
        obj.input_columns_ = data.get("input_columns", [])
        return obj
