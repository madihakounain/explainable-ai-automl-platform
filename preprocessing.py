"""
ml/preprocessing.py
--------------------
Reusable data-preparation helpers.

Extracted from xai_app.py:
  - Feature cleaning / imputation / encoding  (train route, lines 1182-1216)
  - Target encoding                           (train route, lines 1218-1230)
  - Row encoding for inference                (predict route, lines 1377-1397)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import MAX_MISSING_DROP, REGRESSION_UNIQUE_THRESHOLD, RANDOM_STATE


# ── Task detection ─────────────────────────────────────────────────────────────

def detect_task(y_raw: pd.Series) -> str:
    """
    Return 'regression' when the target is numeric with many unique values,
    otherwise 'classification'.  Mirrors the original logic exactly.
    """
    is_numeric = pd.api.types.is_numeric_dtype(y_raw)
    unique_count = y_raw.nunique()
    if is_numeric and unique_count > REGRESSION_UNIQUE_THRESHOLD:
        return "regression"
    return "classification"


# ── Feature preprocessing ──────────────────────────────────────────────────────

def preprocess_features(X: pd.DataFrame):
    """
    Clean, impute, encode, and scale the feature matrix.

    Returns
    -------
    X_scaled    : np.ndarray  — scaled feature array ready for sklearn
    feature_cols: list[str]   — column names after dropping high-missing cols
    encoders    : dict        — {col: LabelEncoder} for categorical columns
    feature_info: dict        — per-column metadata for the prediction form
    scaler      : StandardScaler fitted on X
    X_clean     : pd.DataFrame — imputed + encoded (unscaled) DataFrame
    """
    X = X.copy()

    # 1. Drop columns that are mostly missing
    drop_cols = [c for c in X.columns if X[c].isnull().mean() > MAX_MISSING_DROP]
    X.drop(columns=drop_cols, inplace=True)
    feature_cols = X.columns.tolist()

    # 2. Impute remaining nulls
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            mode_val = X[c].mode()
            X[c] = X[c].fillna(mode_val[0] if len(mode_val) else "unknown")

    # 3. Encode categoricals; collect metadata for the UI form
    encoders: dict = {}
    feature_info: dict = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            feature_info[c] = {
                "kind": "numeric",
                "min":  float(X[c].min()),
                "max":  float(X[c].max()),
                "mean": float(X[c].mean()),
            }
        else:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            encoders[c] = le
            feature_info[c] = {
                "kind":   "categorical",
                "values": le.classes_.tolist(),
            }

    # 4. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, encoders, feature_info, scaler, X


# ── Target encoding ────────────────────────────────────────────────────────────

def encode_target(y_raw: pd.Series, task: str):
    """
    Encode the target series into a numpy array suitable for sklearn.

    Returns
    -------
    y              : np.ndarray
    target_encoder : LabelEncoder | None  (None for regression or numeric classes)
    classes        : list[str] | None
    """
    is_numeric = pd.api.types.is_numeric_dtype(y_raw)

    if task == "classification":
        if not is_numeric:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_raw.fillna("unknown").astype(str))
            classes = target_encoder.classes_.tolist()
        else:
            target_encoder = None
            y = y_raw.fillna(y_raw.mode()[0]).astype(int).values
            classes = sorted(y_raw.dropna().unique().astype(int).astype(str).tolist())
    else:
        target_encoder = None
        y = y_raw.fillna(y_raw.median()).values
        classes = None

    return y, target_encoder, classes


# ── Single-row encoding for inference ─────────────────────────────────────────

def encode_input_row(row_raw: dict, feature_cols: list,
                     feature_info: dict, encoders: dict,
                     scaler: StandardScaler) -> tuple:
    """
    Convert a raw dict from the prediction form into a scaled numpy array.

    Returns
    -------
    X_row_scaled : np.ndarray shape (1, n_features)
    row          : list — unscaled numeric values (used by SHAP approximation)
    """
    row = []
    for c in feature_cols:
        val = row_raw.get(c, "")
        if feature_info[c]["kind"] == "numeric":
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(feature_info[c].get("mean", 0))
        else:
            le = encoders.get(c)
            if le is not None:
                try:
                    row.append(int(le.transform([str(val)])[0]))
                except Exception:
                    row.append(0)
            else:
                row.append(0)

    X_row = np.array(row).reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)
    return X_row_scaled, row
