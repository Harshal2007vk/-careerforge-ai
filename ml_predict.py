"""
Inference module for CareerForge-AI ML pipeline.
Load preprocessor and trained models to predict career, skill gaps, readiness, and weeks.
Ready for integration with Streamlit or Flask.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from ml_config import (
    ARTIFACTS_DIR,
    PREPROCESSOR_PATH,
    CAREER_MODEL_PATH,
    ALT_CAREER_1_PATH,
    ALT_CAREER_2_PATH,
    SKILL_GAP_MODEL_PATH,
    READINESS_MODEL_PATH,
    WEEKS_READY_MODEL_PATH,
    FEATURE_NAMES_PATH,
    TARGET_RECOMMENDED_CAREER,
    TARGET_ALT_CAREER_1,
    TARGET_ALT_CAREER_2,
    TARGET_SKILL_GAPS,
    TARGET_READINESS,
    TARGET_WEEKS_READY,
)
from ml_preprocessing import CareerForgePreprocessor


def load_artifacts(
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load preprocessor, label encoders, and all trained models.
    Returns a dict with keys: preprocessor, encoders, career_model, alt_career_1_model,
    alt_career_2_model, skill_gap_model, readiness_model, weeks_model, feature_names.
    """
    root = Path(artifacts_dir or ARTIFACTS_DIR)
    preprocessor = CareerForgePreprocessor.load(root / "preprocessor.joblib")
    encoders = joblib.load(root / "label_encoders.joblib")
    feature_names = json.loads((root / "feature_names.json").read_text(encoding="utf-8"))

    artifacts = {
        "preprocessor": preprocessor,
        "encoders": encoders,
        "feature_names": feature_names,
        "career_model": joblib.load(root / "career_classifier.joblib"),
        "alt_career_1_model": joblib.load(root / "alt_career_1_classifier.joblib") if (root / "alt_career_1_classifier.joblib").exists() else None,
        "alt_career_2_model": joblib.load(root / "alt_career_2_classifier.joblib") if (root / "alt_career_2_classifier.joblib").exists() else None,
        "skill_gap_model": joblib.load(root / "skill_gap_classifier.joblib"),
        "readiness_model": joblib.load(root / "readiness_regressor.joblib"),
        "weeks_model": joblib.load(root / "weeks_ready_regressor.joblib"),
    }
    return artifacts


def _ensure_dataframe(data: Any, input_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert dict or array to DataFrame with correct columns for preprocessor.
    If data is a dict and input_columns is provided, build a row with those columns,
    filling missing keys with NaN so the preprocessor can impute.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        row = dict(data)
        if input_columns:
            for col in input_columns:
                if col not in row:
                    row[col] = np.nan
            return pd.DataFrame([row])[input_columns]
        return pd.DataFrame([data])
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    raise TypeError("data must be DataFrame, dict, or ndarray")


def predict(
    data: pd.DataFrame | Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full prediction pipeline on one or more rows.
    data: DataFrame or single row as dict (keys = column names).
    Returns dict with: recommended_career, alt_career_1, alt_career_2, skill_gap_1/2/3,
    readiness_score, est_weeks_to_ready. Lists if multiple rows.
    """
    if artifacts is None:
        artifacts = load_artifacts(artifacts_dir)
    input_columns = getattr(artifacts["preprocessor"], "input_columns_", None) or artifacts.get("input_columns")
    df = _ensure_dataframe(data, input_columns=input_columns)
    X = artifacts["preprocessor"].transform(df)
    enc = artifacts["encoders"]

    out = {}
    # Classification: index -> label
    le_career = enc[TARGET_RECOMMENDED_CAREER]
    pred_career_idx = artifacts["career_model"].predict(X)
    out["recommended_career"] = le_career.inverse_transform(pred_career_idx.astype(int)).tolist()

    if artifacts.get("alt_career_1_model") and TARGET_ALT_CAREER_1 in enc:
        le1 = enc[TARGET_ALT_CAREER_1]
        out["alt_career_1"] = le1.inverse_transform(artifacts["alt_career_1_model"].predict(X).astype(int)).tolist()
    if artifacts.get("alt_career_2_model") and TARGET_ALT_CAREER_2 in enc:
        le2 = enc[TARGET_ALT_CAREER_2]
        out["alt_career_2"] = le2.inverse_transform(artifacts["alt_career_2_model"].predict(X).astype(int)).tolist()

    # Skill gaps (multi-output)
    Y_skill = artifacts["skill_gap_model"].predict(X)
    for i, col in enumerate(TARGET_SKILL_GAPS):
        if col in enc:
            out[col] = enc[col].inverse_transform(Y_skill[:, i].astype(int)).tolist()

    # Regression
    out[TARGET_READINESS] = artifacts["readiness_model"].predict(X).tolist()
    out[TARGET_WEEKS_READY] = np.round(artifacts["weeks_model"].predict(X)).astype(int).tolist()

    # If single row, return scalars
    if len(df) == 1:
        return {k: v[0] for k, v in out.items()}
    return out


def predict_single(profile: Dict[str, Any], artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience: predict for a single profile dict (e.g. from Flask/Streamlit form)."""
    return predict(profile, artifacts=artifacts)


# ---------------------------------------------------------------------------
# Streamlit/Flask usage example
# ---------------------------------------------------------------------------
# Streamlit:
#   from ml_predict import load_artifacts, predict_single
#   artifacts = load_artifacts()
#   result = predict_single({"age": 22, "cgpa": 8.0, ...})
#   st.write("Recommended career:", result["recommended_career"])
#
# Flask:
#   from ml_predict import load_artifacts, predict_single
#   artifacts = load_artifacts()
#   @app.route("/predict", methods=["POST"])
#   def predict_endpoint():
#       result = predict_single(request.json, artifacts=artifacts)
#       return jsonify(result)
# ---------------------------------------------------------------------------
