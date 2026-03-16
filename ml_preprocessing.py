"""
ml_preprocessing.py  —  CareerForge AI
Robust preprocessing: validation, composite feature engineering, median imputation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Raw feature columns ───────────────────────────────────────────────────────

NUMERIC_FEATURES: List[str] = [
    "age",
    "cgpa",
    "backlogs",
    "attendance_percentage",
    "aptitude_test_score",
    "coding_skill_score",
    "problem_solving_score",
    "teamwork_score",
    "communication_score",
    "leadership_score",
    "mock_interview_score",
    "extracurricular_score",
]

TARGET_COL = "recommended_career"
CONFIDENCE_COL = "match_confidence"

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age":                    (16.0, 35.0),
    "cgpa":                   (0.0,  10.0),
    "backlogs":               (0.0,  20.0),
    "attendance_percentage":  (0.0, 100.0),
    "aptitude_test_score":    (0.0, 100.0),
    "coding_skill_score":     (0.0, 100.0),
    "problem_solving_score":  (0.0, 100.0),
    "teamwork_score":         (0.0, 100.0),
    "communication_score":    (0.0, 100.0),
    "leadership_score":       (0.0, 100.0),
    "mock_interview_score":   (0.0, 100.0),
    "extracurricular_score":  (0.0, 100.0),
}

# Derived composite columns added by engineer_features()
COMPOSITE_FEATURES: List[str] = [
    "technical_composite",
    "soft_skills_composite",
    "academic_composite",
    "experience_bonus",
]

# Full model feature set (raw + composite)
ALL_MODEL_FEATURES: List[str] = NUMERIC_FEATURES + COMPOSITE_FEATURES


# ── Validation ────────────────────────────────────────────────────────────────

def validate_profile(profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a raw student profile dict.
    Returns (is_valid, warnings). Warnings are non-fatal; imputation handles them.
    """
    warnings: List[str] = []
    for feat, (lo, hi) in FEATURE_BOUNDS.items():
        raw = profile.get(feat)
        if raw is None:
            warnings.append(f"'{feat}' missing — median imputation will be used.")
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"'{feat}' is not numeric ({raw!r}) — will use median.")
            continue
        if not (lo <= v <= hi):
            warnings.append(
                f"'{feat}' value {v} outside expected range [{lo}, {hi}]."
            )
    return True, warnings


# ── Composite feature engineering ────────────────────────────────────────────

def _f(d: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def engineer_features(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compute composite features from raw profile. Returns an enriched copy."""
    p = dict(profile)

    coding      = _f(p, "coding_skill_score")
    problem     = _f(p, "problem_solving_score")
    aptitude    = _f(p, "aptitude_test_score")
    comm        = _f(p, "communication_score")
    team        = _f(p, "teamwork_score")
    lead        = _f(p, "leadership_score")
    extra       = _f(p, "extracurricular_score")
    cgpa        = _f(p, "cgpa")
    attendance  = _f(p, "attendance_percentage")
    backlogs    = _f(p, "backlogs")
    internships = _f(p, "internships_done")
    projects    = _f(p, "projects_count")
    hackathon   = 1.0 if str(p.get("hackathon_participation", "")).lower() in (
        "yes", "1", "true"
    ) else 0.0
    mock        = _f(p, "mock_interview_score")

    p["technical_composite"] = (
        coding * 0.40 + problem * 0.35 + aptitude * 0.25
    )
    p["soft_skills_composite"] = (
        comm * 0.35 + team * 0.30 + lead * 0.20 + extra * 0.15
    )
    backlog_penalty = max(0.0, 20.0 - backlogs * 2.0)
    p["academic_composite"] = (
        min(cgpa / 10.0 * 100, 100) * 0.60
        + min(attendance, 100) * 0.25
        + backlog_penalty * 0.15
    )
    p["experience_bonus"] = min(
        100.0,
        internships * 15.0 + projects * 8.0 + hackathon * 10.0 + mock * 0.30,
    )
    return p


# ── DataFrame-level helpers ───────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply engineer_features row-wise. NaNs filled with column medians first."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    enriched_rows = [engineer_features(row) for row in df.to_dict("records")]
    enriched_df = pd.DataFrame(enriched_rows)
    keep = [c for c in ALL_MODEL_FEATURES if c in enriched_df.columns]
    return enriched_df[keep]


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """Preprocess a raw DataFrame, scale it. Used during training only."""
    enriched = preprocess_dataframe(df)
    feature_cols = [c for c in ALL_MODEL_FEATURES if c in enriched.columns]
    X = enriched[feature_cols].values.astype(float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Feature matrix: %d rows x %d features", *X_scaled.shape)
    return X_scaled, scaler, feature_cols


def profile_to_vector(
    profile: Dict[str, Any],
    feature_cols: List[str],
    scaler: MinMaxScaler,
    median_fallback: Dict[str, float],
) -> np.ndarray:
    """Convert a single student profile dict → scaled numpy vector."""
    enriched = engineer_features(profile)
    vec = np.array(
        [float(enriched.get(col, median_fallback.get(col, 0.0))) for col in feature_cols],
        dtype=float,
    )
    return scaler.transform(vec.reshape(1, -1))
