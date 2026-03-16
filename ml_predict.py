"""
ml_predict.py  —  CareerForge AI
Cosine-similarity recommender with confidence calibration, gap scoring,
peer benchmarking, and placement probability estimation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from ml_preprocessing import (
    ALL_MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
    engineer_features,
    profile_to_vector,
    validate_profile,
)

logger = logging.getLogger(__name__)

DATA_PATH = Path("career_guidance_perfect.csv")
MODEL_PATH = Path("career_model.pkl")

# Columns stored in the artifact DataFrame
_STORE_COLS = [
    "student_id",
    "recommended_career",
    "alt_career_1",
    "alt_career_2",
    "match_confidence",
    "skill_gap_1",
    "skill_gap_2",
    "skill_gap_3",
    "gap_severity",
    "readiness_score",
    "est_weeks_to_ready",
    "peer_benchmark_percentile",
    "learning_roadmap_url",
]

# Career → required skill weights for gap scoring (0–100)
CAREER_SKILL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AI / ML Engineer": {
        "coding_skill_score": 0.30, "problem_solving_score": 0.25,
        "aptitude_test_score": 0.20, "communication_score": 0.10,
        "leadership_score": 0.05, "extracurricular_score": 0.05,
        "teamwork_score": 0.05,
    },
    "Data Scientist": {
        "coding_skill_score": 0.25, "aptitude_test_score": 0.25,
        "problem_solving_score": 0.20, "communication_score": 0.15,
        "teamwork_score": 0.10, "extracurricular_score": 0.05,
    },
    "Full Stack Developer": {
        "coding_skill_score": 0.35, "problem_solving_score": 0.25,
        "aptitude_test_score": 0.15, "communication_score": 0.10,
        "teamwork_score": 0.10, "leadership_score": 0.05,
    },
    "Cloud Engineer": {
        "coding_skill_score": 0.25, "problem_solving_score": 0.25,
        "aptitude_test_score": 0.20, "leadership_score": 0.10,
        "communication_score": 0.10, "teamwork_score": 0.10,
    },
    "DevOps Engineer": {
        "coding_skill_score": 0.30, "problem_solving_score": 0.25,
        "aptitude_test_score": 0.20, "teamwork_score": 0.10,
        "communication_score": 0.10, "leadership_score": 0.05,
    },
    "Product Manager": {
        "communication_score": 0.30, "leadership_score": 0.25,
        "teamwork_score": 0.20, "aptitude_test_score": 0.15,
        "problem_solving_score": 0.10,
    },
    "Cybersecurity Analyst": {
        "coding_skill_score": 0.25, "problem_solving_score": 0.30,
        "aptitude_test_score": 0.20, "communication_score": 0.10,
        "teamwork_score": 0.10, "leadership_score": 0.05,
    },
    "Mobile App Developer": {
        "coding_skill_score": 0.35, "problem_solving_score": 0.25,
        "aptitude_test_score": 0.15, "communication_score": 0.10,
        "teamwork_score": 0.10, "extracurricular_score": 0.05,
    },
    "UI/UX Designer": {
        "extracurricular_score": 0.25, "communication_score": 0.25,
        "teamwork_score": 0.20, "problem_solving_score": 0.15,
        "leadership_score": 0.15,
    },
    "Blockchain Developer": {
        "coding_skill_score": 0.35, "problem_solving_score": 0.30,
        "aptitude_test_score": 0.20, "communication_score": 0.10,
        "teamwork_score": 0.05,
    },
}


# ── Artifact dataclass ────────────────────────────────────────────────────────

@dataclass
class CareerRecommenderArtifacts:
    career_df: pd.DataFrame
    feature_matrix: np.ndarray
    feature_columns: List[str]
    scaler: MinMaxScaler
    median_values: Dict[str, float] = field(default_factory=dict)

    def save(self, path: Path = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Artifacts saved → %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CareerRecommenderArtifacts":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Artifacts loaded ← %s", path)
        return obj


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> CareerRecommenderArtifacts:
    """Load CSV, build feature matrix, persist artifacts."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET_COL := "recommended_career", "match_confidence"])
    df = df.reset_index(drop=True)

    feature_matrix, scaler, feature_columns = build_feature_matrix(df)

    # Store median values for imputation at inference time
    medians: Dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            medians[col] = float(df[col].median())

    available_store_cols = [c for c in _STORE_COLS if c in df.columns]
    artifacts = CareerRecommenderArtifacts(
        career_df=df[available_store_cols].copy(),
        feature_matrix=feature_matrix,
        feature_columns=feature_columns,
        scaler=scaler,
        median_values=medians,
    )
    artifacts.save(model_path)
    return artifacts


# ── Inference helpers ─────────────────────────────────────────────────────────

def _calibrate_confidence(raw_similarity: float) -> float:
    """
    Sigmoid-stretch raw cosine similarity [0,1] → calibrated confidence [0,1].
    Prevents artificially high scores for mediocre matches.
    """
    x = (raw_similarity - 0.5) * 10          # centre & scale
    calibrated = 1.0 / (1.0 + np.exp(-x))    # sigmoid
    return float(np.clip(calibrated, 0.05, 0.98))


def _compute_skill_gap(
    profile: Dict[str, Any],
    career: str,
) -> Dict[str, Any]:
    """
    Compute per-skill gap scores and overall gap severity for a given career.
    Returns a dict with gap details.
    """
    weights = CAREER_SKILL_WEIGHTS.get(career, {})
    if not weights:
        return {"gap_severity": "Medium", "gaps": {}, "gap_score": 50.0}

    gaps: Dict[str, float] = {}
    weighted_deficit = 0.0

    for skill, weight in weights.items():
        current = float(profile.get(skill, 0.0))
        # Target is 80th percentile (80/100) for senior roles, 70 for others
        target = 80.0 if career in {
            "AI / ML Engineer", "Data Scientist", "Product Manager"
        } else 70.0
        deficit = max(0.0, target - current)
        gaps[skill] = round(deficit, 1)
        weighted_deficit += deficit * weight

    gap_score = min(100.0, weighted_deficit)
    if gap_score < 20:
        severity = "Low"
    elif gap_score < 45:
        severity = "Medium"
    else:
        severity = "High"

    top_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        "gap_severity": severity,
        "gap_score": round(gap_score, 1),
        "gaps": dict(top_gaps),
        "top_gap_skills": [g[0] for g in top_gaps],
    }


def _estimate_placement_probability(profile: Dict[str, Any]) -> float:
    """
    Rule-based placement probability (0–100) using weighted profile signals.
    Mirrors the JS predictPlacement() but more granular.
    """
    score = 35.0

    cgpa = float(profile.get("cgpa", 0))
    if cgpa >= 9.0:   score += 20
    elif cgpa >= 8.0: score += 14
    elif cgpa >= 7.0: score += 8
    elif cgpa >= 6.0: score += 3

    coding = float(profile.get("coding_skill_score", 0))
    if coding >= 80: score += 12
    elif coding >= 60: score += 7
    elif coding >= 40: score += 3

    internships = float(profile.get("internships_done", 0))
    score += min(internships * 6, 18)

    projects = float(profile.get("projects_count", 0))
    score += min(projects * 3, 15)

    mock = float(profile.get("mock_interview_score", 0))
    if mock >= 70: score += 8
    elif mock >= 50: score += 4

    if str(profile.get("hackathon_participation", "")).lower() in ("yes", "1", "true"):
        score += 6

    backlogs = float(profile.get("backlogs", 0))
    score -= min(backlogs * 3, 15)

    return round(float(np.clip(score, 10.0, 97.0)), 1)


def _estimate_weeks_to_ready(
    profile: Dict[str, Any],
    gap_score: float,
    placement_prob: float,
) -> int:
    """Estimate weeks of focused study to become job-ready."""
    base_weeks = max(4, int(gap_score / 5))
    if placement_prob >= 75:
        base_weeks = max(4, base_weeks - 4)
    elif placement_prob < 50:
        base_weeks += 6
    return base_weeks


def _peer_percentile(
    profile: Dict[str, Any],
    career_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_cols: List[str],
    scaler: MinMaxScaler,
    median_fallback: Dict[str, float],
) -> int:
    """
    Compute approximate peer benchmark percentile using the already-loaded matrix.
    Returns integer 0–99.
    """
    user_vec = profile_to_vector(profile, feature_cols, scaler, median_fallback)
    sims = cosine_similarity(user_vec, feature_matrix)[0]
    rank = int(np.sum(sims < sims.mean()) / len(sims) * 100)
    return max(1, min(99, rank))


# ── Main inference entrypoint ─────────────────────────────────────────────────

def recommend_for_profile(
    profile: Dict[str, Any],
    top_k: int = 3,
    artifacts: Optional[CareerRecommenderArtifacts] = None,
) -> Dict[str, Any]:
    """
    Full inference pipeline for a single student profile.

    Returns a rich dict:
    {
        "placement_probability": float,
        "peer_percentile": int,
        "recommendations": [
            {
                "rank": 1,
                "career": str,
                "confidence": float,        # calibrated 0–1
                "confidence_pct": int,      # 0–100
                "alt_careers": [str, str],
                "gap_severity": str,
                "gap_score": float,
                "top_gap_skills": [str],
                "est_weeks_to_ready": int,
                "readiness_score": float,
            },
            ...
        ],
        "warnings": [str],
    }
    """
    if artifacts is None:
        artifacts = CareerRecommenderArtifacts.load()

    # Validate
    _, warnings = validate_profile(profile)

    # Build user vector
    user_vec = profile_to_vector(
        profile,
        artifacts.feature_columns,
        artifacts.scaler,
        artifacts.median_values,
    )

    # Cosine similarity against entire training set
    sims = cosine_similarity(user_vec, artifacts.feature_matrix)[0]

    df = artifacts.career_df.copy()
    df["_sim"] = sims

    # Score = 60% similarity + 40% stored match_confidence (normalised)
    if "match_confidence" in df.columns:
        conf_norm = df["match_confidence"] / 100.0
        df["_score"] = df["_sim"] * 0.60 + conf_norm * 0.40
    else:
        df["_score"] = df["_sim"]

    top_rows = df.nlargest(top_k, "_score")

    placement_prob = _estimate_placement_probability(profile)
    peer_pct = _peer_percentile(
        profile,
        artifacts.career_df,
        artifacts.feature_matrix,
        artifacts.feature_columns,
        artifacts.scaler,
        artifacts.median_values,
    )

    recommendations = []
    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        career = str(row.get("recommended_career", "Unknown"))
        raw_sim = float(row["_sim"])
        calibrated = _calibrate_confidence(raw_sim)

        gap_info = _compute_skill_gap(profile, career)
        est_weeks = _estimate_weeks_to_ready(
            profile, gap_info["gap_score"], placement_prob
        )

        readiness = float(row.get("readiness_score", 0)) or round(
            (calibrated * 60) + (placement_prob * 0.40), 1
        )

        recommendations.append(
            {
                "rank": rank,
                "career": career,
                "confidence": round(calibrated, 3),
                "confidence_pct": int(calibrated * 100),
                "alt_careers": [
                    str(row.get("alt_career_1", "")),
                    str(row.get("alt_career_2", "")),
                ],
                "gap_severity": gap_info["gap_severity"],
                "gap_score": gap_info["gap_score"],
                "top_gap_skills": gap_info["top_gap_skills"],
                "skill_gaps": {
                    k: v
                    for k, v in (
                        (row.get("skill_gap_1"), None),
                        (row.get("skill_gap_2"), None),
                        (row.get("skill_gap_3"), None),
                    )
                    if k and str(k) not in ("nan", "None", "")
                },
                "est_weeks_to_ready": est_weeks,
                "est_months_to_ready": round(est_weeks / 4.3, 1),
                "readiness_score": round(readiness, 1),
                "peer_benchmark_percentile": int(
                    row.get("peer_benchmark_percentile", peer_pct) or peer_pct
                ),
                "learning_roadmap_url": str(row.get("learning_roadmap_url", "")),
            }
        )

    return {
        "placement_probability": placement_prob,
        "peer_percentile": peer_pct,
        "recommendations": recommendations,
        "warnings": warnings,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    sample = {
        "age": 21, "cgpa": 7.8, "backlogs": 0,
        "attendance_percentage": 82,
        "aptitude_test_score": 70, "coding_skill_score": 65,
        "problem_solving_score": 72, "teamwork_score": 68,
        "communication_score": 70, "leadership_score": 55,
        "mock_interview_score": 60, "extracurricular_score": 50,
        "internships_done": 2, "projects_count": 4,
        "hackathon_participation": "Yes",
    }

    result = recommend_for_profile(sample)
    print(json.dumps(result, indent=2))
