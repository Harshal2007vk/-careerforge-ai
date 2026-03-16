"""
train_model.py  —  CareerForge AI
Single-command training: loads CSV, builds feature matrix, saves artifacts.
Run: python train_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_save(
    data_path: str = "career_guidance_perfect.csv",
    model_path: str = "career_model.pkl",
) -> None:
    from ml_predict import train_and_save as _train

    data = Path(data_path)
    model = Path(model_path)

    if not data.exists():
        logger.error("Dataset not found at %s", data)
        sys.exit(1)

    logger.info("Starting training pipeline...")
    artifacts = _train(data_path=data, model_path=model)

    logger.info("Training complete.")
    logger.info("  Rows in training set : %d", len(artifacts.career_df))
    logger.info("  Feature columns      : %d", len(artifacts.feature_columns))
    logger.info("  Features used        : %s", ", ".join(artifacts.feature_columns))
    logger.info("  Model saved to       : %s", model_path)

    # Quick sanity check
    sample = {
        "age": 21, "cgpa": 7.5, "backlogs": 0,
        "attendance_percentage": 80,
        "aptitude_test_score": 65, "coding_skill_score": 60,
        "problem_solving_score": 65, "teamwork_score": 65,
        "communication_score": 65, "leadership_score": 55,
        "mock_interview_score": 55, "extracurricular_score": 50,
        "internships_done": 1, "projects_count": 3,
        "hackathon_participation": "Yes",
    }
    from ml_predict import recommend_for_profile
    result = recommend_for_profile(sample, top_k=3, artifacts=artifacts)
    logger.info("Sanity check — top recommendation: %s (%.0f%% confidence)",
                result["recommendations"][0]["career"],
                result["recommendations"][0]["confidence_pct"])
    logger.info("Placement probability for sample: %.1f%%", result["placement_probability"])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train CareerForge AI recommender")
    p.add_argument("--data",  default="career_guidance_perfect.csv", help="Path to CSV dataset")
    p.add_argument("--model", default="career_model.pkl",            help="Output model path")
    args = p.parse_args()
    train_and_save(args.data, args.model)
