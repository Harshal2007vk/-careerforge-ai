import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = "career_guidance_perfect.csv"
MODEL_PATH = "career_model.pkl"


NUMERIC_FEATURES = [
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


@dataclass
class CareerRecommenderArtifacts:
    career_df: pd.DataFrame
    feature_matrix: np.ndarray
    feature_columns: List[str]
    scaler: MinMaxScaler


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["recommended_career", "match_confidence"])
    df = df.reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> (np.ndarray, MinMaxScaler, List[str]):
    features = [col for col in NUMERIC_FEATURES if col in df.columns]
    X = df[features].fillna(df[features].median())
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, features


def train_and_save() -> None:
    df = load_raw_data()
    feature_matrix, scaler, feature_columns = build_feature_matrix(df)

    artifacts = CareerRecommenderArtifacts(
        career_df=df[
            [
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
        ].copy(),
        feature_matrix=feature_matrix,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Recommender artifacts saved to {MODEL_PATH}")


def recommend_for_profile(profile: Dict[str, Any], top_k: int = 3) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        artifacts: CareerRecommenderArtifacts = pickle.load(f)

    user_vec = np.zeros(len(artifacts.feature_columns))
    for i, col in enumerate(artifacts.feature_columns):
        user_vec[i] = profile.get(col, artifacts.career_df[col].median())

    user_scaled = artifacts.scaler.transform(user_vec.reshape(1, -1))
    sims = cosine_similarity(user_scaled, artifacts.feature_matrix)[0]

    result = artifacts.career_df.copy()
    result["similarity"] = sims
    result = result.sort_values(["similarity", "match_confidence"], ascending=False)
    return result.head(top_k)


if __name__ == "__main__":
    train_and_save()