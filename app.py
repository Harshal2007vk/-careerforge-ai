import pickle
from typing import Dict, Any

import numpy as np

# Required for pickle to load career_model.pkl (saved when train_model.py was run as __main__)
from train_model import CareerRecommenderArtifacts  # noqa: F401

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="CareerForge AI", page_icon="🎯", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("career_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


def build_user_profile(form_values: Dict[str, Any], artifacts) -> np.ndarray:
    user_vec = np.zeros(len(artifacts.feature_columns))
    for i, col in enumerate(artifacts.feature_columns):
        if col in form_values:
            user_vec[i] = form_values[col]
        else:
            user_vec[i] = artifacts.career_df[col].median()
    return artifacts.scaler.transform(user_vec.reshape(1, -1))


def compute_recommendations(user_scaled: np.ndarray, artifacts, top_k: int = 3) -> pd.DataFrame:
    sims = cosine_similarity(user_scaled, artifacts.feature_matrix)[0]
    result = artifacts.career_df.copy()
    result["similarity"] = sims
    result = result.sort_values(["similarity", "match_confidence"], ascending=False)
    return result.head(top_k)


def main() -> None:
    artifacts = load_artifacts()

    st.sidebar.title("CareerForge-AI")
    st.sidebar.markdown(
        "AI-powered **career guidance** with **skill-gap analysis** and "
        "a tailored **learning roadmap** built on real student data."
    )

    st.title("CareerForge-AI: Smart Career & Skill Navigator")
    st.markdown(
        "Tell us a bit about your academic profile and skills. "
        "We will recommend careers, highlight your strengths, and show what to learn next."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=40, value=21)
        cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
        backlogs = st.number_input("Backlogs (current / past)", min_value=0, max_value=10, value=0)
        attendance = st.slider("Attendance %", 40.0, 100.0, 80.0, 0.5)
    with col2:
        aptitude = st.slider("Aptitude test score", 0.0, 100.0, 60.0, 1.0)
        coding = st.slider("Coding skill score", 0.0, 100.0, 60.0, 1.0)
        problem_solving = st.slider("Problem solving score", 0.0, 100.0, 60.0, 1.0)
        communication = st.slider("Communication score", 0.0, 100.0, 60.0, 1.0)
    with col3:
        teamwork = st.slider("Teamwork score", 0.0, 100.0, 60.0, 1.0)
        leadership = st.slider("Leadership score", 0.0, 100.0, 50.0, 1.0)
        mock_interview = st.slider("Mock interview score", 0.0, 100.0, 60.0, 1.0)
        extracurricular = st.slider("Extracurricular score", 0.0, 100.0, 50.0, 1.0)

    if st.button("Analyze my profile", type="primary"):
        form_values = {
            "age": age,
            "cgpa": cgpa,
            "backlogs": backlogs,
            "attendance_percentage": attendance,
            "aptitude_test_score": aptitude,
            "coding_skill_score": coding,
            "problem_solving_score": problem_solving,
            "communication_score": communication,
            "teamwork_score": teamwork,
            "leadership_score": leadership,
            "mock_interview_score": mock_interview,
            "extracurricular_score": extracurricular,
        }

        user_scaled = build_user_profile(form_values, artifacts)
        recs = compute_recommendations(user_scaled, artifacts, top_k=3)

        st.subheader("Recommended career paths")
        st.dataframe(
            recs[
                [
                    "recommended_career",
                    "alt_career_1",
                    "alt_career_2",
                    "match_confidence",
                    "readiness_score",
                    "gap_severity",
                    "similarity",
                ]
            ].rename(
                columns={
                    "recommended_career": "Primary career",
                    "alt_career_1": "Option 2",
                    "alt_career_2": "Option 3",
                    "match_confidence": "Model confidence",
                    "readiness_score": "Readiness score",
                    "gap_severity": "Gap severity",
                    "similarity": "Profile match",
                }
            ),
            use_container_width=True,
        )

        top_row = recs.iloc[0]
        st.subheader("Skill-gap analysis for your top match")
        skills_have = []
        skills_need = []
        for col in ["skill_gap_1", "skill_gap_2", "skill_gap_3"]:
            skill = str(top_row.get(col, "")).strip()
            if skill and skill.lower() != "none":
                skills_need.append(skill)

        if skills_need:
            gap_df = pd.DataFrame(
                {
                    "Skill": skills_need,
                    "Status": ["Missing / Improve"] * len(skills_need),
                }
            )
            st.table(gap_df)
        else:
            st.info("No major skill gaps detected for this role in the dataset.")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Readiness score", f"{top_row['readiness_score']:.1f} / 100")
        with metric_col2:
            st.metric("Weeks to job-ready", int(top_row["est_weeks_to_ready"]))
        with metric_col3:
            st.metric("Peer benchmark percentile", f"{top_row['peer_benchmark_percentile']:.0f}th")

        st.subheader("Learning roadmap")
        st.markdown(
            f"Based on your gaps, we recommend following a roadmap like: "
            f"[View roadmap]({top_row['learning_roadmap_url']})."
        )


if __name__ == "__main__":
    main()