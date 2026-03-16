"""
Configuration for CareerForge-AI ML pipeline.
Column definitions, paths, and model hyperparameters.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "career_guidance_perfect.csv"
ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Saved artifact filenames
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
CAREER_MODEL_PATH = ARTIFACTS_DIR / "career_classifier.joblib"
ALT_CAREER_1_PATH = ARTIFACTS_DIR / "alt_career_1_classifier.joblib"
ALT_CAREER_2_PATH = ARTIFACTS_DIR / "alt_career_2_classifier.joblib"
SKILL_GAP_MODEL_PATH = ARTIFACTS_DIR / "skill_gap_classifier.joblib"
READINESS_MODEL_PATH = ARTIFACTS_DIR / "readiness_regressor.joblib"
WEEKS_READY_MODEL_PATH = ARTIFACTS_DIR / "weeks_ready_regressor.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
LABEL_ENCODERS_PATH = ARTIFACTS_DIR / "label_encoders.joblib"

# ---------------------------------------------------------------------------
# Column groups (must match CSV)
# ---------------------------------------------------------------------------
ID_COLUMN = "student_id"

# Numeric features (will be imputed with median and optionally scaled)
NUMERIC_FEATURES = [
    "age",
    "cgpa",
    "backlogs",
    "attendance_percentage",
    "target_salary_lpa",
    "internships_count",
    "projects_count",
    "certifications_count",
    "online_course_count",
    "hackathon_count",
    "github_activity_score",
    "resume_score",
    "aptitude_test_score",
    "coding_skill_score",
    "ml_knowledge_score",
    "problem_solving_score",
    "teamwork_score",
    "communication_score",
    "leadership_score",
    "mock_interview_score",
    "extracurricular_score",
    "sleep_hours",
    "study_hours_per_day",
    "match_confidence",
]

# Categorical features (one-hot or target encoding)
CATEGORICAL_FEATURES = [
    "gender",
    "branch",
    "college_tier",
    "interest_domain",
    "industry_preference",
    "preferred_work_style",
    "location_preference",
    "learning_style",
    "tool_proficiency",
    "internship_domain",
    "college_club_role",
    "volunteer_experience",
    "placement_status",
    "linkedin_profile",
]

# Multi-label features (semicolon-separated; use MultiLabelBinarizer)
MULTILABEL_FEATURES = [
    "programming_languages",
    "project_domains",
    "certification_names",
]

# Target columns
TARGET_RECOMMENDED_CAREER = "recommended_career"
TARGET_ALT_CAREER_1 = "alt_career_1"
TARGET_ALT_CAREER_2 = "alt_career_2"
TARGET_SKILL_GAPS = ["skill_gap_1", "skill_gap_2", "skill_gap_3"]
TARGET_READINESS = "readiness_score"
TARGET_WEEKS_READY = "est_weeks_to_ready"

# Columns to drop from features (IDs, targets, or high-cardinality free text)
DROP_FROM_FEATURES = [
    ID_COLUMN,
    "salary_package_lpa",  # leakage / post-placement
    "learning_roadmap_url",  # URL, not a feature
    "peer_benchmark_percentile",  # can keep or drop
    "gap_severity",  # derived; optional to keep
] + [TARGET_RECOMMENDED_CAREER, TARGET_ALT_CAREER_1, TARGET_ALT_CAREER_2] + TARGET_SKILL_GAPS + [TARGET_READINESS, TARGET_WEEKS_READY]

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# For large dataset: use n_jobs=-1 and moderate tree depth
CLASSIFICATION_ESTIMATOR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

REGRESSION_ESTIMATOR_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Max number of one-hot categories to keep (rest go to "other")
MAX_CARDINALITY_ONEHOT = 50
