"""
CareerForge-AI: Full ML training pipeline.
Trains classifiers (recommended_career, alt_career_1/2, skill_gaps) and
regressors (readiness_score, est_weeks_to_ready). Saves models and supports
feature importance visualization.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib

# Local modules
from ml_config import (
    DATA_PATH,
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
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    CLASSIFICATION_ESTIMATOR_PARAMS,
    REGRESSION_ESTIMATOR_PARAMS,
)
from ml_preprocessing import CareerForgePreprocessor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Data loading (memory-efficient for ~50MB)
# ---------------------------------------------------------------------------
def load_data(path: Path, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load CSV with optional sampling for quick runs. For 50MB, full load is fine.
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, low_memory=False)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing critical targets so we have valid labels for training.
    Fill missing skill_gap_1/2/3 with "Unknown" for multi-output model.
    """
    required = [TARGET_RECOMMENDED_CAREER, TARGET_READINESS, TARGET_WEEKS_READY]
    for col in required:
        if col in df.columns:
            df = df.dropna(subset=[col])
    # Clean string targets: drop empty or "nan" string for career targets
    for col in [TARGET_RECOMMENDED_CAREER, TARGET_ALT_CAREER_1, TARGET_ALT_CAREER_2]:
        if col in df.columns:
            df = df[df[col].astype(str).str.strip().str.lower() != ""]
            df = df[df[col].astype(str).str.lower() != "nan"]
    # Skill gaps: fill missing with "Unknown" so multi-output has valid labels
    for col in TARGET_SKILL_GAPS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            df.loc[df[col].str.strip().str.lower().isin(["", "nan"]), col] = "Unknown"
    df = df.reset_index(drop=True)
    print(f"After dropping missing targets: {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Train / evaluate classification
# ---------------------------------------------------------------------------
def train_classifier(X_train: np.ndarray, y_train: np.ndarray, name: str):
    clf = RandomForestClassifier(**CLASSIFICATION_ESTIMATOR_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray, target_name: str) -> dict:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"  {target_name} — Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc, "f1_weighted": f1}


# ---------------------------------------------------------------------------
# Train / evaluate multi-output classification (skill_gap_1, 2, 3)
# ---------------------------------------------------------------------------
def train_multioutput_classifier(X_train: np.ndarray, Y_train: np.ndarray):
    base = RandomForestClassifier(**CLASSIFICATION_ESTIMATOR_PARAMS)
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, Y_train)
    return clf


def evaluate_multioutput(clf, X_test: np.ndarray, Y_test: np.ndarray, target_names: list) -> dict:
    Y_pred = clf.predict(X_test)
    accs = []
    for i, name in enumerate(target_names):
        acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
        accs.append(acc)
        print(f"  {name} — Accuracy: {acc:.4f}")
    return {"accuracy_per_output": accs, "mean_accuracy": np.mean(accs)}


# ---------------------------------------------------------------------------
# Train / evaluate regression
# ---------------------------------------------------------------------------
def train_regressor(X_train: np.ndarray, y_train: np.ndarray):
    reg = RandomForestRegressor(**REGRESSION_ESTIMATOR_PARAMS)
    reg.fit(X_train, y_train)
    return reg


def evaluate_regressor(reg, X_test: np.ndarray, y_test: np.ndarray, target_name: str) -> dict:
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  {target_name} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Label encoding for classification targets (string -> int)
# ---------------------------------------------------------------------------
def fit_label_encoders(df: pd.DataFrame) -> dict:
    """Fit label encoders for each classification target; return encoders and encoded y."""
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in [TARGET_RECOMMENDED_CAREER, TARGET_ALT_CAREER_1, TARGET_ALT_CAREER_2] + TARGET_SKILL_GAPS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode_labels(df: pd.DataFrame, encoders: dict) -> dict:
    """Encode string labels to integers using fitted encoders."""
    out = {}
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        vals = df[col].astype(str)
        mask = vals.isin(le.classes_)
        out[col] = np.where(mask, le.transform(vals), 0)
    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(importance: np.ndarray, feature_names: list, title: str, save_path: Path, top_n: int = 25):
    """Save a horizontal bar plot of feature importance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping feature importance plot)")
        return
    idx = np.argsort(importance)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]
    fig, ax = plt.subplots(figsize=(8, min(10, top_n * 0.35)))
    ax.barh(range(len(names)), vals, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(data_path: Path = DATA_PATH, sample_frac: float = 1.0, save_plots: bool = True) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, sample_frac=sample_frac)
    df = prepare_targets(df)

    # Build feature matrix using preprocessor (numeric + categorical + multi-label)
    preprocessor = CareerForgePreprocessor()
    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.feature_names_
    preprocessor.save(PREPROCESSOR_PATH)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    df_train = df.loc[idx_train].reset_index(drop=True)
    df_test = df.loc[idx_test].reset_index(drop=True)

    # Label encoders for classification targets
    encoders = fit_label_encoders(df_train)
    joblib.dump(encoders, ARTIFACTS_DIR / "label_encoders.joblib")

    # ----- 1. recommended_career (primary classification) -----
    print("\n--- Recommended career (classification) ---")
    y_career_train = encoders[TARGET_RECOMMENDED_CAREER].transform(df_train[TARGET_RECOMMENDED_CAREER].astype(str))
    y_career_test = encode_labels(df_test, encoders)[TARGET_RECOMMENDED_CAREER]
    clf_career = train_classifier(X_train, y_career_train, TARGET_RECOMMENDED_CAREER)
    evaluate_classifier(clf_career, X_test, y_career_test, TARGET_RECOMMENDED_CAREER)
    cv_scores = cross_val_score(clf_career, X_train, y_career_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
    print(f"  CV accuracy ({CV_FOLDS}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    joblib.dump(clf_career, CAREER_MODEL_PATH)
    if save_plots and hasattr(clf_career, "feature_importances_"):
        plot_feature_importance(
            clf_career.feature_importances_, feature_names,
            "Feature importance: recommended_career",
            ARTIFACTS_DIR / "importance_recommended_career.png",
        )

    # ----- 2. alt_career_1, alt_career_2 (optional) -----
    for col, path in [(TARGET_ALT_CAREER_1, ALT_CAREER_1_PATH), (TARGET_ALT_CAREER_2, ALT_CAREER_2_PATH)]:
        if col not in encoders:
            continue
        print(f"\n--- {col} (classification) ---")
        y_train = encoders[col].transform(df_train[col].astype(str))
        y_test = encode_labels(df_test, encoders)[col]
        clf = train_classifier(X_train, y_train, col)
        evaluate_classifier(clf, X_test, y_test, col)
        joblib.dump(clf, path)
        if save_plots and hasattr(clf, "feature_importances_"):
            plot_feature_importance(
                clf.feature_importances_, feature_names,
                f"Feature importance: {col}",
                ARTIFACTS_DIR / f"importance_{col}.png",
            )

    # ----- 3. skill_gap_1, skill_gap_2, skill_gap_3 (multi-output) -----
    print("\n--- Skill gaps (multi-output classification) ---")
    Y_skill_train = np.column_stack([
        encoders[c].transform(df_train[c].astype(str)) for c in TARGET_SKILL_GAPS if c in encoders
    ])
    Y_skill_test = np.column_stack([
        encode_labels(df_test, encoders)[c] for c in TARGET_SKILL_GAPS if c in encoders
    ])
    clf_skill = train_multioutput_classifier(X_train, Y_skill_train)
    evaluate_multioutput(clf_skill, X_test, Y_skill_test, TARGET_SKILL_GAPS)
    joblib.dump(clf_skill, SKILL_GAP_MODEL_PATH)

    # ----- 4. readiness_score (regression) -----
    print("\n--- Readiness score (regression) ---")
    y_read_train = df_train[TARGET_READINESS].values
    y_read_test = df_test[TARGET_READINESS].values
    reg_read = train_regressor(X_train, y_read_train)
    evaluate_regressor(reg_read, X_test, y_read_test, TARGET_READINESS)
    cv_rmse = cross_val_score(reg_read, X_train, y_read_train, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"  CV RMSE ({CV_FOLDS}-fold): {-cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    joblib.dump(reg_read, READINESS_MODEL_PATH)
    if save_plots and hasattr(reg_read, "feature_importances_"):
        plot_feature_importance(
            reg_read.feature_importances_, feature_names,
            "Feature importance: readiness_score",
            ARTIFACTS_DIR / "importance_readiness_score.png",
        )

    # ----- 5. est_weeks_to_ready (regression) -----
    print("\n--- Est. weeks to ready (regression) ---")
    y_weeks_train = df_train[TARGET_WEEKS_READY].values.astype(float)
    y_weeks_test = df_test[TARGET_WEEKS_READY].values.astype(float)
    reg_weeks = train_regressor(X_train, y_weeks_train)
    evaluate_regressor(reg_weeks, X_test, y_weeks_test, TARGET_WEEKS_READY)
    joblib.dump(reg_weeks, WEEKS_READY_MODEL_PATH)
    if save_plots and hasattr(reg_weeks, "feature_importances_"):
        plot_feature_importance(
            reg_weeks.feature_importances_, feature_names,
            "Feature importance: est_weeks_to_ready",
            ARTIFACTS_DIR / "importance_est_weeks_to_ready.png",
        )

    print("\n--- Done. Artifacts saved under:", ARTIFACTS_DIR)


if __name__ == "__main__":
    sample = 1.0
    if len(sys.argv) > 1:
        try:
            sample = float(sys.argv[1])
        except ValueError:
            pass
    run_pipeline(data_path=DATA_PATH, sample_frac=sample, save_plots=True)
