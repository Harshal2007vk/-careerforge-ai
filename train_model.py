"""
CareerForge AI — ML Model Trainer
Trains a Random Forest classifier on synthetic placement data
Run: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import os

np.random.seed(42)

# ─────────────────────────────────────────
#  GENERATE SYNTHETIC PLACEMENT DATASET
#  (Based on real patterns from Indian college placement data)
# ─────────────────────────────────────────
def generate_dataset(n=5000):
    data = []

    for _ in range(n):
        # Academic
        cgpa = round(np.random.uniform(5.0, 10.0), 1)
        internships = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.25, 0.30, 0.25, 0.12, 0.05, 0.03])
        projects = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02])
        hackathon = np.random.choice([0, 1], p=[0.55, 0.45])
        coding_skill = np.random.randint(10, 101)
        comm_skill = np.random.randint(10, 101)
        skills_count = np.random.randint(0, 15)
        branch_cs = np.random.choice([0, 1], p=[0.45, 0.55])  # 1 = CS/IT/DS/AI

        # Placement probability logic (mimics real patterns)
        score = 30
        score += (cgpa - 5.0) * 6          # CGPA weight
        score += internships * 7            # Internship weight
        score += min(projects * 4, 20)      # Projects (capped)
        score += hackathon * 5              # Hackathon bonus
        score += (coding_skill / 100) * 15  # Coding skill
        score += (comm_skill / 100) * 8     # Communication
        score += min(skills_count * 1.5, 12) # Skills
        score += branch_cs * 8              # CS branch bonus
        score += np.random.normal(0, 8)     # Real-world noise

        placed = 1 if score >= 55 else 0

        data.append({
            'cgpa': cgpa,
            'internships': internships,
            'projects': projects,
            'hackathon': hackathon,
            'coding_skill': coding_skill,
            'comm_skill': comm_skill,
            'skills_count': skills_count,
            'branch_cs': branch_cs,
            'placed': placed
        })

    df = pd.DataFrame(data)
    print(f"Dataset generated: {len(df)} rows")
    print(f"Placement rate: {df['placed'].mean()*100:.1f}%")
    return df


# ─────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────
def train():
    print("\n🔧 Generating training data...")
    df = generate_dataset(5000)

    features = ['cgpa', 'internships', 'projects', 'hackathon',
                'coding_skill', 'comm_skill', 'skills_count', 'branch_cs']
    X = df[features]
    y = df['placed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
    print(f"✅ Cross-Val Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    print("\n🔑 Feature Importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Save model
    os.makedirs('model', exist_ok=True)
    with open('model/placement_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    meta = {
        'features': features,
        'accuracy': round(accuracy * 100, 2),
        'cv_score': round(cv_scores.mean() * 100, 2),
        'feature_importance': {k: round(v, 4) for k, v in importance.items()},
        'training_samples': len(X_train),
        'model_type': 'RandomForestClassifier'
    }
    with open('model/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n💾 Model saved to model/placement_model.pkl")
    print("💾 Metadata saved to model/model_meta.json")
    print("\n🚀 Now run: python app.py")
    return model


if __name__ == '__main__':
    train()
