# CareerForge-AI

CareerForge-AI is an AI-powered **career guidance and skill-gap analysis** prototype built for a 24-hour hackathon.
It analyzes a student's academic and skill profile and recommends suitable career paths with an estimated readiness
score, key missing skills, and a learning roadmap link.

## Tech stack

- Python
- Streamlit for the web UI
- pandas / numpy for data handling
- scikit-learn for feature scaling and similarity

## Project structure

- `career_guidance_perfect.csv` – synthetic student and career dataset used to drive recommendations.
- `train_model.py` – legacy: prepares the feature matrix and saves recommender artifacts into `career_model.pkl`.
- **ML pipeline (production):**
  - `ml_config.py` – column definitions, paths, and model hyperparameters.
  - `ml_preprocessing.py` – preprocessor (numeric, categorical, multi-label, missing values).
  - `train_pipeline.py` – full training: career/alt-career/skill-gap classifiers, readiness/weeks regressors; saves to `ml_artifacts/`.
  - `ml_predict.py` – inference: load artifacts and run predictions (ready for Streamlit/Flask).
- `app.py` – Streamlit app that collects user inputs, runs the recommender, and visualizes results.
- `requirements.txt` – Python dependencies.

## Getting started

1. **Create a virtual environment** (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train / (re)generate artifacts**:

   ```bash
   python train_model.py
   ```

   This will read `career_guidance_perfect.csv` and create `career_model.pkl`.

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in the terminal (usually `http://localhost:8501`) to access CareerForge-AI.

## Full ML pipeline (train + predict)

To train all models (classification + regression) on the full dataset and save artifacts:

```bash
python train_pipeline.py
```

Optional: use a fraction of data for a quick run, e.g. 10%:

```bash
python train_pipeline.py 0.1
```

Artifacts are saved under `ml_artifacts/`: preprocessor, label encoders, classifiers (recommended_career, alt_career_1/2, skill_gaps), regressors (readiness_score, est_weeks_to_ready), feature names, and feature-importance plots.

To use the trained models in your app (Streamlit or Flask):

```python
from ml_predict import load_artifacts, predict_single

artifacts = load_artifacts()
result = predict_single({"age": 22, "cgpa": 8.0, "branch": "CSE", ...}, artifacts=artifacts)
# result["recommended_career"], result["readiness_score"], result["skill_gap_1"], etc.
```

## How recommendations work

- We use numeric features like age, CGPA, backlogs, attendance, and several skill/soft-skill scores from
  `career_guidance_perfect.csv`.
- These features are scaled and used to build a student feature matrix.
- When you submit your profile, we build a matching feature vector, scale it with the same scaler, and compute
  **cosine similarity** against all rows.
- The top-matching rows are surfaced as **recommended careers**, along with:
  - Primary and alternative career paths
  - Model confidence and readiness score
  - Estimated weeks to job-ready
  - Peer benchmark percentile
  - Top skill gaps and a curated learning roadmap URL

## Hackathon deliverables

- **PPT**: High-level problem statement, architecture diagram, data description, screenshots of the Streamlit UI,
  and a roadmap for future improvements.
- **Data report**: Short write-up on the dataset, preprocessing steps, and how recommendations and gaps are computed.
- **GitHub repo**: Public repository containing this code and dataset.
- **Demo video**: 2–3 minute walkthrough of the app, showing input → recommendations → skill-gap analytics.

