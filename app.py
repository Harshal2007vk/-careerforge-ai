"""
CareerForge AI — Flask Backend API
Serves the trained scikit-learn ML model via REST API

Run:
  pip install flask flask-cors scikit-learn pandas numpy pickle5
  python train_model.py   # first time only
  python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ─────────────────────────────────────────
#  LOAD MODEL ON STARTUP
# ─────────────────────────────────────────
MODEL = None
META = None

def load_model():
    global MODEL, META
    model_path = 'model/placement_model.pkl'
    meta_path  = 'model/model_meta.json'

    if not os.path.exists(model_path):
        print("⚠️  Model not found! Run: python train_model.py")
        return False

    with open(model_path, 'rb') as f:
        MODEL = pickle.load(f)

    with open(meta_path, 'r') as f:
        META = json.load(f)

    print(f"✅ Model loaded | Accuracy: {META['accuracy']}%")
    return True


# ─────────────────────────────────────────
#  HELPER: BRANCH ENCODING
# ─────────────────────────────────────────
CS_BRANCHES = [
    'computer science', 'information technology',
    'data science', 'artificial intelligence',
    'cs', 'it', 'cse', 'ai', 'ds'
]

def is_cs_branch(branch: str) -> int:
    return 1 if any(b in branch.lower() for b in CS_BRANCHES) else 0


# ─────────────────────────────────────────
#  ROUTE: HEALTH CHECK
# ─────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'model_accuracy': META['accuracy'] if META else None,
        'model_type': META['model_type'] if META else None
    })


# ─────────────────────────────────────────
#  ROUTE: PREDICT PLACEMENT
# ─────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500

    try:
        data = request.get_json()

        # Extract & validate inputs
        cgpa         = float(data.get('cgpa', 7.0))
        internships  = int(data.get('internships', 0))
        projects     = int(data.get('projects', 0))
        hackathon    = 1 if str(data.get('hackathon', 'No')).lower() in ['yes', '1', 'true'] else 0
        coding_skill = int(data.get('coding_skill', 50))
        comm_skill   = int(data.get('comm_skill', 50))
        skills       = data.get('skills', [])
        branch       = data.get('branch', '')

        skills_count = len(skills) if isinstance(skills, list) else int(skills)
        branch_cs    = is_cs_branch(branch)

        # Feature vector (must match training order)
        features = np.array([[
            cgpa, internships, projects, hackathon,
            coding_skill, comm_skill, skills_count, branch_cs
        ]])

        # Predict
        placement_prob = MODEL.predict_proba(features)[0][1]  # Probability of being placed
        placement_pct  = round(placement_prob * 100, 1)
        placed         = bool(MODEL.predict(features)[0])

        # Readiness score (composite)
        readiness = round(
            (placement_pct * 0.5) +
            (min(skills_count * 3, 20)) +
            (min(projects * 4, 20)) +
            (min(internships * 6, 18)) +
            ((coding_skill / 100) * 12)
        )
        readiness = min(97, max(25, readiness))

        # Months to ready estimate
        months_to_ready = max(2, round((100 - readiness) / 9))

        # Skill gap count for target goal
        goal = data.get('goal', '')
        gap_count = compute_gap_count(goal, skills)

        return jsonify({
            'placement_probability': placement_pct,
            'placed_prediction': placed,
            'readiness_score': readiness,
            'skills_count': skills_count,
            'gap_count': gap_count,
            'months_to_ready': months_to_ready,
            'model_accuracy': META['accuracy'],
            'feature_importance': META['feature_importance']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────
#  ROUTE: CAREER MATCH SCORING
# ─────────────────────────────────────────
CAREER_REQUIREMENTS = {
    'AI / ML Engineer':      ['python', 'machine learning', 'tensorflow', 'pytorch', 'sql', 'mlops', 'git'],
    'Full Stack Developer':  ['javascript', 'react', 'node.js', 'sql', 'rest apis', 'docker', 'git'],
    'Data Scientist':        ['python', 'statistics', 'machine learning', 'sql', 'pandas', 'visualization'],
    'Cloud Engineer':        ['aws', 'linux', 'docker', 'kubernetes', 'networking', 'security'],
    'DevOps Engineer':       ['linux', 'docker', 'git', 'ci/cd', 'aws', 'monitoring', 'scripting'],
    'Product Manager':       ['communication', 'analytics', 'leadership', 'sql', 'agile'],
    'Cybersecurity Analyst': ['networking', 'linux', 'ethical hacking', 'risk analysis', 'programming'],
    'Mobile App Developer':  ['javascript', 'react native', 'git', 'rest apis', 'ui'],
    'Blockchain Developer':  ['solidity', 'javascript', 'web3', 'cryptography', 'git'],
    'UI/UX Designer':        ['figma', 'user research', 'prototyping', 'design thinking', 'communication']
}

def compute_gap_count(goal, skills):
    req = CAREER_REQUIREMENTS.get(goal, [])
    if not req:
        return 0
    skills_lower = [s.lower() for s in skills]
    matched = sum(1 for r in req if any(r in s or s in r for s in skills_lower))
    return len(req) - matched

@app.route('/api/career-match', methods=['POST'])
def career_match():
    try:
        data   = request.get_json()
        skills = data.get('skills', [])
        branch = data.get('branch', '')
        goal   = data.get('goal', '')
        branch_cs = is_cs_branch(branch)

        scores = []
        for career, req_skills in CAREER_REQUIREMENTS.items():
            skills_lower = [s.lower() for s in skills]
            matched = sum(1 for r in req_skills if any(r in s or s in r for s in skills_lower))
            base_score = (matched / len(req_skills)) * 100

            bonus = 0
            if career == goal:
                bonus += 20
            if career in ['AI / ML Engineer', 'Data Scientist'] and branch_cs:
                bonus += 5

            final_score = min(98, round(base_score + bonus))
            scores.append({'career': career, 'score': final_score, 'matched_skills': matched, 'required': len(req_skills)})

        scores.sort(key=lambda x: -x['score'])
        return jsonify({'matches': scores[:5]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────
#  ROUTE: MODEL INFO
# ─────────────────────────────────────────
@app.route('/api/model-info', methods=['GET'])
def model_info():
    if META is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify(META)


# ─────────────────────────────────────────
#  START SERVER
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 CareerForge AI Backend Starting...")
    if load_model():
        print("✅ Model ready!")
    else:
        print("⚠️  Starting without model. Run train_model.py first.")
    print("\n📡 API running at: http://localhost:5000\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
