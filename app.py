"""
app.py  —  CareerForge AI  |  FastAPI Backend
Endpoints:
  POST /api/recommend          → ML career recommendations
  POST /api/predict-placement  → Placement probability only
  GET  /api/health             → Health check
  GET  /                       → Serves careerforge.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ml_predict import CareerRecommenderArtifacts, recommend_for_profile, _estimate_placement_probability

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("careerforge.api")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CareerForge AI API",
    description="ML-powered career guidance for engineering students",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /public folder as static files (CSS, JS, images if any)
PUBLIC_DIR = Path("public")
if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# ── Load ML artifacts once at startup ────────────────────────────────────────
_artifacts: Optional[CareerRecommenderArtifacts] = None


@app.on_event("startup")
async def load_model() -> None:
    global _artifacts
    model_path = Path(os.getenv("MODEL_PATH", "career_model.pkl"))
    if not model_path.exists():
        logger.warning("career_model.pkl not found — running train_model.py first!")
        try:
            from train_model import train_and_save as _train
            _train()
        except Exception as exc:
            logger.error("Auto-train failed: %s", exc)
            return
    try:
        _artifacts = CareerRecommenderArtifacts.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


def _get_artifacts() -> CareerRecommenderArtifacts:
    if _artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run `python train_model.py` first.",
        )
    return _artifacts


# ── Request / Response schemas ────────────────────────────────────────────────

class StudentProfile(BaseModel):
    # Identity (optional, not used by model)
    name:   Optional[str] = Field(None, description="Student's name")
    branch: Optional[str] = Field(None, description="Engineering branch")
    year_of_study: Optional[str] = Field(None)

    # Numeric model features
    age:                   float = Field(21.0, ge=16, le=35)
    cgpa:                  float = Field(7.0,  ge=0,  le=10)
    backlogs:              float = Field(0.0,  ge=0,  le=20)
    attendance_percentage: float = Field(80.0, ge=0,  le=100)
    aptitude_test_score:   float = Field(60.0, ge=0,  le=100)
    coding_skill_score:    float = Field(55.0, ge=0,  le=100)
    problem_solving_score: float = Field(55.0, ge=0,  le=100)
    teamwork_score:        float = Field(60.0, ge=0,  le=100)
    communication_score:   float = Field(60.0, ge=0,  le=100)
    leadership_score:      float = Field(50.0, ge=0,  le=100)
    mock_interview_score:  float = Field(50.0, ge=0,  le=100)
    extracurricular_score: float = Field(50.0, ge=0,  le=100)

    # Experiential (used in composite features)
    internships_done:         int  = Field(0, ge=0, le=10)
    projects_count:           int  = Field(0, ge=0, le=20)
    hackathon_participation:  str  = Field("No")
    career_goal:              Optional[str] = Field(None)

    # UI skill tags (not used by model directly, returned for gap display)
    selected_skills: List[str] = Field(default_factory=list)

    @field_validator("hackathon_participation")
    @classmethod
    def normalise_hackathon(cls, v: str) -> str:
        return "Yes" if str(v).lower() in ("yes", "1", "true") else "No"

    def to_feature_dict(self) -> Dict[str, Any]:
        return {
            "age":                   self.age,
            "cgpa":                  self.cgpa,
            "backlogs":              self.backlogs,
            "attendance_percentage": self.attendance_percentage,
            "aptitude_test_score":   self.aptitude_test_score,
            "coding_skill_score":    self.coding_skill_score,
            "problem_solving_score": self.problem_solving_score,
            "teamwork_score":        self.teamwork_score,
            "communication_score":   self.communication_score,
            "leadership_score":      self.leadership_score,
            "mock_interview_score":  self.mock_interview_score,
            "extracurricular_score": self.extracurricular_score,
            "internships_done":      self.internships_done,
            "projects_count":        self.projects_count,
            "hackathon_participation": self.hackathon_participation,
        }


class RecommendRequest(BaseModel):
    profile: StudentProfile
    top_k:   int = Field(3, ge=1, le=10)


class PlacementRequest(BaseModel):
    profile: StudentProfile


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    html_path = Path("careerforge.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/api/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": _artifacts is not None,
        "version": "2.0.0",
    })


@app.post("/api/recommend")
async def recommend(req: RecommendRequest) -> JSONResponse:
    """
    Full ML career recommendation.

    Returns placement probability, peer percentile, and top-k career recommendations
    each with confidence, skill gaps, weeks-to-ready, and roadmap URL.
    """
    artifacts = _get_artifacts()
    profile_dict = req.profile.to_feature_dict()

    try:
        result = recommend_for_profile(
            profile=profile_dict,
            top_k=req.top_k,
            artifacts=artifacts,
        )
    except Exception as exc:
        logger.exception("Recommendation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # Attach non-model metadata back to response
    result["student"] = {
        "name":           req.profile.name or "Student",
        "branch":         req.profile.branch or "Engineering",
        "year_of_study":  req.profile.year_of_study or "3rd Year",
        "career_goal":    req.profile.career_goal,
        "selected_skills": req.profile.selected_skills,
    }

    logger.info(
        "Recommend | %s | goal=%s | placement=%.1f%%",
        req.profile.name or "anon",
        req.profile.career_goal,
        result["placement_probability"],
    )
    return JSONResponse(result)


@app.post("/api/predict-placement")
async def predict_placement(req: PlacementRequest) -> JSONResponse:
    """Lightweight endpoint — returns placement probability only (no model needed)."""
    prob = _estimate_placement_probability(req.profile.to_feature_dict())
    return JSONResponse({
        "placement_probability": prob,
        "name": req.profile.name or "Student",
    })


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc: Any) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
