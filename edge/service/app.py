import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from .redis_client import get_active_model as _get_active_model
from .state import ensure_learner, get_theta_row
from ..models.counterfactuals import generate_counterfactual_item
from ..models.irt import Topic2PLIRT
from ..models.misconception import MisconceptionGMM, build_feature_matrix
from ..policy.scheduler import UCB1Bandit, ThompsonBandit, recommend_items_for_learner

ART = Path(__file__).resolve().parents[2] / "artifacts"
# Bandit choice: UCB (Upper Confidence Bound) or TS (Thompson Sampling)
BANDIT_KIND = "UCB"  # Change to "TS" to use Thompson Sampling instead

# Reward function weights
REWARD_ALPHA = 1.0  # Mastery improvement weight
REWARD_BETA = 0.5  # Misconception reduction weight
REWARD_GAMMA = 0.1  # Time penalty weight

# ====== Load artifacts ======
items_df = pd.read_csv(ART / "items.csv")
irt = Topic2PLIRT.load(ART / "irt_state.npz")
rf = joblib.load(ART / "rf_miscon_cal.joblib") if (ART / "rf_miscon_cal.joblib").exists() else joblib.load(
    ART / "rf_miscon.joblib")
gmm = MisconceptionGMM.load(ART / "gmm_miscon.joblib")
bandit = UCB1Bandit(n_actions=3, state_file=ART / "bandit_state.pkl") if BANDIT_KIND == "UCB" else ThompsonBandit(
    n_actions=3, state_file=ART / "ts_state.pkl")

# Last-practice retention timestamps
if (ART / "last_practice.npy").exists():
    last_practice = np.load(ART / "last_practice.npy")
else:
    last_practice = np.zeros((irt.theta.shape[0], irt.theta.shape[1]))
    np.save(ART / "last_practice.npy", last_practice)

time_step = 0


# ====== Pydantic models with Swagger examples ======

class LearnerState(BaseModel):
    learner_id: int = Field(..., example=42, description="Unique learner identifier (int)")
    concept_id: Optional[int] = Field(None, example=7, description="Optional concept/topic id if known")
    item_id: Optional[int] = Field(None, example=128, description="Optional current item id")
    difficulty_est: float = Field(..., example=0.6, description="Estimated difficulty of the current content")
    prior_mastery: float = Field(..., example=0.35, description="Estimated prior mastery in [0,1]")
    attempts: int = Field(..., example=2, description="Attempts in this session/context")
    hints: int = Field(..., example=1, description="Number of hints used")
    response_time: float = Field(..., example=27.5, description="Response time in seconds")
    correct: int = Field(..., example=0, description="Was the last response correct? 1=yes, 0=no")
    fatigue_factor: int = Field(..., example=2, description="Coarse fatigue proxy (0=low…higher=worse)")
    text_quality: float = Field(..., example=0.7, description="Quality score for learner explanation in [0,1]")
    time_of_day: str = Field(..., example="morning", description="'morning'|'afternoon'|'evening'")
    device_type: str = Field(..., example="desktop", description="'desktop'|'tablet'|'phone'")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "learner_id": 42,
                    "concept_id": 7,
                    "item_id": 128,
                    "difficulty_est": 0.6,
                    "prior_mastery": 0.35,
                    "attempts": 2,
                    "hints": 1,
                    "response_time": 27.5,
                    "correct": 0,
                    "fatigue_factor": 2,
                    "text_quality": 0.7,
                    "time_of_day": "morning",
                    "device_type": "desktop"
                }
            ]
        }


class PredictResponse(BaseModel):
    misconception_prob: float
    policy_action: str
    bandit_action: Optional[str] = None
    mastery_used: Optional[float] = None


class RecommendResponse(BaseModel):
    learner_id: int = Field(..., example=42)
    recommended_items: List[int] = Field(..., example=[101, 202, 303, 404, 505])


class Feedback(BaseModel):
    learner_id: int = Field(..., example=42)
    item_id: int = Field(..., example=128)
    correct: int = Field(..., example=0)
    response_time: float = Field(..., example=27.5)
    attempts: int = Field(..., example=2)
    hints: int = Field(..., example=1)


class RewardUpdate(BaseModel):
    action: int = Field(..., example=1,
                        description="Index of action taken by the policy (0=practice,1=remediate,2=challenge)")
    reward: float = Field(..., example=0.35, description="Scalar reward after the action (can be shaped)")


class CounterfactualResponse(BaseModel):
    counterfactual: Dict[str, Any] = Field(..., example={
        "stem": "Original stem (Consider edge-case #2 explicitly.)",
        "options": ["B.", "A.", "D.", "C."],
        "correct_index": 0
    })


# ====== FastAPI app with metadata & tags ======
tags_metadata = [
    {"name": "Health", "description": "Health checks"},
    {"name": "Inference", "description": "Misconception prediction & item recommendations"},
    {"name": "Learning", "description": "Feedback & bandit updates"},
    {"name": "Generation", "description": "Counterfactual item generation (heuristic/LLM)"},
    {"name": "Admin", "description": "Metrics & diagnostics"},
]

app = FastAPI(
    title="EDGE Final PoC API",
    description="Evaluate → Diagnose → Generate → Exercise. Calibrated RF + IRT + Bayesian GMM + Bandits.\n\nSwagger "
                "examples included.",
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={"name": "@13shivam", "email": "@13shivam"}
)


@app.get("/health", tags=["Health"], summary="Service health check")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"],
          summary="Predict misconception probability & policy action")
def predict(state: LearnerState = Body(...)):
    """
    Predict misconception probability for the given learner state and
    choose a policy action using calibrated thresholds.
    """
    # Ensure learner exists (lazy onboarding)
    try:
        ensure_learner(int(state.learner_id))
    except Exception:
        # If IRT state missing, we'll fall back to using prior_mastery or 0.0
        pass

    # Build DataFrame
    df = pd.DataFrame([state.dict()])

    # Compute mastery: prefer IRT theta if available, else use prior_mastery or 0.0
    mastery_val = None
    try:
        theta_row = get_theta_row(int(state.learner_id))
        topic_idx = None
        if hasattr(state, "concept_id") and state.concept_id is not None:
            topic_idx = int(state.concept_id)
        elif hasattr(state, "topic") and state.topic is not None:
            topic_idx = int(state.topic)
        if topic_idx is not None and 0 <= topic_idx < len(theta_row):
            mastery_val = float(theta_row[topic_idx])
        else:
            mastery_val = float(np.mean(theta_row))
    except Exception:
        pass

    if mastery_val is None:
        mastery_val = float(getattr(state, "prior_mastery", 0.0) or 0.0)

    df["mastery"] = mastery_val

    # Features expected by the trained pipeline
    features = [
        "mastery",
        "response_time",
        "attempts",
        "hints",
        "text_quality",
        "fatigue_factor",
        "time_of_day",
        "device_type",
    ]

    # Check columns
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for prediction: {missing}")

    prob = float(rf.predict_proba(df[features])[:, 1][0])

    # Policy thresholds (tunable via env/config)
    THRESH_REMEDIATE = float(os.getenv("EDGE_THRESH_REMEDIATE", 0.70))
    THRESH_PRACTICE = float(os.getenv("EDGE_THRESH_PRACTICE", 0.40))

    if prob >= THRESH_REMEDIATE:
        policy_action = "remediate_now"
    elif prob >= THRESH_PRACTICE:
        policy_action = "practice_later"
    else:
        policy_action = "challenge"

    # Bandit action (kept for backwards compatibility)
    actions = ["practice", "remediate", "challenge"]
    try:
        bandit_action = actions[bandit.select()]
    except Exception:
        bandit_action = None

    return {"misconception_prob": prob, "policy_action": policy_action, "bandit_action": bandit_action,
            "mastery_used": mastery_val}


@app.post("/recommend", response_model=RecommendResponse, tags=["Inference"],
          summary="Recommend next items for a learner")
def recommend(learner_id: int = Body(..., embed=True, example=42), n: int = Body(5, embed=True, example=5)):
    global time_step
    recs = recommend_items_for_learner(learner_id, items_df, irt.theta, last_practice, time_step, n=n)
    return {"learner_id": learner_id, "recommended_items": recs}


@app.post("/feedback", tags=["Learning"], summary="Send learner feedback; updates retention & misconception profiles")
def feedback(fb: Feedback = Body(...)):
    global time_step
    time_step += 1
    topic = int(items_df.loc[fb.item_id, "topic"])
    last_practice[fb.learner_id, topic] = time_step
    np.save(ART / "last_practice.npy", last_practice)

    # (Optional) misconception profile update when incorrect
    if fb.correct == 0:
        dfw = pd.DataFrame([{"response_time": fb.response_time, "attempts": fb.attempts, "hints": fb.hints}])
        X = build_feature_matrix(dfw)
        # Typically you'd persist this responsibility vector into a learner profile matrix
        _ = gmm.predict_proba(X)[0]

    # Reward shaping example (Δ mastery + misconception reduction - time penalty)
    prev_mastery = irt.theta[fb.learner_id, topic]
    post_mastery = prev_mastery + (0.02 if fb.correct == 1 else -0.01)  # simple proxy
    delta_mastery = post_mastery - prev_mastery
    miscon_prev, miscon_now = (1 - fb.correct), 0 if fb.correct == 1 else 1
    delta_miscon = miscon_prev - miscon_now
    time_pen = fb.response_time / 60.0

    reward = REWARD_ALPHA * delta_mastery + REWARD_BETA * delta_miscon - REWARD_GAMMA * time_pen
    # Note: to properly update the bandit, pass the action index used earlier.
    # For demo simplicity we map correct->remediate(1) else practice(0).
    action_idx = 1 if fb.correct == 0 else 0
    bandit.update(action_idx, reward)

    return {"status": "recorded", "time_step": time_step, "reward": reward}


@app.post("/updateReward", tags=["Learning"], summary="Manually update bandit with a reward")
def update_reward(update: RewardUpdate = Body(...)):
    bandit.update(update.action, update.reward)
    return {"status": "ok"}


@app.post("/counterfactual", response_model=CounterfactualResponse, tags=["Generation"],
          summary="Generate counterfactual (heuristic)")
def counterfactual(item_id: int = Body(..., embed=True, example=10),
                   misconception: int = Body(2, embed=True, example=2)):
    row = items_df.loc[item_id].to_dict()
    row["distractors"] = row["distractors"].split("|")
    cf = generate_counterfactual_item(row, misconception)
    return {"counterfactual": cf}


# Optional LLM endpoint (needs OPENAI_API_KEY)
try:
    from ..models.llm_adapter import generate_counterfactual_llm

    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False


@app.post("/counterfactual_llm", tags=["Generation"],
          summary="Generate counterfactual via LLM (requires OPENAI_API_KEY)")
def counterfactual_llm(item_id: int = Body(..., embed=True, example=10),
                       misconception_label: str = Body(..., embed=True,
                                                       example="confuses_derivative_with_slope_at_point")):
    if not LLM_AVAILABLE:
        return {"error": "LLM not configured (set OPENAI_API_KEY) or package missing."}
    row = items_df.loc[item_id].to_dict()
    distractors = row["distractors"].split("|")
    cf = generate_counterfactual_llm(
        stem=row.get("stem", ""),
        distractors=distractors,
        misconception_label=misconception_label,
    )
    if cf is None:
        return {"error": "LLM generation failed or not configured."}
    return {"counterfactual": cf}


@app.get("/metrics", tags=["Admin"], summary="Policy & model metrics snapshot")
def metrics():
    return {
        "bandit_kind": BANDIT_KIND,
        "bandit_state": {
            "keys": ["counts_values" if hasattr(bandit, "values") else "alpha_beta"],
            "counts": getattr(bandit, "counts", None).tolist() if hasattr(bandit, "counts") else None,
            "values": getattr(bandit, "values", None).tolist() if hasattr(bandit, "values") else None,
            "alpha": getattr(bandit, "alpha", None).tolist() if hasattr(bandit, "alpha") else None,
            "beta": getattr(bandit, "beta", None).tolist() if hasattr(bandit, "beta") else None,
        }
    }


@app.post("/reload_model", tags=["Admin"], summary="Reload model from active_model.json")
def reload_model():
    """
    Load model specified in artifacts/active_model.json and replace in-memory model.
    """
    rec = _get_active_model()
    if not rec:
        return {"status": "no_active_model"}
    path = Path(rec.get("model_path"))
    if not path.exists():
        return {"status": "model_not_found", "path": str(path)}
    # load model file (joblib)
    try:
        global rf
        rf = joblib.load(path)
        return {"status": "reloaded", "model_path": str(path)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
