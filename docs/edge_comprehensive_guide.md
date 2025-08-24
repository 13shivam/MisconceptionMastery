# EDGE PoC - Comprehensive Guide

This document provides a complete reference for the EDGE (Educational Data-driven Guidance Engine) Proof of Concept, including data dictionary, configuration parameters, and tuning guidelines.

## Data Dictionary

### items.csv
- **item_id** (int): Unique identifier for the question/item.
- **topic** (int): Topic index the item is associated with.
- **a** (float): Item discrimination parameter (IRT slope).
- **b** (float): Item difficulty parameter.
- **stem** (string): The question text or stem.
- **distractors** (string): Pipe-separated incorrect answer options.

### interactions_*.csv
- **learner_id** (int): Unique identifier for the learner/student.
- **item_id** (int): ID of the presented item.
- **topic** (int): Topic index (duplicated from items.csv for convenience).
- **a** (float): Item discrimination copied from items.csv.
- **b** (float): Item difficulty copied from items.csv.
- **response_time** (float): Time in seconds the learner took to answer.
- **attempts** (int): Number of attempts taken for the item.
- **hints** (int): Number of hints used.
- **correct** (0/1): Whether learner answered correctly.
- **misconception** (0/1): Simulated label indicating presence of misconception.
- **time_of_day** (string): 'morning'|'afternoon'|'evening'.
- **device_type** (string): 'desktop'|'tablet'|'phone'.
- **text_quality** (float): 0..1 quality score of textual answer/explanation.
- **fatigue_factor** (int): Proxy for learner fatigue.
- **chosen_distractor** (int): Index of incorrect option selected, -1 if none.
- **mastery** (float): Learner mastery for the topic (0..1), can be computed via IRT theta.

## Configuration and Environment Variables

### Environment Variables
- **EDGE_THRESH_REMEDIATE**: float (default 0.70) — threshold above which remediation is recommended.
- **EDGE_THRESH_PRACTICE**: float (default 0.40) — threshold above which practice is scheduled.
- **EDGE_UCB_EXPLORATION_C**: float — UCB exploration constant (if using UCB bandit).
- **EDGE_BANDIT_KIND**: string — 'ucb' or 'ts' to select bandit type.

### Files and Artifacts
- **artifacts/irt_state.npz**: contains arrays `theta`, `a`, `b`, `item_topics`.
- **artifacts/last_practice.npy**: per-learner per-topic last practice times.
- **artifacts/rf_miscon_cal.joblib**: calibrated RF classifier.
- **artifacts/gmm_miscon.joblib**: GMM misconception clusters.
- **artifacts/learners.json**: registry of seen learner IDs.

## Tuning and Production Guidelines

### Data Generation
**Location**: `edge/data/dataset_gen.py`
- **Scale Controls**: 
  - `n_learners`, `n_items`, `n_topics`, `n_samples`: control dataset scale.
- **Response Dynamics**:
  - `response_time` mean/variance (fatigue, device effects).
  - `attempts`, `hints` rate functions — increase difficulty sensitivity.
- **Misconception Modeling**:
  - `n_miscon` (mixture components) and Dirichlet prior for learner profiles.
  - Increase `miscon_flag` probability when wrong to simulate trickier domains.

### IRT Evaluation (2PL)
**Location**: `edge/models/irt.py`
- **lr** (learning rate): start 0.01—0.05.
- **epochs**: 2—5 for synthetic; >10 for real data.
- **reg**: L2 regularization 1e-4…1e-3 to stabilize a/b/theta.
- **Batch size**: 4k—16k for speed vs. stability.
- For harder content, allow higher `a` (discrimination) range in the item bank.

### Misconception Diagnosis (Bayesian GMM)
- **n_components**: try 4—12. Validate with BIC/AIC and interpretability.
- **Feature Matrix**:
  - Add distractor embeddings, per-distractor error rates, or semantic features.
  - Add recency-weighted features (last k interactions).
- Persist per-learner posterior by averaging responsibilities across mistakes.

### Supervised RF Misconception Model
- **n_estimators**: 100—300; increase for noisy labels.
- **Features**: Include IRT mastery, time, attempts, hints, fatigue, text_quality, device/time-of-day.
- Consider calibrated models: Platt/Isotonic (sklearn CalibratedClassifierCV).

### Exercise Scheduling
**Location**: `edge/policy/scheduler.py`
- **Priority Index**:
  - Weights for mastery, retention decay (lambda), pace, misconception penalty.
  - Retention decay: `retention = exp(-lambda * time_since)`; set `lambda` by domain half-life.
- **Bandit (UCB1)**:
  - UCB exploration strength (constant "2" inside sqrt). Increase to explore more.
  - Reward shaping (e.g., +1 for improvement, 0 otherwise; or Δ mastery, or -penalty for timeouts).
- **Optional**: Replace UCB1 with Thompson Sampling or add DQN for stateful policies.

### FastAPI Service
**Location**: `edge/service/app.py`
- Add rate limiting (e.g., behind API gateway).
- **Logging**: add structured logs and request IDs.
- Batch scoring endpoint for high-throughput use.
- Add authentication if exposed externally.

### LLM Counterfactuals
**Location**: `edge/models/llm_adapter.py`
- Set `OPENAI_API_KEY` and `OPENAI_MODEL` (e.g., `gpt-4o-mini`).
- **Tune temperature** for diversity (0.2—0.7).
- **Validate Outputs**:
  - Parseable JSON.
  - Plausibility checks (e.g., options length = 4; non-duplicated).

### Validation & Monitoring
**Location**: `edge/validate.py`
- **Track Metrics**: Accuracy, Precision/Recall/F1, **ECE**, Brier.
- Add reliability diagrams; retrain or calibrate if ECE high.
- **Online Monitoring**:
  - Track bandit reward over time; guardrails for regret spikes.
  - Shadow-test new models before switching traffic.

### Persistence Strategy
- **All artifacts location**: `/artifacts`
  - `irt_state.npz`, `gmm_miscon.joblib`, `rf_miscon.joblib`, `bandit_state.pkl`, `last_practice.npy`.
- Back up and version these; include data drift checks before reuse.

## Production Optimization Recommendations

- Increase `n_samples` and epochs to stabilize IRT and RF.
- Use CalibratedClassifierCV for probability calibration.
- Add feature store & schema validation to avoid training-serving skew.
- Add retries/timeouts around LLM calls.