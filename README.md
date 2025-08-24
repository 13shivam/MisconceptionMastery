
# EDGE: Misconception-Aware Adaptive Learning Framework

This repository provides a **production-quality research artifact** for **EDGE** — a framework for adaptive, misconception-aware learning. EDGE integrates psychometric models (IRT), machine learning classifiers (Random Forests), probabilistic clustering (Bayesian GMM), and multi-armed bandit algorithms to dynamically adjust learning paths, detect misconceptions, and personalize learner experiences.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Components](#key-components)
- [Installation Instructions](#installation-instructions)
- [Data Generation](#data-generation)
- [Training](#training)
- [Validation](#validation)
- [Serving API](#serving-api)
- [Tuning and Production Adjustments](#tuning-and-production-adjustments)
- [Metrics Interpretation](#metrics-interpretation)
- [End-User Documentation](#end-user-documentation)
- [License](#license)

---

## Project Overview

### Core Framework
1. **Evaluate**: IRT-style ability estimation (per-topic 2PL approximation)
2. **Diagnose**: Bayesian GMM misconception inference using incorrect-response features
3. **Generate**: Counterfactual item generator (heuristic, with pluggable LLM hook)
4. **Exercise**: Practical scheduler with EdgeScore-like priority + online UCB1 bandit
5. **Validation**: Metrics including accuracy, precision/recall/F1, and calibration (ECE, Brier)

### Key Components

1. **Data Simulation**:
   - Synthetic learner–item interaction data generated using **Item Response Theory (2PL model)**.
   - Learner features include response time, attempts, hints, fatigue, device type, and text quality.
   - Misconception signals simulated via distractor embeddings and GMM clusters.

2. **Models**:
   - **IRT (2PL)**: estimates learner mastery and item difficulty/discrimination.
   - **Random Forest Classifier (calibrated)**: predicts misconception probability from features.
   - **Bayesian GMM**: identifies latent misconception clusters from wrong attempts.

3. **Adaptive Policy**:
   - **Multi-armed bandit policies** (UCB, Thompson Sampling) select instructional actions:
     - *Practice* (reinforce current concept)
     - *Remediate* (address misconceptions)
     - *Challenge* (advance difficulty)

4. **Serving Layer**:
   - **FastAPI service** exposing endpoints:
     - `/predict`: predict misconception probability and bandit action
     - `/recommend`: recommend next items
     - `/feedback`: update learner state and bandit
     - `/counterfactual`: generate counterfactual items
     - `/metrics`: inspect policy and state

5. **Validation**:
   - Offline validation pipeline with accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
   - Calibration analysis via Expected Calibration Error (ECE) and Brier score.
   - Reliability diagrams and confusion matrices for interpretability.

---

## Installation Instructions

### Clone the Repository
```bash
gh repo clone 13shivam/MisconceptionMastery
cd MisconceptionMastery
```

### Set up Virtual Environment and Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Generation

The **data generation** process simulates synthetic learner-item interactions using Item Response Theory (IRT). Learner responses, attempts, hints, and other features like **response time**, **device type**, and **fatigue factor** are modeled. Misconceptions are simulated using distractor embeddings, clustered with **Bayesian Gaussian Mixture Models (GMM)**.

1. **Generate Synthetic Data**:
   - Use the following command to generate the synthetic dataset:
     ```bash
     python -m edge.data.dataset_gen
     ```
   - This will generate:
     - `artifacts/items.csv` — A CSV file containing item information, topics, and associated parameters.
     - `artifacts/interactions_train.csv` — Training data for the learners.
     - `artifacts/interactions_test.csv` — Testing data for learner interactions.

---

## Training

The training process involves training the **Random Forest** model to predict misconceptions, based on learner interaction features. Additionally, **Item Response Theory (IRT)** parameters are estimated for learner mastery prediction.

1. **Train Models**:
   - Use the following command to start the training process:
     ```bash
     python -m edge.train
     ```
   - This will produce the following artifacts:
     - `irt_state.npz` — IRT-based learner mastery estimates.
     - `gmm_miscon.joblib` — Bayesian GMM model for misconception clustering.
     - `rf_miscon_cal.joblib` — Calibrated Random Forest model for misconception prediction.

---

## Validation

The validation process involves evaluating the trained models on test data, measuring performance using metrics such as **accuracy**, **precision**, **recall**, **F1 score**, **ROC-AUC**, **PR-AUC**, **Expected Calibration Error (ECE)**, and **Brier Score**.

1. **Run Validation**:
   - Use the following command to run the validation script:
     ```bash
     python -m edge.validate
     ```
   - This outputs:
     - Console metrics:
       - Accuracy
       - Precision, Recall, F1
       - ROC-AUC, PR-AUC
       - Expected Calibration Error (ECE)
       - Brier Score
     - Plots in `artifacts/plots/`
     - HTML report: `artifacts/validate_report.html`

---

## Serving API

Run the FastAPI service using Uvicorn:
```bash
uvicorn edge.service.app:app --reload --host 0.0.0.0 --port 8000
```

Access Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

### Example Request: Predict
```bash
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "learner_id": 42,
  "concept_id": 2,
  "item_id": 10,
  "prior_mastery": 0.3,
  "response_time": 28.5,
  "attempts": 1,
  "hints": 0,
  "correct": 0,
  "fatigue_factor": 2,
  "text_quality": 0.7,
  "time_of_day": "morning",
  "device_type": "desktop"
}'
```

Response:
```json
{ "misconception_prob": 0.42, "bandit_action": "remediate" }
```

---

## Notable Artifacts
- `artifacts/items.csv` — item bank with topics and parameters
- `artifacts/interactions_train.csv` / `interactions_test.csv`
- `artifacts/irt_state.npz` — per-learner per-topic ability estimates
- `artifacts/gmm_miscon.joblib` — misconception GMM
- `artifacts/rf_miscon.joblib` — supervised misconception predictor
- `artifacts/bandit_state.pkl` — UCB1 bandit persistence
- `artifacts/last_practice.npy` — scheduling retention timestamps


## Tuning and Production Adjustments

### Data Generation (`edge/data/dataset_gen.py`)
- **Control Parameters**:
  - `n_learners`, `n_items`, `n_topics`, `n_samples`: scale control
  - Misconception modeling: increase `miscon_flag` probability for more complex domains

### IRT (2PL) Tuning (`edge/models/irt.py`)
- **Hyperparameters**:
  - Learning rate (`lr`): 0.01–0.05
  - Epochs: 2–5 for synthetic, >10 for real data
  - Regularization: L2 regularization from 1e-4 to 1e-3

### Misconception Diagnosis (Bayesian GMM)
- **n_components**: 4–12, based on BIC/AIC validation
- **Feature Matrix**: Add distractor embeddings, recency-weighted features

### Supervised Random Forest
- **Hyperparameters**: 
  - `n_estimators`: 100–300; consider CalibratedClassifierCV for better probability estimates.

### Scheduling & Bandit Policies
- **Priority Index**: Customize for your domain with weights for mastery, retention decay, etc.
- **Bandit (UCB1)**: Adjust exploration strength and reward shaping

### FastAPI Service Adjustments
- Add rate limiting, structured logging, and authentication if necessary

### LLM Counterfactuals
- Set `OPENAI_API_KEY`, adjust `temperature` for diversity (0.2–0.7)

---

## Metrics Interpretation

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Fraction of predicted misconceptions that were true.
- **Recall**: Fraction of actual misconceptions detected.
- **F1**: Harmonic mean of precision and recall.
- **ECE (Expected Calibration Error)**: Measures the gap between confidence and accuracy.
- **Brier Score**: Mean squared error of probability predictions.

---

## End-User Documentation

- **Onboarding New Learners**: Dynamic learner profiles are supported. New learners are tracked automatically without retraining.
- **Persistence**: All model state is persisted in `/artifacts` and can be versioned or backed up as necessary.

---

## Onboarding New Learners

- New learners are supported dynamically.
- First time a learner ID is seen:
  - `irt_state.npz` and `last_practice.npy` expand with a new row.
  - Learners tracked in `artifacts/learners.json`.
- No retraining required for dynamic learners.

---

## Architecture

### Sequence Diagram
![Sequence Diagram.png](docs%2FSequence%20Diagram.png)

### Component Diagram
![HLD Component.png](docs%2FHLD%20Component.png)

---

## Production Hardening Recommendations

- Replace file-based state with Redis or DynamoDB for concurrency safety.
- Add message queue (Kafka, RabbitMQ, SQS) for durable feedback ingestion.
- Introduce CI/CD pipeline for retraining and promotion.
- Add monitoring with Prometheus and Grafana (latency, calibration drift, model accuracy).
- Support A/B testing for safe model deployment.

---


## License

MIT License

## Acknowledgment

I’d like to thank Anand Verma, author of the white paper “EDGE: A Theoretical Framework for Misconception-Aware Adaptive Learning” (https://arxiv.org/abs/2508.07224
).
This project builds on the ideas from his paper, and I’m grateful to him for sharing his research openly, it made this implementation possible.
