import json
from pathlib import Path
import numpy as np
import os

ART = Path(__file__).resolve().parents[2] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
REG = ART / "learners.json"
IRT = ART / "irt_state.npz"
LP = ART / "last_practice.npy"


def _load_registry():
    if REG.exists():
        try:
            return json.loads(REG.read_text())
        except Exception:
            return {"known": []}
    data = {"known": []}
    REG.write_text(json.dumps(data))
    return data


def _save_registry(reg):
    REG.write_text(json.dumps(reg))


def _ensure_last_practice(n_learners: int, n_topics: int):
    if LP.exists():
        arr = np.load(LP)
        if arr.shape[0] >= n_learners and arr.shape[1] == n_topics:
            return arr
        if arr.shape[0] < n_learners:
            extra = np.zeros((n_learners - arr.shape[0], n_topics))
            arr = np.vstack([arr, extra])
            np.save(LP, arr)
            return arr
    arr = np.zeros((n_learners, n_topics))
    np.save(LP, arr)
    return arr


def ensure_learner(learner_id: int):
    """
    Ensure that irt_state.npz and last_practice.npy have rows up to learner_id.
    Expands files with zeros when needed.
    """
    if not IRT.exists():
        raise RuntimeError("irt_state.npz not found. Run training first.")
    data = np.load(IRT, allow_pickle=True)
    theta = data["theta"]
    item_topics = data.get("item_topics", None)
    nL, nT = theta.shape
    reg = _load_registry()
    known = set(reg.get("known", []))
    if learner_id < nL:
        known.add(learner_id)
        reg["known"] = sorted(list(known))
        _save_registry(reg)
        _ensure_last_practice(nL, nT)
        return
    # expand theta
    new_rows = learner_id - nL + 1
    theta_exp = np.vstack([theta, np.zeros((new_rows, nT))])
    a = data.get("a", None)
    b = data.get("b", None)
    np.savez(IRT, theta=theta_exp, a=a, b=b, item_topics=item_topics)
    _ensure_last_practice(theta_exp.shape[0], nT)
    known.add(learner_id)
    reg["known"] = sorted(list(known))
    _save_registry(reg)


def get_theta_row(learner_id: int):
    if not IRT.exists():
        raise RuntimeError("irt_state.npz not found.")
    data = np.load(IRT, allow_pickle=True)
    theta = data["theta"]
    if learner_id >= theta.shape[0]:
        raise IndexError("learner not initialized")
    return theta[learner_id]
