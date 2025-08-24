import os, sys, json, tempfile
import numpy as np
from pathlib import Path

# Import ensure_learner to test expansion logic
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.service.state import ensure_learner, get_theta_row
from pathlib import Path


def test_ensure_learner_creates_row(tmp_path):
    ART = Path(__file__).resolve().parents[1] / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    # create a minimal irt_state.npz with 2 learners and 3 topics
    irt = ART / "irt_state.npz"
    theta = np.zeros((2, 3))
    a = np.zeros(5)
    b = np.zeros(5)
    item_topics = np.zeros(5, dtype=int)
    np.savez(irt, theta=theta, a=a, b=b, item_topics=item_topics)
    # ensure learner id 5 is created (expands to 6 rows)
    ensure_learner(5)
    data = np.load(irt, allow_pickle=True)
    assert data['theta'].shape[0] >= 6
