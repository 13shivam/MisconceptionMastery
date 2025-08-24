import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import joblib
from pathlib import Path


class MisconceptionGMM:
    """
    Bayesian Gaussian Mixture for misconception clustering on *incorrect* responses.

    Why Bayesian GMM?
      - It can automatically deactivate redundant components via Dirichlet priors.
      - Provides responsibilities (posterior probabilities) per component, which we use
        as a soft misconception profile (rather than hard clustering).

    Typical features (handcrafted):
      - response_time (and transforms): captures hesitation vs. impulsive errors
      - attempts, hints: proxy for struggle and help usage
      - (optionally) distractor embeddings or per-distractor error rates

    Usage:
      - fit(X): X is N x D feature matrix from wrong answers
      - predict_proba(X): returns responsibilities (N x K), i.e., P(component | x_n)
    """

    def __init__(self, n_components=6, random_state=42):
        self.model = BayesianGaussianMixture(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.model = joblib.load(path)
        return obj


def build_feature_matrix(df_wrong: pd.DataFrame) -> np.ndarray:
    """Compose a simple feature set for misconception clustering.

    Columns expected in df_wrong:
      - response_time, attempts, hints
    Returns:
      - numpy array shape [N, D] with a few time-based transforms for richer geometry.
    """
    t = df_wrong["response_time"].values
    feats = np.stack([
        t,  # raw time
        np.log(t + 1),  # log-time (compresses long tails)
        1 / (t + 1),  # inverse-time (emphasizes very fast clicks)
        df_wrong["attempts"].values,
        df_wrong["hints"].values
    ], axis=1)
    return feats
