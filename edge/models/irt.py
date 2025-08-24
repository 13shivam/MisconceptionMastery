import numpy as np
from pathlib import Path


class Topic2PLIRT:
    """
    Per-topic 2PL IRT trained with simple SGD.

    Model (2PL for an item j in topic t):
        P(correct | theta_i,t, a_j, b_j) = sigmoid( a_j * ( theta[i, t] - b_j ) )

    Parameters:
      - theta[i, t] : learner i's ability for topic t (matrix shape [n_learners, n_topics])
      - a[j]        : item j discrimination (slope of the logistic)
      - b[j]        : item j difficulty (location / shift of the logistic)
      - item_topics[j] : topic index for item j

    Training with SGD:
      For each observation (i, j, y):
        logit   = a[j] * (theta[i,t] - b[j])
        p       = sigmoid(logit)
        err     = y - p     # gradient points to reducing cross-entropy
        dtheta  = a[j] * err - reg*theta[i,t]        # gradient wrt theta
        da      = (theta[i,t] - b[j]) * err - reg*a[j]
        db      = (-a[j]) * err - reg*b[j]

      Then update with learning rate lr:
        theta[i,t] += lr * dtheta
        a[j]       += lr * da
        b[j]       += lr * db

    Notes:
      - This is a compact trainer for demonstration/PoC; for large-scale use, consider
        batched vectorized updates or probabilistic frameworks (e.g., variational Bayes).
      - reg is an L2-style shrinkage to prevent parameter blow-up on small data.
    """

    def __init__(self, n_learners, n_topics, item_topics, a_init, b_init, lr=0.02, reg=1e-4):
        self.n_learners = n_learners
        self.n_topics = n_topics
        self.item_topics = np.array(item_topics)
        self.a = np.array(a_init, dtype=float)
        self.b = np.array(b_init, dtype=float)
        self.theta = np.zeros((n_learners, n_topics), dtype=float)
        self.lr = lr
        self.reg = reg

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, interactions, epochs=3, batch_size=4096, seed=7):
        """SGD over interactions = array[:, [learner_id, item_id, correct]]"""
        rng = np.random.default_rng(seed)
        X = interactions
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)  # shuffle each epoch
            for s in range(0, n, batch_size):
                batch = X[idx[s:s + batch_size]]
                if len(batch) == 0: break
                self._sgd_step(batch)

    def _sgd_step(self, batch):
        lr = self.lr
        for row in batch:
            i, j, y = int(row[0]), int(row[1]), float(row[2])
            t = self.item_topics[j]
            # Forward pass (logistic)
            logit = self.a[j] * (self.theta[i, t] - self.b[j])
            p = self._sigmoid(logit)
            # Error signal (y - p) for cross-entropy gradient
            err = y - p
            # Parameter gradients (with small L2 regularization)
            dtheta = self.a[j] * err - self.reg * self.theta[i, t]
            da = (self.theta[i, t] - self.b[j]) * err - self.reg * self.a[j]
            db = (-self.a[j]) * err - self.reg * self.b[j]
            # SGD updates
            self.theta[i, t] += lr * dtheta
            self.a[j] += lr * da
            self.b[j] += lr * db

    def save(self, path: Path):
        np.savez(path, theta=self.theta, a=self.a, b=self.b, item_topics=self.item_topics)

    @classmethod
    def load(cls, path: Path):
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_learners=data["theta"].shape[0],
            n_topics=data["theta"].shape[1],
            item_topics=data["item_topics"],
            a_init=data["a"],
            b_init=data["b"],
        )
        obj.theta = data["theta"]
        obj.a = data["a"]
        obj.b = data["b"]
        return obj

    def prob_correct(self, learner_id, item_id):
        """Convenience helper for scoring a single learner-item pair."""
        t = self.item_topics[item_id]
        logit = self.a[item_id] * (self.theta[learner_id, t] - self.b[item_id])
        return self._sigmoid(logit)
