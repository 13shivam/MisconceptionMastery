import numpy as np
import joblib
from pathlib import Path

# ===============================================================
# Bandit + Scheduling utilities
# - UCB1 (deterministic, optimistic exploration)
# - Thompson Sampling (Bayesian, randomized exploration)
# - Retention-aware priority index (EdgeScore-style)
# ===============================================================
# UCB Exploration Constant (tune for exploration vs exploitation)
UCB_EXPLORATION_C = 2.0  # You can tune this constant to control exploration behavior in UCB1

# Retention Decay (λ) for the forgetting curve
RET_LAMBDA_DEFAULT = 0.05  # L


class UCB1Bandit:
    """
    UCB1 multi-armed bandit.
    References: Auer et al., 2002.

    State:
      - counts[k]: number of times arm k was selected
      - values[k]: running mean reward estimate for arm k
      - total: total number of bandit pulls

    Selection rule (per arm k):
      UCB_k = mean_k + c * sqrt( (2 * ln(total)) / counts[k] )
      where:
        - mean_k is the current average reward for arm k
        - c is exploration constant (UCB_EXPLORATION_C). c > 0 increases exploration
        - the second term is the "optimism" bonus shrinking as counts[k] grows

    Update rule when receiving reward r for action a:
      counts[a] += 1
      values[a] <- values[a] + (r - values[a]) / counts[a]   # incremental mean
    """

    def __init__(self, n_actions, state_file=Path("artifacts/bandit_state.pkl")):
        self.n_actions = n_actions
        self.state_file = Path(state_file)
        if self.state_file.exists():
            self.counts, self.values = joblib.load(self.state_file)
        else:
            self.counts = np.zeros(n_actions, dtype=int)  # N_k
            self.values = np.zeros(n_actions, dtype=float)  # \hat{\mu}_k
        self.total = int(self.counts.sum())  # total pulls so far

    def select(self):
        """Select an action according to the UCB1 rule.

        - Step 1 (Initialization): if any arm was never tried, pick an untried arm to ensure
          every arm gets at least one sample (pure exploration).
        - Step 2 (Compute UCB): for each arm k, compute mean_k + exploration_bonus.
        - Step 3 (Argmax): choose arm with maximum UCB.
        """
        # (Step 1) Ensure each arm is tried at least once
        if 0 in self.counts:
            return int(np.argmin(self.counts))

        # (Step 2) UCB score = empirical mean + exploration bonus
        exploration = UCB_EXPLORATION_C * np.sqrt(2 * np.log(self.total) / self.counts)
        ucb = self.values + exploration

        # (Step 3) Greedy on UCB
        return int(np.argmax(ucb))

    def update(self, action, reward):
        """Update running statistics with observed reward.

        - Increment counts and total pulls
        - Update incremental mean for the chosen arm
        - Persist to disk for durability across restarts
        """
        self.counts[action] += 1
        self.total += 1
        n = self.counts[action]

        # Incremental mean: new_mean = old_mean + (r - old_mean)/n
        self.values[action] += (reward - self.values[action]) / n

        # Persist
        joblib.dump((self.counts, self.values), self.state_file)


class ThompsonBandit:
    """
    Thompson Sampling for Bernoulli rewards.

    We maintain independent Beta posteriors for each arm's success probability:
      p_k ~ Beta(alpha_k, beta_k)

    Selection:
      - Sample \tilde{p}_k ~ Beta(alpha_k, beta_k) for all k
      - Choose argmax_k \tilde{p}_k (randomized exploration via posterior sampling)

    Update (with binary reward r in {0,1}):
      - alpha_a += r
      - beta_a  += (1 - r)

    Notes:
      - Works well in non-stationary or sparse data regimes.
      - If your reward is not binary, consider scaling/clipping to [0,1] for TS.
    """

    def __init__(self, n_actions, state_file=Path("artifacts/ts_state.pkl")):
        self.n_actions = n_actions
        self.state_file = Path(state_file)
        if self.state_file.exists():
            self.alpha, self.beta = joblib.load(self.state_file)
        else:
            self.alpha = np.ones(n_actions)  # successes + 1 (prior)
            self.beta = np.ones(n_actions)  # failures  + 1 (prior)

    def select(self):
        """Sample from each arm's posterior and pick the max."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action, reward):
        """Update Beta posterior for the chosen arm.

        reward should be in [0,1]. If you have real-valued reward, you may cast to Bernoulli
        by thresholding or use a Beta-Binomial generalization.
        """
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
        joblib.dump((self.alpha, self.beta), self.state_file)


# ---------------- Retention-aware priority index ----------------
def priority_index(mastery, retention, pace, miscon_penalty):
    """Compute EdgeScore-style priority.

    Components:
      - (1 - mastery): focus weaker skills
      - (1 - retention): revisit items whose memory has decayed
      - 0.5*(1 - pace): throttle over-challenging pacing
      - 0.3*miscon_penalty: emphasize topics with recent misconception signals

    All terms are designed to be in [0,1] to keep magnitudes comparable.
    """
    return (1 - mastery) + (1 - retention) + 0.5 * (1 - pace) + 0.3 * miscon_penalty


def retention_decay(last_t, now_t, lam=RET_LAMBDA_DEFAULT):
    """Exponential forgetting curve: retention = exp(-lambda * delta_time)."""
    return np.exp(-lam * np.clip(now_t - last_t, 0, None))


def recommend_items_for_learner(learner_id, items_df, theta, last_practice, now_step, n=5):
    """Select n items for a learner by ranking topics with the priority index.

    Steps:
      1) Compute per-topic mastery from IRT theta[learner, topic].
      2) Compute retention via exponential decay from last practice timestamps.
      3) Estimate pace (here a fixed heuristic; you can learn per-topic pace).
      4) Set misconception penalty (hook in learner misconception profiles if available).
      5) Rank topics by priority_index and choose items of medium difficulty (b≈0).
    """
    topics = items_df["topic"].values
    n_topics = theta.shape[1]

    # (1) Mastery from IRT
    mastery = theta[learner_id]

    # (2) Retention from spacing effect
    time_since = now_step - last_practice[learner_id]
    retention = retention_decay(time_since, 0)  # now_t - last_t is inside decay

    # (3) Pace heuristic (could be learned)
    pace = np.full(n_topics, 0.8)

    # (4) Misconception penalty (wire in profiles to upweight flagged topics)
    miscon_penalty = np.zeros(n_topics)

    # Rank topics by priority
    idx = np.argsort(-priority_index(mastery, retention, pace, miscon_penalty))[:n]

    # Map top topics to concrete items (choose near-medium difficulty)
    chosen = []
    for t in idx:
        cand = items_df[items_df["topic"] == t]
        cand = cand.iloc[(cand["b"] - 0).abs().argsort()[:10]]
        if len(cand) > 0:
            chosen.append(int(cand.sample(1, random_state=42)["item_id"].iloc[0]))
    return chosen
