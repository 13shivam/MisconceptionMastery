
# UCB Exploration Constant (tune for exploration vs exploitation)
UCB_EXPLORATION_C = 2.0  # tune this constant to control exploration behavior in UCB1

# Retention Decay (λ) for the forgetting curve
RET_LAMBDA_DEFAULT = 0.05  # Lower values mean slower forgetting; larger = quicker decay

# Bandit choice: UCB (Upper Confidence Bound) or TS (Thompson Sampling)
BANDIT_KIND = "UCB"  # Change to "TS" to use Thompson Sampling instead

# Reward function weights
REWARD_ALPHA = 1.0  # Mastery improvement weight
REWARD_BETA  = 0.5  # Misconception reduction weight
REWARD_GAMMA = 0.1  # Time penalty weight
