import numpy as np
import pandas as pd
from pathlib import Path

ART = Path(__file__).resolve().parents[2] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def generate_item_bank(n_items=800, n_topics=20, seed=7):
    rng = np.random.default_rng(seed)
    items = []
    for j in range(n_items):
        topic = rng.integers(0, n_topics)
        a = np.clip(rng.normal(1.0, 0.3), 0.3, 3.0)  # discrimination
        b = rng.normal(0.0, 1.0)  # difficulty
        distractors = [f"D{d}" for d in range(4)]
        items.append([j, topic, a, b, "Item stem text", "|".join(distractors)])
    df = pd.DataFrame(items, columns=["item_id", "topic", "a", "b", "stem", "distractors"])
    return df


def simulate_interactions(n_learners=1000, n_items=800, n_topics=20, n_miscon=8, n_samples=60000, seed=7):
    rng = np.random.default_rng(seed)

    # Learner per-topic abilities
    theta = rng.normal(0, 1, size=(n_learners, n_topics))

    # Adding the 'mastery' column for learners (how good each learner is in each topic)
    mastery = rng.uniform(0, 1, size=(n_learners, n_topics))  # Mastery is between 0 (no mastery) and 1 (full mastery)

    # Misconception profiles (learners over mixture)
    learner_pi = rng.dirichlet(alpha=np.ones(n_miscon), size=n_learners)
    miscon_centers = rng.normal(0, 1, size=(n_miscon, 8))  # 8D embedding space

    # Build item bank and distractor embeddings
    items = generate_item_bank(n_items, n_topics, seed)
    item_distr_emb = rng.normal(0, 1, size=(n_items, 4, 8))

    rows = []
    for _ in range(n_samples):
        i = rng.integers(0, n_learners)
        j = rng.integers(0, n_items)
        topic = int(items.loc[j, "topic"])
        a = float(items.loc[j, "a"])
        b = float(items.loc[j, "b"])

        # 2PL probability
        logit = a * (theta[i, topic] - b)
        p_correct = 1 / (1 + np.exp(-logit))

        response_time = float(np.maximum(3, rng.normal(30 - 6 * (theta[i, topic] - b), 8)))
        attempts = int(np.clip(rng.poisson(1.2 + max(0, 0.5 - p_correct) * 2.0), 1, None))
        hints = int(np.clip(rng.poisson(0.4 + max(0, 0.6 - p_correct) * 1.5), 0, None))
        time_of_day = rng.choice(["morning", "afternoon", "evening"])
        device_type = rng.choice(["desktop", "tablet", "phone"])
        text_quality = float(rng.uniform(0, 1))
        fatigue_factor = int(rng.poisson(2))

        correct = int(rng.random() < p_correct)

        chosen_dist = -1
        miscon_flag = 0
        if not correct:
            z = rng.choice(np.arange(n_miscon), p=learner_pi[i])
            miscon_flag = 1 if rng.random() < 0.7 else 0
            emb = item_distr_emb[j]
            dists = np.linalg.norm(emb - miscon_centers[z], axis=1)
            chosen_dist = int(np.argmin(dists))

        # Now include the learner's mastery level for the topic
        learner_mastery = mastery[i, topic]  # Assign mastery to the learner-topic combination

        rows.append([i, j, topic, a, b, response_time, attempts, hints, correct, miscon_flag,
                     time_of_day, device_type, text_quality, fatigue_factor, chosen_dist, learner_mastery])

    cols = ["learner_id", "item_id", "topic", "a", "b", "response_time", "attempts", "hints",
            "correct", "misconception", "time_of_day", "device_type", "text_quality",
            "fatigue_factor", "chosen_distractor", "mastery"]  # Add 'mastery' column here
    df = pd.DataFrame(rows, columns=cols)
    return items, df


def main():
    items, interactions = simulate_interactions()
    items.to_csv(ART / "items.csv", index=False)
    msk = np.random.rand(len(interactions)) < 0.8
    interactions.loc[msk].to_csv(ART / "interactions_train.csv", index=False)
    interactions.loc[~msk].to_csv(ART / "interactions_test.csv", index=False)
    print("✅ Generated: artifacts/items.csv, interactions_train.csv, interactions_test.csv")


if __name__ == "__main__":
    main()
