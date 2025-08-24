import time, json, os
from pathlib import Path
from edge.service.redis_client import blpop_feedback, get_active_model, get_redis
import redis, joblib, numpy as np

R = get_redis()


# helper to update per-learner bandit state in Redis
def update_bandit_state(learner_id, action, reward):
    key = f"bandit:{learner_id}"
    # stored as JSON string in Redis
    cur = R.get(key)
    if cur:
        state = json.loads(cur)
    else:
        # initialize counts and values for 3 arms
        state = {"counts": [0, 0, 0], "values": [0.0, 0.0, 0.0]}
    idx = int(action)
    state["counts"][idx] += 1
    # incremental update of mean
    n = state["counts"][idx]
    old = state["values"][idx]
    state["values"][idx] = old + (reward - old) / n
    R.set(key, json.dumps(state))


def process_feedback(ev):
    # expected fields: learner_id, item_id, correct, response_time, attempts, hints, topic (optional)
    learner = ev.get("learner_id")
    correct = int(ev.get("correct", 0))
    # simple reward: 1 for correct, 0 for incorrect (could be shaped)
    reward = 1.0 if correct == 1 else 0.0
    # choose action index heuristic: map (practice=0, remediate=1, challenge=2) from feedback or default 0
    action = int(ev.get("action", 0))
    update_bandit_state(learner, action, reward)
    # update last_practice map
    topic = ev.get("topic", None)
    if topic is not None:
        lp_key = f"learner:{learner}:last_practice"
        val = R.get(lp_key)
        if val:
            obj = json.loads(val)
        else:
            obj = {}
        obj[str(topic)] = int(ev.get("_ts", int(time.time())))
        R.set(lp_key, json.dumps(obj))
    # append to artifacts/rewards.jsonl for audit
    ART = Path(__file__).resolve().parents[2] / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "rewards.jsonl", "a") as fh:
        fh.write(json.dumps(ev) + "\\n")


def main():
    print("Policy worker started, listening on Redis feedback_queue...")
    while True:
        item = blpop_feedback(timeout=5)
        if item is None:
            time.sleep(0.1)
            continue
        try:
            process_feedback(item)
        except Exception as e:
            print("Error processing feedback:", e)


if __name__ == '__main__':
    main()
