import os
import json
import redis
from pathlib import Path

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_client = None


def get_redis():
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def enqueue_feedback(payload: dict):
    r = get_redis()
    # push JSON string to feedback queue (list)
    r.rpush("feedback_queue", json.dumps(payload))


def blpop_feedback(timeout=0):
    r = get_redis()
    item = r.blpop("feedback_queue", timeout=timeout)
    if not item:
        return None
    # item is (key, value)
    return json.loads(item[1])


def set_active_model(record: dict):
    ART = Path(__file__).resolve().parents[2] / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "active_model.json"
    path.write_text(json.dumps(record))
    return path


def get_active_model():
    ART = Path(__file__).resolve().parents[2] / "artifacts"
    path = ART / "active_model.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
