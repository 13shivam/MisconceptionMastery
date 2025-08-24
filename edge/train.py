import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
ART = Path(__file__).resolve().parents[2] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def load_data():
    interactions_train = pd.read_csv(ART / "interactions_train.csv")
    interactions_test = pd.read_csv(ART / "interactions_test.csv")
    items = pd.read_csv(ART / "items.csv")
    return interactions_train, interactions_test, items


def feature_engineering(df, items):
    df = add_mastery_from_irt(df, ART / "irt_state.npz")
    df = pd.get_dummies(df, columns=["time_of_day", "device_type"], drop_first=True)
    features = ["mastery", "response_time", "attempts", "hints",
                "text_quality", "fatigue_factor"] + [col for col in df.columns if
                                                     col.startswith('time_of_day') or col.startswith('device_type')]
    return df[features], df["misconception"]


def add_mastery_from_irt(df, irt_path):
    if not irt_path.exists():
        df = df.copy()
        df["mastery"] = 0.0
        return df

    data = np.load(irt_path, allow_pickle=True)
    theta = data["theta"]
    df = df.copy()
    mastery = []
    for _, r in df.iterrows():
        learner_id = int(r["learner_id"])
        topic = int(r["topic"])
        if 0 <= learner_id < theta.shape[0] and 0 <= topic < theta.shape[1]:
            mastery.append(float(theta[learner_id, topic]))
        else:
            mastery.append(0.0)
    df["mastery"] = mastery
    return df


def train_misconception_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, ART / "rf_miscon.joblib")
    return rf


def validate_model(rf, X_test, y_test):
    preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {accuracy:.3f}")


def main():
    interactions_train, interactions_test, items = load_data()
    X_train, y_train = feature_engineering(interactions_train, items)
    X_test, y_test = feature_engineering(interactions_test, items)
    rf = train_misconception_model(X_train, y_train)
    validate_model(rf, X_test, y_test)


if __name__ == "__main__":
    main()
