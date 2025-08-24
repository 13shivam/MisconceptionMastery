#!/usr/bin/env python3
"""
validate.py

Run offline validation of the misconception classifier and produce:
 - console summary
 - reliability (calibration) plot
 - ROC and PR plots
 - per-topic confusion heatmaps (for topics with at least 5 examples)
 - artifacts/validate_report.html (simple HTML summary with embedded plots)
"""

import os
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# -------- Paths & config --------
ART = Path(__file__).resolve().parents[2] / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
PLOTS = ART / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

THRESH = float(os.getenv("EDGE_THRESH", 0.5))  # default decision threshold


# -------- Utilities --------
def png_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def save_plot(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -------- Feature engineering (matching training pipeline) --------
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same categorical encoding used during training.
    This converts raw categorical columns to one-hot encoded features.
    """
    df = df.copy()

    # One-hot encode device_type
    if 'device_type' in df.columns:
        device_dummies = pd.get_dummies(df['device_type'], prefix='device_type', dtype=int)
        # Ensure all expected device types are present
        for device in ['phone', 'tablet', 'desktop']:
            col_name = f'device_type_{device}'
            if col_name not in device_dummies.columns:
                device_dummies[col_name] = 0
        df = pd.concat([df, device_dummies], axis=1)
        df = df.drop('device_type', axis=1)

    # One-hot encode time_of_day
    if 'time_of_day' in df.columns:
        time_dummies = pd.get_dummies(df['time_of_day'], prefix='time_of_day', dtype=int)
        # Ensure all expected time periods are present
        for time_period in ['morning', 'afternoon', 'evening']:
            col_name = f'time_of_day_{time_period}'
            if col_name not in time_dummies.columns:
                time_dummies[col_name] = 0
        df = pd.concat([df, time_dummies], axis=1)
        df = df.drop('time_of_day', axis=1)

    return df


# -------- Calibration (ECE) --------
def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: average gap between confidence and accuracy across bins."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


# -------- Plots --------
def reliability_plot(probs, labels, path: Path, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() == 0:
            continue
        xs.append(probs[m].mean())
        ys.append(labels[m].mean())
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.scatter(xs, ys, label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    save_plot(fig, path)


def roc_plot(y_true, probs, path: Path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    save_plot(fig, path)


def pr_plot(y_true, probs, path: Path):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    fig = plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    save_plot(fig, path)


# -------- Feature construction helpers --------
def add_mastery_from_irt(df: pd.DataFrame, irt_path: Path) -> pd.DataFrame:
    """
    Fill df['mastery'] using IRT theta[learner_id, topic].
    If any index out-of-range, default to 0.0 for that row.
    Only used as fallback if mastery column doesn't already exist.
    """
    if not irt_path.exists():
        print("Warning: No IRT state file found, using mastery = 0.0")
        df = df.copy()
        df["mastery"] = 0.0
        return df

    try:
        data = np.load(irt_path, allow_pickle=True)
        theta = data["theta"]  # shape [n_learners, n_topics]
        df = df.copy()
        m = []
        for _, r in df.iterrows():
            i = int(r["learner_id"])
            t = int(r["topic"])
            if 0 <= i < theta.shape[0] and 0 <= t < theta.shape[1]:
                m.append(float(theta[i, t]))
            else:
                m.append(0.0)
        df["mastery"] = m
        print("Added mastery from IRT state")
        return df
    except Exception as e:
        print(f"Error loading IRT state: {e}, using mastery = 0.0")
        df = df.copy()
        df["mastery"] = 0.0
        return df


# -------- Model selection & evaluation helpers --------
def pick_model():
    cal = ART / "rf_miscon_cal.joblib"
    uncal = ART / "rf_miscon.joblib"
    if cal.exists():
        return joblib.load(cal), cal.name
    if uncal.exists():
        return joblib.load(uncal), uncal.name
    raise FileNotFoundError("No model artifact found in artifacts/ (rf_miscon_cal.joblib or rf_miscon.joblib)")


def eval_at_threshold(y, probs, thresh):
    preds = (probs >= thresh).astype(int)
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y, preds)
    return acc, prec, rec, f1, cm, preds


def best_f1_threshold(y, probs, steps=101):
    """
    Robust search for the threshold in [0,1] that yields best F1.
    steps: how many grid points to evaluate (default 101 -> 0.00,0.01,...,1.00)
    """
    ths = np.linspace(0.0, 1.0, steps)
    best_t = 0.5
    best_f1 = -1.0
    for t in ths:
        _, _, rec, f1, _, _ = eval_at_threshold(y, probs, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


# -------- Per-topic analysis --------
def per_topic_analysis(df_te: pd.DataFrame, y: np.ndarray, probs: np.ndarray, out_dir: Path, min_count=5):
    """
    Compute per-topic confusion matrices and save heatmaps into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = sorted(df_te["topic"].unique())
    for t in topics:
        mask = df_te["topic"] == t
        if mask.sum() < min_count:
            continue
        y_t = y[mask.values]
        probs_t = probs[mask.values]
        preds_t = (probs_t >= THRESH).astype(int)
        cm = confusion_matrix(y_t, preds_t, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion matrix (topic={t})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        out_file = out_dir / f"confusion_topic_{t}.png"
        save_plot(fig, out_file)


# -------- HTML report generator --------
def generate_html_report(metrics: dict, paths: dict, out_path: Path):
    html = f"""<!DOCTYPE html>
<html>
<head><title>EDGE Validation Report</title></head>
<body bgcolor="#FFFFFF" text="#000000">
<h1>EDGE – Validation Report</h1>
<p>Model file: <code>{metrics['model_name']}</code></p>
<p>Test rows: <b>{metrics['n_rows']}</b></p>

<h2>Summary (threshold = {metrics['threshold']:.2f})</h2>
<ul>
  <li>Accuracy: <b>{metrics['acc']:.3f}</b></li>
  <li>Precision: <b>{metrics['prec']:.3f}</b></li>
  <li>Recall: <b>{metrics['rec']:.3f}</b></li>
  <li>F1: <b>{metrics['f1']:.3f}</b></li>
  <li>ECE: <b>{metrics['ece']:.3f}</b> (lower is better)</li>
  <li>Brier: <b>{metrics['brier']:.3f}</b> (lower is better)</li>
  <li>ROC-AUC: <b>{metrics['roc_auc']:.3f}</b></li>
  <li>PR-AUC: <b>{metrics['pr_auc']:.3f}</b></li>
</ul>

<h3>Best-F1 Threshold</h3>
<p>Best-F1 threshold ≈ <b>{metrics['best_f1_thresh']:.3f}</b> with F1 ≈ <b>{metrics['best_f1']:.3f}</b>.
Use this when you want a balanced precision–recall tradeoff.</p>

<h2>Confusion Matrix (threshold = {metrics['threshold']:.2f})</h2>
<pre>
TN={metrics['cm'][0][0]}   FP={metrics['cm'][0][1]}
FN={metrics['cm'][1][0]}   TP={metrics['cm'][1][1]}
</pre>

<h2>Plots</h2>
<p><b>Reliability Diagram</b></p>
<img src="data:image/png;base64,{png_to_base64(paths['reliability'])}" width="480" height="360"/>

<p><b>ROC Curve</b></p>
<img src="data:image/png;base64,{png_to_base64(paths['roc'])}" width="480" height="360"/>

<p><b>Precision–Recall Curve</b></p>
<img src="data:image/png;base64,{png_to_base64(paths['pr'])}" width="480" height="360"/>

<hr>
<h2>How to use these numbers in practice</h2>
<ul>
  <li><b>High recall needed?</b> (don't miss struggling students) — lower threshold toward <code>{max(0.1, metrics['best_f1_thresh'] - 0.1):.2f}</code>.</li>
  <li><b>High precision needed?</b> (avoid over-remediation) — raise threshold toward <code>{min(0.9, metrics['best_f1_thresh'] + 0.1):.2f}</code>.</li>
  <li><b>Calibrated probabilities</b> (ECE ≈ {metrics['ece']:.3f}) enable policy rules like:
    <ul>
      <li>If <i>p(misconception)</i> ≥ 0.70 → show targeted remediation now.</li>
      <li>If 0.40 ≤ <i>p</i> &lt; 0.70 → schedule practice in 24h (spaced repetition).</li>
      <li>If <i>p</i> &lt; 0.40 → challenge or move on.</li>
    </ul>
  </li>
  <li><b>Watch drift:</b> if ECE/Brier worsen over weeks, re-calibrate or re-train.</li>
</ul>

<p><i>Report generated by validate.py</i></p>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# -------- Main routine --------
def main():
    # ---- Load data & model ----
    try:
        df_te = pd.read_csv(ART / "interactions_test.csv")
        print(f"Loaded test data with shape: {df_te.shape}")
        print(f"Columns: {list(df_te.columns)}")
    except FileNotFoundError:
        print("Error: interactions_test.csv not found in artifacts/")
        return

    try:
        model, model_name = pick_model()
        print(f"Loaded model: {model_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Check if mastery column already exists, otherwise build from IRT
    if 'mastery' not in df_te.columns:
        df_te = add_mastery_from_irt(df_te, ART / "irt_state.npz")
    else:
        print("Using existing mastery column from CSV")

    # Apply categorical encoding to match training pipeline
    df_te = encode_categorical_features(df_te)

    # Required features (after encoding - must match training pipeline)
    expected_features = [
        "mastery",
        "response_time",
        "attempts",
        "hints",
        "text_quality",
        "fatigue_factor",
        "device_type_desktop",
        "device_type_phone",
        "device_type_tablet",
        "time_of_day_afternoon",
        "time_of_day_evening",
        "time_of_day_morning",
    ]

    # Check which features the model actually expects
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        print(f"Model expects features: {model_features}")
        feat_cols = model_features
    else:
        # Fall back to expected features
        feat_cols = expected_features

    # Ensure all required columns exist
    missing = set(feat_cols) - set(df_te.columns)
    if missing:
        print(f"Warning: Missing columns {missing}, will be filled with 0")
        for col in missing:
            df_te[col] = 0

    # Select features in the right order
    X = df_te[feat_cols]
    y = df_te["misconception"].astype(int).values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")

    # ---- Predict ----
    probs = model.predict_proba(X)[:, 1]

    # Core metrics @ default threshold
    acc, prec, rec, f1, cm, preds = eval_at_threshold(y, probs, THRESH)
    ece = expected_calibration_error(probs, y, n_bins=10)
    brier = brier_score_loss(y, probs)

    # Curves
    try:
        roc_auc = roc_auc_score(y, probs)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y, probs)
    except Exception:
        pr_auc = float("nan")

    # Best-F1 threshold (helpful default)
    t_best, f1_best = best_f1_threshold(y, probs)

    # ---- Plots ----
    rel_path = PLOTS / "reliability.png"
    reliability_plot(probs, y, rel_path, n_bins=10)

    roc_path = PLOTS / "roc.png"
    roc_plot(y, probs, roc_path)

    pr_path = PLOTS / "pr.png"
    pr_plot(y, probs, pr_path)

    # Per-topic confusion heatmaps
    per_topic_analysis(df_te, y, probs, PLOTS, min_count=5)

    # ---- HTML report ----
    metrics = {
        "model_name": model_name,
        "n_rows": len(df_te),
        "threshold": THRESH,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "ece": ece,
        "brier": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1_thresh": t_best,
        "best_f1": f1_best,
        "cm": cm.tolist(),
    }
    paths = {"reliability": rel_path, "roc": roc_path, "pr": pr_path}
    out_html = ART / "validate_report.html"
    generate_html_report(metrics, paths, out_html)

    # ---- Console summary ----
    print(
        f"✅ Validation: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}, "
        f"ECE={ece:.3f}, Brier={brier:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}"
    )
    print(f"ℹ️ Best-F1 threshold ≈ {t_best:.3f} (F1≈{f1_best:.3f})")
    print(f"🖼  Plots saved under: {PLOTS}")
    print(f"📄 HTML report: {out_html}")


if __name__ == "__main__":
    main()