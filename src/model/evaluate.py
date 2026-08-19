import os, sys, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATASET, OUTPUTS_DIR, RANDOM_SEED

KNOWN_EVENTS_WITH_DATE = {
    "Wayanad":    "2024-07-29",
    "Raigad":     "2021-07-22",
    "Ratnagiri":  "2021-07-23",
    "Malappuram": "2019-08-10",
    "Idukki":     "2021-10-16",
}

def load_model(mode):
    path = os.path.join(OUTPUTS_DIR, f"rainrisk_xgb_{mode}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def evaluate_model(mode):
    bundle   = load_model(mode)
    model    = bundle["model"]
    features = bundle["features"]
    df       = pd.read_csv(PROCESSED_DATASET).fillna(0)
    X        = df[features]
    y        = df["landslide"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n{mode.upper()} — AUC: {auc:.4f}  F1: {f1:.4f}")
    return auc, f1, y_test, y_prob

def ablation_study():
    print("\n=== Ablation Study ===")
    df = pd.read_csv(PROCESSED_DATASET).fillna(0)
    y  = df["landslide"]
    results = {}
    for mode in ["terrain_only", "full"]:
        bundle   = load_model(mode)
        X        = df[bundle["features"]]
        _, X_t, _, y_t = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
        )
        auc = roc_auc_score(y_t, bundle["model"].predict_proba(X_t)[:, 1])
        results[mode] = auc
        print(f"  {mode:20s} AUC: {auc:.4f}")
    print(f"\nTerrain only AUC:        {results['terrain_only']:.4f}")
    print(f"Terrain + Rainfall AUC:  {results['full']:.4f}")
    print(f"Improvement:             +{results['full'] - results['terrain_only']:.4f}")
    return results

def plot_roc_comparison():
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.read_csv(PROCESSED_DATASET).fillna(0)
    y  = df["landslide"]
    for mode, label in [("terrain_only", "Terrain Only (Baseline)"),
                         ("full",         "Terrain + Rainfall (RainRisk)")]:
        bundle = load_model(mode)
        X      = df[bundle["features"]]
        _, X_t, _, y_t = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
        )
        y_prob      = bundle["model"].predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_t, y_prob)
        auc         = roc_auc_score(y_t, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Baseline vs RainRisk")
    ax.legend()
    out = os.path.join(OUTPUTS_DIR, "roc_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nROC plot saved: {out}")

def lead_time_analysis():
    print("\n=== Lead Time Analysis (Known Historical Events) ===")
    bundle   = load_model("full")
    model    = bundle["model"]
    features = bundle["features"]
    df       = pd.read_csv(PROCESSED_DATASET).fillna(0)
    for district, event_date in KNOWN_EVENTS_WITH_DATE.items():
        rows = df[df["district"] == district]
        if rows.empty:
            continue
        probs     = model.predict_proba(rows[features])[:, 1]
        high_risk = (probs >= 0.70).sum()
        print(f"  {district:20s}  event={event_date}  "
              f"HIGH-risk samples={high_risk}/{len(rows)}")

if __name__ == "__main__":
    evaluate_model("terrain_only")
    evaluate_model("full")
    ablation_study()
    plot_roc_comparison()
    lead_time_analysis()