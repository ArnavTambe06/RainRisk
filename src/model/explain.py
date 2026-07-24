import pickle, sys, os, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATASET, OUTPUTS_DIR

def load_model(mode="full"):
    path = os.path.join(OUTPUTS_DIR, f"rainrisk_xgb_{mode}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def explain_prediction(district, rain_1d, rain_3d, rain_7d, rain_15d,
                        rain_max3d, is_monsoon, model_bundle):
    model    = model_bundle["model"]
    features = model_bundle["features"]
    df       = pd.read_csv(PROCESSED_DATASET)
    row      = df[df["district"] == district].iloc[0]

    sample = {f: row[f] for f in features if f in row}
    sample.update({
        "rain_1d":    rain_1d,
        "rain_3d":    rain_3d,
        "rain_7d":    rain_7d,
        "rain_15d":   rain_15d,
        "rain_max3d": rain_max3d,
        "is_monsoon": is_monsoon,
    })
    if "slope_x_rain7d" in features:
        sample["slope_x_rain7d"] = sample.get("slope_deg", 0) * rain_7d / 100
    if "twi_x_rain15d" in features:
        sample["twi_x_rain15d"]  = sample.get("twi", 0) * rain_15d / 100

    X      = pd.DataFrame([sample])[features]
    score  = float(model.predict_proba(X)[0, 1])

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    contrib = dict(zip(features, np.abs(sv)))
    total   = sum(contrib.values())
    top3    = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]
    top_factors = [{"factor": k, "contribution": f"{v/total:.0%}"} for k, v in top3]

    return score, top_factors

def global_shap_plot(mode="full"):
    bundle   = load_model(mode)
    model    = bundle["model"]
    features = bundle["features"]
    df       = pd.read_csv(PROCESSED_DATASET).fillna(0)
    X        = df[features]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, show=False)
    out = os.path.join(OUTPUTS_DIR, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"SHAP summary saved: {out}")

if __name__ == "__main__":
    bundle = load_model("full")
    score, top = explain_prediction(
        district="Wayanad",
        rain_1d=95, rain_3d=220, rain_7d=380, rain_15d=610,
        rain_max3d=110, is_monsoon=1,
        model_bundle=bundle
    )
    print(f"\nRisk score: {score:.2f}")
    for t in top:
        print(f"  {t['factor']}: {t['contribution']}")
    global_shap_plot("full")