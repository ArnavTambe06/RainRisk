import os, sys, pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATASET, MODEL_PATH, OUTPUTS_DIR, XGBOOST_PARAMS, RANDOM_SEED

TERRAIN_FEATURES = [
    "slope_deg","elevation_m","aspect_deg","curvature",
    "twi","lulc_class","lithology","dist_river_km","drainage_density","ndvi"
]

RAINFALL_FEATURES = [
    "rain_1d","rain_3d","rain_7d","rain_15d","rain_max3d","is_monsoon",
    "slope_x_rain7d","twi_x_rain15d"
]

def load_data(mode="terrain_only"):
    df = pd.read_csv(PROCESSED_DATASET)
    if mode == "terrain_only":
        features = TERRAIN_FEATURES
    elif mode == "full":
        features = TERRAIN_FEATURES + RAINFALL_FEATURES
    else:
        raise ValueError("mode must be 'terrain_only' or 'full'")
    X = df[features].fillna(0)
    y = df["landslide"]
    return X, y, features

def train(mode="terrain_only"):
    X, y, features = load_data(mode)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)

    print(f"\n=== {mode.upper()} MODEL RESULTS ===")
    print(f"AUC:  {auc:.4f}")
    print(f"F1:   {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Event","Landslide"]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_auc = cross_val_score(XGBClassifier(**XGBOOST_PARAMS), X, y,
                              cv=cv, scoring="roc_auc")
    print(f"5-Fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    save_path = MODEL_PATH.replace(".pkl", f"_{mode}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "features": features, "mode": mode,
                     "auc": auc, "f1": f1}, f)
    print(f"\nModel saved: {save_path}")

    results = {"mode": mode, "auc": auc, "f1": f1,
               "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std()}
    return model, features, results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="terrain_only", choices=["terrain_only","full"])
    args = p.parse_args()
    train(args.mode)