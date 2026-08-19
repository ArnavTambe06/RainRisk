import os
import sys
import pickle
import math
import time
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date as date_type, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OUTPUTS_DIR, DISTRICT_NAMES, DISTRICTS, score_to_level
from model.explain import explain_prediction, load_model


app = FastAPI(title="RainRisk API", version="1.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE = None

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache = {
    "data":       None,
    "fetched_at": None,
    "ttl":        3600,   # 60 minutes
}

# ── Safe float ────────────────────────────────────────────────────────────────
def safe_float(value, default=0.0):
    try:
        value = float(value)
        if pd.isna(value) or math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default

# ── Background cache warmup ───────────────────────────────────────────────────
def warm_cache():
    import requests
    time.sleep(5)  # wait for server to fully start
    print("[WARMUP] Pre-fetching all districts in background...")
    try:
        requests.get("http://localhost:8000/all-districts", timeout=300)
        print("[WARMUP] Cache warmed successfully")
    except Exception as e:
        print(f"[WARMUP] Failed: {e}")

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load():
    global BUNDLE
    BUNDLE = load_model("full")
    print("Model loaded.")
    threading.Thread(target=warm_cache, daemon=True).start()

# ── Request models ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    district:   str
    date:       date_type
    rain_1d:    float
    rain_3d:    float
    rain_7d:    float
    rain_15d:   float
    rain_max3d: float
    is_monsoon: int = 1

class LocationRequest(BaseModel):
    lat:        float
    lon:        float
    place_name: str = "Custom Location"

# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    if req.district not in DISTRICT_NAMES:
        raise HTTPException(400, f"Unknown district. Valid: {DISTRICT_NAMES}")

    score, top_factors = explain_prediction(
        district     = req.district,
        rain_1d      = req.rain_1d,
        rain_3d      = req.rain_3d,
        rain_7d      = req.rain_7d,
        rain_15d     = req.rain_15d,
        rain_max3d   = req.rain_max3d,
        is_monsoon   = req.is_monsoon,
        model_bundle = BUNDLE,
    )
    return {
        "district":    req.district,
        "date":        str(req.date),
        "risk_score":  round(safe_float(score), 4),
        "risk_level":  score_to_level(score),
        "top_factors": top_factors,
    }

# ── /all-districts ────────────────────────────────────────────────────────────
@app.get("/all-districts")
def all_districts_risk():
    from data_pipeline.gpm_rainfall import fetch_daily_rainfall, engineer_rainfall_features

    # Return cached data if still fresh
    if _cache["data"] and _cache["fetched_at"]:
        age = time.time() - _cache["fetched_at"]
        if age < _cache["ttl"]:
            print(f"[CACHE] Returning cached data (age: {int(age)}s)")
            return _cache["data"]

    print("[FETCH] Cache miss — fetching NASA POWER for all districts...")
    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=25)).strftime("%Y-%m-%d")

    results = []
    for d in DISTRICTS:
        try:
            series   = fetch_daily_rainfall(d["lat"], d["lon"], start, end)
            feat_df  = engineer_rainfall_features(series)
            features = feat_df.iloc[-1]
            obs_date = feat_df.index[-1].strftime("%Y-%m-%d")

            score, top = explain_prediction(
                district     = d["name"],
                rain_1d      = float(features["rain_1d"]),
                rain_3d      = float(features["rain_3d"]),
                rain_7d      = float(features["rain_7d"]),
                rain_15d     = float(features["rain_15d"]),
                rain_max3d   = float(features["rain_max3d"]),
                is_monsoon   = int(features["is_monsoon"]),
                model_bundle = BUNDLE,
            )
            results.append({
                "district":      d["name"],
                "state":         d["state"],
                "lat":           d["lat"],
                "lon":           d["lon"],
                "risk_score":    round(safe_float(score), 4),
                "risk_level":    score_to_level(score),
                "top_factors":   top,
                "rain_today":    round(float(features["rain_1d"]),  1),
                "rain_7d":       round(float(features["rain_7d"]),  1),
                "rain_15d":      round(float(features["rain_15d"]), 1),
                "obs_date":      obs_date,
                "data_lag_days": (datetime.utcnow() - datetime.strptime(obs_date, "%Y-%m-%d")).days,
                "fetched_at":    end,
            })
        except Exception as e:
            print(f"[WARN] {d['name']}: {e}")

    response = {
        "districts":  results,
        "fetched_at": end,
        "model_auc":  0.8606,
        "note":       "obs_date is actual NASA POWER observation date (2-3 day lag expected)",
        "cached":     False,
    }

    _cache["data"]       = {**response, "cached": True}
    _cache["fetched_at"] = time.time()
    print(f"[CACHE] Stored {len(results)} districts")

    return response

# ── /predict-location ─────────────────────────────────────────────────────────
@app.post("/predict-location")
def predict_location(req: LocationRequest):
    from data_pipeline.gpm_rainfall import fetch_daily_rainfall, engineer_rainfall_features

    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=25)).strftime("%Y-%m-%d")

    series   = fetch_daily_rainfall(req.lat, req.lon, start, end)
    feat_df  = engineer_rainfall_features(series)
    features = feat_df.iloc[-1]
    obs_date = feat_df.index[-1].strftime("%Y-%m-%d")

    # Use nearest trained district for terrain features
    def dist(d):
        return math.sqrt((d["lat"] - req.lat)**2 + (d["lon"] - req.lon)**2)
    nearest = min(DISTRICTS, key=dist)

    score, top = explain_prediction(
        district     = nearest["name"],
        rain_1d      = float(features["rain_1d"]),
        rain_3d      = float(features["rain_3d"]),
        rain_7d      = float(features["rain_7d"]),
        rain_15d     = float(features["rain_15d"]),
        rain_max3d   = float(features["rain_max3d"]),
        is_monsoon   = int(features["is_monsoon"]),
        model_bundle = BUNDLE,
    )

    return {
        "place_name":        req.place_name,
        "lat":               req.lat,
        "lon":               req.lon,
        "nearest_district":  nearest["name"],
        "risk_score":        round(safe_float(score), 4),
        "risk_level":        score_to_level(score),
        "top_factors":       top,
        "rain_today":        round(float(features["rain_1d"]),  1),
        "rain_7d":           round(float(features["rain_7d"]),  1),
        "rain_15d":          round(float(features["rain_15d"]), 1),
        "obs_date":          obs_date,
        "data_lag_days":     (datetime.utcnow() - datetime.strptime(obs_date, "%Y-%m-%d")).days,
        "fetched_at":        end,
        "note":              f"Terrain proxied from nearest trained district: {nearest['name']}",
    }

# ── /cache-status ─────────────────────────────────────────────────────────────
@app.get("/cache-status")
def cache_status():
    if not _cache["fetched_at"]:
        return {"status": "empty", "age_seconds": None, "ready": False}
    age = time.time() - _cache["fetched_at"]
    return {
        "status":      "warm" if age < _cache["ttl"] else "stale",
        "age_seconds": int(age),
        "ttl_seconds": _cache["ttl"],
        "ready":       age < _cache["ttl"],
        "fetched_at":  _cache["data"]["fetched_at"] if _cache["data"] else None,
    }

# ── /districts ────────────────────────────────────────────────────────────────
@app.get("/districts")
def districts():
    return {"districts": DISTRICT_NAMES}

# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": BUNDLE is not None,
        "cache_ready":  _cache["data"] is not None,
    }