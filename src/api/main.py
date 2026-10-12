import os
import sys
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date as date_type, datetime, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OUTPUTS_DIR, DISTRICT_NAMES, DISTRICTS, score_to_level
from model.explain import explain_prediction, load_model


BUNDLE = None

# ── In-memory cache ───────────────────────────────────────────────────────────
CACHE_TTL      = 3600        # 60 minutes for a fully successful fetch
PARTIAL_TTL    = 300         # 5 minutes if some districts failed (retry sooner)
REFRESH_EVERY  = 55 * 60     # background refresh, just before the TTL expires
FETCH_WORKERS  = 4           # parallel NASA POWER calls; set to 1 if NASA throttles

_cache = {
    "data":       None,
    "fetched_at": None,
    "expires_at": 0.0,
    "ttl":        CACHE_TTL,
}
_refresh_lock = threading.Lock()


def _cache_is_fresh() -> bool:
    return _cache["data"] is not None and time.time() < _cache["expires_at"]


# ── Safe float ────────────────────────────────────────────────────────────────
def safe_float(value, default=0.0):
    try:
        value = float(value)
        if pd.isna(value) or math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


# ── Core: compute risk for every district ─────────────────────────────────────
def _fetch_features(d, start, end):
    """Network-bound part: runs in a thread pool."""
    from data_pipeline.gpm_rainfall import fetch_daily_rainfall, engineer_rainfall_features
    series = fetch_daily_rainfall(d["lat"], d["lon"], start, end)
    return engineer_rainfall_features(series)


def compute_all_districts():
    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=25)).strftime("%Y-%m-%d")

    print("[FETCH] Fetching NASA POWER for all districts...")
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        jobs = [(d, pool.submit(_fetch_features, d, start, end)) for d in DISTRICTS]

    # Model inference stays sequential (SHAP/XGBoost, keeps memory low)
    results = []
    for d, fut in jobs:
        try:
            feat_df  = fut.result()
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

    if not results:
        raise RuntimeError("No districts could be fetched")

    return {
        "districts":  results,
        "fetched_at": end,
        "model_auc":  0.8606,   # TODO: hardcoded - standardize with your paper/synopsis figures
        "note":       "obs_date is actual NASA POWER observation date (2-3 day lag expected)",
        "cached":     True,
    }


def refresh_cache(force: bool = False):
    """Only one refresh runs at a time; others wait, then reuse the result."""
    with _refresh_lock:
        if not force and _cache_is_fresh():
            return  # another thread refreshed while we were waiting

        response = compute_all_districts()
        complete = len(response["districts"]) == len(DISTRICTS)
        ttl      = CACHE_TTL if complete else PARTIAL_TTL

        now = time.time()
        _cache["data"]       = response
        _cache["fetched_at"] = now
        _cache["expires_at"] = now + ttl
        print(f"[CACHE] Stored {len(response['districts'])}/{len(DISTRICTS)} districts (ttl {ttl}s)")


def cache_refresher():
    """Runs for the life of the process: warms at startup, then refreshes hourly."""
    while True:
        try:
            print("[WARMUP] Refreshing cache in background...")
            refresh_cache(force=True)
        except Exception as e:
            print(f"[WARMUP] Failed: {e}")
        time.sleep(REFRESH_EVERY)


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global BUNDLE
    BUNDLE = load_model("full")
    print("Model loaded.")
    threading.Thread(target=cache_refresher, daemon=True).start()
    yield


app = FastAPI(title="RainRisk API", version="1.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS on the host, e.g. "https://rainrisk.vercel.app,http://localhost:5173"
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
    if not _cache_is_fresh():
        try:
            refresh_cache()
        except Exception as e:
            print(f"[WARN] refresh failed: {e}")

    # Serve stale data rather than nothing if the refresh failed
    if _cache["data"] is None:
        raise HTTPException(503, "Rainfall data temporarily unavailable. Retry shortly.")
    return _cache["data"]


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
    age   = time.time() - _cache["fetched_at"]
    fresh = _cache_is_fresh()
    return {
        "status":      "warm" if fresh else "stale",
        "age_seconds": int(age),
        "ttl_seconds": _cache["ttl"],
        "ready":       fresh,
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