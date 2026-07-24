import os, sys, pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date as date_type
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR, DISTRICT_NAMES, score_to_level
from model.explain import explain_prediction, load_model

app    = FastAPI(title="RainRisk API", version="1.0")
BUNDLE = None

@app.on_event("startup")
def load():
    global BUNDLE
    BUNDLE = load_model("full")
    print("Model loaded.")

class PredictRequest(BaseModel):
    district:    str
    date:        date_type
    rain_1d:     float
    rain_3d:     float
    rain_7d:     float
    rain_15d:    float
    rain_max3d:  float
    is_monsoon:  int = 1

@app.post("/predict")
def predict(req: PredictRequest):
    if req.district not in DISTRICT_NAMES:
        raise HTTPException(400, f"Unknown district. Valid: {DISTRICT_NAMES}")
    score, top_factors = explain_prediction(
        district   = req.district,
        rain_1d    = req.rain_1d,
        rain_3d    = req.rain_3d,
        rain_7d    = req.rain_7d,
        rain_15d   = req.rain_15d,
        rain_max3d = req.rain_max3d,
        is_monsoon = req.is_monsoon,
        model_bundle = BUNDLE,
    )
    return {
        "district":    req.district,
        "date":        str(req.date),
        "risk_score":  round(score, 4),
        "risk_level":  score_to_level(score),
        "top_factors": top_factors,
    }

@app.get("/districts")
def districts():
    return {"districts": DISTRICT_NAMES}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None}