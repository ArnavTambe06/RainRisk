# RainRisk

**Dynamic Rainfall-Aware Landslide Susceptibility Prediction for the Western Ghats, India**

MCA Research Project | BVIMTECH Navi Mumbai | Arnav | 2026–27

RainRisk is a machine learning system that predicts landslide risk for Western Ghats districts by combining static terrain features with live satellite rainfall data, and explains every prediction using SHAP.

> **Status:** Core ML pipeline, FastAPI backend, and React dashboard are built and operational end-to-end. Model validated against the July 2024 Wayanad disaster. NRSC inventory integration and model retrain (Path B) still in progress. See [Known Issues](#known-issues--open-items) before citing figures externally.

---

## Table of Contents

- [Problem](#problem)
- [Project Structure](#project-structure)
- [Current Build Status](#current-build-status)
- [Setup](#setup)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Model & Validation Results](#model--validation-results)
- [Study Region](#study-region)
- [Data Sources](#data-sources)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Known Issues / Open Items](#known-issues--open-items)
- [Author](#author)

---

## Problem

Existing landslide susceptibility maps in India are static — drawn once from terrain data (slope, soil, elevation) and never updated. They give the same risk level on a dry day as during a monsoon downpour. The July 2024 Wayanad disaster (400+ deaths) exposed this gap directly: the terrain was already known to be high-risk, but no system could translate live rainfall into a real-time warning. RainRisk adds a dynamic rainfall layer on top of static terrain susceptibility to close that gap.

## Project Structure

```
rainrisk/
├── data/
│   ├── raw/
│   │   ├── terrain/        # SRTM DEM, slope, LULC (downloaded once from USGS/GEE)
│   │   ├── rainfall/       # NASA POWER / GPM rainfall, historical + live
│   │   └── inventory/      # NRSC Landslide Atlas 2023 event records
│   └── processed/          # Merged feature matrix (terrain + rainfall + labels)
├── src/
│   ├── data_pipeline/      # Download + preprocessing scripts
│   │   ├── terrain_features.py
│   │   ├── gpm_rainfall.py
│   │   └── build_dataset.py
│   ├── model/              # Training, evaluation, SHAP
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   └── api/                # FastAPI deployment
│       └── main.py
├── frontend/                # React dashboard
├── notebooks/               # EDA and experiments
├── outputs/                 # Trained models, evaluation reports, plots
└── logs/
```

## Current Build Status

| Component | Status | Notes |
|---|---|---|
| Terrain feature layer | ✅ Built | Hardcoded from published literature for target districts |
| Rainfall feature layer | ✅ Built | NASA POWER GPM data (2015–2024), 3/7/15-day rolling windows |
| Feature engineering | ✅ Built | Includes `slope_x_rain7d`, `twi_x_rain15d`, `rain_intensity_ratio`, `soil_saturation_proxy` |
| XGBoost model | ✅ Built | Outperforms terrain-only baseline on historical events |
| SHAP explainability | ✅ Built | `rain_3d` and `slope_x_rain7d` confirmed as top predictors |
| FastAPI backend | ✅ Built | Live, serving real data — see [API Reference](#api-reference) |
| React dashboard | ✅ Built | 4 screens: Overview, Regional, Modeling, Alerts |
| NRSC inventory integration | 🔄 In progress | Part of "Path B" — new features + retrain |
| Research paper | ⏳ Not started | Deliberately sequenced after system completion |

## Setup

```bash
git clone <repo-url>
cd rainrisk
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd frontend
npm install
```

For local configuration, create a `.env` file in the project root:

```env
NASA_POWER_URL=https://power.larc.nasa.gov/api/temporal/daily/point
API_BASE_URL=http://localhost:8000
ALLOWED_ORIGINS=http://localhost:5173
```

**Provider credentials for authenticated alternatives:**
- NASA Earthdata account (free) — needed for GPM/POWER rainfall API access
- If required, set them in `.env`:
  ```
  NASA_EARTHDATA_USERNAME=your_username
  NASA_EARTHDATA_PASSWORD=your_password
  ```

## Running the System

**Backend:**
```bash
cd src/api
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

The dashboard polls the backend's cache-status endpoint and refreshes hourly via a `useLiveDistricts` hook.

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Returns risk score + SHAP breakdown for a given district and date |
| `/all-districts` | GET | Returns risk scores for all tracked districts (cached, background warm-up) |
| `/predict-location` | POST | Returns risk score for an arbitrary lat/long, not just tracked districts |
| `/cache-status` | GET | Reports cache freshness — used by the frontend's polling hook |
| `/health` | GET | Basic liveness check |

CORS is enabled on all endpoints. NASA POWER's reporting lag is handled by dropping trailing NaN rows; responses surface `obs_date` and `data_lag_days` so consumers know how current the rainfall data actually is.

**Example response:**
```json
{
  "district": "Wayanad",
  "date": "2024-07-29",
  "risk_score": 0.91,
  "risk_level": "HIGH",
  "top_factors": [
    { "factor": "7_day_rainfall_mm", "contribution": "71%" },
    { "factor": "slope_degrees", "contribution": "18%" },
    { "factor": "lulc_class", "contribution": "11%" }
  ]
}
```

## Model & Validation Results

**Training data:** 42 verified historical landslide events, augmented with monsoon-season negative samples to correct class imbalance.

**Wayanad 2024 replay:** Replaying the disaster through the trained model, the disaster day scored **0.869** and was correctly classified **HIGH risk**, with 3-day maximum rainfall (`rain_max3d`) as the top SHAP-attributed factor. The model flagged HIGH risk with an **8-day lead time** ahead of the event.

> ⚠️ Two slightly different versions of the Wayanad replay figures (score and lead time) currently exist across project documents. Verify and standardize before any mentor presentation or submission.

**Key technical findings:**
- Class imbalance correction (monsoon-season negative sampling) fixed an early failure mode where the model scored all cases LOW.
- Engineered interaction features (`rain_intensity_ratio`, `soil_saturation_proxy`) outperformed raw terrain/rainfall variables individually.
- AUC-PR is used as the primary evaluation metric over AUC-ROC, given the class imbalance inherent to landslide event data.

## Study Region

Target districts span the top-risk Western Ghats zones per the NRSC Landslide Atlas of India 2023:

- **Kerala:** Wayanad, Idukki, Thrissur
- **Karnataka:** Kodagu (Coorg), Chikmagalur, Uttara Kannada
- **Maharashtra:** Ratnagiri, Sindhudurg, Satara, Raigad
- **Tamil Nadu:** Nilgiris

> ⚠️ District count is inconsistent across project docs — some reference the top 20 districts, others (the tracker) reference 90. Resolve before finalizing scope language for the paper.

## Data Sources

All sources are free; some require a free account registration.

| Dataset | Source | Used For |
|---|---|---|
| NRSC Landslide Atlas of India 2023 | Geological Survey of India | Training labels — historical landslide locations |
| SRTM DEM 30m | USGS EarthExplorer | Slope, elevation, aspect, curvature |
| NASA POWER / GPM IMERG | NASA Earthdata | Real-time + historical rainfall, 3-hour intervals |
| Bhukosh Portal | Geological Survey of India | Soil type / lithology maps |
| IMD Rainfall Records | India Meteorological Department | Validation, cross-comparison |
| Sentinel-2 / Landsat 8 | ESA / USGS via Google Earth Engine | Land use / land cover (LULC) |

## Tech Stack

**ML/Data:** Python, XGBoost, SHAP, scikit-learn (5-fold CV), pandas, geopandas
**Backend:** FastAPI
**Frontend:** React, Recharts, Leaflet
**Rainfall data:** NASA POWER GPM API

## Roadmap

| Month | Phase | Deliverable |
|---|---|---|
| 1 | Data collection & setup | Merged dataset: terrain + rainfall + labels |
| 2 | Base model (terrain only) | Baseline XGBoost, AUC/F1 report |
| 3 | Add dynamic rainfall | Improved model with rolling rainfall features |
| 4 | API + Explainability | FastAPI endpoint + SHAP explanations |
| 5 | Evaluation & Validation | Comparison vs. baseline, ablation study |
| 6 | Paper writing | Draft for IJDRR / Natural Hazards |

**Currently:** Months 1–4 complete; Month 5 (evaluation/ablation) partially complete via the Wayanad replay; Path B (NRSC inventory + retrain) in progress ahead of finalizing Month 5–6.

## Known Issues / Open Items

- Wayanad replay figures (score, lead time) differ slightly across documents — needs a single source of truth.
- District scope (20 vs. 90) is inconsistent between the tracker and other project docs.
- No GitHub commit history / build log formalized yet — noted as a gap, not yet started.
- NRSC inventory integration and the two new engineered features are not yet merged into the production model.

## Author

Arnav T
