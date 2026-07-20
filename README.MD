# RainRisk
**Dynamic Rainfall-Aware Landslide Susceptibility Prediction for the Western Ghats, India**

MCA Research Project | BVIMTECH Navi Mumbai | Arnav | 2026–27

---

## Project Structure
```
rainrisk/
├── data/
│   ├── raw/
│   │   ├── terrain/        # SRTM DEM, slope, LULC (downloaded once from USGS/GEE)
│   │   ├── rainfall/       # NASA GPM IMERG historical + live downloads
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
├── notebooks/              # EDA and experiments
├── outputs/                # Trained models, evaluation reports, plots
└── logs/
```

## Month-by-Month Plan
| Month | Phase | Key Deliverable |
|-------|-------|----------------|
| 1 | Data collection & setup | Merged dataset: terrain + GPM rainfall + NRSC labels |
| 2 | Base model (terrain only) | Baseline XGBoost, AUC/F1 report |
| 3 | Add dynamic rainfall | Improved model with rolling rainfall features |
| 4 | API + Explainability | FastAPI endpoint + SHAP explanations |
| 5 | Evaluation & Validation | Comparison vs baseline, ablation study |
| 6 | Paper writing | Draft for IJDRR / Natural Hazards |

## Study Region
Top 20 highest-risk Western Ghats districts from NRSC Landslide Atlas 2023.

## Data Sources (all free)
- NRSC Landslide Atlas 2023 — historical inventory
- SRTM DEM 30m — USGS EarthExplorer
- NASA GPM IMERG — real-time + historical rainfall API
- Bhukosh GSI — lithology / soil maps
- IMD rainfall — validation
- Sentinel-2 / Landsat 8 — LULC via Google Earth Engine

## Tech Stack
Python | XGBoost | SHAP | scikit-learn | pandas | geopandas | FastAPI | PostgreSQL