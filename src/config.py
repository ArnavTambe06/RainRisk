import os

DISTRICTS = [
    {"name": "Wayanad",          "state": "Kerala",      "lat": 11.6854, "lon": 76.1320},
    {"name": "Idukki",           "state": "Kerala",      "lat": 9.9189,  "lon": 77.1025},
    {"name": "Malappuram",       "state": "Kerala",      "lat": 11.0730, "lon": 76.0740},
    {"name": "Kozhikode",        "state": "Kerala",      "lat": 11.2588, "lon": 75.7804},
    {"name": "Thrissur",         "state": "Kerala",      "lat": 10.5276, "lon": 76.2144},
    {"name": "Kodagu",           "state": "Karnataka",   "lat": 12.3375, "lon": 75.8069},
    {"name": "Chikmagalur",      "state": "Karnataka",   "lat": 13.3161, "lon": 75.7720},
    {"name": "Uttara Kannada",   "state": "Karnataka",   "lat": 14.7860, "lon": 74.6948},
    {"name": "Shivamogga",       "state": "Karnataka",   "lat": 13.9299, "lon": 75.5681},
    {"name": "Dakshina Kannada", "state": "Karnataka",   "lat": 12.8438, "lon": 75.2479},
    {"name": "Ratnagiri",        "state": "Maharashtra", "lat": 17.0,    "lon": 73.3},
    {"name": "Sindhudurg",       "state": "Maharashtra", "lat": 16.3,    "lon": 73.9},
    {"name": "Satara",           "state": "Maharashtra", "lat": 17.6805, "lon": 74.0183},
    {"name": "Raigad",           "state": "Maharashtra", "lat": 18.5158, "lon": 73.1298},
    {"name": "Pune",             "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567},
    {"name": "Kolhapur",         "state": "Maharashtra", "lat": 16.7,    "lon": 74.2},
    {"name": "Nilgiris",         "state": "Tamil Nadu",  "lat": 11.4916, "lon": 76.7337},
    {"name": "Coimbatore",       "state": "Tamil Nadu",  "lat": 11.0168, "lon": 76.9558},
    {"name": "Dindigul",         "state": "Tamil Nadu",  "lat": 10.3624, "lon": 77.9695},
    {"name": "Theni",            "state": "Tamil Nadu",  "lat": 10.0104, "lon": 77.4771},
]

DISTRICT_NAMES   = [d["name"] for d in DISTRICTS]
MONSOON_MONTHS   = [6, 7, 8, 9]
RAINFALL_WINDOWS = [3, 7, 15]
RANDOM_SEED      = 42
TEST_SIZE        = 0.30

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
}

RISK_LEVELS = {"LOW": (0.0, 0.4), "MEDIUM": (0.4, 0.7), "HIGH": (0.7, 1.0)}

def score_to_level(score):
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "HIGH"

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW          = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED    = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR       = os.path.join(BASE_DIR, "outputs")
TERRAIN_DIR       = os.path.join(DATA_RAW, "terrain")
RAINFALL_DIR      = os.path.join(DATA_RAW, "rainfall")
INVENTORY_DIR     = os.path.join(DATA_RAW, "inventory")
PROCESSED_DATASET = os.path.join(DATA_PROCESSED, "feature_matrix.csv")
MODEL_PATH        = os.path.join(OUTPUTS_DIR, "rainrisk_xgb.pkl")