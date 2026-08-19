import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DISTRICTS, DISTRICT_NAMES, MONSOON_MONTHS,
                    PROCESSED_DATASET, DATA_PROCESSED, INVENTORY_DIR, RAINFALL_DIR)
from terrain_features import get_terrain_df

KNOWN_EVENTS = [
    ("Wayanad",          "2024-07-29"), ("Wayanad",        "2019-08-08"),
    ("Wayanad",          "2018-08-15"), ("Wayanad",        "2020-08-06"),
    ("Wayanad",          "2021-07-30"), ("Idukki",         "2021-10-16"),
    ("Idukki",           "2018-08-17"), ("Idukki",         "2020-07-14"),
    ("Malappuram",       "2019-08-10"), ("Malappuram",     "2018-08-10"),
    ("Kozhikode",        "2021-07-26"), ("Kozhikode",      "2019-08-09"),
    ("Kodagu",           "2018-08-17"), ("Kodagu",         "2019-08-09"),
    ("Kodagu",           "2020-08-06"), ("Nilgiris",       "2021-11-18"),
    ("Nilgiris",         "2019-08-09"), ("Nilgiris",       "2018-08-16"),
    ("Uttara Kannada",   "2022-09-04"), ("Uttara Kannada", "2020-08-10"),
    ("Chikmagalur",      "2020-08-22"), ("Chikmagalur",    "2019-07-14"),
    ("Ratnagiri",        "2021-07-23"), ("Ratnagiri",      "2019-08-04"),
    ("Raigad",           "2021-07-22"), ("Raigad",         "2019-08-10"),
    ("Satara",           "2021-07-23"), ("Satara",         "2019-07-22"),
    ("Sindhudurg",       "2021-07-25"), ("Sindhudurg",     "2020-08-05"),
    ("Thrissur",         "2018-08-16"), ("Thrissur",       "2019-08-09"),
    ("Theni",            "2018-07-01"), ("Theni",          "2021-11-19"),
    ("Shivamogga",       "2020-08-07"), ("Shivamogga",     "2022-08-18"),
    ("Dakshina Kannada", "2022-07-30"), ("Dakshina Kannada","2019-08-09"),
    ("Kolhapur",         "2021-07-22"), ("Pune",           "2021-07-25"),
    ("Coimbatore",       "2021-11-18"), ("Dindigul",       "2021-11-19"),
]

def load_inventory():
    path = os.path.join(INVENTORY_DIR, "nrsc_landslide_events.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df["landslide"] = 1
        return df[["district","date","landslide"]]
    rows = [{"district":d,"date":pd.Timestamp(dt),"landslide":1}
            for d,dt in KNOWN_EVENTS if d in DISTRICT_NAMES]
    return pd.DataFrame(rows)

def make_negatives(events_df, ratio=5):
    np.random.seed(42)
    rows = []
    for _ in range(len(events_df) * ratio):
        d = np.random.choice(DISTRICT_NAMES)
        if np.random.random() < 0.6:
            month = np.random.choice([6, 7, 8, 9])
        else:
            month = np.random.choice([11, 12, 1, 2, 3, 4])
        year = np.random.randint(2015, 2024)
        day  = np.random.randint(1, 28)
        try:
            date = pd.Timestamp(year, month, day)
            is_known = any(
                e["district"] == d and pd.Timestamp(e["date"]).date() == date.date()
                for e in [{"district": dd, "date": dt} for dd, dt in KNOWN_EVENTS]
            )
            if not is_known:
                rows.append({"district": d, "date": date, "landslide": 0})
        except ValueError:
            pass
    return pd.DataFrame(rows)

def synthetic_rain(district, date, is_event):
    np.random.seed(hash(f"{district}{date}") % (2**31))
    if is_event:
        r1=max(0,np.random.normal(85,30));  r3=max(r1,np.random.normal(180,50))
        r7=max(r3,np.random.normal(310,70)); r15=max(r7,np.random.normal(520,100))
    else:
        r1=max(0,np.random.normal(8,12));   r3=max(r1,np.random.normal(20,18))
        r7=max(r3,np.random.normal(40,25)); r15=max(r7,np.random.normal(80,40))
    return {"rain_1d":round(r1,1),"rain_3d":round(r3,1),
            "rain_7d":round(r7,1),"rain_15d":round(r15,1),
            "rain_max3d":round(r1*np.random.uniform(0.8,1.2),1),
            "is_monsoon":int(date.month in MONSOON_MONTHS)}

def load_real_rainfall(district, date):
    path = os.path.join(RAINFALL_DIR, f"{district.replace(' ', '_')}_rainfall.csv")
    if not os.path.exists(path):
        return None
    df  = pd.read_csv(path, index_col=0, parse_dates=True)
    key = pd.Timestamp(date).normalize()
    if key in df.index:
        row = df.loc[key]
        return {
            "rain_1d":    round(float(row.get("rain_1d",    0)), 1),
            "rain_3d":    round(float(row.get("rain_3d",    0)), 1),
            "rain_7d":    round(float(row.get("rain_7d",    0)), 1),
            "rain_15d":   round(float(row.get("rain_15d",   0)), 1),
            "rain_max3d": round(float(row.get("rain_max3d", 0)), 1),
            "is_monsoon": int(row.get("is_monsoon", 0)),
        }
    return None

def build_feature_matrix():
    terrain = get_terrain_df()
    pos     = load_inventory()
    neg     = make_negatives(pos, ratio=5)
    df      = pd.concat([pos, neg], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df      = df.merge(terrain.reset_index(), on="district", how="left")

    rain_rows = []
    for _, row in df.iterrows():
        real = load_real_rainfall(row["district"], row["date"])
        rain_rows.append(real if real else
                         synthetic_rain(row["district"], row["date"], row["landslide"]))
    rain_df = pd.DataFrame(rain_rows)

    df = pd.concat([df.reset_index(drop=True), rain_df], axis=1)

    # interaction features
    df["slope_x_rain7d"]       = df["slope_deg"] * df["rain_7d"] / 100
    df["twi_x_rain15d"]        = df["twi"]        * df["rain_15d"] / 100
    df["rain_intensity_ratio"]  = df["rain_1d"] / (df["rain_7d"] / 7 + 0.1)
    df["soil_saturation_proxy"] = df["rain_15d"] / (df["elevation_m"] / 100 + 1)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset: {len(df)} rows | Landslide rate: {df['landslide'].mean():.1%}")
    return df

if __name__ == "__main__":
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df = build_feature_matrix()
    df.to_csv(PROCESSED_DATASET, index=False)
    print(f"Saved: {PROCESSED_DATASET}")