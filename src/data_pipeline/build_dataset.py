import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DISTRICTS, DISTRICT_NAMES, MONSOON_MONTHS,
                    PROCESSED_DATASET, DATA_PROCESSED, INVENTORY_DIR, RAINFALL_DIR)
from terrain_features import get_terrain_df

KNOWN_EVENTS = [
    ("Wayanad","2024-07-29"), ("Wayanad","2019-08-08"), ("Wayanad","2018-08-15"),
    ("Idukki","2021-10-16"),  ("Idukki","2018-08-17"),
    ("Malappuram","2019-08-10"), ("Kozhikode","2021-07-26"),
    ("Kodagu","2018-08-17"),  ("Kodagu","2019-08-09"),
    ("Nilgiris","2021-11-18"),("Nilgiris","2019-08-09"),
    ("Uttara Kannada","2022-09-04"), ("Chikmagalur","2020-08-22"),
    ("Ratnagiri","2021-07-23"), ("Raigad","2021-07-22"),
    ("Satara","2021-07-23"),  ("Sindhudurg","2021-07-25"),
    ("Thrissur","2018-08-16"), ("Theni","2018-07-01"),
    ("Shivamogga","2020-08-07"), ("Dakshina Kannada","2022-07-30"),
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
        d     = np.random.choice(DISTRICT_NAMES)
        month = np.random.choice([10,11,12,1,2,3,4,5])
        year  = np.random.randint(2015,2024)
        day   = np.random.randint(1,28)
        try:
            rows.append({"district":d,"date":pd.Timestamp(year,month,day),"landslide":0})
        except ValueError:
            pass
    return pd.DataFrame(rows)

def synthetic_rain(district, date, is_event):
    np.random.seed(hash(f"{district}{date}") % (2**31))
    if is_event:
        r1=max(0,np.random.normal(85,30)); r3=max(r1,np.random.normal(180,50))
        r7=max(r3,np.random.normal(310,70)); r15=max(r7,np.random.normal(520,100))
    else:
        r1=max(0,np.random.normal(8,12));  r3=max(r1,np.random.normal(20,18))
        r7=max(r3,np.random.normal(40,25)); r15=max(r7,np.random.normal(80,40))
    return {"rain_1d":round(r1,1),"rain_3d":round(r3,1),
            "rain_7d":round(r7,1),"rain_15d":round(r15,1),
            "rain_max3d":round(r1*np.random.uniform(0.8,1.2),1),
            "is_monsoon":int(date.month in MONSOON_MONTHS)}

def build_feature_matrix(use_synthetic_rainfall=False):
    terrain = get_terrain_df()
    pos     = load_inventory()
    neg     = make_negatives(pos, ratio=5)
    df      = pd.concat([pos, neg], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df      = df.merge(terrain.reset_index(), on="district", how="left")
    rain    = pd.DataFrame([synthetic_rain(r.district, r.date, r.landslide)
                            for _, r in df.iterrows()])
    df      = pd.concat([df.reset_index(drop=True), rain], axis=1)
    df["slope_x_rain7d"] = df["slope_deg"] * df["rain_7d"] / 100
    df["twi_x_rain15d"]  = df["twi"]       * df["rain_15d"] / 100
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset: {len(df)} rows | Landslide rate: {df['landslide'].mean():.1%}")
    return df

def load_real_rainfall(district, date):
    path = os.path.join(RAINFALL_DIR, f"{district.replace(' ','_')}_rainfall.csv")
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    if date in df.index:
        row = df.loc[date]
        return {col: row[col] for col in
                ["rain_1d","rain_3d","rain_7d","rain_15d","rain_max3d","is_monsoon"]
                if col in df.columns}
    return {col: 0 for col in ["rain_1d","rain_3d","rain_7d","rain_15d","rain_max3d","is_monsoon"]}

if __name__ == "__main__":
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df = build_feature_matrix()
    df.to_csv(PROCESSED_DATASET, index=False)
    print(f"Saved: {PROCESSED_DATASET}")