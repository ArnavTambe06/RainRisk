import os, sys, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DISTRICTS, MONSOON_MONTHS, RAINFALL_WINDOWS, RAINFALL_DIR

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_daily_rainfall(lat, lon, start, end):
    params = {
        "parameters": "PRECTOTCORR",
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start.replace("-", ""),
        "end":        end.replace("-", ""),
        "format":     "JSON",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=60)
    r.raise_for_status()
    daily = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    s = pd.Series({datetime.strptime(k, "%Y%m%d"): v for k, v in daily.items()})
    return s.replace(-999.0, np.nan).sort_index()

def engineer_rainfall_features(series):
    df = pd.DataFrame({"rain_1d": series})
    for w in RAINFALL_WINDOWS:
        df[f"rain_{w}d"] = df["rain_1d"].rolling(w, min_periods=1).sum()
    df["rain_max3d"] = df["rain_1d"].rolling(3, min_periods=1).max()
    df["is_monsoon"] = df.index.month.isin(MONSOON_MONTHS).astype(int)

    # ── Drop trailing NaN rows (NASA POWER reporting delay) ──────────────────
    # NASA POWER has a 2-3 day lag. The last 1-3 rows may have NaN rain_1d.
    # Drop them so .iloc[-1] always returns the last VALID observation.
    df = df.dropna(subset=["rain_1d"])

    return df

def download_district(district, start="2015-01-01", end="2024-12-31"):
    name = district["name"].replace(" ", "_")
    out  = os.path.join(RAINFALL_DIR, f"{name}_rainfall.csv")
    if os.path.exists(out):
        print(f"  [SKIP] {district['name']}")
        return out
    print(f"  [FETCH] {district['name']} ...")
    series = fetch_daily_rainfall(district["lat"], district["lon"], start, end)
    df = engineer_rainfall_features(series)
    df["district"] = district["name"]
    df["state"]    = district["state"]
    os.makedirs(RAINFALL_DIR, exist_ok=True)
    df.to_csv(out)
    print(f"  [DONE] {len(df)} days saved")
    return out

def download_all(start="2015-01-01", end="2024-12-31"):
    dfs = []
    for d in DISTRICTS:
        try:
            path = download_district(d, start, end)
            dfs.append(pd.read_csv(path, index_col=0, parse_dates=True))
        except Exception as e:
            print(f"  [ERROR] {d['name']}: {e}")
    combined = pd.concat(dfs)
    out = os.path.join(RAINFALL_DIR, "all_districts_rainfall.csv")
    combined.to_csv(out)
    print(f"\nCombined saved: {out}  ({len(combined):,} rows)")
    return combined

def fetch_live():
    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=20)).strftime("%Y-%m-%d")
    rows  = []
    for d in DISTRICTS:
        try:
            s    = fetch_daily_rainfall(d["lat"], d["lon"], start, end)
            feat = engineer_rainfall_features(s).iloc[-1]
            rows.append({"district": d["name"], "state": d["state"],
                         "date": s.index[-1].date(), **feat.to_dict()})
        except Exception as e:
            print(f"  [WARN] {d['name']}: {e}")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end",   default="2024-12-31")
    p.add_argument("--live",  action="store_true")
    args = p.parse_args()
    if args.live:
        print(fetch_live()[["district","date","rain_1d","rain_3d","rain_7d","rain_15d"]])
    else:
        download_all(args.start, args.end)