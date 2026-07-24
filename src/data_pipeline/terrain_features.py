import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DISTRICTS, TERRAIN_DIR

TERRAIN_DATA = {
    "Wayanad":          {"slope_deg":24.5,"elevation_m":870, "aspect_deg":112,"curvature":-0.42,"twi":7.8,"lulc_class":0,"lithology":1,"dist_river_km":1.2,"drainage_density":2.8,"ndvi":0.72},
    "Idukki":           {"slope_deg":28.2,"elevation_m":1050,"aspect_deg":95, "curvature":-0.55,"twi":7.2,"lulc_class":0,"lithology":2,"dist_river_km":0.9,"drainage_density":3.1,"ndvi":0.68},
    "Malappuram":       {"slope_deg":18.3,"elevation_m":420, "aspect_deg":78, "curvature":-0.28,"twi":8.1,"lulc_class":1,"lithology":1,"dist_river_km":2.1,"drainage_density":2.4,"ndvi":0.61},
    "Kozhikode":        {"slope_deg":15.7,"elevation_m":320, "aspect_deg":88, "curvature":-0.22,"twi":8.4,"lulc_class":1,"lithology":1,"dist_river_km":1.8,"drainage_density":2.2,"ndvi":0.58},
    "Thrissur":         {"slope_deg":12.1,"elevation_m":210, "aspect_deg":102,"curvature":-0.15,"twi":8.9,"lulc_class":1,"lithology":1,"dist_river_km":2.8,"drainage_density":1.9,"ndvi":0.52},
    "Kodagu":           {"slope_deg":26.8,"elevation_m":1100,"aspect_deg":85, "curvature":-0.61,"twi":7.0,"lulc_class":0,"lithology":2,"dist_river_km":0.7,"drainage_density":3.4,"ndvi":0.74},
    "Chikmagalur":      {"slope_deg":22.4,"elevation_m":890, "aspect_deg":92, "curvature":-0.48,"twi":7.5,"lulc_class":0,"lithology":2,"dist_river_km":1.4,"drainage_density":2.9,"ndvi":0.69},
    "Uttara Kannada":   {"slope_deg":19.6,"elevation_m":610, "aspect_deg":105,"curvature":-0.35,"twi":7.9,"lulc_class":0,"lithology":3,"dist_river_km":1.1,"drainage_density":2.6,"ndvi":0.71},
    "Shivamogga":       {"slope_deg":17.3,"elevation_m":540, "aspect_deg":118,"curvature":-0.29,"twi":8.2,"lulc_class":0,"lithology":3,"dist_river_km":1.6,"drainage_density":2.5,"ndvi":0.65},
    "Dakshina Kannada": {"slope_deg":20.1,"elevation_m":490, "aspect_deg":96, "curvature":-0.38,"twi":7.7,"lulc_class":0,"lithology":1,"dist_river_km":1.0,"drainage_density":2.7,"ndvi":0.70},
    "Ratnagiri":        {"slope_deg":16.8,"elevation_m":350, "aspect_deg":75, "curvature":-0.25,"twi":8.3,"lulc_class":1,"lithology":3,"dist_river_km":2.0,"drainage_density":2.1,"ndvi":0.55},
    "Sindhudurg":       {"slope_deg":14.2,"elevation_m":280, "aspect_deg":82, "curvature":-0.19,"twi":8.6,"lulc_class":1,"lithology":3,"dist_river_km":2.4,"drainage_density":2.0,"ndvi":0.53},
    "Satara":           {"slope_deg":21.5,"elevation_m":780, "aspect_deg":120,"curvature":-0.44,"twi":7.6,"lulc_class":1,"lithology":3,"dist_river_km":1.5,"drainage_density":2.7,"ndvi":0.48},
    "Raigad":           {"slope_deg":18.9,"elevation_m":520, "aspect_deg":88, "curvature":-0.33,"twi":7.8,"lulc_class":1,"lithology":3,"dist_river_km":1.3,"drainage_density":2.4,"ndvi":0.51},
    "Pune":             {"slope_deg":13.4,"elevation_m":580, "aspect_deg":130,"curvature":-0.17,"twi":8.8,"lulc_class":2,"lithology":3,"dist_river_km":3.2,"drainage_density":1.8,"ndvi":0.38},
    "Kolhapur":         {"slope_deg":15.1,"elevation_m":460, "aspect_deg":110,"curvature":-0.21,"twi":8.5,"lulc_class":1,"lithology":3,"dist_river_km":2.2,"drainage_density":2.1,"ndvi":0.46},
    "Nilgiris":         {"slope_deg":27.3,"elevation_m":1820,"aspect_deg":98, "curvature":-0.59,"twi":6.9,"lulc_class":0,"lithology":2,"dist_river_km":0.8,"drainage_density":3.5,"ndvi":0.76},
    "Coimbatore":       {"slope_deg":11.8,"elevation_m":440, "aspect_deg":125,"curvature":-0.14,"twi":9.1,"lulc_class":2,"lithology":2,"dist_river_km":3.5,"drainage_density":1.7,"ndvi":0.41},
    "Dindigul":         {"slope_deg":17.6,"elevation_m":620, "aspect_deg":108,"curvature":-0.31,"twi":8.0,"lulc_class":1,"lithology":2,"dist_river_km":2.0,"drainage_density":2.3,"ndvi":0.49},
    "Theni":            {"slope_deg":23.9,"elevation_m":940, "aspect_deg":92, "curvature":-0.51,"twi":7.3,"lulc_class":0,"lithology":2,"dist_river_km":1.1,"drainage_density":3.0,"ndvi":0.66},
}

TERRAIN_FEATURES = [
    "slope_deg","elevation_m","aspect_deg","curvature",
    "twi","lulc_class","lithology","dist_river_km","drainage_density","ndvi"
]

def get_terrain_df():
    rows = [{"district": d["name"], "state": d["state"], **TERRAIN_DATA[d["name"]]} for d in DISTRICTS]
    return pd.DataFrame(rows).set_index("district")

def save_terrain_features():
    os.makedirs(TERRAIN_DIR, exist_ok=True)
    df = get_terrain_df()
    out = os.path.join(TERRAIN_DIR, "terrain_features.csv")
    df.to_csv(out)
    print(f"Saved: {out}")
    return out

if __name__ == "__main__":
    save_terrain_features()
    print(get_terrain_df()[TERRAIN_FEATURES].describe().round(2))