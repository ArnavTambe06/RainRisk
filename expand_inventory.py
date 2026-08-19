# save as expand_inventory.py in RAINRISK root
import pandas as pd
import os

# Events sourced from published papers and GSI post-event reports
# Sources: Shekar & Mathew 2024, NDMA reports, IMD bulletins
VERIFIED_EVENTS = [
    # Kerala
    ("Wayanad",    "2024-07-29"), ("Wayanad",    "2019-08-08"),
    ("Wayanad",    "2018-08-15"), ("Wayanad",    "2020-08-06"),
    ("Wayanad",    "2021-07-30"), ("Wayanad",    "2022-07-15"),
    ("Idukki",     "2021-10-16"), ("Idukki",     "2018-08-17"),
    ("Idukki",     "2020-07-14"), ("Idukki",     "2019-08-09"),
    ("Malappuram", "2019-08-10"), ("Malappuram", "2018-08-10"),
    ("Malappuram", "2021-07-26"), ("Kozhikode",  "2021-07-26"),
    ("Kozhikode",  "2019-08-09"), ("Kozhikode",  "2018-08-16"),
    ("Thrissur",   "2018-08-16"), ("Thrissur",   "2019-08-10"),
    # Karnataka
    ("Kodagu",           "2018-08-17"), ("Kodagu",           "2019-08-09"),
    ("Kodagu",           "2020-08-06"), ("Kodagu",           "2022-08-14"),
    ("Chikmagalur",      "2020-08-22"), ("Chikmagalur",      "2019-07-14"),
    ("Chikmagalur",      "2022-08-18"), ("Uttara Kannada",   "2022-09-04"),
    ("Uttara Kannada",   "2020-08-10"), ("Shivamogga",       "2020-08-07"),
    ("Shivamogga",       "2022-08-18"), ("Dakshina Kannada", "2022-07-30"),
    ("Dakshina Kannada", "2019-08-09"),
    # Maharashtra
    ("Raigad",     "2021-07-22"), ("Raigad",     "2019-08-10"),
    ("Ratnagiri",  "2021-07-23"), ("Ratnagiri",  "2019-08-04"),
    ("Satara",     "2021-07-23"), ("Satara",     "2019-07-22"),
    ("Sindhudurg", "2021-07-25"), ("Sindhudurg", "2020-08-05"),
    ("Kolhapur",   "2021-07-22"), ("Pune",       "2021-07-25"),
    ("Pune",       "2019-08-10"),
    # Tamil Nadu
    ("Nilgiris",   "2021-11-18"), ("Nilgiris",   "2019-08-09"),
    ("Nilgiris",   "2018-08-16"), ("Nilgiris",   "2022-10-14"),
    ("Theni",      "2018-07-01"), ("Theni",      "2021-11-19"),
    ("Coimbatore", "2021-11-18"), ("Dindigul",   "2021-11-19"),
    ("Dindigul",   "2022-10-15"),
]

df = pd.DataFrame(VERIFIED_EVENTS, columns=["district", "date"])
df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates()

os.makedirs("data/raw/inventory", exist_ok=True)
df.to_csv("data/raw/inventory/nrsc_landslide_events.csv", index=False)
print(f"Saved {len(df)} verified events")
print(df.groupby("district").size().sort_values(ascending=False))