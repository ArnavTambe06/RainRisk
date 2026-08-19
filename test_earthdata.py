# save as test_earthdata.py in RAINRISK root
import requests

r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params={
    "parameters": "PRECTOTCORR",
    "community":  "RE",
    "longitude":  76.1320,
    "latitude":   11.6854,
    "start":      "20240725",
    "end":        "20240730",
    "format":     "JSON",
})

print("Status:", r.status_code)
data = r.json()["properties"]["parameter"]["PRECTOTCORR"]
for date, mm in data.items():
    print(f"{date}: {mm} mm/day")