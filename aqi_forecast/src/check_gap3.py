"""Gap diagnostic 3: do ANY PM2.5 reference monitors near Delhi/Patna/Lucknow/Gurugram have data in Nov 2023?
Lists all locations (not only the long-running ones), classifies them by start/end, and samples raw data for Nov 2023.
Run (key loaded in this terminal):  python src/check_gap3.py 2>&1 | Tee-Object -FilePath gap3_results.txt"""
import os, sys, time, requests, numpy as np, pandas as pd

key = os.environ.get("OPENAQ_API_KEY")
if not key:
    sys.exit("OPENAQ_API_KEY is not set in this terminal")
BASE = "https://api.openaq.org/v3"
CITIES = {"Delhi": (28.61, 77.21), "Patna": (25.59, 85.14), "Lucknow": (26.85, 80.95), "Gurugram": (28.46, 77.03)}
selected = set(pd.read_csv("data/openaq_station_day.csv", usecols=["location_id"])["location_id"].unique())

def get(path, params):
    time.sleep(1.3)
    for k in range(4):
        r = requests.get(BASE + path, params=params, headers={"X-API-Key": key}, timeout=60)
        if r.status_code == 429:
            time.sleep(65); continue
        return r.status_code, (r.json() if r.status_code == 200 else {})
    return 429, {}

def ts(x):
    return pd.to_datetime(x, utc=True).tz_localize(None) if x else pd.NaT

rows = []
for city, (lat, lon) in CITIES.items():
    page = 1
    while True:
        code, j = get("/locations", {"coordinates": f"{lat},{lon}", "radius": 25000, "parameters_id": 2,
                                     "monitor": "true", "limit": 100, "page": page})
        res = j.get("results", [])
        for L in res:
            pm = [s["id"] for s in L.get("sensors", []) if (s.get("parameter") or {}).get("id") == 2]
            rows.append({"city": city, "id": L["id"], "name": str(L.get("name"))[:28],
                         "provider": str((L.get("provider") or {}).get("name"))[:18],
                         "first": ts((L.get("datetimeFirst") or {}).get("utc")), "last": ts((L.get("datetimeLast") or {}).get("utc")),
                         "pm_sensors": pm, "selected": L["id"] in selected})
        if len(res) < 100:
            break
        page += 1
L = pd.DataFrame(rows).drop_duplicates("id")
L["covers_nov2023"] = (L["first"] <= "2023-10-01") & (L["last"] >= "2023-12-01")
def cat(r):
    if r["first"] < pd.Timestamp("2021-01-01") and r["last"] >= pd.Timestamp("2026-01-01"):
        return "A long-running (selected before)"
    if r["first"] < pd.Timestamp("2021-01-01"):
        return "B started before 2021, ended earlier"
    if r["last"] >= pd.Timestamp("2025-02-01"):
        return "C started 2021 or later, still active"
    return "D started 2021+, ended before 2025"
L["category"] = L.apply(cat, axis=1)
print("=== 1. LOCATIONS BY CATEGORY (count) and how many span Nov 2023 at location level ===")
print(L.groupby(["city", "category"]).agg(locations=("id", "size"), span_nov2023=("covers_nov2023", "sum")).to_string())
print("\nprovider mix of locations that span Nov 2023:", L[L.covers_nov2023].provider.value_counts().to_dict())

print("\n=== 2. RAW DATA FOR NOV 2023 (first PM2.5 sensor of up to 4 locations per city that span Nov 2023) ===")
out = []
for city in CITIES:
    c = L[(L.city == city) & L.covers_nov2023]
    pick = pd.concat([c[~c.selected].head(3), c[c.selected].head(1)])       # mostly NOT-selected ones, plus one control
    for _, r in pick.iterrows():
        if not r.pm_sensors:
            continue
        code, j = get(f"/sensors/{r.pm_sensors[0]}/measurements",
                      {"datetime_from": "2023-11-01T00:00:00Z", "datetime_to": "2023-12-01T00:00:00Z", "limit": 1000})
        res = j.get("results", [])
        st = sorted(str(((x.get("period") or {}).get("datetimeFrom") or {}).get("utc") or "")[:16] for x in res)
        out.append({"city": city, "location": r["id"], "name": r["name"], "provider": r["provider"], "selected before": r["selected"],
                    "sensor": r.pm_sensors[0], "http": code, "rows Nov 2023": len(res),
                    "first": st[0] if st else "-", "last": st[-1] if st else "-"})
print(pd.DataFrame(out).to_string(index=False) if out else "(no location spans Nov 2023 in these cities)")
print("\nReading: locations marked selected before=False with rows > 0 are the missing 2023 coverage. "
      "If every sampled location shows 0 rows, the gap is in OpenAQ's archive.")
