"""Where are the holes in the OpenAQ daily data? Monthly coverage, date-alignment check, and a direct API look at one month of 2023.
Run (key loaded in this terminal):  python src/check_gap.py 2>&1 | Tee-Object -FilePath gap_results.txt"""
import os, sys, time, requests, numpy as np, pandas as pd

sd = pd.read_csv("data/openaq_station_day.csv", parse_dates=["date"])
sd["month"] = sd.date.dt.to_period("M").astype(str)
m = sd.groupby("month").agg(sensors=("sensor_id", "nunique"), station_days=("pm25", "size"), cities=("city", "nunique"))
print("=== 1. MONTHLY COVERAGE (sensors reporting, station-days, cities) ===")
print(m.loc["2021-01":].to_string())

print("\n=== 2. DATE ALIGNMENT vs KAGGLE (correlation of daily city means when OpenAQ is shifted by k days) ===")
kg = pd.read_csv("data/city_day.csv", parse_dates=["Date"])[["City", "Date", "PM2.5"]].dropna()
rows = []
for city in ["Delhi", "Gurugram", "Lucknow", "Patna", "Kolkata", "Mumbai"]:
    oq = sd[sd.city == city].groupby("date")["pm25"].mean()
    k = kg[kg.City == city].set_index("Date")["PM2.5"]
    row = {"city": city}
    for s in (-1, 0, 1):
        j = pd.concat([oq.shift(s, freq="D"), k], axis=1, join="inner").dropna()
        row[f"shift {s:+d}d"] = round(j.corr().iloc[0, 1], 3) if len(j) > 60 else np.nan
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("(if shift +1d or -1d is clearly higher than 0d, the two sources' dates are offset)")

print("\n=== 3. DIRECT API CHECK: Nov 2023 for three sensors that report in both 2021 and 2025 ===")
key = os.environ.get("OPENAQ_API_KEY")
if not key:
    sys.exit("OPENAQ_API_KEY is not set in this terminal")
def api(path, params):
    time.sleep(1.3)
    r = requests.get("https://api.openaq.org/v3" + path, params=params, headers={"X-API-Key": key}, timeout=60)
    return r.status_code, (r.json() if r.status_code == 200 else {})
yr = sd.groupby(["sensor_id", sd.date.dt.year]).size().unstack(fill_value=0)
good = yr[(yr.get(2021, 0) >= 200) & (yr.get(2025, 0) >= 200)].sum(axis=1).sort_values(ascending=False).head(3).index.tolist()
print("sensors checked:", good)
win = {"datetime_from": "2023-11-01T00:00:00Z", "datetime_to": "2023-12-01T00:00:00Z", "limit": 1000}
for sid in good:
    out = {}
    for label, path in [("days", f"/sensors/{sid}/days"), ("hours", f"/sensors/{sid}/hours"), ("raw", f"/sensors/{sid}/measurements")]:
        code, j = api(path, win)
        out[label] = f"HTTP {code}, {len((j or {}).get('results', []))} rows"
    code, j = api(f"/sensors/{sid}", {})
    s = ((j or {}).get("results") or [{}])[0]
    print(f"sensor {sid}: {out} | first {((s.get('datetimeFirst') or {}).get('utc'))} | last {((s.get('datetimeLast') or {}).get('utc'))}")
print("\ndone")
