"""Corrected gap diagnostic: sensors that report in 2021 and 2025 but have NO daily data in 2023, tested with
several endpoint / date-parameter variants for November 2023.
Run (key loaded in this terminal):  python src/check_gap2.py 2>&1 | Tee-Object -FilePath gap2_results.txt"""
import os, sys, time, requests, numpy as np, pandas as pd

key = os.environ.get("OPENAQ_API_KEY")
if not key:
    sys.exit("OPENAQ_API_KEY is not set in this terminal")
sd = pd.read_csv("data/openaq_station_day.csv", parse_dates=["date"])
yr = sd.groupby(["sensor_id", sd.date.dt.year]).size().unstack(fill_value=0)
for y in (2021, 2023, 2025):
    if y not in yr.columns:
        yr[y] = 0
cand = yr[(yr[2021] >= 150) & (yr[2025] >= 150) & (yr[2023] == 0)]
city_of = sd.drop_duplicates("sensor_id").set_index("sensor_id")["city"]
pick = []
for c in ["Delhi", "Patna", "Lucknow", "Gurugram", "Kolkata", "Mumbai"]:
    ids = [s for s in cand.index if city_of.get(s) == c]
    if ids:
        pick.append(ids[0])
pick = pick[:3]
print(f"{len(cand)} sensors have >=150 days in 2021 and 2025 but none in 2023. Testing: "
      f"{[(int(s), city_of[s], int(yr.loc[s, 2021]), int(yr.loc[s, 2025])) for s in pick]}  (id, city, days 2021, days 2025)\n")

W_DT = {"datetime_from": "2023-11-01T00:00:00Z", "datetime_to": "2023-12-01T00:00:00Z"}
W_DATE = {"date_from": "2023-11-01", "date_to": "2023-11-30"}
VARIANTS = [("days, no date filter", "/days", {}), ("days, datetime_from/to", "/days", W_DT),
            ("days, date_from/to", "/days", W_DATE), ("hours, datetime_from/to", "/hours", W_DT),
            ("raw measurements, datetime_from/to", "/measurements", W_DT)]

def stamp(x):
    per = (x.get("period") or {}).get("datetimeFrom") or {}
    dt = x.get("datetime") or {}
    return str(per.get("utc") or per.get("local") or dt.get("utc") or x.get("date") or "")[:16]

for sid in pick:
    print(f"--- sensor {int(sid)} ({city_of[sid]}) ---")
    for label, path, extra in VARIANTS:
        time.sleep(1.3)
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{int(sid)}{path}", params={**extra, "limit": 1000},
                             headers={"X-API-Key": key}, timeout=60)
        except requests.RequestException as e:
            print(f"  {label:38s} network error {repr(e)[:60]}"); continue
        if r.status_code != 200:
            print(f"  {label:38s} HTTP {r.status_code}: {r.text[:80]}"); continue
        res = r.json().get("results", [])
        st = sorted(s for s in map(stamp, res) if s)
        nov = sum(1 for s in st if s.startswith("2023-11"))
        print(f"  {label:38s} rows {len(res):4d} | first {st[0] if st else '-':16s} last {st[-1] if st else '-':16s} | rows in Nov 2023: {nov}")
print("\nReading: a variant whose first/last dates fall inside Nov 2023 is honouring the date filter. If daily rows are 0 but "
      "hourly or raw rows exist, the gap is only in the daily rollup and daily means can be rebuilt from hours.")
