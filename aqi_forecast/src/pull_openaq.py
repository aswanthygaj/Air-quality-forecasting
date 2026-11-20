"""Pull daily PM2.5 from OpenAQ (2018 -> today) for long-running reference monitors near each city.
Results are cached per sensor, so the script can be stopped and restarted safely.
Run (OPENAQ_API_KEY must be set in this terminal):  python src/pull_openaq.py 2>&1 | Tee-Object -FilePath pull_results.txt"""
import os, sys, time, math, requests, pandas as pd, numpy as np
from common import CITY_COORDS

BASE = "https://api.openaq.org/v3"
START = "2018-01-01T00:00:00Z"
END = pd.Timestamp.today().strftime("%Y-%m-%dT23:59:59Z")
FIRST_BEFORE, LAST_AFTER = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01")   # "long-running" locations
PAUSE = 1.3                                   # seconds between requests (limits: 60/min, 2000/hour)
CACHE = "data/openaq_cache"
N_REQ = 0

def get(path, params=None, tries=6):
    global N_REQ
    key = os.environ.get("OPENAQ_API_KEY")
    for k in range(tries):
        time.sleep(PAUSE)
        try:
            r = requests.get(BASE + path, params=params, headers={"X-API-Key": key}, timeout=60)
        except requests.RequestException as e:
            print("  network error, retrying:", repr(e)[:80], flush=True); time.sleep(10); continue
        N_REQ += 1
        if r.status_code == 429:
            wait = 65 + 30 * k
            print(f"  rate limited, waiting {wait}s", flush=True); time.sleep(wait); continue
        if r.status_code in (401, 403):
            sys.exit(f"API key rejected (HTTP {r.status_code}) - reload the key in this terminal")
        if r.status_code == 404:
            return None
        if r.status_code == 422:
            print("  request rejected (422):", r.text[:200], flush=True); return None
        if r.status_code >= 500:
            time.sleep(10 * (k + 1)); continue
        r.raise_for_status()
        return r.json()
    print("  giving up on", path, flush=True)
    return None

def ts(s):
    return pd.to_datetime(s, utc=True).tz_localize(None) if s else pd.NaT

def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))

def find_locations(lat, lon):
    out, page = [], 1
    while True:
        j = get("/locations", {"coordinates": f"{lat},{lon}", "radius": 25000, "parameters_id": 2,
                               "monitor": "true", "limit": 100, "page": page})
        res = (j or {}).get("results", [])
        out += res
        if len(res) < 100:
            return out
        page += 1

def pull_sensor(sid):
    f = f"{CACHE}/{sid}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    rows, page = [], 1
    while True:
        j = get(f"/sensors/{sid}/days", {"datetime_from": START, "datetime_to": END, "limit": 1000, "page": page})
        res = (j or {}).get("results", [])
        for x in res:
            dfm = (x.get("period") or {}).get("datetimeFrom") or {}
            cov = x.get("coverage") or {}
            rows.append({"date": (dfm.get("local") or dfm.get("utc") or "")[:10], "value": x.get("value"),
                         "pct": cov.get("percentComplete", cov.get("percentCoverage")), "n": cov.get("observedCount")})
        if len(res) < 1000:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["date", "value", "pct", "n"])
    df.to_csv(f, index=False)
    return df

def main():
    if not os.environ.get("OPENAQ_API_KEY"):
        sys.exit("OPENAQ_API_KEY is not set in this terminal")
    os.makedirs(CACHE, exist_ok=True)
    # ---- 1. long-running reference monitors, each assigned to its nearest city centre
    locs = {}
    for city, (lat, lon) in CITY_COORDS.items():
        for L in find_locations(lat, lon):
            first, last = ts(((L.get("datetimeFirst") or {}).get("utc"))), ts(((L.get("datetimeLast") or {}).get("utc")))
            if not (first < FIRST_BEFORE and last >= LAST_AFTER):
                continue
            c = L.get("coordinates") or {}
            if c.get("latitude") is None:
                continue
            sensors = [s["id"] for s in L.get("sensors", []) if (s.get("parameter") or {}).get("id") == 2
                       or (s.get("parameter") or {}).get("name") == "pm25"]
            if not sensors:
                continue
            near = min(CITY_COORDS, key=lambda k: km(c["latitude"], c["longitude"], *CITY_COORDS[k]))
            locs[L["id"]] = {"city": near, "name": L.get("name"), "sensors": sensors}
        print(f"searched {city}: {len(locs)} distinct long-running locations so far", flush=True)
    per_city = pd.Series([v["city"] for v in locs.values()]).value_counts()
    n_sens = sum(len(v["sensors"]) for v in locs.values())
    print(f"\nselected {len(locs)} locations / {n_sens} PM2.5 sensors in {len(per_city)} cities")
    print("locations per city:", per_city.to_dict(), flush=True)
    # ---- 2. daily series per sensor
    frames, done = [], 0
    for lid, v in locs.items():
        for sid in v["sensors"]:
            d = pull_sensor(sid)
            d = d.assign(city=v["city"], location_id=lid, location_name=v["name"], sensor_id=sid)
            frames.append(d); done += 1
            if done % 10 == 0:
                print(f"  {done}/{n_sens} sensors done ({N_REQ} requests so far)", flush=True)
    sd = pd.concat(frames, ignore_index=True)
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd = sd.dropna(subset=["date", "value"]).rename(columns={"value": "pm25"})
    sd = sd[(sd.pm25 >= 0) & (sd.pm25 < 2000)]
    sd.to_csv("data/openaq_station_day.csv", index=False)
    print(f"\nsaved data/openaq_station_day.csv: {len(sd)} station-days, {sd.date.min().date()} to {sd.date.max().date()}")
    print("station-days per year:", sd.groupby(sd.date.dt.year).size().to_dict())
    print("sensors reporting per year:", sd.groupby(sd.date.dt.year)["sensor_id"].nunique().to_dict())
    # ---- 3. sanity check against the Kaggle (CPCB) city series on the overlapping dates
    if os.path.exists("data/city_day.csv"):
        kg = pd.read_csv("data/city_day.csv", parse_dates=["Date"])[["City", "Date", "PM2.5"]].dropna()
        kg.columns = ["city", "date", "kaggle"]
        oq = sd.groupby(["city", "date"])["pm25"].mean().rename("openaq").reset_index()
        m = oq.merge(kg, on=["city", "date"])
        rows = []
        for c, g in m.groupby("city"):
            if len(g) >= 60:
                rows.append({"city": c, "overlap days": len(g), "corr": round(g.openaq.corr(g.kaggle), 3),
                             "median OpenAQ/Kaggle": round(float((g.openaq / g.kaggle).median()), 2)})
        print("\nOpenAQ vs Kaggle city series on overlapping days (corr near 1 and ratio near 1 = consistent):")
        print(pd.DataFrame(rows).sort_values("overlap days", ascending=False).to_string(index=False) if rows else "  (no city had >= 60 overlapping days)")
    print(f"\ndone ({N_REQ} API requests)")

if __name__ == "__main__":
    main()
