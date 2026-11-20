"""Download daily weather (Open-Meteo archive) for the cities in data/city_day_oaq.csv, 2018 -> latest day. Resumable.
Run (venv active):  python src/fetch_weather_oaq.py 2>&1 | Tee-Object -FilePath weather_oaq_results.txt"""
import os, time, requests, pandas as pd
from common import CITY_COORDS

OUT = "data/weather_oaq.csv"
DAILY = ("temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,"
         "precipitation_sum,surface_pressure_mean")

def fetch(lat, lon, start, end, tries=6):
    for k in range(tries):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120,
                         params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                     timezone="Asia/Kolkata", daily=DAILY))
        if r.status_code == 429:
            wait = 30 * (k + 1)
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            print(r.text[:200])
        r.raise_for_status()
        d = pd.DataFrame(r.json()["daily"])
        d.columns = ["date", "temp", "rh", "wind", "rain", "pressure"]
        d["date"] = pd.to_datetime(d["date"])
        return d
    raise RuntimeError("Still rate limited. Wait about an hour and rerun; finished cities are kept.")

if __name__ == "__main__":
    cd = pd.read_csv("data/city_day_oaq.csv", parse_dates=["Date"])
    start, end = "2018-01-01", cd.Date.max().date()
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set(done["city"]) if len(done) else set()
    for c in sorted(cd["City"].unique()):
        if c in have:
            print("already have", c); continue
        if c not in CITY_COORDS:
            print("skip (no coords):", c); continue
        lat, lon = CITY_COORDS[c]
        w = fetch(lat, lon, start, end)
        w["city"] = c
        done = pd.concat([done, w]); done.to_csv(OUT, index=False)
        print("fetched", c, f"({done['city'].nunique()} cities saved)", flush=True)
        time.sleep(5)
    print("done:", len(done), "rows,", done["date"].min(), "to", done["date"].max())
