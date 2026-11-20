"""Live PM2.5 from OpenAQ v3 (CPCB stations are included). Needs a free API key:
https://explore.openaq.org -> register -> API key.  export OPENAQ_API_KEY=xxxx"""
import os, requests, pandas as pd
from datetime import datetime, timedelta, timezone
BASE = "https://api.openaq.org/v3"
def _h(): return {"X-API-Key": os.environ["OPENAQ_API_KEY"]}

def pm25_sensors(lat, lon, radius=25000, limit=10):
    r = requests.get(f"{BASE}/locations", headers=_h(), timeout=60, params=dict(
        coordinates=f"{lat},{lon}", radius=radius, parameters_id=2, limit=limit))   # 2 = pm25
    r.raise_for_status(); out = []
    for loc in r.json()["results"]:
        for s in loc.get("sensors", []):
            if s["parameter"]["name"] == "pm25":
                out.append({"sensor_id": s["id"], "location": loc["name"]})
    return out

def hourly_pm25(sensor_id, hours=72):
    end = datetime.now(timezone.utc); start = end - timedelta(hours=hours)
    r = requests.get(f"{BASE}/sensors/{sensor_id}/hours", headers=_h(), timeout=60, params=dict(
        datetime_from=start.isoformat(), datetime_to=end.isoformat(), limit=1000))
    r.raise_for_status()
    rows = [{"time": pd.to_datetime(x["period"]["datetimeFrom"]["utc"]), "pm25": x["value"]}
            for x in r.json()["results"]]
    return pd.DataFrame(rows).sort_values("time") if rows else pd.DataFrame(columns=["time", "pm25"])

def city_live_hourly(lat, lon, hours=72):
    """Average across all PM2.5 sensors within 25 km."""
    frames = []
    for s in pm25_sensors(lat, lon):
        d = hourly_pm25(s["sensor_id"], hours)
        if len(d): frames.append(d.set_index("time")["pm25"].rename(s["location"]))
    if not frames: return pd.DataFrame(columns=["time", "pm25"])
    m = pd.concat(frames, axis=1).clip(lower=0, upper=1000)
    return m.mean(axis=1).rename("pm25").reset_index()

def live_to_daily(df):   # use to append latest days to data/processed.csv
    return df.set_index("time")["pm25"].resample("D").mean().reset_index().rename(columns={"time": "date"})
