"""Creates a realistic *synthetic* dataset in the exact Kaggle schema so the whole
pipeline runs immediately. Replace data/city_day.csv with the real Kaggle file later."""
import numpy as np, pandas as pd
from common import pm25_to_aqi, aqi_bucket
rng = np.random.default_rng(42)
CITIES = {  # base PM2.5, seasonal amplitude, base temp, base wind
    "Delhi": (95, 0.65, 25, 2.2), "Lucknow": (85, 0.6, 26, 2.0), "Kolkata": (65, 0.55, 27, 2.4),
    "Mumbai": (40, 0.35, 28, 3.2), "Chennai": (32, 0.2, 29, 3.5), "Bengaluru": (30, 0.25, 24, 3.0)}
dates = pd.date_range("2015-01-01", "2020-07-01")
n = len(dates); doy = dates.dayofyear.values
def ar1(phi, sig, n):
    e = np.zeros(n)
    for i in range(1, n): e[i] = phi * e[i - 1] + rng.normal(0, sig)
    return e
shared = ar1(0.85, 0.12, n)                         # regional shock => spatial correlation
monsoon = ((doy > 165) & (doy < 275))
cd, wx = [], []
for c, (base, amp, bt, bw) in CITIES.items():
    temp = bt + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 1.5, n)
    wind = np.clip(bw * np.exp(ar1(0.6, 0.25, n)), 0.3, 15)
    rain = np.where(rng.random(n) < np.where(monsoon, 0.55, 0.06), rng.exponential(8, n), 0.0)
    rh = np.clip(55 + 25 * monsoon + 0.8 * np.minimum(rain, 20) + rng.normal(0, 8, n), 15, 100)
    pres = 1000 + 6 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2, n)
    seas = amp * np.cos(2 * np.pi * (doy - 15) / 365)   # peaks mid-Jan (winter)
    lp = (np.log(base) + seas + 0.7 * shared + ar1(0.8, 0.14, n)
          - 0.30 * (wind - bw) / bw - 0.55 * (rain > 2) - 0.01 * (temp - bt))
    pm = np.exp(lp)
    pm = pm * np.where(rng.random(n) < 0.03, rng.uniform(1.4, 2.2, n), 1)   # random spikes
    wx.append(pd.DataFrame({"city": c, "date": dates, "temp": temp, "rh": rh, "wind": wind,
                            "rain": rain, "pressure": pres}))
    aqi = np.array([pm25_to_aqi(v) for v in pm])
    pm_obs = pm.copy(); pm_obs[rng.random(n) < 0.05] = np.nan       # 5% missing like real data
    cd.append(pd.DataFrame({"City": c, "Date": dates, "PM2.5": pm_obs.round(2),
                            "PM10": (pm * 1.8).round(2), "AQI": aqi.round(0),
                            "AQI_Bucket": [aqi_bucket(a) for a in aqi]}))
pd.concat(cd).to_csv("data/city_day.csv", index=False)
pd.concat(wx).to_csv("data/weather.csv", index=False)
print("wrote data/city_day.csv and data/weather.csv")
