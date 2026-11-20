"""Build a FIXED-STATION city panel from station_day.csv: each city's series uses only stations that were
already reporting before 2018 and are still active in mid-2020, so the monitor set does not change over the
evaluation period.
Run (venv active):  python src/build_fixed_panel.py FOLDER_WITH_THE_CSV_FILES 2>&1 | Tee-Object -FilePath fixed_panel_results.txt"""
import sys, os, pandas as pd, numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads", "aqreal")
FIRST_BY = pd.Timestamp("2017-12-31")      # station must have reported on or before this date
ACTIVE_AFTER = pd.Timestamp("2020-06-01")  # ... and still be reporting after this date
MIN_COVER = 0.6                             # ... with PM2.5 on >= 60% of days between its first and last report
MIN_FRAC = 0.5                              # a city-day needs >= 50% of the city's core stations reporting
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

sd = pd.read_csv(os.path.join(folder, "station_day.csv"), usecols=["StationId", "Date", "PM2.5"], parse_dates=["Date"])
st = pd.read_csv(os.path.join(folder, "stations.csv"))[["StationId", "City"]]
sd = sd.merge(st, on="StationId", how="left").dropna(subset=["PM2.5", "City"])

g = sd.groupby("StationId").agg(first=("Date", "min"), last=("Date", "max"), n=("Date", "size"))
g["cover"] = g["n"] / ((g["last"] - g["first"]).dt.days + 1)
core_ids = g[(g["first"] <= FIRST_BY) & (g["last"] >= ACTIVE_AFTER) & (g["cover"] >= MIN_COVER)].index
core = sd[sd.StationId.isin(core_ids)]
n_core = core.groupby("City")["StationId"].nunique().rename("n_core")

out = core.groupby(["City", "Date"]).agg(**{"PM2.5": ("PM2.5", "mean"), "n_reporting": ("StationId", "nunique")}).reset_index()
out = out.merge(n_core, left_on="City", right_index=True)
out = out[out["n_reporting"] >= np.ceil(MIN_FRAC * out["n_core"])]
out[["City", "Date", "PM2.5", "n_reporting", "n_core"]].to_csv("data/city_day_fixed.csv", index=False)

orig = pd.read_csv(os.path.join(folder, "city_day.csv"), usecols=["City", "Date", "PM2.5"], parse_dates=["Date"]).dropna()
n_all = sd.groupby("City")["StationId"].nunique().rename("stations_ever")
rows = []
for c in sorted(set(sd.City)):
    f = out[out.City == c]
    o = orig[orig.City == c]
    m = f.merge(o, on=["City", "Date"], suffixes=("_fixed", "_orig"))
    rows.append({"city": c, "stations_ever": int(n_all.get(c, 0)), "core_stations": int(n_core.get(c, 0)),
                 "fixed_days": len(f), "orig_days": len(o),
                 "corr_with_original": round(m["PM2.5_fixed"].corr(m["PM2.5_orig"]), 3) if len(m) > 30 else np.nan,
                 "first_day": f.Date.min().date() if len(f) else None})
rep = pd.DataFrame(rows).sort_values("core_stations", ascending=False)
print(f"rules: first report <= {FIRST_BY.date()}, active after {ACTIVE_AFTER.date()}, coverage >= {MIN_COVER:.0%}, "
      f">= {MIN_FRAC:.0%} of core stations reporting per day")
print(f"core stations: {len(core_ids)} of {g.shape[0]} stations with PM2.5; cities kept: {int((rep.core_stations > 0).sum())} of {len(rep)}")
print(rep.to_string(index=False))
print("\ncities with NO core station (dropped from the fixed panel):", ", ".join(rep[rep.core_stations == 0].city) or "none")
print("\nsaved data/city_day_fixed.csv")
