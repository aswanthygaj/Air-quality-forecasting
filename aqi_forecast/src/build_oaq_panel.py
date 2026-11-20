"""Build a fixed-location city series from the OpenAQ station-day file (merging each location's old and new sensors).
Run (venv active):  python src/build_oaq_panel.py 2>&1 | Tee-Object -FilePath oaq_panel_results.txt"""
import numpy as np, pandas as pd

FIRST_BY = pd.Timestamp("2020-06-30")                      # location must have reported by this date
ERA_A, ERA_B = ("2020-01-01", "2022-10-31"), ("2025-02-01", "2026-10-31")
MIN_COVER, MIN_FRAC, MIN_PCT = 0.4, 0.25, 50                # coverage per era; share of core locations per day; hours-complete %
pd.set_option("display.width", 200)

sd = pd.read_csv("data/openaq_station_day.csv", parse_dates=["date"])
sd = sd[sd.date >= "2018-01-01"]
print("rows from 2018:", len(sd), "| percent-complete column present:", int(sd["pct"].notna().sum()), "non-missing values")
if sd["pct"].notna().any():
    print("percent-complete quantiles (0/10/25/50/100%):", sd["pct"].quantile([0, .1, .25, .5, 1]).round(0).tolist())
    sd = sd[sd["pct"].isna() | (sd["pct"] >= MIN_PCT)]
    print(f"rows kept after requiring >= {MIN_PCT}% of hours:", len(sd))
loc = sd.groupby(["city", "location_id", "date"])["pm25"].mean().reset_index()      # old and new sensors of a location become one series

def cover(g, a, b):
    return g[(g.date >= a) & (g.date <= b)].date.nunique() / ((pd.Timestamp(b) - pd.Timestamp(a)).days + 1)
core = [(c, l) for (c, l), g in loc.groupby(["city", "location_id"])
        if g.date.min() <= FIRST_BY and cover(g, *ERA_A) >= MIN_COVER and cover(g, *ERA_B) >= MIN_COVER]
core = pd.DataFrame(core, columns=["city", "location_id"])
n_all = loc.groupby("city")["location_id"].nunique()
n_core = core.groupby("city").size()
print(f"\ncore locations (reported by {FIRST_BY.date()}, >= {MIN_COVER:.0%} coverage in both eras): {len(core)} of {loc.location_id.nunique()}")

cs = loc.merge(core, on=["city", "location_id"])
out = cs.groupby(["city", "date"]).agg(pm25=("pm25", "mean"), n_reporting=("location_id", "nunique")).reset_index()
out["n_core"] = out.city.map(n_core)
out = out[out.n_reporting >= np.ceil(MIN_FRAC * out.n_core)]
out.rename(columns={"city": "City", "date": "Date", "pm25": "PM2.5"}).to_csv("data/city_day_oaq.csv", index=False)

def winter_days(g, y):
    return int(((g.date >= f"{y}-11-01") & (g.date <= f"{y + 1}-02-28")).sum())
rows = []
for c, g in out.groupby("city"):
    rows.append({"city": c, "locations all/core": f"{n_all.get(c, 0)}/{n_core.get(c, 0)}", "days 2018-2020": int((g.date < "2020-07-01").sum()),
                 "days 2020-07..2022-10": int(((g.date >= "2020-07-01") & (g.date <= "2022-10-31")).sum()),
                 "days 2025+": int((g.date >= "2025-02-01").sum()), "winter 20-21": winter_days(g, 2020),
                 "winter 21-22": winter_days(g, 2021), "winter 25-26": winter_days(g, 2025)})
rep = pd.DataFrame(rows).sort_values("days 2025+", ascending=False)
print("\nper city (winter columns = days with data out of ~120 between Nov 1 and Feb 28):")
print(rep.to_string(index=False))
ok = rep[(rep["winter 20-21"] >= 80) & (rep["winter 21-22"] >= 80) & (rep["winter 25-26"] >= 80)]
print(f"\ncities with >= 80 winter days in ALL three new winters: {len(ok)} -> {', '.join(ok.city)}")
print("\nsaved data/city_day_oaq.csv:", len(out), "city-days")
