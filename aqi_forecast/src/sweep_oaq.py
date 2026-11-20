"""Threshold sweep for the fixed-location OpenAQ panel: how many core locations / cities survive under each rule?
Run (venv active):  python src/sweep_oaq.py 2>&1 | Tee-Object -FilePath sweep_results.txt"""
import itertools, numpy as np, pandas as pd

KEY = ["Delhi", "Patna", "Lucknow", "Gurugram", "Kolkata", "Mumbai"]
ERA_A, ERA_B = ("2020-01-01", "2022-10-31"), ("2025-02-01", "2026-10-31")
sd = pd.read_csv("data/openaq_station_day.csv", parse_dates=["date"])
sd = sd[sd.date >= "2018-01-01"]
days_a = (pd.Timestamp(ERA_A[1]) - pd.Timestamp(ERA_A[0])).days + 1
days_b = (pd.Timestamp(ERA_B[1]) - pd.Timestamp(ERA_B[0])).days + 1
rows = []
for pct in (0, 50, 75):
    d = sd[sd["pct"].isna() | (sd["pct"] >= pct)]
    loc = d.groupby(["city", "location_id", "date"]).size().reset_index()[["city", "location_id", "date"]]
    first = loc.groupby(["city", "location_id"])["date"].min()
    def cov(a, b, n):
        s = loc[(loc.date >= a) & (loc.date <= b)].groupby(["city", "location_id"]).size()
        return s.reindex(first.index, fill_value=0) / n
    t = pd.DataFrame({"first": first, "ca": cov(*ERA_A, days_a), "cb": cov(*ERA_B, days_b)}).reset_index()
    for fb, mc in itertools.product(("2019-12-31", "2020-06-30", "2020-12-31"), (0.2, 0.3, 0.4, 0.5)):
        c = t[(t["first"] <= fb) & (t.ca >= mc) & (t.cb >= mc)]
        per = c.groupby("city").size()
        rows.append({"min %hours": pct, "first by": fb, "min coverage": mc, "core locations": len(c), "cities": len(per),
                     "key cities (locations)": " ".join(f"{k[:3]}{int(per.get(k, 0))}" for k in KEY)})
print("percent-complete values above 100 occur for", int((sd['pct'] > 100).sum()), "rows (probably sub-hourly reporting)")
print(pd.DataFrame(rows).to_string(index=False))
