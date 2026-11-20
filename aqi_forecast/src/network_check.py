"""Network-composition check: did the city-level PM2.5 level shift come from changing monitor networks?
Run (venv active):  python src/network_check.py FOLDER_WITH_THE_CSV_FILES 2>&1 | Tee-Object -FilePath network_results.txt"""
import sys, os, pandas as pd, numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads", "aqreal")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
path = lambda fn: os.path.join(folder, fn)

sd = pd.read_csv(path("station_day.csv"), usecols=["StationId", "Date", "PM2.5"], parse_dates=["Date"])
st = pd.read_csv(path("stations.csv"))[["StationId", "City"]]
sd = sd.merge(st, on="StationId", how="left").dropna(subset=["PM2.5", "City"])
sd["year"] = sd.Date.dt.year

print("=== 1. NUMBER OF STATIONS WITH PM2.5 DATA, BY CITY AND YEAR ===")
tab = sd.groupby(["City", "year"])["StationId"].nunique().unstack(fill_value=0)
print(tab.sort_values(2019, ascending=False).head(26).to_string())

print("\n=== 2. IS city_day.csv THE AVERAGE OVER STATIONS? ===")
cd = pd.read_csv(path("city_day.csv"), usecols=["City", "Date", "PM2.5"], parse_dates=["Date"]).dropna()
calc = sd.groupby(["City", "Date"])["PM2.5"].mean().rename("calc").reset_index()
m = cd.merge(calc, on=["City", "Date"])
diff = (m["PM2.5"] - m["calc"]).abs()
print(f"{len(m)} city-days compared; median |difference| = {diff.median():.2f}; "
      f"share within 1 ug/m3 = {100 * (diff < 1).mean():.0f}%; correlation = {m['PM2.5'].corr(m['calc']):.3f}")

print("\n=== 3. LEVEL CHANGE, ALL STATIONS vs THE SAME STATIONS IN BOTH ERAS (Jul-Aug excluded in both) ===")
t0, t1 = cd.Date.min(), cd.Date.max()
cut_tr, cut_va = t0 + (t1 - t0) * 0.70, t0 + (t1 - t0) * 0.85
print(f"training era: {t0.date()} to {cut_tr.date()}   test era: {cut_va.date()} to {t1.date()}")
sd = sd[~sd.Date.dt.month.isin([7, 8])]
tr, te = sd[sd.Date < cut_tr], sd[sd.Date >= cut_va]
def era_mean(d):
    return d.groupby("Date")["PM2.5"].mean().mean() if len(d) else np.nan
rows = []
for city in sorted(set(tr.City) & set(te.City)):
    a, b = tr[tr.City == city], te[te.City == city]
    na, nb = a.groupby("StationId").size(), b.groupby("StationId").size()
    core = sorted(set(na[na >= 180].index) & set(nb[nb >= 90].index))
    if not core:
        continue
    ma, mb = era_mean(a), era_mean(b)
    ca, cb = era_mean(a[a.StationId.isin(core)]), era_mean(b[b.StationId.isin(core)])
    rows.append({"city": city, "stations_train": int((na >= 30).sum()), "stations_test": int((nb >= 30).sum()),
                 "core": len(core), "all_train": round(ma, 1), "all_test": round(mb, 1),
                 "change_all_%": round(100 * (mb / ma - 1), 1), "core_train": round(ca, 1),
                 "core_test": round(cb, 1), "change_core_%": round(100 * (cb / ca - 1), 1)})
out = pd.DataFrame(rows).sort_values("all_train", ascending=False)
print(out.to_string(index=False))
if len(out):
    print(f"\nmedian change over {len(out)} cities: all stations {out['change_all_%'].median():+.1f}%, "
          f"same stations only {out['change_core_%'].median():+.1f}%")
print("\ndone")
