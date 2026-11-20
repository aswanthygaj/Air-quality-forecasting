"""Audit of the Kaggle files (schema, coverage, stations per city).
Run (venv active):  python src/inspect_data.py FOLDER_WITH_THE_CSV_FILES 2>&1 | Tee-Object -FilePath inspect_results.txt"""
import sys, os, pandas as pd, numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads", "aqreal")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def col(df, *keys):
    for c in df.columns:
        if all(k in c.lower() for k in keys):
            return c
    return None

def path(fn):
    return os.path.join(folder, fn)

print(f"folder: {folder}")
for fn in ["stations.csv", "station_day.csv", "city_day.csv", "station_hour.csv", "city_hour.csv"]:
    if not os.path.exists(path(fn)):
        print(f"\n{fn}: MISSING"); continue
    head = pd.read_csv(path(fn), nrows=3)
    print(f"\n=== {fn} ({os.path.getsize(path(fn)) / 1e6:.1f} MB) ===")
    print("columns:", list(head.columns))
    print(head.head(2).to_string(index=False))

# ---- stations.csv
if os.path.exists(path("stations.csv")):
    st = pd.read_csv(path("stations.csv"))
    print(f"\n--- stations.csv: {len(st)} stations ---")
    for c in st.columns:
        if st[c].nunique() < 12:
            print(f"{c}:", st[c].value_counts().to_dict())
    cc = col(st, "city")
    if cc:
        print("stations per city:", st[cc].value_counts().to_dict())
    print("coordinate columns present:", [c for c in st.columns if any(k in c.lower() for k in ("lat", "lon"))] or "NONE")

# ---- station_day.csv
if os.path.exists(path("station_day.csv")):
    sd = pd.read_csv(path("station_day.csv"))
    idc, dc, pc = col(sd, "station", "id"), col(sd, "date"), col(sd, "pm2")
    sd[dc] = pd.to_datetime(sd[dc])
    obs = sd[sd[pc].notna()]
    per = obs.groupby(idc).size()
    print(f"\n--- station_day.csv: {len(sd)} rows, {sd[idc].nunique()} stations, {sd[dc].min().date()} to {sd[dc].max().date()} ---")
    print(f"PM2.5 non-missing: {100 * sd[pc].notna().mean():.1f}% of rows")
    print("PM2.5 days per station (quantiles 0/25/50/75/100%):", per.quantile([0, .25, .5, .75, 1]).round(0).tolist())
    for k in (365, 730, 1095):
        print(f"stations with >= {k} PM2.5 days: {int((per >= k).sum())}")
    if os.path.exists(path("stations.csv")) and col(st, "station", "id") and col(st, "city"):
        m = per.rename("days").reset_index().merge(st[[col(st, "station", "id"), col(st, "city")]],
                                                   left_on=idc, right_on=col(st, "station", "id"), how="left")
        g = m.groupby(col(st, "city")).agg(stations=("days", "size"), long_history=("days", lambda x: int((x >= 730).sum())))
        print("per city: stations with PM2.5 / stations with >= 730 days")
        print(g.sort_values("stations", ascending=False).to_string())
    print("PM2.5 days per year:", obs.groupby(obs[dc].dt.year).size().to_dict())

# ---- city_day.csv
if os.path.exists(path("city_day.csv")):
    cd = pd.read_csv(path("city_day.csv"))
    dc, pc, cc = col(cd, "date"), col(cd, "pm2"), col(cd, "city")
    cd[dc] = pd.to_datetime(cd[dc])
    print(f"\n--- city_day.csv: {len(cd)} rows, {cd[cc].nunique()} cities, {cd[dc].min().date()} to {cd[dc].max().date()} ---")

# ---- hourly files (chunked)
for fn in ["station_hour.csv", "city_hour.csv"]:
    if not os.path.exists(path(fn)):
        continue
    head = pd.read_csv(path(fn), nrows=3)
    idc = col(head, "station", "id") or col(head, "city"); dc = col(head, "date"); pc = col(head, "pm2")
    n = 0; nn = 0; lo = hi = None; counts = pd.Series(dtype="float64")
    for ch in pd.read_csv(path(fn), usecols=[idc, dc, pc], chunksize=500_000):
        t = pd.to_datetime(ch[dc], errors="coerce")
        n += len(ch); nn += int(ch[pc].notna().sum())
        lo = t.min() if lo is None else min(lo, t.min()); hi = t.max() if hi is None else max(hi, t.max())
        counts = counts.add(ch.groupby(idc)[pc].count(), fill_value=0)
    print(f"\n--- {fn}: {n} rows, {counts.shape[0]} ids ({idc}), {lo} to {hi} ---")
    print(f"PM2.5 non-missing: {100 * nn / n:.1f}% of rows")
    print("PM2.5 hours per id (quantiles 0/25/50/75/100%):", counts.quantile([0, .25, .5, .75, 1]).round(0).tolist())
    print(f"ids with >= 17520 PM2.5 hours (2 years): {int((counts >= 17520).sum())}")
print("\ndone")
