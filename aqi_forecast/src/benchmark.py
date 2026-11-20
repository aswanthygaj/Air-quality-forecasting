"""Benchmark v2: causal imputation, observed-only scoring, rolling-origin validation,
Ridge + gradient boosting (LightGBM) + Diebold-Mariano tests.
Run (project root, venv active):  python src/benchmark.py | Tee-Object -FilePath benchmark_results.txt"""
import math, time, numpy as np, pandas as pd
from common import WX
from final_lib import city2grp, ridge_params

H = 3                      # horizons: 1, 2, 3 days
LAGS = 7
MAX_FFILL = 14             # longest gap (days) that is carried forward
MIN_TRAIN_DAYS = 365       # a city enters a fold's evaluation only with >= 1 year of training origins
ALPHA = 100.0              # ridge strength (chosen on validation in the earlier study)
DM_LAG = 7                 # Newey-West lag (days) for Diebold-Mariano
# test blocks (expanding-window training before each block)
BOUNDS = pd.to_datetime(["2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01", "2020-01-01", "2020-07-02"])

try:
    import lightgbm as lgb
    def make_gbm():
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 verbose=-1, random_state=0)
    GBM = "LightGBM"
except Exception:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        def make_gbm():
            return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                 min_samples_leaf=40, random_state=0)
        GBM = "HistGradientBoosting"
    except Exception:
        make_gbm, GBM = None, None
print("gradient boosting backend:", GBM)

# ------------------------------------------------------------------ data with CAUSAL imputation
cd = pd.read_csv("data/city_day.csv", parse_dates=["Date"]).rename(
    columns={"City": "city", "Date": "date", "PM2.5": "pm_raw"})[["city", "date", "pm_raw"]]
wx = pd.read_csv("data/weather.csv", parse_dates=["date"])
full = pd.date_range(cd.date.min(), cd.date.max())
cap = cd.pm_raw.quantile(0.999)
parts = []
for c in sorted(set(cd.city) & set(wx.city)):
    s = cd[cd.city == c].drop_duplicates("date").set_index("date")["pm_raw"].reindex(full)
    if s.notna().sum() < 200:
        continue
    s = s.loc[s.first_valid_index():s.last_valid_index()].clip(lower=1, upper=cap)
    g = pd.DataFrame({"pm_raw": s})
    g["obs"] = g["pm_raw"].notna()
    g["pm"] = g["pm_raw"].ffill(limit=MAX_FFILL)           # carry forward only: never uses future values
    g["city"] = c
    g.index.name = "date"
    parts.append(g.reset_index())
df = pd.concat(parts).merge(wx[["city", "date"] + WX], on=["city", "date"], how="left")
df[WX] = df.groupby("city")[WX].transform(lambda x: x.ffill().bfill())
cities = sorted(df.city.unique())
print(f"{len(cities)} cities; {100 * (~df.obs).mean():.1f}% of city-days are missing readings "
      f"(carried forward for features, never used as targets)")

df["lp"] = np.log1p(df["pm"])
piv = df.pivot(index="date", columns="city", values="lp")
reg = {}
for c in piv.columns:
    mates = [m for m in piv.columns if m != c and city2grp.get(m, "other") == city2grp.get(c, "other")]
    reg[c] = piv[mates].mean(axis=1) if mates else pd.Series(np.nan, index=piv.index)

FEATS = ([f"lp_l{k}" for k in range(LAGS)] + ["lp_m14", "lp_m30", "lp_sd7", "reg0", "reg1", "doy_s", "doy_c"]
         + WX + [w + "_m3" for w in WX])
frames = []
for c in cities:
    g = df[df.city == c].sort_values("date").reset_index(drop=True)
    lp = g["lp"]
    o = pd.DataFrame({"city": c, "date": g["date"], "city_id": cities.index(c)})
    for k in range(LAGS):
        o[f"lp_l{k}"] = lp.shift(k)
    o["lp_m14"] = lp.rolling(14).mean()
    o["lp_m30"] = lp.rolling(30, min_periods=20).mean()
    o["lp_sd7"] = lp.rolling(7).std()
    r0 = reg[c].reindex(g["date"]).values
    r0 = np.where(np.isnan(r0), lp.values, r0)
    o["reg0"] = r0
    o["reg1"] = pd.Series(r0).shift(1).values
    doy = g["date"].dt.dayofyear
    o["doy_s"], o["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for w in WX:
        o[w] = g[w]
        o[w + "_m3"] = g[w].rolling(3).mean()
    o["pm_now"] = g["pm"]
    for h in range(1, H + 1):
        o[f"y{h}"] = g["pm_raw"].shift(-h)                 # NaN unless the target day was really observed
    frames.append(o)
data = pd.concat(frames).dropna(subset=FEATS).reset_index(drop=True)
print(f"origins: {len(data)}")

# ------------------------------------------------------------------ rolling-origin evaluation
def ridge_X(frame, mu, sd):
    oh = (np.asarray(frame["city"].astype(object))[:, None] == np.array(cities, dtype=object)[None, :]).astype(float)
    return np.hstack([(frame[FEATS].values - mu) / sd, oh])

out = []
for k in range(len(BOUNDS) - 1):
    a, b = BOUNDS[k], BOUNDS[k + 1]
    train = data[data.date + pd.Timedelta(days=H) < a]
    n_tr = train.groupby("city").size()
    elig = [c for c in cities if n_tr.get(c, 0) >= MIN_TRAIN_DAYS]
    test = data[(data.date >= a) & (data.date < b) & data.city.isin(elig)]
    print(f"fold {k + 1}: test {a.date()} to {(b - pd.Timedelta(days=1)).date()}, "
          f"train origins {len(train)}, eligible cities {len(elig)}/{len(cities)}", flush=True)
    for h in range(1, H + 1):
        tr_h = train[train[f"y{h}"].notna()]
        te_h = test[test[f"y{h}"].notna()]
        if len(te_h) == 0:
            continue
        rec = te_h[["city", "date"]].copy()
        rec["fold"], rec["h"], rec["y"] = k + 1, h, te_h[f"y{h}"].values
        rec["Persistence"] = te_h["pm_now"].values
        mu, sd = tr_h[FEATS].mean().values, tr_h[FEATS].std().values + 1e-6
        m_, ym_, w_ = ridge_params(ridge_X(tr_h, mu, sd), np.log1p(tr_h[f"y{h}"].values), ALPHA)
        rec["Ridge"] = np.clip(np.expm1((ridge_X(te_h, mu, sd) - m_) @ w_ + ym_), 0, None)
        if make_gbm is not None:
            gbm = make_gbm()
            gbm.fit(tr_h[FEATS + ["city_id"]], np.log1p(tr_h[f"y{h}"]))
            rec[GBM] = np.clip(np.expm1(gbm.predict(te_h[FEATS + ["city_id"]])), 0, None)
        out.append(rec)
R = pd.concat(out).reset_index(drop=True)
MODELS = [m for m in ["Persistence", "Ridge", GBM] if m and m in R.columns]
for m in MODELS:
    R[m + "_ae"] = (R[m] - R["y"]).abs()
R.to_csv("benchmark_predictions.csv", index=False)

# ------------------------------------------------------------------ tables
print(f"\n=== 1. POOLED TEST RESULTS across {len(BOUNDS) - 1} rolling folds (observed targets only, ug/m3) ===")
rows = []
for m in MODELS:
    row = {"model": m}
    for h in range(1, H + 1):
        s = R[R.h == h]
        row[f"MAE_{24 * h}h"] = round(s[m + "_ae"].mean(), 2)
    for h in range(1, H + 1):
        s = R[R.h == h]
        row[f"RMSE_{24 * h}h"] = round(float(np.sqrt(((s[m] - s["y"]) ** 2).mean())), 2)
    hi = R[R.y > 90]
    row["MAE_high"] = round(hi[m + "_ae"].mean(), 2)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("test windows per horizon:", R.groupby("h").size().to_dict(),
      "| high-pollution targets:", int((R.y > 90).sum()))

print("\n=== 2. MAE BY FOLD (24 h / 72 h) ===")
rows = []
for k in sorted(R.fold.unique()):
    s = R[R.fold == k]
    row = {"fold": k, "test block": f"{BOUNDS[k - 1].date()} -> {(BOUNDS[k] - pd.Timedelta(days=1)).date()}",
           "cities": s.city.nunique()}
    for m in MODELS:
        row[f"{m} 24h"] = round(s[s.h == 1][m + "_ae"].mean(), 1)
        row[f"{m} 72h"] = round(s[s.h == 3][m + "_ae"].mean(), 1)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

def dm(Rh, a, b, lag=DM_LAG):
    d = (Rh[a + "_ae"] - Rh[b + "_ae"]).groupby(Rh["date"]).mean().sort_index()
    T, mean = len(d), d.mean()
    x = d.values - mean
    lrv = (x ** 2).mean()
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    stat = mean / math.sqrt(max(lrv, 1e-12) / T)
    return mean, stat, math.erfc(abs(stat) / math.sqrt(2))

print(f"\n=== 3. DIEBOLD-MARIANO TESTS (A minus B in MAE; negative = A better; Newey-West lag {DM_LAG} days) ===")
pairs = [("Ridge", "Persistence")]
if GBM:
    pairs += [(GBM, "Persistence"), (GBM, "Ridge")]
rows = []
for a_, b_ in pairs:
    row = {"A minus B": f"{a_} - {b_}"}
    for h in range(1, H + 1):
        d_, s_, p_ = dm(R[R.h == h], a_, b_)
        row[f"{24 * h}h"] = f"{d_:+.2f} (DM {s_:+.1f}, p={p_:.3f})"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\nsaved benchmark_predictions.csv")
