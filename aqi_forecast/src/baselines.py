"""Simple baselines + per-city + lockdown/pre-lockdown evaluation.
Run (project root, venv active):  python src/baselines.py
Needs models/lstm.pt and models/meta.json from train_lstm.py (the LSTM is skipped if absent)."""
import json, numpy as np, pandas as pd
from common import *

LAGS = 7
LOCKDOWN = np.datetime64("2020-03-25")

raw = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))
meta = json.load(open("models/meta.json"))
cities = meta["cities"]
if sorted(raw.city.unique()) != cities:
    print("WARNING: cities differ from models/meta.json - retrain with train_lstm.py")
t0, t1 = raw.date.min(), raw.date.max()
cut_tr = (t0 + (t1 - t0) * 0.70).to_datetime64()
cut_va = (t0 + (t1 - t0) * 0.85).to_datetime64()

# ---- seasonal climatology (per city, day-of-year mean over TRAIN period, smoothed +/-15 days)
def _profile(g):
    m = g.groupby(g.date.dt.dayofyear)["pm25"].mean().reindex(range(1, 367))
    ext = pd.concat([m.iloc[-15:], m, m.iloc[:15]]).reset_index(drop=True)
    sm = ext.rolling(31, center=True, min_periods=5).mean().iloc[15:-15]
    return sm.fillna(g.pm25.mean()).values
_train = raw[raw.date < pd.Timestamp(cut_tr)]
_glob = _profile(_train)          # all-cities profile, used for cities with little/no training data
clim = {c: (_profile(_train[_train.city == c]) if (_train.city == c).sum() >= 365 else _glob)
        for c in raw.city.unique()}

# ---- build one record per forecast origin (same origins as the LSTM windows)
std = standardize(raw, meta)
feats = meta["feats"]
X, L, Y, P, D, C, S = [], [], [], [], [], [], []
for c in cities:
    g = std[std.city == c].reset_index(drop=True)
    r = raw[raw.city == c].reset_index(drop=True)
    F = g[feats].values.astype("float32")
    pm, lpm, dt = r["pm25"].values, r["lpm"].values, r["date"].values
    sn, cs, doy = r["doy_sin"].values, r["doy_cos"].values, r["date"].dt.dayofyear.values
    onehot = np.zeros(len(cities)); onehot[cities.index(c)] = 1
    for t in range(SEQ - 1, len(r) - HOR):
        X.append(F[t - SEQ + 1:t + 1])
        L.append(np.concatenate([lpm[t - LAGS + 1:t + 1], [sn[t], cs[t]], onehot]))
        Y.append(pm[t + 1:t + 1 + HOR]); P.append(pm[t]); D.append(dt[t]); C.append(c)
        S.append(clim[c][doy[t + 1:t + 1 + HOR] - 1])
X = np.array(X); L = np.array(L); Y = np.array(Y); P = np.array(P)
D = np.array(D, dtype="datetime64[ns]"); C = np.array(C); S = np.array(S)
tr, va, te = D < cut_tr, (D >= cut_tr) & (D < cut_va), D >= cut_va
print(f"origins: train {tr.sum()}, val {va.sum()}, test {te.sum()}")

# ---- ridge regression on last 7 days + season + city (closed form, no sklearn)
def ridge_fit(A, y, alpha):
    mu, ym = A.mean(0), y.mean(0)
    Ac = A - mu
    w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (y - ym))
    return lambda B: (B - mu) @ w + ym
logY = np.log1p(Y)
best = None
for a in [0.01, 0.1, 1, 10, 100]:
    f = ridge_fit(L[tr], logY[tr], a)
    v = np.abs(np.expm1(f(L[va])) - Y[va]).mean()
    if best is None or v < best[0]:
        best = (v, a, f)
print("ridge alpha chosen on validation:", best[1])
preds = {
    "Persistence": np.repeat(P[:, None], HOR, 1),
    "Seasonal avg": S,
    "Ridge (7d lags)": np.expm1(best[2](L)),
}

# ---- saved LSTM (all features: history + weather + regional)
try:
    import torch
    from model import Net
    net = Net(len(feats))
    net.load_state_dict(torch.load("models/lstm.pt", map_location="cpu", weights_only=True))
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(net(torch.tensor(X[i:i + 2048])).numpy())
    preds["LSTM"] = inv(np.concatenate(out), meta)
except Exception as e:
    print("LSTM skipped:", repr(e)[:150])

# ---- tables
def mae(p, m): return np.abs(p[m] - Y[m]).mean(0)
def rmse(p, m): return np.sqrt(((p[m] - Y[m]) ** 2).mean(0))
sev = te[:, None] & (Y > 90)

rows = []
for n, p in preds.items():
    rows.append([n, *mae(p, te).round(2), *rmse(p, te).round(2), np.abs(p - Y)[sev].mean().round(2)])
overall = pd.DataFrame(rows, columns=["model", "MAE_24h", "MAE_48h", "MAE_72h",
                                      "RMSE_24h", "RMSE_48h", "RMSE_72h", "MAE_severe(>90)"])
print("\n=== 1. OVERALL TEST (ug/m3) ===\n" + overall.to_string(index=False))

rows = []
for c in cities:
    m = te & (C == c)
    if m.sum() == 0: continue
    row = {"city": c, "n": int(m.sum()), "mean_pm25": Y[m].mean().round(1)}
    for n, p in preds.items():
        e = mae(p, m); row[f"{n} 24h"], row[f"{n} 72h"] = e[0].round(1), e[2].round(1)
    rows.append(row)
percity = pd.DataFrame(rows).sort_values("mean_pm25", ascending=False)
print("\n=== 2. PER CITY (test MAE, sorted by pollution level) ===\n" + percity.to_string(index=False))

rows = []
for label, m in [("before lockdown (origin < 25 Mar 2020)", te & (D < LOCKDOWN)),
                 ("lockdown period (origin >= 25 Mar 2020)", te & (D >= LOCKDOWN))]:
    if m.sum() == 0: continue
    row = {"period": label, "n": int(m.sum())}
    for n, p in preds.items():
        e = mae(p, m); row[f"{n} 24h"], row[f"{n} 72h"] = e[0].round(1), e[2].round(1)
    rows.append(row)
period = pd.DataFrame(rows)
print("\n=== 3. BY PERIOD (test MAE) ===\n" + period.to_string(index=False))

overall.to_csv("baselines_overall.csv", index=False)
percity.to_csv("baselines_per_city.csv", index=False)
period.to_csv("baselines_by_period.csv", index=False)
print("\nsaved baselines_overall.csv, baselines_per_city.csv, baselines_by_period.csv")
