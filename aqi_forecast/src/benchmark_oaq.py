"""Benchmark phase 2: LSTM + tuned ridge + tuned gradient boosting in the SAME rolling folds,
Diebold-Mariano tests, block-bootstrap on high-pollution days, and a legacy-window check.
Run (project root, venv active):  python src/benchmark2.py 2>&1 | Tee-Object -FilePath benchmark2_results.txt"""
import math, copy, os, time, numpy as np, pandas as pd
from common import WX
from final_lib import city2grp, ridge_params

CITY_FILE = os.environ.get("CITY_FILE", "data/city_day_oaq.csv")
TAG = os.environ.get("TAG", "_oaq")
WEATHER_FILE = os.environ.get("WEATHER_FILE", "data/weather_oaq.csv")
FOLDS = [tuple(pd.Timestamp(x) for x in f.split(":")) for f in os.environ.get("FOLDS", "2020-07-01:2021-03-01,2021-03-01:2022-11-01,2025-02-01:2026-11-06").split(",")]
H, LAGS, SEQ = 3, 7, 14
MAX_FFILL, MIN_TRAIN_DAYS, DM_LAG = 14, 365, 7
N_SEEDS, BOOT_B, BLOCK_DAYS, HIGH = 3, 2000, 14, 90
ALPHAS = [1, 10, 100, 1000, 10000]
GBM_CONFIGS = [dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80),
               dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40),
               dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0)]
BOUNDS = pd.to_datetime(["2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01", "2020-01-01", "2020-07-02"])

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_GBM = True
except Exception as e:
    print("scikit-learn unavailable, boosting skipped:", repr(e)[:100]); HAVE_GBM = False
try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from model import Net
    HAVE_TORCH = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print("PyTorch unavailable, LSTM skipped:", repr(e)[:100]); HAVE_TORCH = False

# ------------------------------------------------------------------ data (causal imputation, observed-only targets)
cd = pd.read_csv(CITY_FILE, parse_dates=["Date"]).rename(
    columns={"City": "city", "Date": "date", "PM2.5": "pm_raw"})[["city", "date", "pm_raw"]]
wx = pd.read_csv(WEATHER_FILE, parse_dates=["date"])
full = pd.date_range(cd.date.min(), cd.date.max())
cap = cd.pm_raw.quantile(0.999)
parts = []
for c in sorted(set(cd.city) & set(wx.city)):
    s = cd[cd.city == c].drop_duplicates("date").set_index("date")["pm_raw"].reindex(full)
    if s.notna().sum() < 200:
        continue
    s = s.loc[s.first_valid_index():s.last_valid_index()].clip(lower=1, upper=cap)
    g = pd.DataFrame({"pm_raw": s}); g["pm"] = g["pm_raw"].ffill(limit=MAX_FFILL); g["city"] = c
    g.index.name = "date"; parts.append(g.reset_index())
df = pd.concat(parts).merge(wx[["city", "date"] + WX], on=["city", "date"], how="left")
df[WX] = df.groupby("city")[WX].transform(lambda x: x.ffill().bfill())
cities = sorted(df.city.unique()); NC = len(cities)
df["lp"] = np.log1p(df["pm"])
piv = df.pivot(index="date", columns="city", values="lp")
reg = {}
for c in piv.columns:
    mates = [m for m in piv.columns if m != c and city2grp.get(m, "other") == city2grp.get(c, "other")]
    reg[c] = piv[mates].mean(axis=1) if mates else pd.Series(np.nan, index=piv.index)

FEATS = ([f"lp_l{k}" for k in range(LAGS)] + ["lp_m14", "lp_m30", "lp_sd7", "reg0", "reg1", "doy_s", "doy_c"]
         + WX + [w + "_m3" for w in WX])
DYN_STD = ["lp_l0", "reg0"] + WX           # channels standardised for the LSTM; sin/cos appended unstandardised
DYN, ONEHOT, frames = {}, {}, []
for c in cities:
    g = df[df.city == c].sort_values("date").reset_index(drop=True)
    lp = g["lp"]
    o = pd.DataFrame({"city": c, "date": g["date"], "city_id": cities.index(c), "pos": np.arange(len(g))})
    for k in range(LAGS):
        o[f"lp_l{k}"] = lp.shift(k)
    o["lp_m14"] = lp.rolling(14).mean(); o["lp_m30"] = lp.rolling(30, min_periods=20).mean()
    o["lp_sd7"] = lp.rolling(7).std()
    r0 = reg[c].reindex(g["date"]).values
    r0 = np.where(np.isnan(r0), lp.values, r0)
    o["reg0"] = r0; o["reg1"] = pd.Series(r0).shift(1).values
    doy = g["date"].dt.dayofyear
    ds, dc = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    o["doy_s"], o["doy_c"] = ds, dc
    for w in WX:
        o[w] = g[w]; o[w + "_m3"] = g[w].rolling(3).mean()
    o["pm_now"] = g["pm"]
    for h in range(1, H + 1):
        o[f"y{h}"] = g["pm_raw"].shift(-h)
    frames.append(o)
    DYN[c] = np.column_stack([lp.values, r0, g[WX].values, ds.values, dc.values]).astype("float32")
    ONEHOT[c] = np.eye(NC, dtype="float32")[cities.index(c)]
data = pd.concat(frames).dropna(subset=FEATS).reset_index(drop=True)
print(f"{NC} cities, {len(data)} origins | GBM: {'HistGradientBoosting' if HAVE_GBM else 'none'} | LSTM: {HAVE_TORCH}", flush=True)

# ------------------------------------------------------------------ model helpers
def ridge_X(frame, mu, sd):
    oh = (np.asarray(frame["city"].astype(object))[:, None] == np.array(cities, dtype=object)[None, :]).astype(float)
    return np.hstack([(frame[FEATS].values - mu) / sd, oh])

def fit_ridge(trn, h, alpha):
    t = trn[trn[f"y{h}"].notna()]
    mu, sd = t[FEATS].mean().values, t[FEATS].std().values + 1e-6
    m_, ym_, w_ = ridge_params(ridge_X(t, mu, sd), np.log1p(t[f"y{h}"].values), alpha)
    return lambda f: np.clip(np.expm1((ridge_X(f, mu, sd) - m_) @ w_ + ym_), 0, None)

def fit_gbm(trn, h, cfg):
    t = trn[trn[f"y{h}"].notna()]
    m = HistGradientBoostingRegressor(random_state=0, **cfg).fit(t[FEATS + ["city_id"]], np.log1p(t[f"y{h}"]))
    return lambda f: np.clip(np.expm1(m.predict(f[FEATS + ["city_id"]])), 0, None)

def inner_mae(pred_fns, va):
    errs = []
    for h in range(1, H + 1):
        v = va[va[f"y{h}"].notna()]
        errs.append(np.abs(pred_fns[h - 1](v) - v[f"y{h}"].values).mean())
    return float(np.mean(errs))

def make_seq(fr, mu, sd):
    nd = len(mu) + 2
    X = np.empty((len(fr), SEQ, nd + NC), dtype="float32")
    mu2 = np.concatenate([mu, [0, 0]]).astype("float32"); sd2 = np.concatenate([sd, [1, 1]]).astype("float32")
    for i, (c, p) in enumerate(zip(np.asarray(fr["city"].astype(object)), fr["pos"].values)):
        X[i, :, :nd] = (DYN[c][p - SEQ + 1:p + 1] - mu2) / sd2
        X[i, :, nd:] = ONEHOT[c]
    return X

def lstm_fold_predict(train, inner_tr, inner_va, test):
    """Returns (list of per-seed predictions (n_test, 3) in ug/m3, mean best epoch)."""
    mu = inner_tr[DYN_STD].mean().values.astype("float32"); sd = inner_tr[DYN_STD].std().values.astype("float32") + 1e-6
    mu_lp, sd_lp = float(mu[0]), float(sd[0])
    def Z(fr):
        return ((np.log1p(fr[[f"y{h}" for h in range(1, H + 1)]].values.astype("float64")) - mu_lp) / sd_lp).astype("float32")
    Xtr, Xva, Xte = make_seq(inner_tr, mu, sd), make_seq(inner_va, mu, sd), make_seq(test, mu, sd)
    Ztr, Zva = torch.tensor(Z(inner_tr)), torch.tensor(Z(inner_va))
    Xtr, Xva, Xte = torch.tensor(Xtr), torch.tensor(Xva), torch.tensor(Xte)
    huber = nn.HuberLoss(reduction="none")
    def mloss(p, z):
        m = ~torch.isnan(z)
        l = huber(p, torch.where(m, z, torch.zeros_like(z))) * m
        return l.sum() / m.sum().clamp(min=1)
    outs, epochs = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        net = Net(Xtr.shape[2]).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        dl = DataLoader(TensorDataset(Xtr, Ztr), 128, shuffle=True)
        xv, zv = Xva.to(dev), Zva.to(dev)
        best, bad, state, best_ep = 1e9, 0, None, 0
        for ep in range(40):
            net.train()
            for xb, zb in dl:
                xb, zb = xb.to(dev), zb.to(dev)
                opt.zero_grad(); mloss(net(xb), zb).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = mloss(net(xv), zv).item()
            if v < best:
                best, bad, state, best_ep = v, 0, copy.deepcopy(net.state_dict()), ep
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            z = np.concatenate([net(Xte[i:i + 4096].to(dev)).cpu().numpy() for i in range(0, len(Xte), 4096)])
        outs.append(np.expm1(z * sd_lp + mu_lp)); epochs.append(best_ep + 1)
        print(f"      LSTM seed {s} done ({best_ep + 1} epochs)", flush=True)
    return outs, float(np.mean(epochs))

def run_fold(name, train_before, test_start, test_end):
    train = data[data.date + pd.Timedelta(days=H) < train_before]
    n_tr = train.groupby("city").size()
    elig = [c for c in cities if n_tr.get(c, 0) >= MIN_TRAIN_DAYS]
    test = data[(data.date >= test_start) & (data.date < test_end) & data.city.isin(elig)].reset_index(drop=True)
    split = train.date.min() + (train.date.max() - train.date.min()) * 0.85
    inner_tr, inner_va = train[train.date < split], train[train.date >= split]
    info = f"fold {name}: train origins {len(train)}, eligible cities {len(elig)}/{NC}, test rows {len(test)}"
    # ridge: alpha chosen on the inner validation window
    best_a = min(ALPHAS, key=lambda a: inner_mae([fit_ridge(inner_tr, h, a) for h in range(1, H + 1)], inner_va))
    ridge_fns = [fit_ridge(train, h, best_a) for h in range(1, H + 1)]
    info += f" | ridge alpha {best_a}"
    gbm_fns = None
    if HAVE_GBM:
        best_c = min(range(len(GBM_CONFIGS)), key=lambda j: inner_mae(
            [fit_gbm(inner_tr, h, GBM_CONFIGS[j]) for h in range(1, H + 1)], inner_va))
        gbm_fns = [fit_gbm(train, h, GBM_CONFIGS[best_c]) for h in range(1, H + 1)]
        info += f" | GBM config {best_c}"
    print(info, flush=True)
    lstm_preds = None
    if HAVE_TORCH:
        lstm_preds, ep = lstm_fold_predict(train, inner_tr, inner_va, test)
        print(f"      LSTM mean best epoch {ep:.1f}", flush=True)
    recs = []
    for h in range(1, H + 1):
        mk = test[f"y{h}"].notna().values
        te_h = test[mk]
        if len(te_h) == 0:
            continue
        rec = te_h[["city", "date"]].copy()
        rec["fold"], rec["h"], rec["y"] = name, h, te_h[f"y{h}"].values
        rec["Persistence"] = te_h["pm_now"].values
        rec["Ridge"] = ridge_fns[h - 1](te_h)
        if gbm_fns:
            rec["GBM"] = gbm_fns[h - 1](te_h)
        if lstm_preds:
            for s, p in enumerate(lstm_preds):
                rec[f"LSTM_s{s}"] = np.clip(p[mk, h - 1], 0, None)
            rec["LSTM"] = np.mean([rec[f"LSTM_s{s}"] for s in range(len(lstm_preds))], axis=0)
            rec["Blend"] = 0.5 * (rec["LSTM"] + rec["Ridge"])
        recs.append(rec)
    return pd.concat(recs)

# ------------------------------------------------------------------ run folds
t_start = time.time()
R = pd.concat([run_fold(k + 1, a, a, b) for k, (a, b) in enumerate(FOLDS)]).reset_index(drop=True)
t0_, t1_ = df.date.min(), df.date.max()
cut_tr_l, cut_va_l = t0_ + (t1_ - t0_) * 0.70, t0_ + (t1_ - t0_) * 0.85
L = R[R.fold == len(FOLDS)].reset_index(drop=True)
MODELS = [m for m in ["Persistence", "Ridge", "GBM", "LSTM", "Blend"] if m in R.columns]
for T in (R, L):
    for m in MODELS:
        T[m + "_ae"] = (T[m] - T["y"]).abs()
R.to_csv(f"benchmark2_predictions{TAG}.csv", index=False)
print(f"\nall folds done in {(time.time() - t_start) / 60:.1f} min")

def summary(T):
    rows = []
    for m in MODELS:
        row = {"model": m}
        for h in range(1, H + 1):
            row[f"MAE_{24 * h}h"] = round(T[T.h == h][m + "_ae"].mean(), 2)
        for h in range(1, H + 1):
            s = T[T.h == h]; row[f"RMSE_{24 * h}h"] = round(float(np.sqrt(((s[m] - s["y"]) ** 2).mean())), 2)
        row["MAE_high"] = round(T[T.y > HIGH][m + "_ae"].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)

print(f"\n=== 1. POOLED ROLLING-FOLD RESULTS (tuned models, observed targets, ug/m3) ===")
print(summary(R))
print("windows per horizon:", R.groupby("h").size().to_dict(), "| high-pollution targets:", int((R.y > HIGH).sum()))
if "LSTM" in R.columns:
    sp = {f"seed{s}": [round(float(R[R.h == h][f"LSTM_s{s}"].sub(R[R.h == h]["y"]).abs().mean()), 2) for h in (1, 3)]
          for s in range(N_SEEDS)}
    print("single-seed LSTM pooled MAE [24h, 72h]:", sp)

print("\n=== 2. MAE BY FOLD (24 h / 72 h) ===")
rows = []
for k in sorted(R.fold.unique()):
    s = R[R.fold == k]
    row = {"fold": k, "cities": s.city.nunique()}
    for m in MODELS:
        row[f"{m} 24h"] = round(s[s.h == 1][m + "_ae"].mean(), 1); row[f"{m} 72h"] = round(s[s.h == 3][m + "_ae"].mean(), 1)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

def dm(Th, a, b, lag=DM_LAG):
    d = (Th[a + "_ae"] - Th[b + "_ae"]).groupby(Th["date"]).mean().sort_index()
    T, mean = len(d), d.mean(); x = d.values - mean
    lrv = (x ** 2).mean()
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    stat = mean / math.sqrt(max(lrv, 1e-12) / T)
    return mean, stat, math.erfc(abs(stat) / math.sqrt(2))

PAIRS = [("Ridge", "Persistence"), ("GBM", "Ridge"), ("LSTM", "Ridge"), ("LSTM", "GBM"),
         ("LSTM", "Persistence"), ("Blend", "Ridge"), ("Blend", "LSTM")]
PAIRS = [(a, b) for a, b in PAIRS if a in MODELS and b in MODELS]
print(f"\n=== 3. DIEBOLD-MARIANO TESTS on MAE (A minus B; negative = A better; Newey-West lag {DM_LAG}) ===")
rows = []
for a_, b_ in PAIRS:
    row = {"A minus B": f"{a_} - {b_}"}
    for h in range(1, H + 1):
        d_, s_, p_ = dm(R[R.h == h], a_, b_); row[f"{24 * h}h"] = f"{d_:+.2f} (DM {s_:+.1f}, p={p_:.3f})"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

rng = np.random.default_rng(123)
def boot_ci(xs, ws):
    T = len(xs); Lb = min(BLOCK_DAYS, T); nb = int(np.ceil(T / Lb))
    cx, cw = np.concatenate([[0], np.cumsum(xs)]), np.concatenate([[0], np.cumsum(ws)])
    st = rng.integers(0, T - Lb + 1, size=(BOOT_B, nb))
    sx, sw = cx[st + Lb] - cx[st], cw[st + Lb] - cw[st]
    est = np.where(sw.sum(1) > 0, sx.sum(1) / np.maximum(sw.sum(1), 1e-12), np.nan)
    return np.nanpercentile(est, [2.5, 97.5])
cal = pd.date_range(R.date.min(), R.date.max())
print(f"\n=== 4. HIGH-POLLUTION DAYS (true PM2.5 > {HIGH}): MAE difference A minus B, 95% block-bootstrap CI ===")
rows = []
for a_, b_ in [p for p in PAIRS if p[1] == "Ridge" or p == ("LSTM", "GBM")]:
    row = {"A minus B": f"{a_} - {b_}"}
    for h in range(1, H + 1):
        s = R[(R.h == h) & (R.y > HIGH)]
        d = pd.DataFrame({"date": s["date"], "x": (s[a_ + "_ae"] - s[b_ + "_ae"]).values})
        g = d.groupby("date").agg(x=("x", "sum"), w=("x", "size")).reindex(cal, fill_value=0)
        est = g.x.sum() / g.w.sum(); lo, hi = boot_ci(g.x.values.astype(float), g.w.values.astype(float))
        row[f"{24 * h}h"] = f"{est:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print(f"\n=== 5. LAST BLOCK ONLY (2025-26) ===")
print(f"cities: {L.city.nunique()}; windows per horizon: {L.groupby('h').size().to_dict()}")
print(summary(L))
pass
pass
print(f"\nsaved benchmark2_predictions{TAG}.csv")
