"""Statistics round: 10 seeds, block-bootstrap CIs for model differences, LSTM+ridge blend,
and a seasonal-baseline diagnostic.
Run (project root, venv active):  python src/stats_final.py | Tee-Object -FilePath stats_results.txt"""
import copy, time, numpy as np, pandas as pd
from common import SEQ, HOR, WX, add_raw_features, inv

SEEDS = list(range(10))
LAGS = 7
LOCKDOWN = np.datetime64("2020-03-25")
MIN_TRAIN_DAYS = 365
HIGH = 90
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
BOOT_B, BLOCK_DAYS = 2000, 14

GROUPS = {
    "north_plain": ["Delhi", "Gurugram", "Lucknow", "Patna", "Amritsar", "Chandigarh", "Jaipur"],
    "east": ["Kolkata", "Brajrajnagar", "Talcher", "Jorapokhar"],
    "northeast": ["Guwahati", "Shillong", "Aizawl"],
    "west_central": ["Ahmedabad", "Bhopal"],
    "south": ["Hyderabad", "Bengaluru", "Chennai", "Coimbatore", "Kochi", "Ernakulam",
              "Thiruvananthapuram", "Visakhapatnam", "Amaravati"],
}
city2grp = {c: g for g, cs in GROUPS.items() for c in cs}

try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from model import Net
    HAVE_TORCH = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    print("PyTorch unavailable, LSTM rows skipped:", repr(e)[:120])
    HAVE_TORCH = False

# ------------------------------------------------------------------ data + features (as in evaluate_all.py)
raw = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))

def add_group_regional(df):
    piv = df.pivot(index="date", columns="city", values="lpm")
    out = pd.DataFrame(np.nan, index=piv.index, columns=piv.columns)
    for c in piv.columns:
        mates = [m for m in piv.columns if m != c and city2grp.get(m, "other") == city2grp.get(c, "other")]
        if mates:
            out[c] = piv[mates].mean(axis=1)
    reg = out.reset_index().melt(id_vars="date", var_name="city", value_name="gregional")
    df = df.merge(reg, on=["date", "city"], how="left")
    df["gregional"] = df["gregional"].fillna(df["lpm"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)
raw = add_group_regional(raw)

cities = sorted(raw.city.unique())
t0, t1 = raw.date.min(), raw.date.max()
cut_tr = (t0 + (t1 - t0) * 0.70).to_datetime64()
cut_va = (t0 + (t1 - t0) * 0.85).to_datetime64()
train_rows = raw[raw.date < pd.Timestamp(cut_tr)]
n_train_days = train_rows.groupby("city").size().reindex(cities).fillna(0).astype(int)
eligible = [c for c in cities if n_train_days[c] >= MIN_TRAIN_DAYS]
print(f"{len(eligible)} cities with >= {MIN_TRAIN_DAYS} training days (used for all results): {eligible}")

NUMF = ["lpm", "gregional"] + WX
meta = {"mu": {c: float(train_rows[c].mean()) for c in NUMF},
        "sd": {c: float(train_rows[c].std() + 1e-6) for c in NUMF}}
std = raw.copy()
for c in NUMF:
    std[c] = (std[c] - meta["mu"][c]) / meta["sd"][c]
OH = ["c_" + c for c in cities]
for c in cities:
    std["c_" + c] = (std["city"] == c).astype("float32")
feats_all = NUMF + ["doy_sin", "doy_cos"] + OH
col = {n: i for i, n in enumerate(feats_all)}

def profile(g):
    m = g.groupby(g.date.dt.dayofyear)["pm25"].mean().reindex(range(1, 367))
    ext = pd.concat([m.iloc[-15:], m, m.iloc[:15]]).reset_index(drop=True)
    sm = ext.rolling(31, center=True, min_periods=5).mean().iloc[15:-15]
    return sm.fillna(g.pm25.mean()).values
glob, glob_mean = profile(train_rows), train_rows.pm25.mean()
clim = {}
for c in cities:
    gc = train_rows[train_rows.city == c]
    clim[c] = profile(gc) if len(gc) >= MIN_TRAIN_DAYS else (
        glob * gc.pm25.mean() / glob_mean if len(gc) > 0 else glob)

Xl, Zl, Yl, Pl, Dl, Cl, Sl, Srl = [], [], [], [], [], [], [], []
for c in cities:
    g = std[std.city == c].reset_index(drop=True)
    r = raw[raw.city == c].reset_index(drop=True)
    F = g[feats_all].values.astype("float32"); z = g["lpm"].values
    pm, dt, doy = r["pm25"].values, r["date"].values, r["date"].dt.dayofyear.values
    for t in range(SEQ - 1, len(r) - HOR):
        Xl.append(F[t - SEQ + 1:t + 1]); Zl.append(z[t + 1:t + 1 + HOR])
        Yl.append(pm[t + 1:t + 1 + HOR]); Pl.append(pm[t]); Dl.append(dt[t]); Cl.append(c)
        base = clim[c][doy[t + 1:t + 1 + HOR] - 1]
        Sl.append(base)
        w = min(30, t + 1)                                   # recent level vs climatology
        rec = pm[t - w + 1:t + 1].sum() / clim[c][doy[t - w + 1:t + 1] - 1].sum()
        Srl.append(base * float(np.clip(rec, 0.25, 4.0)))
X = np.stack(Xl); Z = np.array(Zl); Y = np.array(Yl); P = np.array(Pl)
D = np.array(Dl, dtype="datetime64[ns]"); C = np.array(Cl); S = np.array(Sl); Sr = np.array(Srl)
tr, va, te = D < cut_tr, (D >= cut_tr) & (D < cut_va), D >= cut_va
elig_o = np.isin(C, eligible)
m_head = te & elig_o
print(f"origins: train {tr.sum()}, val {va.sum()}, test {te.sum()} (eligible-city test windows: {m_head.sum()})")

VARIANTS = {"history": (False, False), "+wx+reg": (True, True)}
def names(wx, reg):
    dyn = ["lpm"] + (["gregional"] if reg else []) + (WX if wx else [])
    return [col[n] for n in dyn], [col[n] for n in ["doy_sin", "doy_cos"] + OH]

preds = {"Persistence": [np.repeat(P[:, None], HOR, 1)], "Seasonal avg": [S],
         "Seasonal x recent level": [Sr]}

def ridge_fit(A, y, alpha):
    mu, ym = A.mean(0), y.mean(0)
    Ac = A - mu
    w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (y - ym))
    return lambda B: (B - mu) @ w + ym
for vn, (wx, reg) in VARIANTS.items():
    di, si = names(wx, reg)
    A = np.concatenate([X[:, -LAGS:, di].reshape(len(X), -1), X[:, -1, si]], 1).astype("float64")
    best = None
    for a in ALPHAS:
        f = ridge_fit(A[tr], Z[tr], a)
        v = np.abs(inv(f(A[va]), meta) - Y[va]).mean()
        if best is None or v < best[0]:
            best = (v, a, f)
    print(f"ridge {vn}: alpha {best[1]}", flush=True)
    preds["Ridge: " + vn] = [inv(best[2](A), meta)]

def lstm_runs(idx):
    Xt = torch.tensor(np.ascontiguousarray(X[:, :, idx]), dtype=torch.float32)
    Yt = torch.tensor(Z, dtype=torch.float32)
    tri, vai = torch.from_numpy(np.where(tr)[0]), torch.from_numpy(np.where(va)[0])
    outs = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        net = Net(len(idx)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        loss_fn = nn.HuberLoss()
        dl = DataLoader(TensorDataset(Xt[tri], Yt[tri]), 128, shuffle=True)
        Xva, Yva = Xt[vai].to(dev), Yt[vai].to(dev)
        best, bad, state = 1e9, 0, None
        for ep in range(40):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xva), Yva).item()
            if v < best:
                best, bad, state = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            z = np.concatenate([net(Xt[i:i + 4096].to(dev)).cpu().numpy() for i in range(0, len(Xt), 4096)])
        outs.append(inv(z, meta))
        print(f"    seed {s} done", flush=True)
    return outs
if HAVE_TORCH:
    for vn, (wx, reg) in VARIANTS.items():
        t_start = time.time()
        di, si = names(wx, reg)
        preds["LSTM: " + vn] = lstm_runs(di + si)
        print(f"LSTM {vn}: {len(SEEDS)} seeds in {(time.time() - t_start) / 60:.1f} min", flush=True)
    ens = np.mean(preds["LSTM: +wx+reg"], axis=0)                  # 10-seed ensemble
    preds["LSTM ens: +wx+reg"] = [ens]
    preds["Blend: LSTM ens + Ridge"] = [0.5 * (ens + preds["Ridge: +wx+reg"][0][:len(ens)])]   # fixed 50/50, no tuning

# ------------------------------------------------------------------ metrics
COLS = ["MAE_24h", "MAE_48h", "MAE_72h", "RMSE_24h", "RMSE_48h", "RMSE_72h", "MAE_high"]
def per_seed(plist, m):
    rows = []
    for p in plist:
        e = np.abs(p[m] - Y[m]); r = np.sqrt(((p[m] - Y[m]) ** 2).mean(0))
        hi = m[:, None] & (Y > HIGH)
        rows.append([*e.mean(0), *r, np.abs(p - Y)[hi].mean() if hi.any() else np.nan])
    return np.array(rows)
def fmt(v):
    mean = np.nanmean(v)
    return f"{mean:.2f}" if len(v) == 1 else f"{mean:.2f}+/-{np.std(v, ddof=1):.2f}"

summary = pd.DataFrame([{"model": n + (f" (mean of {len(p)} seeds)" if len(p) > 1 else ""),
                         **{c: fmt(per_seed(p, m_head)[:, i]) for i, c in enumerate(COLS)}}
                        for n, p in preds.items()])
print(f"\n=== 1. SUMMARY: test, {len(eligible)} eligible cities (MAE_high = days with true PM2.5 > {HIGH}) ===")
print(summary.to_string(index=False))
summary.to_csv("stats_summary.csv", index=False)

# ------------------------------------------------------------------ block bootstrap of differences
abs_err = {n: np.mean([np.abs(p - Y) for p in plist], axis=0) for n, plist in preds.items()}
day = D.astype("datetime64[D]")
rng = np.random.default_rng(123)

def boot_ci(xs, ws):
    T = len(xs); L = min(BLOCK_DAYS, T); nb = int(np.ceil(T / L))
    cx, cw = np.concatenate([[0], np.cumsum(xs)]), np.concatenate([[0], np.cumsum(ws)])
    st = rng.integers(0, T - L + 1, size=(BOOT_B, nb))
    sx, sw = cx[st + L] - cx[st], cw[st + L] - cw[st]
    est = np.where(sw.sum(1) > 0, sx.sum(1) / np.maximum(sw.sum(1), 1e-12), np.nan)
    return np.nanpercentile(est, [2.5, 97.5])

def diff_cell(eA, eB, m, k):
    """Mean(eA-eB) over mask m for column k (0-2 horizons, 3 = high-pollution), block-bootstrap CI."""
    d = eA - eB
    if k < 3:
        x, w = d[:, k], np.ones(len(d))
    else:
        hi = (Y > HIGH)
        x, w = (d * hi).sum(1), hi.sum(1).astype(float)
    df = pd.DataFrame({"day": day[m], "x": x[m], "w": w[m]}).groupby("day").sum()
    if df["w"].sum() < (1 if k < 3 else 30) or len(df) < 3:
        return "n/a (too few)"
    est = df["x"].sum() / df["w"].sum()
    lo, hi_ = boot_ci(df["x"].values, df["w"].values)
    star = "*" if (lo > 0 or hi_ < 0) else " "
    return f"{est:+.2f} [{lo:+.2f},{hi_:+.2f}]{star}"

PAIRS = [("LSTM: +wx+reg", "Ridge: +wx+reg"), ("LSTM ens: +wx+reg", "Ridge: +wx+reg"),
         ("Blend: LSTM ens + Ridge", "Ridge: +wx+reg"), ("Blend: LSTM ens + Ridge", "LSTM ens: +wx+reg"),
         ("LSTM: +wx+reg", "Persistence"), ("Ridge: +wx+reg", "Persistence"),
         ("LSTM: +wx+reg", "LSTM: history"), ("Ridge: +wx+reg", "Ridge: history")]
PAIRS = [(a, b) for a, b in PAIRS if a in abs_err and b in abs_err]
print(f"\n=== 2. MODEL DIFFERENCES in test MAE (ug/m3), A minus B: negative = A is better ===")
print(f"    95% CI from a {BLOCK_DAYS}-day moving-block bootstrap over days ({BOOT_B} resamples); * = CI excludes 0")
print("    'LSTM: ...' = a typical single LSTM (errors averaged over seeds); 'LSTM ens' = average of all seeds")
rows_out = []
for label, m in [("ALL TEST (eligible cities)", m_head),
                 ("PRE-LOCKDOWN origins", m_head & (D < LOCKDOWN)),
                 ("LOCKDOWN origins (>= 25 Mar 2020)", m_head & (D >= LOCKDOWN))]:
    rows = []
    for a, b in PAIRS:
        row = {"A minus B": f"{a}  -  {b}"}
        for k, cname in enumerate(["MAE_24h", "MAE_48h", "MAE_72h", "MAE_high"]):
            row[cname] = diff_cell(abs_err[a], abs_err[b], m, k)
        rows.append(row)
    t = pd.DataFrame(rows)
    print(f"\n--- {label}: {m.sum()} windows, {len(np.unique(day[m]))} days ---")
    print(t.to_string(index=False))
    t.insert(0, "subset", label); rows_out.append(t)
if rows_out:
    pd.concat(rows_out).to_csv("stats_pairs.csv", index=False)

# ------------------------------------------------------------------ seasonal baseline diagnostic
print("\n=== 3. SEASONAL BASELINE DIAGNOSTIC (does the 2015-18 profile still fit 2019-20?) ===")
rows = []
for c in eligible:
    rr = raw[raw.city == c]
    mv, mt = va & (C == c), te & (C == c)
    rows.append({"city": c,
                 "mean train": round(rr[rr.date < pd.Timestamp(cut_tr)].pm25.mean(), 1),
                 "mean val": round(rr[(rr.date >= pd.Timestamp(cut_tr)) & (rr.date < pd.Timestamp(cut_va))].pm25.mean(), 1),
                 "mean test": round(rr[rr.date >= pd.Timestamp(cut_va)].pm25.mean(), 1),
                 "Seas val": round(np.abs(S[mv, 0] - Y[mv, 0]).mean(), 1),
                 "Seas test": round(np.abs(S[mt, 0] - Y[mt, 0]).mean(), 1),
                 "SeasxRecent test": round(np.abs(Sr[mt, 0] - Y[mt, 0]).mean(), 1),
                 "Persist test": round(np.abs(P[mt] - Y[mt, 0]).mean(), 1)})
diag = pd.DataFrame(rows)
print("(24 h MAE; 'mean' columns are average PM2.5 in each period)")
print(diag.to_string(index=False))
diag.to_csv("stats_seasonal_diag.csv", index=False)
print("\nsaved stats_summary.csv, stats_pairs.csv, stats_seasonal_diag.csv")
