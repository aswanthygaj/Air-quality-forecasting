"""Final evaluation: corrected baselines + regional-group feature + ablation with ridge included.
Run (project root, venv active):  python src/evaluate_all.py | Tee-Object -FilePath eval_results.txt
Independent of train_lstm.py: trains the LSTM variants itself (SEEDS below)."""
import copy, time, numpy as np, pandas as pd
from common import SEQ, HOR, WX, add_raw_features, inv

SEEDS = [0, 1, 2]            # fewer seeds = faster, e.g. [0, 1]
LAGS = 7                     # days of history used by ridge
LOCKDOWN = np.datetime64("2020-03-25")
MIN_TRAIN_DAYS = 365         # cities with less training history are reported separately
HIGH = 90                    # ug/m3: above this = CPCB "Poor" or worse
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]

# Regional groups (my own geographic judgement - edit freely). Unlisted cities share one "other" group.
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
    print("PyTorch unavailable, LSTM variants skipped:", repr(e)[:120])
    HAVE_TORCH = False

# ------------------------------------------------------------------ data + features
raw = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))

def add_group_regional(df):
    """Mean log-PM2.5 (same day) of the OTHER cities in the same regional group."""
    piv = df.pivot(index="date", columns="city", values="lpm")
    out = pd.DataFrame(np.nan, index=piv.index, columns=piv.columns)
    for c in piv.columns:
        mates = [m for m in piv.columns if m != c and city2grp.get(m, "other") == city2grp.get(c, "other")]
        if mates:
            out[c] = piv[mates].mean(axis=1)
    reg = out.reset_index().melt(id_vars="date", var_name="city", value_name="gregional")
    df = df.merge(reg, on=["date", "city"], how="left")
    df["gregional"] = df["gregional"].fillna(df["lpm"])      # no group mates -> own value
    return df.sort_values(["city", "date"]).reset_index(drop=True)
raw = add_group_regional(raw)

cities = sorted(raw.city.unique())
t0, t1 = raw.date.min(), raw.date.max()
cut_tr = (t0 + (t1 - t0) * 0.70).to_datetime64()
cut_va = (t0 + (t1 - t0) * 0.85).to_datetime64()
train_rows = raw[raw.date < pd.Timestamp(cut_tr)]
n_train_days = train_rows.groupby("city").size().reindex(cities).fillna(0).astype(int)
eligible = [c for c in cities if n_train_days[c] >= MIN_TRAIN_DAYS]
late = [c for c in cities if c not in eligible]
print(f"cities: {len(cities)} total, {len(eligible)} with >= {MIN_TRAIN_DAYS} training days, late-start: {late}")

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

# ---- seasonal baseline (fixed): own day-of-year profile if >= 1 year of training data,
#      otherwise the all-cities profile rescaled to the city's own training mean
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

# ---- one record per forecast origin
Xl, Zl, Yl, Pl, Dl, Cl, Sl = [], [], [], [], [], [], []
for c in cities:
    g = std[std.city == c].reset_index(drop=True)
    r = raw[raw.city == c].reset_index(drop=True)
    F = g[feats_all].values.astype("float32"); z = g["lpm"].values
    pm, dt, doy = r["pm25"].values, r["date"].values, r["date"].dt.dayofyear.values
    for t in range(SEQ - 1, len(r) - HOR):
        Xl.append(F[t - SEQ + 1:t + 1]); Zl.append(z[t + 1:t + 1 + HOR])
        Yl.append(pm[t + 1:t + 1 + HOR]); Pl.append(pm[t]); Dl.append(dt[t]); Cl.append(c)
        Sl.append(clim[c][doy[t + 1:t + 1 + HOR] - 1])
X = np.stack(Xl); Z = np.array(Zl); Y = np.array(Yl); P = np.array(Pl)
D = np.array(Dl, dtype="datetime64[ns]"); C = np.array(Cl); S = np.array(Sl)
tr, va, te = D < cut_tr, (D >= cut_tr) & (D < cut_va), D >= cut_va
elig_o = np.isin(C, eligible)
print(f"origins: train {tr.sum()}, val {va.sum()}, test {te.sum()} "
      f"(test: {(te & elig_o).sum()} in eligible cities, {(te & ~elig_o).sum()} in late-start cities)")

VARIANTS = {"history": (False, False), "+wx": (True, False),
            "+reg": (False, True), "+wx+reg": (True, True)}
def names(wx, reg):
    dyn = ["lpm"] + (["gregional"] if reg else []) + (WX if wx else [])
    stat = ["doy_sin", "doy_cos"] + OH
    return [col[n] for n in dyn], [col[n] for n in stat]

preds = {"Persistence": [np.repeat(P[:, None], HOR, 1)], "Seasonal avg (scaled)": [S]}

# ------------------------------------------------------------------ ridge variants
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
    print(f"ridge {vn}: alpha {best[1]} (val MAE {best[0]:.2f})", flush=True)
    preds["Ridge: " + vn] = [inv(best[2](A), meta)]

# ------------------------------------------------------------------ LSTM variants
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
    return outs
if HAVE_TORCH:
    for vn, (wx, reg) in VARIANTS.items():
        t_start = time.time()
        di, si = names(wx, reg)
        preds["LSTM: " + vn] = lstm_runs(di + si)
        print(f"LSTM {vn}: {len(SEEDS)} seeds done in {(time.time() - t_start) / 60:.1f} min", flush=True)

# ------------------------------------------------------------------ evaluation
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

m_head = te & elig_o
head = pd.DataFrame([{"model": n, **{c: fmt(per_seed(p, m_head)[:, i]) for i, c in enumerate(COLS)}}
                     for n, p in preds.items()])
print(f"\n=== 1. HEADLINE: test MAE/RMSE (ug/m3), {len(eligible)} cities with >= 1 year of training data ===")
print(f"    (LSTM rows: mean +/- std over {len(SEEDS)} seeds; MAE_high = days where true PM2.5 > {HIGH})")
print(head.to_string(index=False))

per_rows = []
for n, p in preds.items():
    row = {"model": n}
    for lab, m in [("pre", m_head & (D < LOCKDOWN)), ("lockdown", m_head & (D >= LOCKDOWN))]:
        a = per_seed(p, m)
        row[f"{lab} MAE_24h"], row[f"{lab} MAE_72h"] = fmt(a[:, 0]), fmt(a[:, 2])
    per_rows.append(row)
period = pd.DataFrame(per_rows)
print(f"\n=== 2. BY PERIOD (eligible cities; pre-lockdown n={(m_head & (D < LOCKDOWN)).sum()}, "
      f"lockdown n={(m_head & (D >= LOCKDOWN)).sum()}) ===")
print(period.to_string(index=False))

m_late = te & ~elig_o
if m_late.any():
    lrows = [{"model": n, "MAE_24h": fmt(per_seed(p, m_late)[:, 0]), "MAE_72h": fmt(per_seed(p, m_late)[:, 2])}
             for n, p in preds.items()]
    lt = pd.DataFrame(lrows)
    print(f"\n=== 3. LATE-START CITIES (little/no training data; not part of the headline) ===")
    print("    " + ", ".join(f"{c} ({n_train_days[c]} train days)" for c in late))
    print(lt.to_string(index=False))
    lt.to_csv("eval_late_start.csv", index=False)

def val_score(n):
    return per_seed(preds[n], va & elig_o)[:, :3].mean()
best_of = {fam: min([n for n in preds if n.startswith(fam)], key=val_score)
           for fam in ["Ridge", "LSTM"] if any(n.startswith(fam) for n in preds)}
pc_models = ["Persistence"] + list(best_of.values())
rows = []
for c in eligible:
    m = te & (C == c)
    row = {"city": c, "n": int(m.sum()), "mean_pm25": round(float(Y[m].mean()), 1)}
    for n in pc_models:
        a = per_seed(preds[n], m)
        row[f"{n.split(':')[0]} 24h"], row[f"{n.split(':')[0]} 72h"] = round(float(a[:, 0].mean()), 1), round(float(a[:, 2].mean()), 1)
    rows.append(row)
percity = pd.DataFrame(rows).sort_values("mean_pm25", ascending=False)
print(f"\n=== 4. PER CITY (test MAE; best variant per family chosen on VALIDATION: {best_of}) ===")
print(percity.to_string(index=False))

head.to_csv("eval_headline.csv", index=False); period.to_csv("eval_by_period.csv", index=False)
percity.to_csv("eval_per_city.csv", index=False)
print("\nsaved eval_headline.csv, eval_by_period.csv, eval_per_city.csv (and eval_late_start.csv)")
