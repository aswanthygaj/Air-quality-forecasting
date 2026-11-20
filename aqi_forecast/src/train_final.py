"""Train the deployed models (LSTM ensemble + ridge) with weather + regional-group features.
Uses the last 10% of the timeline for early stopping / ridge alpha.   Run: python src/train_final.py"""
import copy, json, time, numpy as np, pandas as pd
from common import SEQ, HOR, WX, inv
from final_lib import prepare, standardize2, ridge_params, ridge_matrix, LAGS

N_SEEDS = 5
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from model import Net
    HAVE_TORCH = True
except Exception as e:
    print("PyTorch unavailable - saving a ridge-only bundle:", repr(e)[:120])
    HAVE_TORCH = False

raw = prepare(pd.read_csv("data/processed.csv", parse_dates=["date"]))
cities = sorted(raw.city.unique())
t0, t1 = raw.date.min(), raw.date.max()
cut = (t0 + (t1 - t0) * 0.90).to_datetime64()
train_rows = raw[raw.date < pd.Timestamp(cut)]
numf = ["lpm", "gregional"] + WX
meta = {"numf": numf, "cities": cities,
        "mu": {c: float(train_rows[c].mean()) for c in numf},
        "sd": {c: float(train_rows[c].std() + 1e-6) for c in numf}}
feats = numf + ["doy_sin", "doy_cos"] + ["c_" + c for c in cities]
meta["feats"] = feats
std = standardize2(raw, meta)

Xl, Zl, Yl, Pl, Dl = [], [], [], [], []
for c in cities:
    g = std[std.city == c].reset_index(drop=True)
    r = raw[raw.city == c].reset_index(drop=True)
    F = g[feats].values.astype("float32"); z = g["lpm"].values
    pm, dt = r["pm25"].values, r["date"].values
    for t in range(SEQ - 1, len(r) - HOR):
        Xl.append(F[t - SEQ + 1:t + 1]); Zl.append(z[t + 1:t + 1 + HOR])
        Yl.append(pm[t + 1:t + 1 + HOR]); Pl.append(pm[t]); Dl.append(dt[t])
X = np.stack(Xl); Z = np.array(Zl); Y = np.array(Yl); P = np.array(Pl)
D = np.array(Dl, dtype="datetime64[ns]")
tr, va = D < cut, D >= cut
print(f"trained on data through {t1.date()}; windows: train {tr.sum()}, validation {va.sum()} "
      f"(validation = origins from {str(pd.Timestamp(cut).date())})")

# ---- ridge: pick alpha on validation, then refit on ALL windows
nd = len(numf)
A = ridge_matrix(X, nd)
best = None
for a in ALPHAS:
    mu_, ym_, w_ = ridge_params(A[tr], Z[tr], a)
    v = np.abs(inv((A[va] - mu_) @ w_ + ym_, meta) - Y[va]).mean()
    if best is None or v < best[0]:
        best = (v, a, mu_, ym_, w_)
_, alpha, mu_, ym_, w_ = best
val_ridge = inv((A[va] - mu_) @ w_ + ym_, meta)              # train-only fit, for an honest validation score
mu_, ym_, w_ = ridge_params(A, Z, alpha)                      # final: refit on everything
meta["ridge"] = {"alpha": alpha, "mu": mu_.tolist(), "ym": ym_.tolist(), "w": w_.tolist()}
print("ridge alpha:", alpha)

# ---- LSTM ensemble
lstm_val = []
n_done = 0
if HAVE_TORCH:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, Yt = torch.tensor(X), torch.tensor(Z, dtype=torch.float32)
    tri, vai = torch.from_numpy(np.where(tr)[0]), torch.from_numpy(np.where(va)[0])
    for s in range(N_SEEDS):
        t_start = time.time()
        torch.manual_seed(s); np.random.seed(s)
        net = Net(len(feats)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        loss_fn = nn.HuberLoss()
        dl = DataLoader(TensorDataset(Xt[tri], Yt[tri]), 128, shuffle=True)
        Xva, Yva = Xt[vai].to(dev), Yt[vai].to(dev)
        best_v, bad, state = 1e9, 0, None
        for ep in range(40):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xva), Yva).item()
            if v < best_v:
                best_v, bad, state = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(state); net.eval()
        torch.save(net.state_dict(), f"models/final_lstm_{s}.pt")
        with torch.no_grad():
            lstm_val.append(inv(net(Xva).cpu().numpy(), meta))
        n_done += 1
        print(f"  LSTM seed {s} saved ({(time.time() - t_start) / 60:.1f} min, best val loss {best_v:.4f})", flush=True)

meta["lstm_seeds"] = n_done
meta["blend_w"] = 0.5
meta["trained_through"] = str(t1.date())
meta["val_start"] = str(pd.Timestamp(cut).date())
meta["min_train_days"] = 365
meta["n_train_days"] = {c: int((train_rows.city == c).sum()) for c in cities}
json.dump(meta, open("models/final_meta.json", "w"))

# ---- sanity report (validation = last 10% of timeline, all cities, ug/m3 MAE at 24/48/72 h)
Yv = Y[va]
def mae3(p): return np.abs(p - Yv).mean(0).round(2)
print("\nVALIDATION MAE (24h / 48h / 72h):")
print("  Persistence :", mae3(np.repeat(P[va][:, None], HOR, 1)))
print("  Ridge       :", mae3(val_ridge))
if lstm_val:
    ens = np.mean(lstm_val, axis=0)
    print("  LSTM ensemble:", mae3(ens))
    print("  Blend 50/50  :", mae3(0.5 * ens + 0.5 * val_ridge))
print("\nsaved models/final_meta.json" + (f" and {n_done} LSTM files" if n_done else " (ridge only)"))
