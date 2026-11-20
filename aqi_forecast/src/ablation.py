"""Ablation: which inputs help? 4 variants x several seeds.  Run: python src/ablation.py"""
import copy, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from common import *
from model import Net
from train_lstm import windows

SEEDS = [0, 1, 2]
dev = "cuda" if torch.cuda.is_available() else "cpu"

raw = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))
t0, t1 = raw.date.min(), raw.date.max()
cut_tr = t0 + (t1 - t0) * 0.70
cut_va = t0 + (t1 - t0) * 0.85
cities = sorted(raw.city.unique())
meta = fit_stats(raw, cut_tr)
meta["cities"] = cities
d = standardize(raw, meta)

base = ["lpm", "doy_sin", "doy_cos"] + ["c_" + c for c in cities]
VARIANTS = {
    "history only": base,
    "+ weather": base + WX,
    "+ regional (other cities)": base + ["regional"],
    "+ weather + regional": base + WX + ["regional"],
}

def metrics(pred, true):
    mae = np.abs(pred - true).mean(0)
    sev = true > 90
    sev_mae = np.abs(pred - true)[sev].mean() if sev.any() else float("nan")
    return list(mae) + [sev_mae]

def train_once(feats, data, seed):
    Xtr, Ytr, Xva, Yva, Xte = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(len(feats)).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    loss_fn = nn.HuberLoss()
    dl = DataLoader(TensorDataset(Xtr, Ytr), 128, shuffle=True)
    best, bad, state = 1e9, 0, None
    for ep in range(40):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva.to(dev)), Yva.to(dev)).item()
        if v < best:
            best, bad, state = v, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(Xte.to(dev)).cpu().numpy()

rows = []
true = pers = None
for name, feats in VARIANTS.items():
    print("Preparing windows:", name, flush=True)
    Xtr, Ytr = windows(d, feats, t0, cut_tr)
    Xva, Yva = windows(d, feats, cut_tr, cut_va)
    Xte, Yte = windows(d, feats, cut_va, t1 + pd.Timedelta(days=1))
    if true is None:
        true = inv(Yte.numpy(), meta)
        pers = np.repeat(inv(Xte[:, -1, 0].numpy(), meta)[:, None], HOR, 1)
        rows.append(("persistence", 0, *metrics(pers, true)))
    for s in SEEDS:
        P = train_once(feats, (Xtr, Ytr, Xva, Yva, Xte), s)
        m = metrics(inv(P, meta), true)
        rows.append((name, s, *m))
        print(f"  {name} seed {s}: MAE 24/48/72h = {m[0]:.2f} {m[1]:.2f} {m[2]:.2f}  severe {m[3]:.2f}", flush=True)

df = pd.DataFrame(rows, columns=["variant", "seed", "MAE_24h", "MAE_48h", "MAE_72h", "MAE_severe"])
df.to_csv("ablation_raw.csv", index=False)
g = df.groupby("variant", sort=False)[["MAE_24h", "MAE_48h", "MAE_72h", "MAE_severe"]]
summary = g.mean().round(2).astype(str) + " +/- " + g.std().fillna(0).round(2).astype(str)
summary.to_csv("ablation_summary.csv")
print("\n=== ABLATION (test MAE in ug/m3, mean +/- std over seeds) ===")
print(summary.to_string())
