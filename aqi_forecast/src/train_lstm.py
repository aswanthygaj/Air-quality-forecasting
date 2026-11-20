"""Multi-city LSTM: 14 days of history (PM2.5 + weather + other-cities signal) -> next 3 days."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from common import *
from model import Net
torch.manual_seed(0); np.random.seed(0)

def windows(d, feats, a, b):
    X, Y = [], []
    for city, g in d.groupby("city"):
        g = g.reset_index(drop=True)
        F = g[feats].values.astype("float32"); y = g["lpm"].values.astype("float32")
        dt = g["date"].values
        for t in range(SEQ - 1, len(g) - HOR):
            if np.datetime64(a) <= dt[t] < np.datetime64(b):
                X.append(F[t - SEQ + 1:t + 1]); Y.append(y[t + 1:t + 1 + HOR])
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

if __name__ == "__main__":
    raw = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))
    t0, t1 = raw.date.min(), raw.date.max()
    cut_tr = t0 + (t1 - t0) * 0.70; cut_va = t0 + (t1 - t0) * 0.85
    cities = sorted(raw.city.unique())
    meta = fit_stats(raw, cut_tr); meta["cities"] = cities
    d = standardize(raw, meta); feats = feat_cols(cities)
    Xtr, Ytr = windows(d, feats, t0, cut_tr)
    Xva, Yva = windows(d, feats, cut_tr, cut_va)
    Xte, Yte = windows(d, feats, cut_va, t1 + pd.Timedelta(days=1))
    print("windows (train/val/test):", len(Xtr), len(Xva), len(Xte))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(len(feats)).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    loss_fn = nn.HuberLoss(); dl = DataLoader(TensorDataset(Xtr, Ytr), 128, shuffle=True)
    best, bad = 1e9, 0
    for ep in range(60):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = loss_fn(model(Xva.to(dev)), Yva.to(dev)).item()
        print(f"epoch {ep:02d} val {v:.4f}")
        if v < best: best, bad = v, 0; torch.save(model.state_dict(), "models/lstm.pt")
        else:
            bad += 1
            if bad >= 8: break
    model.load_state_dict(torch.load("models/lstm.pt")); model.eval()
    with torch.no_grad(): P = model(Xte.to(dev)).cpu().numpy()
    yt, yp = inv(Yte.numpy(), meta), inv(P, meta)
    pers = np.repeat(inv(Xte[:, -1, 0].numpy(), meta)[:, None], HOR, 1)
    print("\nTEST (ug/m3)          24h     48h     72h")
    for name, p in [("LSTM", yp), ("Persistence", pers)]:
        mae = np.abs(p - yt).mean(0); rmse = np.sqrt(((p - yt) ** 2).mean(0))
        print(f"{name:12s} MAE  {mae[0]:7.2f} {mae[1]:7.2f} {mae[2]:7.2f}")
        print(f"{name:12s} RMSE {rmse[0]:7.2f} {rmse[1]:7.2f} {rmse[2]:7.2f}")
    meta["feats"] = feats
    json.dump(meta, open("models/meta.json", "w"), indent=1)
