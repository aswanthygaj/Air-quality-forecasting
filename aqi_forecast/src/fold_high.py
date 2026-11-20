"""High-pollution results fold by fold (each fold has a different amount of training history).
Run (venv active):  python src/fold_high.py 2>&1 | Tee-Object -FilePath fold_high_results.txt
For the fixed-station panel first set:  $env:TAG="_fixed"   (and remove it afterwards)"""
import os, numpy as np, pandas as pd
HIGH, B, BLOCK = 90, 2000, 14
TAG = os.environ.get("TAG", "")
R = pd.read_csv(f"benchmark2_predictions{TAG}.csv", parse_dates=["date"])
MODELS = [m for m in ["Persistence", "Ridge", "GBM", "LSTM", "Blend"] if m in R.columns]
for m in MODELS:
    R[m + "_ae"] = (R[m] - R["y"]).abs()
rng = np.random.default_rng(11)
# training origins per fold, printed by benchmark2.py (original panel / fixed panel)
TRAIN_N = {"": [7782, 10432, 13345, 16635, 20455], "_fixed": [5703, 8064, 10475, 13009, 15543]}.get(TAG, [None] * 5)
SEASON = {1: "Jan-Feb 2018", 2: "Nov-Dec 2018", 3: "Jan-Feb 2019", 4: "Nov-Dec 2019", 5: "Jan-Feb 2020"}

def boot_diff(S, a, b, cal, min_n=30):
    if len(S) < min_n:
        return "n/a (few)"
    d = pd.DataFrame({"date": S["date"].values, "x": S[a + "_ae"].values - S[b + "_ae"].values})
    g = d.groupby("date").agg(x=("x", "sum"), w=("x", "size")).reindex(cal, fill_value=0)
    xs, ws = g.x.values.astype(float), g.w.values.astype(float)
    T = len(xs); Lb = min(BLOCK, T); nb = int(np.ceil(T / Lb))
    cx, cw = np.concatenate([[0], np.cumsum(xs)]), np.concatenate([[0], np.cumsum(ws)])
    st = rng.integers(0, T - Lb + 1, size=(B, nb))
    sx, sw = cx[st + Lb] - cx[st], cw[st + Lb] - cw[st]
    est = np.where(sw.sum(1) > 0, sx.sum(1) / np.maximum(sw.sum(1), 1e-12), np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return f"{xs.sum() / ws.sum():+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"

print(f"panel: {'fixed-station' if TAG else 'original'}; high-pollution = true PM2.5 > {HIGH}; negative = better than ridge; * = 95% CI excludes 0\n")
for a in [m for m in ("LSTM", "Blend") if m in MODELS]:
    rows = []
    for k in sorted(R.fold.unique()):
        F = R[R.fold == k]
        cal = pd.date_range(F.date.min(), F.date.max())
        hi = F[F.y > HIGH]
        rows.append({"fold": k, "high winter": SEASON.get(k, ""), "train origins": TRAIN_N[k - 1],
                     "cities": F.city.nunique(), "n high (24h)": int((hi.h == 1).sum()),
                     **{f"{24 * h}h": boot_diff(hi[hi.h == h], a, "Ridge", cal) for h in (1, 2, 3)}})
    print(f"{a} minus Ridge on high-pollution days, by fold:")
    print(pd.DataFrame(rows).to_string(index=False)); print()
print("Reading: if the LSTM/Blend gain grows with 'train origins' (more training winters), the early-fold weakness is a data-volume effect.")
print("Caveat: folds also differ in cities and in the winter tested, so this is suggestive, not a controlled learning curve.")
