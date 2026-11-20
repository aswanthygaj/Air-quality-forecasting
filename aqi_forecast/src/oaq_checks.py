"""Checks on the OpenAQ benchmark: (1) high-pollution gain by test block, (2) level shift across the data gap.
Run (venv active):  python src/oaq_checks.py 2>&1 | Tee-Object -FilePath oaq_checks_results.txt"""
import numpy as np, pandas as pd
HIGH, B, BLOCK = 90, 2000, 14
R = pd.read_csv("benchmark2_predictions_oaq.csv", parse_dates=["date"])
for m in ("Ridge", "LSTM", "Blend"):
    R[m + "_ae"] = (R[m] - R["y"]).abs()
rng = np.random.default_rng(3)
NAMES = {1: "Jul 2020-Feb 2021", 2: "Mar 2021-Oct 2022", 3: "Feb 2025-Nov 2026"}
TRAIN = {1: 8989, 2: 13822, 3: 20937}

def boot(S, a, cal, min_n=30):
    if len(S) < min_n:
        return "n/a (few)"
    d = pd.DataFrame({"date": S["date"].values, "x": S[a + "_ae"].values - S["Ridge_ae"].values})
    g = d.groupby("date").agg(x=("x", "sum"), w=("x", "size")).reindex(cal, fill_value=0)
    xs, ws = g.x.values.astype(float), g.w.values.astype(float)
    T = len(xs); Lb = min(BLOCK, T); nb = int(np.ceil(T / Lb))
    cx, cw = np.concatenate([[0], np.cumsum(xs)]), np.concatenate([[0], np.cumsum(ws)])
    st = rng.integers(0, T - Lb + 1, size=(B, nb))
    sx, sw = cx[st + Lb] - cx[st], cw[st + Lb] - cw[st]
    est = np.where(sw.sum(1) > 0, sx.sum(1) / np.maximum(sw.sum(1), 1e-12), np.nan)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return f"{xs.sum() / ws.sum():+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"

print("=== 1. HIGH-POLLUTION DAYS (true PM2.5 > 90) BY TEST BLOCK: MAE difference vs ridge (negative = better; * = CI excludes 0) ===")
for a in ("LSTM", "Blend"):
    rows = []
    for k in sorted(R.fold.unique()):
        F = R[R.fold == k]
        hi = F[F.y > HIGH]
        cal = pd.date_range(F.date.min(), F.date.max())
        rows.append({"block": f"{k}: {NAMES.get(k, '')}", "train origins": TRAIN.get(k), "cities": F.city.nunique(),
                     "n high (24h)": int((hi.h == 1).sum()), **{f"{24 * h}h": boot(hi[hi.h == h], a, cal) for h in (1, 2, 3)}})
    print(f"\n{a} minus Ridge:")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 2. LEVEL SHIFT ACROSS THE GAP: mean PM2.5 in the same months before and after (panel file) ===")
P = pd.read_csv("data/city_day_oaq.csv", parse_dates=["Date"])
def mean_in(g, a, b, months):
    s = g[(g.Date >= a) & (g.Date <= b) & g.Date.dt.month.isin(months)]
    return (s["PM2.5"].mean(), len(s)) if len(s) >= 30 else (np.nan, len(s))
rows = []
for c, g in P.groupby("City"):
    a1, n1 = mean_in(g, "2022-01-01", "2022-10-31", range(2, 11)); a2, n2 = mean_in(g, "2025-02-01", "2025-10-31", range(2, 11))
    w1, m1 = mean_in(g, "2021-11-01", "2022-01-31", (11, 12, 1)); w2, m2 = mean_in(g, "2025-11-01", "2026-01-31", (11, 12, 1))
    rows.append({"city": c, "Feb-Oct 2022": round(a1, 1), "Feb-Oct 2025": round(a2, 1), "ratio": round(a2 / a1, 2),
                 "Nov-Jan 21/22": round(w1, 1), "Nov-Jan 25/26": round(w2, 1), "winter ratio": round(w2 / w1, 2), "days (a,b,c,d)": f"{n1},{n2},{m1},{m2}"})
T = pd.DataFrame(rows).sort_values("Feb-Oct 2022", ascending=False)
print(T.to_string(index=False))
print(f"\nmedian ratio Feb-Oct: {T['ratio'].median():.2f} | median winter ratio: {T['winter ratio'].median():.2f} | "
      f"cities with Feb-Oct ratio below 0.85: {int((T['ratio'] < 0.85).sum())}, above 1.15: {int((T['ratio'] > 1.15).sum())} of {int(T['ratio'].notna().sum())}")
print("Reading: ratios near 1 mean no systematic level change across the gap; a large median shift points to instrument or network changes (or real change).")
