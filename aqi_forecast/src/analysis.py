"""Analysis of benchmark2_predictions.csv (no retraining): lockdown split of fold 5, per-city results,
composition and robustness of the high-pollution result, Holm-corrected Diebold-Mariano tests.
Run (project root, venv active):  python src/analysis.py 2>&1 | Tee-Object -FilePath analysis_results.txt"""
import math, os, numpy as np, pandas as pd

HIGH, LOCK, B, BLOCK = 90, pd.Timestamp("2020-03-25"), 2000, 14
R = pd.read_csv(f"benchmark2_predictions{os.environ.get('TAG', '')}.csv", parse_dates=["date"])
MODELS = [m for m in ["Persistence", "Ridge", "GBM", "LSTM", "Blend"] if m in R.columns]
SEEDS = [c for c in R.columns if c.startswith("LSTM_s") and not c.endswith("_ae")]
for m in MODELS + SEEDS:
    R[m + "_ae"] = (R[m] - R["y"]).abs()
HAS_LSTM = "LSTM" in MODELS
rng = np.random.default_rng(7)
FULL_CAL = pd.date_range(R.date.min(), R.date.max())

def boot_diff(S, a, b, cal=None, min_n=30):
    """Mean(a_ae - b_ae) over rows S (one horizon) with a moving-block bootstrap CI over days."""
    if len(S) < min_n:
        return "n/a (too few)"
    cal = FULL_CAL if cal is None else cal
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

# ------------------------------------------------------------------ 1. fold 5: before vs during lockdown
print("=== 1. FOLD 5 (Jan-Jun 2020) SPLIT AT THE LOCKDOWN (origin date) ===")
F5 = R[R.fold == 5]
for label, S in [("Jan 1 - Mar 24 2020 (before lockdown)", F5[F5.date < LOCK]),
                 ("Mar 25 - Jun 2020 (lockdown period)", F5[F5.date >= LOCK])]:
    n_days, cal = S.date.nunique(), pd.date_range(S.date.min(), S.date.max())
    print(f"\n--- {label}: {n_days} origin days, {len(S[S.h == 1])} windows per horizon ---")
    rows = []
    for m in MODELS:
        rows.append({"model": m, **{f"MAE_{24 * h}h": round(S[S.h == h][m + "_ae"].mean(), 2) for h in (1, 2, 3)}})
    print(pd.DataFrame(rows).to_string(index=False))
    if HAS_LSTM:
        rows = []
        for a, b in [("LSTM", "Ridge"), ("Blend", "Ridge"), ("LSTM", "Persistence"), ("Ridge", "Persistence")]:
            rows.append({"A minus B": f"{a} - {b}",
                         **{f"{24 * h}h": boot_diff(S[S.h == h], a, b, cal) for h in (1, 2, 3)}})
        print("differences with 95% block-bootstrap CI (negative = A better; * = excludes 0):")
        print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ 2. per city
print("\n=== 2. PER CITY, all rolling folds pooled (MAE ug/m3; sorted by pollution level) ===")
rows = []
for c, S in R.groupby("city"):
    row = {"city": c, "n": int((S.h == 1).sum()), "mean_pm": round(S[S.h == 1].y.mean(), 1)}
    for h in (1, 3):
        for m in ["Persistence", "Ridge", "LSTM", "Blend"]:
            if m in MODELS:
                row[f"{m[:4]}{24 * h}"] = round(S[S.h == h][m + "_ae"].mean(), 1)
    rows.append(row)
pc = pd.DataFrame(rows).sort_values("mean_pm", ascending=False)
print(pc.to_string(index=False))
if HAS_LSTM:
    for h in (24, 72):
        print(f"{h} h: LSTM beats ridge in {int((pc[f'LSTM{h}'] < pc[f'Ridg{h}']).sum())}/{len(pc)} cities; "
              f"Blend beats ridge in {int((pc[f'Blen{h}'] < pc[f'Ridg{h}']).sum())}/{len(pc)}; "
              f"ridge beats persistence in {int((pc[f'Ridg{h}'] < pc[f'Pers{h}']).sum())}/{len(pc)}")

# ------------------------------------------------------------------ 3. where do high-pollution days come from
H1 = R[(R.h == 1) & (R.y > HIGH)].copy()
mth = H1.date.dt.month
H1["season"] = np.where(mth >= 11, H1.date.dt.year + 1, np.where(mth <= 2, H1.date.dt.year, 0))
print(f"\n=== 3. COMPOSITION OF THE {len(H1)} HIGH-POLLUTION TARGETS (24 h horizon, true PM2.5 > {HIGH}) ===")
by_city = H1.groupby("city").size().sort_values(ascending=False)
print("by city (count, share):", ", ".join(f"{c} {n} ({100 * n / len(H1):.0f}%)" for c, n in by_city.head(8).items()))
print("by winter (Nov-Feb; 0 = other months):", H1.groupby("season").size().to_dict())
print("distinct 14-day blocks containing high days:", H1.date.dt.floor("D").map(lambda d: (d - R.date.min()).days // 14).nunique())

# ------------------------------------------------------------------ 4. robustness of the high-pollution result
print("\n=== 4. ROBUSTNESS OF THE HIGH-POLLUTION RESULT (MAE difference vs ridge; * = 95% CI excludes 0) ===")
HI = R[R.y > HIGH].copy()
mth = HI.date.dt.month
HI["season"] = np.where(mth >= 11, HI.date.dt.year + 1, np.where(mth <= 2, HI.date.dt.year, 0))
if HAS_LSTM:
    subsets = [("all high days", HI)]
    subsets += [(f"excluding {c}", HI[HI.city != c]) for c in by_city.head(4).index]
    subsets += [(f"winter {s} only" if s else "Mar-Oct only", HI[HI.season == s]) for s in sorted(HI.season.unique())]
    for a in ("LSTM", "Blend"):
        rows = []
        for lab, S in subsets:
            rows.append({"subset": lab, "n(24h)": int((S.h == 1).sum()),
                         **{f"{24 * h}h": boot_diff(S[S.h == h], a, "Ridge") for h in (1, 2, 3)}})
        print(f"\n{a} minus Ridge:")
        print(pd.DataFrame(rows).to_string(index=False))
    if SEEDS:
        rows = [{"seed": s, **{f"{24 * h}h": round(float((HI[HI.h == h][s + "_ae"] - HI[HI.h == h]["Ridge_ae"]).mean()), 2)
                               for h in (1, 2, 3)}} for s in SEEDS]
        print("\nsingle-seed LSTM minus Ridge on high days (point estimates; all seeds should agree in sign):")
        print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ 5. Holm-corrected Diebold-Mariano tests
def dm(Th, a, b, lag=7):
    d = (Th[a + "_ae"] - Th[b + "_ae"]).groupby(Th["date"]).mean().sort_index()
    T, mean = len(d), d.mean(); x = d.values - mean
    lrv = (x ** 2).mean()
    for l in range(1, lag + 1):
        lrv += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    stat = mean / math.sqrt(max(lrv, 1e-12) / T)
    return mean, stat, math.erfc(abs(stat) / math.sqrt(2))
pairs = [(a, b) for a, b in [("Ridge", "Persistence"), ("GBM", "Ridge"), ("LSTM", "Ridge"), ("LSTM", "GBM"),
                             ("LSTM", "Persistence"), ("Blend", "Ridge"), ("Blend", "LSTM")]
         if a in MODELS and b in MODELS]
rows = [{"pair": f"{a} - {b}", "h": 24 * h, **dict(zip(["diff", "DM", "p"], dm(R[R.h == h], a, b)))}
        for a, b in pairs for h in (1, 2, 3)]
T = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(T)
T["p_holm"] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * T["p"].values))
T["significant (Holm 5%)"] = np.where(T["p_holm"] < 0.05, "yes", "no")
T[["diff", "DM"]] = T[["diff", "DM"]].round(2); T["p"] = T["p"].round(4); T["p_holm"] = T["p_holm"].round(4)
print(f"\n=== 5. DIEBOLD-MARIANO TESTS WITH HOLM CORRECTION ({m} tests; MAE difference, A minus B) ===")
print(T.to_string(index=False))
print("\ndone")
