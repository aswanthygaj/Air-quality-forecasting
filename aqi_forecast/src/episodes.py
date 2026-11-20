"""How concentrated is the high-pollution gain? Ranks target days by their contribution to (model - ridge).
Run (venv active):  python src/episodes.py 2>&1 | Tee-Object -FilePath episodes_results.txt   (set $env:TAG="_fixed" first for the fixed panel)"""
import os, numpy as np, pandas as pd
HIGH = 90
TAG = os.environ.get("TAG", "")
R = pd.read_csv(f"benchmark2_predictions{TAG}.csv", parse_dates=["date"])
for m in [c for c in ("Ridge", "LSTM", "Blend") if c in R.columns]:
    R[m + "_ae"] = (R[m] - R["y"]).abs()
print(f"panel: {'fixed-station' if TAG else 'original'}; high-pollution targets (true PM2.5 > {HIGH}); negative = better than ridge\n")
for a in [m for m in ("LSTM", "Blend") if m in R.columns]:
    rows, tops = [], {}
    for h in (1, 2, 3):
        S = R[(R.h == h) & (R.y > HIGH)].copy()
        S["target_date"] = S["date"] + pd.to_timedelta(h, unit="D")
        S["d"] = S[a + "_ae"] - S["Ridge_ae"]
        g = S.groupby("target_date").agg(total=("d", "sum"), n=("d", "size"), cities=("city", "nunique"),
                                         mean_pm=("y", "mean")).sort_values("total")
        tot, N = g.total.sum(), int(g.n.sum())
        row = {"horizon": f"{24 * h}h", "target days": len(g), "targets": N, "mean diff": round(tot / N, 2)}
        for k in (5, 10, 20):
            row[f"top{k} days share"] = f"{100 * g.total.head(k).sum() / tot:.0f}%" if tot < 0 else "n/a"
        rest_n = N - int(g.n.head(10).sum())
        row["mean diff w/o top10 days"] = round((tot - g.total.head(10).sum()) / max(rest_n, 1), 2)
        rows.append(row); tops[h] = g.head(6)
    print(f"{a} minus Ridge:")
    print(pd.DataFrame(rows).to_string(index=False))
    for h in (2, 3):
        t = tops[h].copy(); t["total"] = t["total"].round(1); t["mean_pm"] = t["mean_pm"].round(0)
        print(f"  biggest-contribution target days at {24 * h}h (date, summed gain, targets, cities, mean PM2.5):")
        print(t.to_string(header=False))
    print()
print("share = (sum of the k biggest daily gains) / (total gain); a share near or above 100% means a few episodes carry the whole result.")
