"""One-time patch: lets benchmark2.py and analysis.py read an alternative city file via environment variables
CITY_FILE and TAG (defaults keep the old behaviour). Run: python src/apply_tag_patch.py"""
def patch(p, reps):
    s = open(p, encoding="utf-8").read()
    if "TAG" in s and "CITY_FILE" in s or ("TAG" in s and p.endswith("analysis.py")):
        print(p, "already patched"); return
    for a, b in reps:
        assert a in s, f"pattern not found in {p}: {a[:60]}"
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s); print(p, "patched")
patch("src/benchmark2.py", [
    ("import math, copy, time, numpy as np, pandas as pd", "import math, copy, os, time, numpy as np, pandas as pd"),
    ("H, LAGS, SEQ = 3, 7, 14", 'CITY_FILE = os.environ.get("CITY_FILE", "data/city_day.csv")\nTAG = os.environ.get("TAG", "")\nH, LAGS, SEQ = 3, 7, 14'),
    ('pd.read_csv("data/city_day.csv", parse_dates=["Date"])', 'pd.read_csv(CITY_FILE, parse_dates=["Date"])'),
    ('R.to_csv("benchmark2_predictions.csv", index=False)', 'R.to_csv(f"benchmark2_predictions{TAG}.csv", index=False)'),
    ('print("\\nsaved benchmark2_predictions.csv")', 'print(f"\\nsaved benchmark2_predictions{TAG}.csv")')])
patch("src/analysis.py", [
    ("import math, numpy as np, pandas as pd", "import math, os, numpy as np, pandas as pd"),
    ('pd.read_csv("benchmark2_predictions.csv", parse_dates=["date"])',
     'pd.read_csv(f"benchmark2_predictions{os.environ.get(\'TAG\', \'\')}.csv", parse_dates=["date"])')])
