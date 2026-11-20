"""One-time generator: reads src/benchmark2.py and writes src/benchmark_oaq.py, a copy that runs on the OpenAQ panel
with three test blocks (winter 2020-21, winter 2021-22, 2025-26) and without the legacy fold.  Run: python src/make_benchmark_oaq.py"""
s = open("src/benchmark2.py", encoding="utf-8").read()
reps = [
    ('CITY_FILE = os.environ.get("CITY_FILE", "data/city_day.csv")', 'CITY_FILE = os.environ.get("CITY_FILE", "data/city_day_oaq.csv")'),
    ('TAG = os.environ.get("TAG", "")', 'TAG = os.environ.get("TAG", "_oaq")\nWEATHER_FILE = os.environ.get("WEATHER_FILE", "data/weather_oaq.csv")\n'
     'FOLDS = [tuple(pd.Timestamp(x) for x in f.split(":")) for f in os.environ.get("FOLDS", "2020-07-01:2021-03-01,2021-03-01:2022-11-01,2025-02-01:2026-11-06").split(",")]'),
    ('wx = pd.read_csv("data/weather.csv", parse_dates=["date"])', 'wx = pd.read_csv(WEATHER_FILE, parse_dates=["date"])'),
    ('R = pd.concat([run_fold(k + 1, BOUNDS[k], BOUNDS[k], BOUNDS[k + 1]) for k in range(len(BOUNDS) - 1)]).reset_index(drop=True)',
     'R = pd.concat([run_fold(k + 1, a, a, b) for k, (a, b) in enumerate(FOLDS)]).reset_index(drop=True)'),
    ('L = run_fold("legacy", cut_tr_l, cut_va_l, t1_ + pd.Timedelta(days=1)).reset_index(drop=True)', 'L = R[R.fold == len(FOLDS)].reset_index(drop=True)'),
    ('=== 5. LEGACY WINDOW (the old Sep 2019 - Jul 2020 test) re-scored under the new rules ===', '=== 5. LAST BLOCK ONLY (2025-26) ==='),
    ('print("EARLIER STUDY on the same window (interpolated targets, 12 cities), MAE 24/48/72h:")', 'pass'),
    ('print("  Persistence 13.62 / 18.84 / 21.54   Ridge 12.63 / 16.53 / 18.27   LSTM 12.69 / 16.71 / 18.57   Blend 12.44 / 16.39 / 18.14")', 'pass'),
]
for a, b in reps:
    assert a in s, "pattern not found in benchmark2.py: " + a[:70]
    s = s.replace(a, b)
open("src/benchmark_oaq.py", "w", encoding="utf-8").write(s)
print("wrote src/benchmark_oaq.py (", s.count(chr(10)), "lines )")
