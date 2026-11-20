p = "src/build_oaq_panel.py"
s = open(p, encoding="utf-8").read()
for a, b in [('pd.Timestamp("2019-12-31")', 'pd.Timestamp("2020-06-30")'),
             ("MIN_COVER, MIN_FRAC, MIN_PCT = 0.5, 0.5, 75", "MIN_COVER, MIN_FRAC, MIN_PCT = 0.4, 0.5, 50")]:
    assert a in s, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched")
