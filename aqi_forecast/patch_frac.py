p = "src/build_oaq_panel.py"
s = open(p, encoding="utf-8").read()
a = "MIN_COVER, MIN_FRAC, MIN_PCT = 0.4, 0.5, 50"
assert a in s, "threshold line not found"
open(p, "w", encoding="utf-8").write(s.replace(a, "MIN_COVER, MIN_FRAC, MIN_PCT = 0.4, 0.25, 50"))
print("patched MIN_FRAC to 0.25")
