p = "src/baselines.py"
s = open(p, encoding="utf-8").read()
start = s.index("clim = {}")
end_marker = "clim[c] = sm.fillna(g.pm25.mean()).values\n"
end = s.index(end_marker) + len(end_marker)
new = '''def _profile(g):
    m = g.groupby(g.date.dt.dayofyear)["pm25"].mean().reindex(range(1, 367))
    ext = pd.concat([m.iloc[-15:], m, m.iloc[:15]]).reset_index(drop=True)
    sm = ext.rolling(31, center=True, min_periods=5).mean().iloc[15:-15]
    return sm.fillna(g.pm25.mean()).values
_train = raw[raw.date < pd.Timestamp(cut_tr)]
_glob = _profile(_train)          # all-cities profile, used for cities with little/no training data
clim = {c: (_profile(_train[_train.city == c]) if (_train.city == c).sum() >= 365 else _glob)
        for c in raw.city.unique()}
'''
s = s[:start] + new + s[end:]
open(p, "w", encoding="utf-8").write(s)
print("patched")
