"""Helpers for the deployed models: features, ridge, bundle loading, inference."""
import json, numpy as np, pandas as pd
from common import SEQ, HOR, WX, add_raw_features, inv

LAGS = 7
BUNDLE = "models/final_meta.json"
GROUPS = {
    "north_plain": ["Delhi", "Gurugram", "Lucknow", "Patna", "Amritsar", "Chandigarh", "Jaipur"],
    "east": ["Kolkata", "Brajrajnagar", "Talcher", "Jorapokhar"],
    "northeast": ["Guwahati", "Shillong", "Aizawl"],
    "west_central": ["Ahmedabad", "Bhopal"],
    "south": ["Hyderabad", "Bengaluru", "Chennai", "Coimbatore", "Kochi", "Ernakulam",
              "Thiruvananthapuram", "Visakhapatnam", "Amaravati"],
}
city2grp = {c: g for g, cs in GROUPS.items() for c in cs}

def add_group_regional(df):
    """Mean log-PM2.5 (same day) of the OTHER cities in the same regional group."""
    piv = df.pivot(index="date", columns="city", values="lpm")
    out = pd.DataFrame(np.nan, index=piv.index, columns=piv.columns)
    for c in piv.columns:
        mates = [m for m in piv.columns if m != c and city2grp.get(m, "other") == city2grp.get(c, "other")]
        if mates:
            out[c] = piv[mates].mean(axis=1)
    reg = out.reset_index().melt(id_vars="date", var_name="city", value_name="gregional")
    df = df.merge(reg, on=["date", "city"], how="left")
    df["gregional"] = df["gregional"].fillna(df["lpm"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)

def prepare(df_raw):
    return add_group_regional(add_raw_features(df_raw))

def standardize2(df, meta):
    df = df.copy()
    for c in meta["numf"]:
        df[c] = (df[c] - meta["mu"][c]) / meta["sd"][c]
    for c in meta["cities"]:
        df["c_" + c] = (df["city"] == c).astype("float32")
    return df

def ridge_params(A, y, alpha):
    mu, ym = A.mean(0), y.mean(0)
    Ac = A - mu
    w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (y - ym))
    return mu, ym, w

def ridge_matrix(X, nd):
    """X: (n, SEQ, n_feat) standardised windows. Dynamic features (first nd) use the last LAGS days."""
    return np.concatenate([X[:, -LAGS:, :nd].reshape(len(X), -1), X[:, -1, nd:]], 1).astype("float64")

def build_windows(df_raw, meta):
    """{city: (last SEQ standardised rows as float32 array, last date)} for every modelled city."""
    d = standardize2(prepare(df_raw), meta)
    out = {}
    for c in meta["cities"]:
        g = d[d.city == c].sort_values("date").tail(SEQ)
        if len(g) == SEQ:
            out[c] = (g[meta["feats"]].values.astype("float32"), g["date"].max())
    return out

def load_bundle():
    meta = json.load(open(BUNDLE))
    nets = []
    if meta.get("lstm_seeds", 0) > 0:
        import torch
        from model import Net
        for s in range(meta["lstm_seeds"]):
            n = Net(len(meta["feats"]))
            n.load_state_dict(torch.load(f"models/final_lstm_{s}.pt", map_location="cpu", weights_only=True))
            n.eval(); nets.append(n)
    return meta, nets

def forecast_all(F, meta, nets):
    """F: (SEQ, n_feat) window. Returns {model name: array of HOR forecasts in ug/m3}."""
    rg = meta["ridge"]; nd = len(meta["numf"])
    a = ridge_matrix(F[None].astype("float64"), nd)[0]
    out = {"Ridge": inv((a - np.array(rg["mu"])) @ np.array(rg["w"]) + np.array(rg["ym"]), meta)}
    if nets:
        import torch
        x = torch.tensor(F[None])
        with torch.no_grad():
            ens = np.mean([inv(n(x)[0].numpy(), meta) for n in nets], axis=0)
        out["LSTM ensemble"] = ens
        w = meta.get("blend_w", 0.5)
        out["Blend"] = w * ens + (1 - w) * out["Ridge"]
    return out
