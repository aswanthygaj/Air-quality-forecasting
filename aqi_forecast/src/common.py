"""Shared constants, feature engineering, model and inference helpers."""
import numpy as np, pandas as pd

SEQ, HOR = 14, 3                      # 14 days of history -> next 1,2,3 days (24/48/72 h)
WX = ["temp", "rh", "wind", "rain", "pressure"]
NUM = ["lpm", "regional"] + WX        # numeric features that get standardised

CITY_COORDS = {
    "Delhi": (28.61, 77.21), "Kolkata": (22.57, 88.36), "Mumbai": (19.07, 72.88),
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Lucknow": (26.85, 80.95),
    "Hyderabad": (17.38, 78.48), "Ahmedabad": (23.02, 72.57), "Patna": (25.59, 85.14),
    "Jaipur": (26.91, 75.79),
}

# ---------- CPCB AQI (PM2.5 sub-index) ----------
_BP = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
       (90, 120, 201, 300), (120, 250, 301, 400), (250, 380, 401, 500)]
def pm25_to_aqi(pm):
    pm = float(pm)
    for lo, hi, ilo, ihi in _BP:
        if pm <= hi:
            return ilo + (ihi - ilo) * (pm - lo) / (hi - lo)
    return 500.0
def aqi_bucket(aqi):
    for lim, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                      (300, "Poor"), (400, "Very Poor")]:
        if aqi <= lim:
            return name
    return "Severe"

# ---------- features ----------
def add_raw_features(df):
    """df: city, date, pm25, WX cols (daily, contiguous per city)."""
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df["lpm"] = np.log1p(df["pm25"])
    # spatial feature: mean log-PM2.5 of all OTHER cities on the same day
    piv = df.pivot(index="date", columns="city", values="lpm")
    P = piv.values; m = ~np.isnan(P)
    S = np.nansum(P, 1, keepdims=True); C = m.sum(1, keepdims=True)
    oth = (S - np.nan_to_num(P)) / np.maximum(C - m, 1)
    oth = np.where((C - m) > 0, oth, np.nan)
    reg = pd.DataFrame(oth, index=piv.index, columns=piv.columns).reset_index() \
            .melt(id_vars="date", var_name="city", value_name="regional")
    df = df.merge(reg, on=["date", "city"], how="left")
    df["regional"] = df["regional"].fillna(df["lpm"])
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df

def fit_stats(df, cutoff):
    tr = df[df["date"] < cutoff]
    return {"mu": {c: float(tr[c].mean()) for c in NUM},
            "sd": {c: float(tr[c].std() + 1e-6) for c in NUM}}

def standardize(df, meta):
    df = df.copy()
    for c in NUM:
        df[c] = (df[c] - meta["mu"][c]) / meta["sd"][c]
    for c in meta["cities"]:
        df["c_" + c] = (df["city"] == c).astype("float32")
    return df

def feat_cols(cities):
    return NUM + ["doy_sin", "doy_cos"] + ["c_" + c for c in cities]

def inv(z, meta):
    return np.expm1(np.asarray(z) * meta["sd"]["lpm"] + meta["mu"]["lpm"])


CITY_COORDS.update({
    "Aizawl": (23.73, 92.72), "Amaravati": (16.51, 80.52), "Amritsar": (31.63, 74.87),
    "Bhopal": (23.26, 77.41), "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78),
    "Coimbatore": (11.00, 76.96), "Ernakulam": (9.98, 76.28), "Gurugram": (28.46, 77.03),
    "Guwahati": (26.14, 91.74), "Jorapokhar": (23.71, 86.41), "Kochi": (9.93, 76.27),
    "Shillong": (25.57, 91.88), "Talcher": (20.95, 85.22),
    "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22),
})
