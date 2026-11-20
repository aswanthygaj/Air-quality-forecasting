"""Prophet baseline with rolling-origin evaluation (refit at each origin, predict 1-3 days)."""
import numpy as np, pandas as pd
from prophet import Prophet

def prophet_forecast(series_df, horizon=3):
    """series_df: columns date, pm25. Returns array of `horizon` forecasts (ug/m3)."""
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": series_df["date"], "y": np.log1p(series_df["pm25"])}))
    f = m.predict(m.make_future_dataframe(horizon))
    return np.expm1(f["yhat"].tail(horizon).values)

if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "Delhi"
    d = pd.read_csv("data/processed.csv", parse_dates=["date"]); d = d[d.city == city].reset_index(drop=True)
    start = int(len(d) * 0.85); origins = range(start, len(d) - 3, 21)
    errs = []
    for o in origins:
        hist = d.iloc[max(0, o - 730):o + 1]
        p = prophet_forecast(hist); y = d.pm25.values[o + 1:o + 4]
        errs.append(np.abs(p - y))
    print(city, "Prophet MAE by horizon (24/48/72h):", np.mean(errs, 0).round(2))
