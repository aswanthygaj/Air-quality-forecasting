import sys, os
sys.path.insert(0, "src")
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from common import CITY_COORDS, HOR, pm25_to_aqi, aqi_bucket
from final_lib import load_bundle, build_windows, forecast_all

st.set_page_config(page_title="India AQI Forecast", layout="wide")
st.title("Air Quality Forecast - Indian Cities")

if not os.path.exists("models/final_meta.json"):
    st.error("Run  python src/train_final.py  first (models/final_meta.json not found).")
    st.stop()

@st.cache_data
def load():
    return pd.read_csv("data/processed.csv", parse_dates=["date"])

@st.cache_resource
def bundle():
    return load_bundle()

@st.cache_data
def all_forecasts():
    meta, nets = bundle()
    wins = build_windows(load(), meta)
    return {c: (forecast_all(F, meta, nets), d) for c, (F, d) in wins.items()}

df = load()
meta, nets = bundle()
fc_all = all_forecasts()

city = st.sidebar.selectbox("City", sorted(fc_all))
models = list(fc_all[city][0].keys())
choice = st.sidebar.radio("Forecast model", models, index=models.index("Blend") if "Blend" in models else 0)
st.sidebar.caption(f"Models trained on data through {meta['trained_through']} "
                   f"(weather + regional-group features). Blend = average of LSTM ensemble and ridge. "
                   "Append live OpenAQ days (src/openaq_live.py) to keep the data current.")

fcs, last = fc_all[city]
fc = fcs[choice]
days = [last + pd.Timedelta(days=i + 1) for i in range(HOR)]
hist = df[df.city == city]
cur = float(hist[hist.date == last].pm25.iloc[0])

ndays = meta["n_train_days"].get(city, 0)
if ndays < meta["min_train_days"]:
    st.warning(f"{city} has only {ndays} days of training history, so its forecasts are less reliable "
               "than for long-history cities (a simple 'same as today' guess may do as well).")

cols = st.columns(HOR + 1)
cols[0].metric(f"Latest PM2.5 ({last.date()})", f"{cur:.0f} ug/m3", aqi_bucket(pm25_to_aqi(cur)), delta_color="off")
for i, c in enumerate(cols[1:]):
    a = pm25_to_aqi(fc[i])
    c.metric(f"+{24 * (i + 1)} h ({days[i].date()})", f"{fc[i]:.0f} ug/m3",
             f"AQI {a:.0f} - {aqi_bucket(a)}", delta_color="off")

tab1, tab2, tab3 = st.tabs(["Forecast", "All cities", "Live (OpenAQ)"])

with tab1:
    h = hist.tail(90)
    fig = go.Figure()
    fig.add_scatter(x=h.date, y=h.pm25, name="Observed")
    for k, v in fcs.items():
        line = dict(width=3) if k == choice else dict(width=1, dash="dot")
        fig.add_scatter(x=[last] + days, y=[cur] + list(v), name=k, mode="lines+markers", line=line)
    for lim, colr in [(60, "#92d050"), (90, "#ffc000"), (120, "#ff7c00"), (250, "#ff0000")]:
        fig.add_hline(y=lim, line_dash="dot", line_color=colr, opacity=0.5)
    fig.update_layout(yaxis_title="PM2.5 (ug/m3)", height=450)
    st.plotly_chart(fig, width="stretch")
    wx = hist.tail(30).set_index("date")
    st.subheader("Recent weather drivers (last 30 days)")
    c1, c2, c3 = st.columns(3)
    c1.caption("Temperature (C)"); c1.line_chart(wx["temp"])
    c2.caption("Wind speed (km/h)"); c2.line_chart(wx["wind"])
    c3.caption("Rainfall (mm)"); c3.bar_chart(wx["rain"])

with tab2:
    rows = []
    for c in sorted(fc_all):
        f = fc_all[c][0][choice]
        rows.append({"City": c, "Last data": str(fc_all[c][1].date()),
                     **{f"+{24 * (i + 1)}h": round(float(f[i]), 1) for i in range(HOR)},
                     "Category (+24h)": aqi_bucket(pm25_to_aqi(f[0])),
                     "Training days": meta["n_train_days"].get(c, 0)})
    st.dataframe(pd.DataFrame(rows), width="stretch")
    mp = pd.DataFrame([{"city": c, "lat": CITY_COORDS[c][0], "lon": CITY_COORDS[c][1]}
                       for c in fc_all if c in CITY_COORDS])
    st.map(mp)

with tab3:
    if "OPENAQ_API_KEY" not in os.environ:
        st.info("Set the OPENAQ_API_KEY environment variable to enable live data.")
    elif st.button("Fetch last 72 h"):
        from openaq_live import city_live_hourly
        lat, lon = CITY_COORDS[city]
        live = city_live_hourly(lat, lon)
        if len(live):
            st.line_chart(live.set_index("time")["pm25"])
        else:
            st.warning("No PM2.5 sensors returned for this city.")
