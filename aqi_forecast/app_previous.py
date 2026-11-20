import sys, os, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch, streamlit as st, plotly.graph_objects as go
from common import *
from model import Net, forecast

st.set_page_config(page_title="India AQI Forecast", layout="wide")
st.title("Air Quality Forecast - Indian Cities")

@st.cache_data
def load():
    return pd.read_csv("data/processed.csv", parse_dates=["date"])

@st.cache_resource
def load_model():
    meta = json.load(open("models/meta.json"))
    m = Net(len(meta["feats"]))
    m.load_state_dict(torch.load("models/lstm.pt", map_location="cpu", weights_only=True))
    return m.eval(), meta

@st.cache_data
def prophet_pred(city):
    from prophet_baseline import prophet_forecast
    d = load()
    d = d[d.city == city].tail(730)
    return prophet_forecast(d)

df = load()
model, meta = load_model()
city = st.sidebar.selectbox("City", sorted(df.city.unique()))
show_prophet = st.sidebar.checkbox("Compare with Prophet baseline", False)
st.sidebar.caption("Forecast origin = last date in data/processed.csv. "
                   "Append live OpenAQ days (src/openaq_live.py) to keep it current.")

fc, last = forecast(df, city, model, meta)
days = [last + pd.Timedelta(days=i + 1) for i in range(HOR)]
cols = st.columns(HOR + 1)
cur = df[df.city == city].iloc[-1].pm25
cols[0].metric("Latest PM2.5", f"{cur:.0f} ug/m3", aqi_bucket(pm25_to_aqi(cur)), delta_color="off")
for i, c in enumerate(cols[1:]):
    a = pm25_to_aqi(fc[i])
    c.metric(f"+{24 * (i + 1)} h ({days[i].date()})", f"{fc[i]:.0f} ug/m3",
             f"AQI {a:.0f} - {aqi_bucket(a)}", delta_color="off")

tab1, tab2, tab3 = st.tabs(["Forecast", "All cities", "Live (OpenAQ)"])

with tab1:
    h = df[df.city == city].tail(90)
    fig = go.Figure()
    fig.add_scatter(x=h.date, y=h.pm25, name="Observed")
    fig.add_scatter(x=[last] + days, y=[cur] + list(fc), name="LSTM forecast", mode="lines+markers")
    if show_prophet:
        try:
            pf = prophet_pred(city)
            fig.add_scatter(x=days, y=pf, name="Prophet", mode="lines+markers")
        except Exception as e:
            st.warning(f"Prophet unavailable on this machine: {e}")
    for lim, col in [(60, "#92d050"), (90, "#ffc000"), (120, "#ff7c00"), (250, "#ff0000")]:
        fig.add_hline(y=lim, line_dash="dot", line_color=col, opacity=0.5)
    fig.update_layout(yaxis_title="PM2.5 (ug/m3)", height=450)
    st.plotly_chart(fig, width="stretch")
    wx = df[df.city == city].tail(30).set_index("date")
    st.subheader("Recent weather drivers (last 30 days)")
    c1, c2, c3 = st.columns(3)
    c1.caption("Temperature (C)"); c1.line_chart(wx["temp"])
    c2.caption("Wind speed (km/h)"); c2.line_chart(wx["wind"])
    c3.caption("Rainfall (mm)"); c3.bar_chart(wx["rain"])

with tab2:
    rows = []
    for c in sorted(df.city.unique()):
        f, _ = forecast(df, c, model, meta)
        rows.append({"City": c, **{f"+{24 * (i + 1)}h": round(float(f[i]), 1) for i in range(HOR)},
                     "Category (+24h)": aqi_bucket(pm25_to_aqi(f[0]))})
    st.dataframe(pd.DataFrame(rows), width="stretch")
    mp = pd.DataFrame([{"city": c, "lat": CITY_COORDS[c][0], "lon": CITY_COORDS[c][1]}
                       for c in df.city.unique() if c in CITY_COORDS])
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
