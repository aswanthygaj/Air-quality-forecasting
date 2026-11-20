"""Clean city_day.csv, merge weather, write a contiguous daily panel: data/processed.csv"""
import pandas as pd
from common import WX
def build(city_day="data/city_day.csv", weather="data/weather.csv", out="data/processed.csv"):
    df = pd.read_csv(city_day, parse_dates=["Date"]).rename(
        columns={"City": "city", "Date": "date", "PM2.5": "pm25"})[["city", "date", "pm25"]]
    wx = pd.read_csv(weather, parse_dates=["date"])
    cov = df.groupby("city")["pm25"].apply(lambda s: s.notna().mean())
    cities = cov[cov > 0.6].index.tolist()
    cities = [c for c in cities if c in set(wx.city)]
    print("cities kept:", cities)
    full = pd.date_range(df["date"].min(), df["date"].max())
    parts = []
    for c in cities:
        s = df[df.city == c].drop_duplicates("date").set_index("date")["pm25"].reindex(full)
        s = s.loc[s.first_valid_index():s.last_valid_index()].interpolate()   # contiguous
        g = s.rename("pm25").rename_axis("date").reset_index(); g["city"] = c
        parts.append(g)
    d = pd.concat(parts).merge(wx, on=["city", "date"], how="left")
    d[WX] = d.groupby("city")[WX].transform(lambda x: x.interpolate().bfill().ffill())
    d["pm25"] = d["pm25"].clip(lower=1, upper=d["pm25"].quantile(0.999))
    d.to_csv(out, index=False); print(d.shape, "->", out)
if __name__ == "__main__":
    build()
