"""LSTM model + inference helper (needs torch)."""
import numpy as np, torch, torch.nn as nn
from common import *

class Net(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, HOR))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])

def forecast(df_raw, city, model, meta):
    """Return (array of HOR PM2.5 forecasts in ug/m3, last observed date)."""
    d = standardize(add_raw_features(df_raw), meta)
    g = d[d["city"] == city].tail(SEQ)
    x = torch.tensor(g[feat_cols(meta["cities"])].values, dtype=torch.float32)[None]
    model.eval()
    with torch.no_grad():
        z = model(x)[0].numpy()
    return inv(z, meta), g["date"].max()
