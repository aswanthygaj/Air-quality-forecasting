"""OPTIONAL research model: Temporal Fusion Transformer (pytorch-forecasting).
pip install pytorch-forecasting lightning   (API varies by version - adjust imports if needed)"""
import pandas as pd, numpy as np, lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from common import WX, add_raw_features

df = add_raw_features(pd.read_csv("data/processed.csv", parse_dates=["date"]))
df["time_idx"] = (df["date"] - df["date"].min()).dt.days
cut = int(df["time_idx"].max() * 0.85)
tr = TimeSeriesDataSet(df[df.time_idx <= cut], time_idx="time_idx", target="pm25", group_ids=["city"],
    max_encoder_length=30, max_prediction_length=3, static_categoricals=["city"],
    time_varying_known_reals=["time_idx", "doy_sin", "doy_cos"],
    time_varying_unknown_reals=["pm25", "regional"] + WX,
    target_normalizer=GroupNormalizer(groups=["city"], transformation="softplus"),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)
va = TimeSeriesDataSet.from_dataset(tr, df, min_prediction_idx=cut + 1, stop_randomization=True)
tl, vl = tr.to_dataloader(True, batch_size=64), va.to_dataloader(False, batch_size=128)
tft = TemporalFusionTransformer.from_dataset(tr, learning_rate=3e-3, hidden_size=32,
    attention_head_size=4, dropout=0.1, hidden_continuous_size=16, loss=QuantileLoss())
pl.Trainer(max_epochs=20, gradient_clip_val=0.1, accelerator="auto").fit(tft, tl, vl)
raw, x = tft.predict(vl, mode="raw", return_x=True)
tft.plot_prediction(x, raw, idx=0)   # also: tft.interpret_output(raw) -> variable importance
