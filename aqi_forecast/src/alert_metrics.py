"""Compute categorical alert metrics (Precision, Recall, F1, Critical Success Index)
for Air Quality Early Warning at thresholds PM2.5 > 90 ug/m3 and > 120 ug/m3.
Outputs summary CSV and Markdown tables for manuscript inclusion.
"""
import os, sys
import numpy as np
import pandas as pd

THRESHOLDS = [90, 120]
HORIZONS = [1, 2, 3]

def evaluate_alerts(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_path} (not found)")
        return None
    df = pd.read_csv(csv_path)
    models = [m for m in ["Persistence", "Ridge", "GBM", "LSTM", "Blend"] if m in df.columns]
    
    rows = []
    for th in THRESHOLDS:
        for h in HORIZONS:
            sub = df[df.h == h]
            y_true = sub.y > th
            n_events = int(y_true.sum())
            total = len(y_true)
            base_rate = n_events / total
            
            for m in models:
                y_pred = sub[m] > th
                tp = int((y_true & y_pred).sum())
                fp = int((~y_true & y_pred).sum())
                fn = int((y_true & ~y_pred).sum())
                tn = int((~y_true & ~y_pred).sum())
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                acc = (tp + tn) / total
                
                rows.append({
                    "Dataset": label,
                    "Threshold_ugm3": th,
                    "Horizon_h": 24 * h,
                    "Model": m,
                    "Total_Windows": total,
                    "Event_Count": n_events,
                    "Base_Rate_%": round(100 * base_rate, 1),
                    "Precision": round(prec, 3),
                    "Recall": round(rec, 3),
                    "F1_Score": round(f1, 3),
                    "CSI_ThreatScore": round(csi, 3),
                    "Accuracy": round(acc, 3)
                })
    res_df = pd.DataFrame(rows)
    return res_df

if __name__ == "__main__":
    datasets = [
        ("Kaggle_Fixed_Panel_14cities", "benchmark2_predictions_fixed.csv"),
        ("OpenAQ_Extended_21cities", "benchmark2_predictions_oaq.csv")
    ]
    
    all_res = []
    for label, fn in datasets:
        # Check both current directory and aqi_forecast
        path = fn if os.path.exists(fn) else os.path.join("aqi_forecast", fn)
        df_res = evaluate_alerts(path, label)
        if df_res is not None:
            all_res.append(df_res)
            
    if all_res:
        final_df = pd.concat(all_res, ignore_index=True)
        final_df.to_csv("alert_warning_metrics.csv", index=False)
        if os.path.exists("aqi_forecast"):
            final_df.to_csv("aqi_forecast/alert_warning_metrics.csv", index=False)
        print("Alert metrics saved to alert_warning_metrics.csv!")
        
        # Print concise markdown comparison for manuscript
        print("\n=== SUMMARY TABLE: 72-HOUR ALERT WARNING PERFORMANCE (PM2.5 > 90 ug/m3) ===")
        sub72 = final_df[(final_df.Threshold_ugm3 == 90) & (final_df.Horizon_h == 72)]
        print(sub72[["Dataset", "Model", "Precision", "Recall", "F1_Score", "CSI_ThreatScore"]].to_string(index=False))

        print("\n=== SUMMARY TABLE: 72-HOUR SEVERE EVENT DETECTION (PM2.5 > 120 ug/m3) ===")
        sub120 = final_df[(final_df.Threshold_ugm3 == 120) & (final_df.Horizon_h == 72)]
        print(sub120[["Dataset", "Model", "Precision", "Recall", "F1_Score", "CSI_ThreatScore"]].to_string(index=False))

