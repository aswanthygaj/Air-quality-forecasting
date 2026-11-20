"""Calculate comprehensive categorical early warning alert metrics for PM2.5 > 90 ug/m3
(and PM2.5 > 120 ug/m3) across 24h, 48h, and 72h forecasting lead times.
Computes:
  - Hits (TP), False Alarms (FP), Misses (FN), Correct Rejections (TN)
  - Recall / Probability of Detection (POD)
  - Precision / Positive Predictive Value (PPV)
  - False Alarm Ratio (FAR)
  - F1-Score
  - Critical Success Index (CSI / Threat Score)
  - Heidke Skill Score (HSS)
Outputs CSV tables and manuscript-ready markdown tables.
"""
import os, sys
import numpy as np
import pandas as pd

def compute_contingency_metrics(y_true, y_pred):
    """Compute meteorological 2x2 contingency table and skill scores."""
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    total = tp + fp + fn + tn
    
    # Standard metrics
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # POD / Recall
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # Precision / PPV
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0     # False Alarm Ratio
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0   # Threat Score / CSI
    
    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / total if total > 0 else 0.0
    hss = (tp + tn - expected_correct) / (total - expected_correct) if (total - expected_correct) > 0 else 0.0
    
    return {
        "Total": total,
        "Hits_TP": tp,
        "FalseAlarms_FP": fp,
        "Misses_FN": fn,
        "CorrectRej_TN": tn,
        "BaseRate_%": round(100 * (tp + fn) / total, 2) if total > 0 else 0.0,
        "Precision": round(prec, 3),
        "Recall_POD": round(rec, 3),
        "FAR": round(far, 3),
        "F1_Score": round(f1, 3),
        "CSI_ThreatScore": round(csi, 3),
        "HSS": round(hss, 3)
    }

def run_alert_analysis():
    datasets = [
        ("Kaggle Fixed Panel (14 Cities)", "benchmark2_predictions_fixed.csv"),
        ("OpenAQ Extended (21 Cities, 2018-2026)", "benchmark2_predictions_oaq.csv")
    ]
    
    thresholds = [90, 120]
    horizons = [1, 2, 3]
    
    all_rows = []
    
    for ds_name, filename in datasets:
        path = filename if os.path.exists(filename) else os.path.join("aqi_forecast", filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        df = pd.read_csv(path)
        models = [m for m in ["Persistence", "Ridge", "GBM", "LSTM", "Blend"] if m in df.columns]
        
        for th in thresholds:
            for h in horizons:
                sub = df[df.h == h]
                y_true = sub["y"] > th
                
                for m in models:
                    y_pred = sub[m] > th
                    metrics = compute_contingency_metrics(y_true, y_pred)
                    metrics.update({
                        "Dataset": ds_name,
                        "Threshold": f"PM2.5 > {th}",
                        "Horizon": f"{24 * h}h",
                        "Model": m
                    })
                    all_rows.append(metrics)
                    
    results_df = pd.DataFrame(all_rows)
    
    # Save CSVs
    results_df.to_csv("alert_metrics_all.csv", index=False)
    results_df[results_df["Threshold"] == "PM2.5 > 90"].to_csv("alert_metrics_pm90.csv", index=False)
    results_df[results_df["Threshold"] == "PM2.5 > 120"].to_csv("alert_metrics_pm120.csv", index=False)
    
    if os.path.exists("aqi_forecast"):
        results_df.to_csv("aqi_forecast/alert_metrics_all.csv", index=False)
        results_df[results_df["Threshold"] == "PM2.5 > 90"].to_csv("aqi_forecast/alert_metrics_pm90.csv", index=False)
        results_df[results_df["Threshold"] == "PM2.5 > 120"].to_csv("aqi_forecast/alert_metrics_pm120.csv", index=False)
        
    print("Successfully generated alert_metrics_all.csv and alert_metrics_pm90.csv!")
    return results_df

if __name__ == "__main__":
    df = run_alert_analysis()
    
    # Print formatted markdown table for PM2.5 > 90
    print("\n" + "="*85)
    print("ALERT VERIFICATION TABLE FOR PM2.5 > 90 ug/m3 (CPCB Poor/Unhealthy Threshold)")
    print("="*85)
    
    pm90 = df[df["Threshold"] == "PM2.5 > 90"]
    cols_to_show = ["Dataset", "Horizon", "Model", "Hits_TP", "Misses_FN", "FalseAlarms_FP", "Precision", "Recall_POD", "F1_Score", "CSI_ThreatScore", "HSS"]
    print(pm90[cols_to_show].to_string(index=False))
    
    print("\n" + "="*85)
    print("SUMMARY COMPARISON: 72-HOUR ALERT SKILL (PM2.5 > 90 ug/m3)")
    print("="*85)
    pm90_72h = pm90[pm90["Horizon"] == "72h"]
    print(pm90_72h[["Dataset", "Model", "Precision", "Recall_POD", "F1_Score", "CSI_ThreatScore", "HSS"]].to_string(index=False))

