"""Publication-quality figure generation for Air Quality Forecasting Benchmark.
Outputs high-res PNG (300 DPI) and vector PDF to 'figures/' directory.
Run (from aqi_forecast subfolder):
    python src/make_figures.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Academic Color Palette
COLORS = {
    "Persistence": "#7f7f7f",   # Neutral grey
    "Ridge":       "#1f77b4",   # Deep blue
    "GBM":         "#2ca02c",   # Green
    "LSTM":        "#d62728",   # Crimson red
    "Blend":       "#9467bd",   # Purple
}

# ==================================================================
# FIGURE 1: Network Composition Bias & Monitor Expansion
# ==================================================================
def figure1_network_bias():
    print("Generating Figure 1: Network Expansion Bias...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Station expansion across key cities
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    cities_stations = {
        "Delhi": [10, 8, 17, 37, 37, 36],
        "Mumbai": [0, 0, 0, 1, 10, 10],
        "Bengaluru": [2, 2, 2, 8, 8, 8],
        "Kolkata": [0, 0, 0, 2, 7, 7],
        "Hyderabad": [2, 3, 6, 6, 6, 6],
        "Patna": [1, 1, 1, 1, 4, 6],
        "Lucknow": [2, 2, 4, 4, 5, 4]
    }
    markers = ["o", "s", "^", "v", "D", "p", "h"]
    for (city, counts), m in zip(cities_stations.items(), markers):
        ax1.plot(years, counts, marker=m, linewidth=1.8, markersize=6, label=city)
        
    ax1.set_title("(a) CPCB Station Network Expansion (2015–2020)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Reporting Monitoring Stations")
    ax1.set_xticks(years)
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel B: Level change artifact (All stations vs Core stations)
    cities = ["Patna", "Lucknow", "Delhi", "Gurugram"]
    all_change = [-38.8, -28.8, -21.6, -38.0]
    core_change = [-12.7, -24.6, -22.4, -39.3]
    
    y = np.arange(len(cities))
    height = 0.35
    
    r1 = ax2.barh(y - height/2, all_change, height, label="Uncurated (All Reporting Stations)", color="#e74c3c", alpha=0.85)
    r2 = ax2.barh(y + height/2, core_change, height, label="Curated (Fixed Core Stations)", color="#2980b9", alpha=0.85)
    
    ax2.set_title("(b) Apparent PM2.5 Level Shift: Training vs Test Era", fontweight="bold")
    ax2.set_xlabel("Mean PM2.5 Percentage Change (%)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cities, fontweight="semibold")
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(True, axis="x")
    ax2.legend(frameon=True, loc="lower left", facecolor="white", edgecolor="#cccccc")
    
    # Annotate the Patna discrepancy
    ax2.annotate("Spurious -26.1% drift\ndue to suburban station additions",
                 xy=(-38.8, 0 - height/2), xytext=(-35, 0.4),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                 fontsize=9, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure1_Network_Expansion_Bias.png")
    fig.savefig(f"{OUT_DIR}/Figure1_Network_Expansion_Bias.pdf")
    plt.close()

# ==================================================================
# FIGURE 2: Multi-Horizon Benchmark Error Across Models
# ==================================================================
def figure2_benchmark_errors():
    print("Generating Figure 2: Multi-Horizon Forecast Errors...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Data from benchmark2_fixed_results.txt (Pooled Rolling Folds)
    models = ["Persistence", "Ridge", "GBM", "LSTM", "Blend"]
    horizons = ["24h", "48h", "72h"]
    
    mae_data = {
        "Persistence": [15.49, 20.40, 22.61],
        "Ridge":       [13.99, 17.14, 18.43],
        "GBM":         [13.81, 16.99, 18.21],
        "LSTM":        [14.51, 17.37, 18.65],
        "Blend":       [13.97, 16.96, 18.24],
    }
    
    rmse_data = {
        "Persistence": [28.21, 35.67, 39.15],
        "Ridge":       [25.06, 29.79, 31.68],
        "GBM":         [24.84, 29.43, 31.26],
        "LSTM":        [25.34, 29.41, 31.28],
        "Blend":       [24.82, 29.24, 31.07],
    }
    
    x = np.arange(len(horizons))
    width = 0.16
    
    for i, m in enumerate(models):
        offset = (i - 2) * width
        ax1.bar(x + offset, mae_data[m], width, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax2.bar(x + offset, rmse_data[m], width, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        
    ax1.set_title("(a) Mean Absolute Error (MAE)", fontweight="bold")
    ax1.set_ylabel("MAE ($\mu$g/m$^3$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["24 Hours", "48 Hours", "72 Hours"], fontweight="semibold")
    ax1.set_ylim(10, 25)
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, ncol=3, loc="upper left")
    
    # Statistical significance annotation
    ax1.annotate("Ridge & GBM beat LSTM\n(p < 0.001, DM test)", xy=(0 + width, 14.51), xytext=(0.05, 17.5),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                 fontsize=8.5, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="#cccc00"))

    ax2.set_title("(b) Root Mean Squared Error (RMSE)", fontweight="bold")
    ax2.set_ylabel("RMSE ($\mu$g/m$^3$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["24 Hours", "48 Hours", "72 Hours"], fontweight="semibold")
    ax2.set_ylim(20, 42)
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, ncol=3, loc="upper left")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure2_Benchmark_Error_Profiles.png")
    fig.savefig(f"{OUT_DIR}/Figure2_Benchmark_Error_Profiles.pdf")
    plt.close()

# ==================================================================
# FIGURE 3: Severe Pollution Regime & Learning Curve Scaling
# ==================================================================
def figure3_high_pollution_scaling():
    print("Generating Figure 3: Severe Pollution Performance & Scaling...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Panel A: Bootstrap Difference vs Ridge on High Days (PM2.5 > 90)
    horizons = ["24h", "48h", "72h"]
    x = np.arange(len(horizons))
    
    # From benchmark2_fixed_results.txt (95% block-bootstrap CI)
    # A minus Ridge: negative = better than ridge
    lstm_diff = [-0.54, -1.90, -2.34]
    lstm_err_lo = [-0.54 - (-1.82), -1.90 - (-3.52), -2.34 - (-4.30)]
    lstm_err_hi = [0.71 - (-0.54), -0.45 - (-1.90), -0.69 - (-2.34)]
    
    blend_diff = [-0.86, -1.64, -1.93]
    blend_err_lo = [-0.86 - (-1.55), -1.64 - (-2.56), -1.93 - (-2.96)]
    blend_err_hi = [-0.18 - (-0.86), -0.82 - (-1.64), -0.94 - (-1.93)]
    
    gbm_diff = [-0.01, -0.03, -0.25]
    gbm_err_lo = [-0.01 - (-0.98), -0.03 - (-1.50), -0.25 - (-1.97)]
    gbm_err_hi = [1.00 - (-0.01), 1.27 - (-0.03), 1.47 - (-0.25)]
    
    w = 0.22
    ax1.axhline(0, color="black", linestyle="-", linewidth=1.0)
    
    ax1.errorbar(x - w, lstm_diff, yerr=[lstm_err_lo, lstm_err_hi], fmt="o", color=COLORS["LSTM"],
                 capsize=5, elinewidth=1.6, capthick=1.5, markersize=7, label="LSTM minus Ridge")
    ax1.errorbar(x, blend_diff, yerr=[blend_err_lo, blend_err_hi], fmt="s", color=COLORS["Blend"],
                 capsize=5, elinewidth=1.6, capthick=1.5, markersize=7, label="Blend minus Ridge")
    ax1.errorbar(x + w, gbm_diff, yerr=[gbm_err_lo, gbm_err_hi], fmt="^", color=COLORS["GBM"],
                 capsize=5, elinewidth=1.6, capthick=1.5, markersize=7, label="GBM minus Ridge")
                 
    ax1.set_title("(a) MAE Difference vs Ridge on High Days ($PM_{2.5} > 90\,\mu$g/m$^3$)", fontweight="bold")
    ax1.set_ylabel("$\Delta$ MAE ($\mu$g/m$^3$, Negative = Better than Ridge)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["24 Hours", "48 Hours", "72 Hours"], fontweight="semibold")
    ax1.grid(True)
    ax1.legend(frameon=True, loc="lower left")
    
    # Shaded region indicating significant advantage
    ax1.axhspan(-5, 0, color="#27ae60", alpha=0.08)
    ax1.text(0.5, -4.5, "Advantage Region over Regularized Linear Model", color="#1e8449", fontsize=9, fontstyle="italic")

    # Panel B: Fold-by-fold Learning Curve (LSTM gain vs Training Size)
    # From fold_high_fixed_results.txt
    folds = [1, 2, 3, 4, 5]
    train_origins = [5703, 8064, 10475, 13009, 15543]
    gain_24h = [2.02, 0.50, 1.77, -5.53, -2.20]
    gain_48h = [-1.17, -0.59, -0.35, -7.21, 0.36]
    gain_72h = [-1.62, -1.35, -0.96, -8.11, 1.85]
    
    ax2.plot(train_origins, gain_24h, marker="o", linewidth=1.8, label="24h Lead Time", color="#e67e22")
    ax2.plot(train_origins, gain_48h, marker="s", linewidth=1.8, label="48h Lead Time", color="#2980b9")
    ax2.plot(train_origins, gain_72h, marker="^", linewidth=2.0, label="72h Lead Time", color=COLORS["LSTM"])
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    ax2.set_title("(b) Evolution of High-Pollution LSTM Advantage with Training History", fontweight="bold")
    ax2.set_xlabel("Number of Historical Training Days (Origins)")
    ax2.set_ylabel("LSTM minus Ridge ($\mu$g/m$^3$)")
    ax2.grid(True)
    ax2.legend(frameon=True, loc="upper right")
    
    # Annotate Fold 4 gain
    ax2.annotate("Fold 4 (Winter 2019-20):\n-8.11 $\mu$g/m$^3$ improvement at 72h",
                 xy=(13009, -8.11), xytext=(10000, -9.5),
                 arrowprops=dict(arrowstyle="->", color=COLORS["LSTM"], lw=1.2),
                 fontsize=9, color=COLORS["LSTM"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure3_High_Pollution_and_Scaling.png")
    fig.savefig(f"{OUT_DIR}/Figure3_High_Pollution_and_Scaling.pdf")
    plt.close()

# ==================================================================
# FIGURE 4: Alert Warning Skill (F1-score & Critical Success Index)
# ==================================================================
def figure4_alert_metrics():
    print("Generating Figure 4: Public Health Alert Classification Skill...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    horizons = ["24h", "48h", "72h"]
    x = np.arange(len(horizons))
    w = 0.18
    
    # Alert metrics for PM2.5 > 90 ug/m3 (from OpenAQ extended dataset)
    # Models: Persistence, Ridge, LSTM, Blend
    f1_scores = {
        "Persistence": [0.759, 0.681, 0.651],
        "Ridge":       [0.756, 0.687, 0.657],
        "LSTM":        [0.740, 0.698, 0.685],
        "Blend":       [0.757, 0.700, 0.684],
    }
    
    recalls = {
        "Persistence": [0.760, 0.679, 0.649],
        "Ridge":       [0.702, 0.607, 0.563],
        "LSTM":        [0.702, 0.646, 0.638],
        "Blend":       [0.708, 0.632, 0.611],
    }
    
    for i, (m, vals) in enumerate(f1_scores.items()):
        ax1.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        
    for i, (m, vals) in enumerate(recalls.items()):
        ax2.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        
    ax1.set_title("(a) Alert F1-Score ($PM_{2.5} > 90\,\mu$g/m$^3$)", fontweight="bold")
    ax1.set_ylabel("F1-Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["24 Hours", "48 Hours", "72 Hours"], fontweight="semibold")
    ax1.set_ylim(0.55, 0.82)
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, ncol=2, loc="upper right")
    
    # Highlight 72h F1 advantage
    ax1.annotate("LSTM leads at 72h\n(F1 = 0.685)", xy=(2 + 0.5 * w, 0.685), xytext=(1.5, 0.75),
                 arrowprops=dict(arrowstyle="->", color=COLORS["LSTM"], lw=1.2),
                 fontsize=8.5, fontweight="bold", color=COLORS["LSTM"])

    ax2.set_title("(b) Alert Recall / Sensitivity ($PM_{2.5} > 90\,\mu$g/m$^3$)", fontweight="bold")
    ax2.set_ylabel("Recall (True Positive Rate)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["24 Hours", "48 Hours", "72 Hours"], fontweight="semibold")
    ax2.set_ylim(0.50, 0.82)
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, ncol=2, loc="upper right")
    
    # Highlight Ridge recall collapse
    ax2.annotate("Ridge misses 44% of\nevents at 72h (Recall = 0.563)", xy=(2 - 0.5 * w, 0.563), xytext=(1.1, 0.53),
                 arrowprops=dict(arrowstyle="->", color=COLORS["Ridge"], lw=1.2),
                 fontsize=8.5, fontweight="bold", color=COLORS["Ridge"])

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure4_Alert_Warning_Skill.png")
    fig.savefig(f"{OUT_DIR}/Figure4_Alert_Warning_Skill.pdf")
    plt.close()

# ==================================================================
# FIGURE 5: Structural Shock Diagnostic (COVID-19 Lockdown)
# ==================================================================
def figure5_lockdown_shock():
    print("Generating Figure 5: COVID-19 Lockdown Structural Shock...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Panel A: Pre-lockdown vs Lockdown MAE breakdown (from analysis_fixed_results.txt)
    periods = ["Pre-Lockdown\n(Jan 1 – Mar 24, 2020)", "Lockdown Period\n(Mar 25 – Jun 30, 2020)"]
    x = np.arange(len(periods))
    w = 0.18
    
    # MAE at 24h
    mae_24h_periods = {
        "Persistence": [16.24, 8.88],
        "Ridge":       [13.80, 7.75],
        "GBM":         [13.27, 7.86],
        "LSTM":        [13.58, 9.22],
        "Blend":       [13.51, 8.28]
    }
    
    for i, (m, vals) in enumerate(mae_24h_periods.items()):
        ax1.bar(x + (i - 2) * w, vals, w, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        
    ax1.set_title("(a) 24h Error Surge During Abrupt Distributional Shift", fontweight="bold")
    ax1.set_ylabel("24h MAE ($\mu$g/m$^3$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontweight="semibold")
    ax1.set_ylim(5, 18)
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, ncol=3, loc="upper right")
    
    # Annotate LSTM degradation
    ax1.annotate("LSTM error surges +1.47 $\mu$g/m$^3$\nabove Ridge during lockdown (p < 0.001)",
                 xy=(1 + w, 9.22), xytext=(0.55, 12.0),
                 arrowprops=dict(arrowstyle="->", color=COLORS["LSTM"], lw=1.2),
                 fontsize=8.5, fontweight="bold", color=COLORS["LSTM"],
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#cc0000"))

    # Panel B: Time-series window around lockdown for Delhi
    pred_file = "benchmark2_predictions_fixed.csv"
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file, parse_dates=["date"])
        sub = df[(df.city == "Delhi") & (df.h == 1) & (df.date >= "2020-02-15") & (df.date <= "2020-05-15")].sort_values("date")
        
        ax2.plot(sub.date, sub.y, label="Observed PM2.5", color="black", linewidth=2.0)
        ax2.plot(sub.date, sub.Ridge, label="Ridge (24h)", color=COLORS["Ridge"], linestyle="--", linewidth=1.5)
        ax2.plot(sub.date, sub.LSTM, label="LSTM (24h)", color=COLORS["LSTM"], linestyle="-", linewidth=1.6)
        
        # Mark Lockdown date
        lock_date = pd.to_datetime("2020-03-25")
        ax2.axvline(lock_date, color="#c0392b", linestyle=":", linewidth=2.0)
        ax2.text(lock_date + pd.Timedelta(days=2), 170, "National Lockdown\n(March 25, 2020)",
                 color="#c0392b", fontweight="bold", fontsize=9)
                 
        ax2.set_title("(b) 24h Forecast Tracking in Delhi Across Lockdown Boundary", fontweight="bold")
        ax2.set_ylabel("PM2.5 ($\mu$g/m$^3$)")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.legend(frameon=True, loc="upper right")
        fig.autofmt_xdate()
    else:
        ax2.text(0.5, 0.5, "benchmark2_predictions_fixed.csv not found", ha="center")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure5_COVID_Lockdown_Shock.png")
    fig.savefig(f"{OUT_DIR}/Figure5_COVID_Lockdown_Shock.pdf")
    plt.close()

if __name__ == "__main__":
    print(f"Generating all publication figures into: {os.path.abspath(OUT_DIR)}")
    figure1_network_bias()
    figure2_benchmark_errors()
    figure3_high_pollution_scaling()
    figure4_alert_metrics()
    figure5_lockdown_shock()
    print("All figures successfully created in PNG (300 DPI) and PDF!")

