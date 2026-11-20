"""Publication-quality figure generation matching exact user specifications.
Generates 4 high-resolution (300 DPI) PNG and vector PDF figures:
- Figure 1: The Network Expansion Artifact & Study Map
- Figure 2: Multi-Horizon Benchmark Performance (24h, 48h, 72h)
- Figure 3: Severe Pollution Advantage (PM2.5 > 90 ug/m3) & Learning Curve
- Figure 4: The COVID-19 Lockdown Shock Response
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
    "legend.fontsize": 9.5,
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
    "GBM":         "#2ca02c",   # Forest green
    "LSTM":        "#d62728",   # Crimson red
    "Blend":       "#9467bd",   # Purple
}

# ==================================================================
# FIGURE 1: The Network Expansion Artifact & Study Map
# ==================================================================
def figure1_study_map_and_network_bias():
    print("Generating Figure 1: Study Map & Network Expansion Bias...")
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_lines = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[0, 2])

    # ----------------- Panel A: Study Map of 14 Fixed Panel Cities
    india_polygon = [
        (68.1, 23.7), (68.8, 24.5), (70.5, 24.6), (71.0, 24.0), (70.2, 22.8), (69.0, 22.4),
        (69.2, 20.8), (72.8, 21.0), (72.8, 19.0), (73.5, 16.0), (74.8, 14.0), (76.0, 10.5),
        (77.5, 8.1), (78.2, 9.2), (79.8, 10.3), (80.3, 13.0), (80.3, 16.0), (82.5, 17.5),
        (84.0, 18.5), (85.8, 19.8), (87.0, 21.5), (88.2, 21.8), (89.0, 22.0), (88.9, 24.0),
        (92.0, 25.5), (93.5, 24.0), (92.5, 22.0), (94.0, 22.0), (95.0, 24.5), (97.0, 27.5),
        (96.0, 28.5), (94.5, 28.5), (92.0, 27.8), (89.0, 27.3), (88.2, 27.8), (88.0, 26.5),
        (85.0, 27.5), (83.0, 28.0), (81.0, 30.0), (79.0, 31.0), (77.8, 32.5), (78.0, 35.5),
        (76.5, 36.5), (74.5, 35.5), (74.0, 33.5), (74.8, 32.5), (74.5, 31.5), (73.8, 30.0),
        (72.5, 30.0), (71.0, 28.0), (70.0, 27.0), (70.5, 25.0), (68.1, 23.7)
    ]
    poly = Polygon(india_polygon, closed=True, facecolor="#f3f5f8", edgecolor="#8fa1b3", linewidth=1.2, zorder=1)
    ax_map.add_patch(poly)

    # 14 Fixed Panel Cities with Coordinates, Mean PM2.5, and Core Stations
    cities_meta = [
        ("Delhi", 28.61, 77.21, 96.4, 12, "top"),
        ("Patna", 25.59, 85.14, 114.5, 1, "top"),
        ("Lucknow", 26.85, 80.95, 103.7, 3, "bottom"),
        ("Gurugram", 28.46, 77.03, 96.1, 2, "left"),
        ("Ahmedabad", 23.02, 72.57, 63.4, 1, "left"),
        ("Brajrajnagar", 21.82, 83.92, 53.2, 1, "right"),
        ("Amritsar", 31.63, 74.87, 46.9, 1, "top"),
        ("Jaipur", 26.91, 75.79, 46.4, 3, "left"),
        ("Visakhapatnam", 17.69, 83.22, 44.5, 1, "right"),
        ("Chennai", 13.08, 80.27, 44.3, 3, "right"),
        ("Hyderabad", 17.38, 78.48, 40.9, 6, "left"),
        ("Amaravati", 16.51, 80.52, 34.9, 1, "bottom"),
        ("Bengaluru", 12.97, 77.59, 34.1, 2, "left"),
        ("Thiruvananthapuram", 8.52, 76.94, 25.8, 1, "bottom")
    ]
    
    lons = [c[2] for c in cities_meta]
    lats = [c[1] for c in cities_meta]
    pm_means = [c[3] for c in cities_meta]
    core_stations = [c[4] for c in cities_meta]
    
    # Bubble scatter: size by core stations, color by mean PM2.5
    scatter = ax_map.scatter(lons, lats, c=pm_means, cmap="YlOrRd", s=[35 + s * 16 for s in core_stations],
                             edgecolor="#222222", linewidth=1.0, alpha=0.92, zorder=3)
    
    # Selected clean labels to avoid crowding
    label_offsets = {
        "Delhi": (0.8, 0.8), "Patna": (1.0, 0.3), "Lucknow": (0.8, -1.2),
        "Ahmedabad": (-1.0, -1.0), "Bengaluru": (-1.0, -0.9), "Chennai": (0.8, -0.3),
        "Hyderabad": (-4.5, 0.2), "Amritsar": (-3.8, 0.4), "Thiruvananthapuram": (0.8, -0.6)
    }
    for name, lat, lon, pm, st_cnt, pos in cities_meta:
        if name in label_offsets:
            ox, oy = label_offsets[name]
            ax_map.text(lon + ox, lat + oy, f"{name}", fontsize=8, fontweight="bold", color="#111111", zorder=4)

    cbar = plt.colorbar(scatter, ax=ax_map, orientation="horizontal", pad=0.04, shrink=0.82)
    cbar.set_label(r"Mean $PM_{2.5}$ ($\mu$g/m$^3$)", fontsize=9)
    ax_map.set_xlim(67, 98)
    ax_map.set_ylim(7, 37.5)
    ax_map.set_xlabel("Longitude (°E)", fontsize=9.5)
    ax_map.set_ylabel("Latitude (°N)", fontsize=9.5)
    ax_map.set_title("(a) Study Area: 14 Fixed-Station Panel Cities", fontweight="bold")
    ax_map.grid(True, linestyle=":", alpha=0.5)

    # ----------------- Panel B: Line plot showing station growth over time
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
        ax_lines.plot(years, counts, marker=m, linewidth=1.8, markersize=5.5, label=city)
        
    ax_lines.set_title("(b) CPCB Monitor Expansion (2015–2020)", fontweight="bold")
    ax_lines.set_xlabel("Year")
    ax_lines.set_ylabel("Reporting Monitoring Stations")
    ax_lines.set_xticks(years)
    ax_lines.grid(True)
    ax_lines.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="upper left")

    # ----------------- Panel C: Bar chart showing level change artifact
    cities_comp = ["Patna", "Lucknow", "Delhi", "Gurugram"]
    all_change = [-38.8, -28.8, -21.6, -38.0]
    core_change = [-12.7, -24.6, -22.4, -39.3]
    
    y = np.arange(len(cities_comp))
    height = 0.35
    
    ax_bars.barh(y - height/2, all_change, height, label="Uncurated (All Stations)", color="#e74c3c", alpha=0.85)
    ax_bars.barh(y + height/2, core_change, height, label="Curated (Fixed Core)", color="#2980b9", alpha=0.85)
    
    ax_bars.set_title("(c) Level Shift Bias: Training vs Test Era", fontweight="bold")
    ax_bars.set_xlabel(r"Mean $PM_{2.5}$ Change (%)")
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(cities_comp, fontweight="semibold")
    ax_bars.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax_bars.grid(True, axis="x")
    ax_bars.legend(frameon=True, loc="lower left", facecolor="white", edgecolor="#cccccc", fontsize=8.5)
    
    # Annotate Patna
    ax_bars.annotate("Spurious -26.1% drop\ndue to suburban stations",
                     xy=(-38.8, 0 - height/2), xytext=(-36, 0.45),
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                     fontsize=8.5, color="#c0392b", fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure1_StudyMap_and_NetworkBias.png")
    fig.savefig(f"{OUT_DIR}/Figure1_StudyMap_and_NetworkBias.pdf")
    plt.close()

# ==================================================================
# FIGURE 2: Multi-Horizon Benchmark Performance (24h, 48h, 72h)
# ==================================================================
def figure2_multi_horizon_benchmark():
    print("Generating Figure 2: Multi-Horizon Benchmark Performance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    horizons = ["24h Lead", "48h Lead", "72h Lead"]
    models = ["Persistence", "Ridge", "GBM", "LSTM", "Blend"]
    
    # Pooled Rolling-Fold Results (Fixed Panel, 9600+ windows per horizon)
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
        
    ax1.set_title(r"(a) Mean Absolute Error (MAE)", fontweight="bold")
    ax1.set_ylabel(r"MAE ($\mu$g/m$^3$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons, fontweight="semibold")
    ax1.set_ylim(10, 25)
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, ncol=3, loc="upper left")
    
    # Statistical significance annotation
    ax1.annotate("Ridge & GBM beat LSTM\n(p < 0.001, DM test)", xy=(0 + width, 14.51), xytext=(0.02, 17.8),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                 fontsize=8.5, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="#cccc00"))

    ax2.set_title(r"(b) Root Mean Squared Error (RMSE)", fontweight="bold")
    ax2.set_ylabel(r"RMSE ($\mu$g/m$^3$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons, fontweight="semibold")
    ax2.set_ylim(20, 42)
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, ncol=3, loc="upper left")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure2_Benchmark_Error_Profiles.png")
    fig.savefig(f"{OUT_DIR}/Figure2_Benchmark_Error_Profiles.pdf")
    plt.close()

# ==================================================================
# FIGURE 3: Severe Pollution Advantage & Learning Curve
# ==================================================================
def figure3_severe_pollution_and_scaling():
    print("Generating Figure 3: Severe Pollution Advantage & Learning Curve...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Panel A: Bootstrap Difference vs Ridge on High Days (PM2.5 > 90 ug/m3)
    horizons = ["24h Lead", "48h Lead", "72h Lead"]
    x = np.arange(len(horizons))
    
    # 95% moving block bootstrap CI
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
                 
    ax1.set_title(r"(a) MAE Difference vs Ridge on High Days ($PM_{2.5} > 90\,\mu$g/m$^3$)", fontweight="bold")
    ax1.set_ylabel(r"$\Delta$ MAE ($\mu$g/m$^3$, Negative = Better than Ridge)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons, fontweight="semibold")
    ax1.grid(True)
    ax1.legend(frameon=True, loc="lower left")
    
    # Shaded region
    ax1.axhspan(-5, 0, color="#27ae60", alpha=0.08)
    ax1.text(0.4, -4.5, "Advantage Region over Regularized Linear Model", color="#1e8449", fontsize=9, fontstyle="italic")

    # Panel B: Learning curve across rolling folds
    train_origins = [5703, 8064, 10475, 13009, 15543]
    gain_24h = [2.02, 0.50, 1.77, -5.53, -2.20]
    gain_48h = [-1.17, -0.59, -0.35, -7.21, 0.36]
    gain_72h = [-1.62, -1.35, -0.96, -8.11, 1.85]
    
    ax2.plot(train_origins, gain_24h, marker="o", linewidth=1.8, label="24h Horizon", color="#e67e22")
    ax2.plot(train_origins, gain_48h, marker="s", linewidth=1.8, label="48h Horizon", color="#2980b9")
    ax2.plot(train_origins, gain_72h, marker="^", linewidth=2.0, label="72h Horizon", color=COLORS["LSTM"])
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    ax2.set_title(r"(b) High-Pollution LSTM Gain vs Training History", fontweight="bold")
    ax2.set_xlabel("Historical Training Window (Total Days / Origins)")
    ax2.set_ylabel(r"LSTM minus Ridge MAE ($\mu$g/m$^3$)")
    ax2.grid(True)
    ax2.legend(frameon=True, loc="upper right")
    
    ax2.annotate("Fold 4 (Winter 2019-20):\n-8.11 $\mu$g/m$^3$ gain at 72h",
                 xy=(13009, -8.11), xytext=(9800, -9.6),
                 arrowprops=dict(arrowstyle="->", color=COLORS["LSTM"], lw=1.2),
                 fontsize=8.5, color=COLORS["LSTM"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure3_Severe_Pollution_and_Scaling.png")
    fig.savefig(f"{OUT_DIR}/Figure3_Severe_Pollution_and_Scaling.pdf")
    plt.close()

# ==================================================================
# FIGURE 4: The COVID-19 Lockdown Shock Response
# ==================================================================
def figure4_covid_lockdown_shock():
    print("Generating Figure 4: COVID-19 Lockdown Shock Response...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # Panel A: Time series of observed vs predicted around lockdown onset for Delhi
    pred_file = "benchmark2_predictions_fixed.csv"
    if not os.path.exists(pred_file):
        pred_file = os.path.join("aqi_forecast", pred_file)
        
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file, parse_dates=["date"])
        sub = df[(df.city == "Delhi") & (df.h == 1) & (df.date >= "2020-02-15") & (df.date <= "2020-05-15")].sort_values("date")
        
        ax1.plot(sub.date, sub.y, label="Observed Ground Truth", color="black", linewidth=2.0)
        ax1.plot(sub.date, sub.Ridge, label="Ridge (24h Forecast)", color=COLORS["Ridge"], linestyle="--", linewidth=1.6)
        ax1.plot(sub.date, sub.LSTM, label="LSTM (24h Forecast)", color=COLORS["LSTM"], linestyle="-", linewidth=1.6)
        
        lock_date = pd.to_datetime("2020-03-25")
        ax1.axvline(lock_date, color="#c0392b", linestyle=":", linewidth=2.0)
        ax1.text(lock_date + pd.Timedelta(days=2), 175, "National Lockdown\n(March 25, 2020)",
                 color="#c0392b", fontweight="bold", fontsize=9)
                 
        ax1.set_title(r"(a) 24h Forecast Tracking in Delhi Across Lockdown", fontweight="bold")
        ax1.set_ylabel(r"$PM_{2.5}$ ($\mu$g/m$^3$)")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.legend(frameon=True, loc="upper right")
        fig.autofmt_xdate()
    else:
        ax1.text(0.5, 0.5, "Prediction data not found", ha="center")

    # Panel B: Quantitative error surge comparison (Pre-Lockdown vs Lockdown)
    periods = ["Pre-Lockdown\n(Jan 1 – Mar 24, 2020)", "Lockdown Period\n(Mar 25 – Jun 30, 2020)"]
    x = np.arange(len(periods))
    w = 0.17
    
    mae_24h_periods = {
        "Persistence": [16.24, 8.88],
        "Ridge":       [13.80, 7.75],
        "GBM":         [13.27, 7.86],
        "LSTM":        [13.58, 9.22],
        "Blend":       [13.51, 8.28]
    }
    
    for i, m in enumerate(["Persistence", "Ridge", "GBM", "LSTM", "Blend"]):
        ax2.bar(x + (i - 2) * w, mae_24h_periods[m], w, label=m, color=COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        
    ax2.set_title(r"(b) 24h Error Surge Under Abrupt Out-of-Distribution Shift", fontweight="bold")
    ax2.set_ylabel(r"24h MAE ($\mu$g/m$^3$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontweight="semibold")
    ax2.set_ylim(5, 18.5)
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, ncol=3, loc="upper right")
    
    ax2.annotate("LSTM error surges +1.47 $\mu$g/m$^3$\nabove Ridge during lockdown (p < 0.001)",
                 xy=(1 + w, 9.22), xytext=(0.52, 12.5),
                 arrowprops=dict(arrowstyle="->", color=COLORS["LSTM"], lw=1.2),
                 fontsize=8.5, fontweight="bold", color=COLORS["LSTM"],
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#cc0000"))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/Figure4_COVID_Lockdown_Shock.png")
    fig.savefig(f"{OUT_DIR}/Figure4_COVID_Lockdown_Shock.pdf")
    plt.close()

if __name__ == "__main__":
    print(f"Generating all quoted publication figures into: {os.path.abspath(OUT_DIR)}")
    figure1_study_map_and_network_bias()
    figure2_multi_horizon_benchmark()
    figure3_severe_pollution_and_scaling()
    figure4_covid_lockdown_shock()
    print("All 4 figures successfully generated in 300 DPI PNG and vector PDF!")

