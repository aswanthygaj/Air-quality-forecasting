# Urban Air Quality Forecasting Across Indian Cities: Empirical Benchmark & Extreme Pollution Dynamics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A multi-horizon (24h, 48h, 72h) empirical forecasting benchmark across 14 to 21 Indian cities evaluating Persistence, Regularized Linear Models (Ridge), Gradient Boosted Decision Trees (HistGradientBoosting), Recurrent Neural Networks (LSTM Multi-Seed Ensembles), and Linear–Neural Hybrids under peak pollution accumulation and structural socio-economic regime shifts.

---

## 🌟 Key Scientific Findings

1. **The "Baseline Reality" on Routine Days:** Contrary to common claims in literature, tuned regularized Ridge regression and Gradient Boosted Trees consistently outperform or match deep LSTM sequence models on routine background days at 24h lead times (MAE: **13.99** and **13.81** vs. **14.51** $\mu\text{g/m}^3$, $p < 0.001$, Diebold–Mariano test with Holm correction).
2. **The "High-Pollution Niche" ($PM_{2.5} > 90\,\mu\text{g/m}^3$):** LSTMs demonstrate statistically significant, robust superiority during severe multi-day accumulation events where non-linear atmospheric inversions dominate, yielding error reductions of **$-1.90\,\mu\text{g/m}^3$ at 48h** and **$-2.34\,\mu\text{g/m}^3$ at 72h** (95% moving-block bootstrap CIs).
3. **Public Health Early Warning Skill:** At a 72-hour lead time on extreme hazardous days ($PM_{2.5} > 120\,\mu\text{g/m}^3$), Ridge regression misses **52.7%** of events (Recall = 0.473) due to L2 penalty shrinkage. In contrast, the LSTM captures **58.8%** of these events (Recall = 0.588, CSI = 0.462), providing an **11.5 percentage point increase in alert sensitivity**.
4. **The "Network Expansion Bias" Artifact:** Public city-aggregated monitoring datasets (CPCB / Kaggle) suffer from non-stationary spatial aggregation bias: rapid sensor network expansion (e.g., Delhi growing from 10 to 37 stations, Patna from 1 to 6) introduces an artificial downward trend in uncurated city means (e.g., a spurious -38.8% decline in Patna, vs. only -12.7% on core continuous monitors). We establish a continuous fixed-station curation protocol (`city_day_fixed.csv`) that eliminates this artifact.
5. **Structural Shock Resilience (COVID-19 Natural Experiment):** Under the abrupt March 2020 national lockdown, regularized linear models adapted immediately, whereas neural networks suffered severe out-of-distribution forecast lag (+1.47 $\mu\text{g/m}^3$ error surge) due to memorized historical seasonal priors.

---

## 📊 Summary of Main Results (Pooled Rolling Folds)

| Model | MAE 24h ($\mu\text{g/m}^3$) | MAE 48h ($\mu\text{g/m}^3$) | MAE 72h ($\mu\text{g/m}^3$) | MAE High ($>90\,\mu\text{g/m}^3$) | 72h Alert Recall ($>90\,\mu\text{g/m}^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 15.49 | 20.40 | 22.61 | 45.46 | 0.715 |
| **Ridge Regression** | 13.99 | 17.14 | 18.43 | 40.31 | 0.712 |
| **Gradient Boosting (GBM)** | **13.81** | **16.99** | **18.21** | 40.22 | 0.718 |
| **LSTM (3-Seed Ensemble)** | 14.51 | 17.37 | 18.65 | **38.72** | **0.763** |
| **Linear–LSTM Blend (50/50)** | 13.97 | 16.96 | 18.24 | 38.84 | 0.749 |

---

## 📁 Repository Structure

```
.
├── figures/                          # 300 DPI publication figures & verification tables
│   ├── Figure1_StudyMap_and_NetworkBias.png (.pdf)
│   ├── Figure2_Benchmark_Error_Profiles.png (.pdf)
│   ├── Figure3_Severe_Pollution_and_Scaling.png (.pdf)
│   ├── Figure4_COVID_Lockdown_Shock.png (.pdf)
│   ├── alert_metrics_pm90.csv
│   └── computational_resource_table.csv
├── aqi_forecast/                     # Core experiment and codebase
│   ├── data/                         # Datasets & fixed panel records
│   │   ├── city_day.csv              # Raw CPCB city-day records
│   │   ├── city_day_fixed.csv        # Continuous fixed-station curated panel
│   │   ├── city_day_oaq.csv          # Extended OpenAQ multi-era panel (2018–2026)
│   │   └── weather.csv               # Open-Meteo reanalysis weather features
│   ├── models/                       # Saved PyTorch LSTM checkpoints & meta
│   ├── src/                          # Modular Python source code
│   │   ├── common.py                 # Feature transforms & spatial groupings
│   │   ├── final_lib.py              # Causal windowing & inference bundle
│   │   ├── model.py                  # PyTorch LSTM sequence architecture
│   │   ├── build_fixed_panel.py      # Station continuity filtering algorithm
│   │   ├── benchmark2.py             # 5 rolling temporal folds & tuning pipeline
│   │   ├── analysis.py               # Diebold-Mariano tests & bootstrap CIs
│   │   ├── make_quoted_figures.py    # Figure generation pipeline
│   │   └── compute_alert_f1_csi.py   # Public health contingency table metrics
│   ├── app.py                        # Streamlit interactive forecasting dashboard
│   ├── requirements.txt              # Environment dependencies
│   └── README.md                     # Local run instructions
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/thuliraj19/Air-quality-forecasting.git
cd Air-quality-forecasting/aqi_forecast
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Benchmark
Execute the rolling-fold evaluation across all models:
```bash
python src/benchmark2.py
```

### 3. Generate Publication Figures & Verification Tables
To re-generate all publication-quality vector PDFs, 300 DPI PNGs, and classification tables:
```bash
python src/make_quoted_figures.py
python src/compute_alert_f1_csi.py
python src/compute_resource_table.py
```

### 4. Launch the Interactive Dashboard
To explore live and rolling predictions across all Indian cities in a browser:
```bash
streamlit run app.py
```
---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
