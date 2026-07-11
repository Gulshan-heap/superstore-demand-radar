# 📊 Superstore Demand Radar

An end-to-end sales forecasting and demand intelligence system built on four years of Superstore retail transaction data. The project covers exploratory analysis, time-series decomposition, multi-model forecasting, anomaly detection, product demand segmentation, and a deployed interactive dashboard — the kind of pipeline a retail data science team would build to answer one question: *how much of each product will we sell next month, and are we stocked to meet it?*

---

## 🔗 Live Demo

**Streamlit Dashboard:** https://superstore-demand-radar-svczmby8rqlmlhrn48xvcu.streamlit.app/

**Notebook (Colab):** https://colab.research.google.com/drive/1G34xKYt5WmecpzgwZ0R9D6cApcfiPq5L?usp=sharing
---

## 📁 Repository Structure

```
superstore-demand-radar/
├── analysis.ipynb        # Full notebook — EDA, forecasting, anomaly detection, clustering
├── app.py                # Streamlit dashboard (4-page interactive app)
├── train.csv             # Superstore sales dataset (raw input)
├── requirements.txt       # Python dependencies
├── summary.pdf            # 2-page executive business report
└── charts/                 # Saved chart images (.png) referenced in the notebook
```

---

## 🎯 Problem Statement

Retail demand planning sits between two costly failure modes: overstocking, which wastes capital and warehouse space, and understocking, which loses sales and customers. This project builds a forecasting and monitoring system to help a supply chain team avoid both — combining classical statistics, machine learning, and an interactive dashboard into one pipeline.

---

## 🧩 What's Inside

### 1. Data Loading, Merging & Deep Exploration
- Parses and cleans 4 years of daily Superstore order data
- Extracts time-based features (year, month, week, quarter, season)
- Aggregates sales into weekly and monthly granularities
- Answers core business questions: top revenue category, most consistent regional growth, shipping delay by region, and seasonal spikes

### 2. Time Series Analysis & Decomposition
- Decomposes monthly sales into trend, seasonal, and residual components (`statsmodels`)
- Runs the Augmented Dickey-Fuller (ADF) test for stationarity, with plain-English interpretation
- Applies differencing where required

### 3. Sales Forecasting — 3 Models Compared
| Model | Type | Role |
|---|---|---|
| **SARIMA** | Statistical | Secondary / backup forecast |
| **Facebook Prophet** | Industry-standard forecasting | **Primary production model** |
| **XGBoost** | ML (lag-feature regression) | Benchmark comparison |

Each model is backtested against held-out historical data (for honest MAE / RMSE / MAPE) and separately refit on the full dataset to generate a genuine 3-month-ahead forecast with confidence intervals.

### 4. Category & Region-Level Forecasting
Repeats the best-performing model (Prophet) across Furniture, Technology, Office Supplies, West, and East segments to compare growth trajectories side by side.

### 5. Anomaly Detection
Two independent methods, cross-compared:
- **Isolation Forest** — flags global, all-time extreme sales weeks
- **Rolling Z-Score** — flags sudden local deviations from recent (8-week) trends

Each flagged anomaly is paired with a plain-language, real-world explanation (e.g., holiday surges, end-of-quarter B2B ordering, post-holiday troughs).

### 6. Product Demand Segmentation
- K-Means clustering (with the Elbow Method for choosing K) groups product sub-categories by volume, growth rate, volatility, and average order value
- PCA visualization of the resulting clusters
- Each cluster mapped to a concrete inventory strategy (JIT, safety-stock buffers, automated reorder points, or SKU rationalization)

### 7. Interactive Dashboard (Streamlit)
A 4-page live app (`app.py`):
- **Sales Overview** — annual/monthly trends, filterable by category and region
- **Forecast Explorer** — pick a category or region, choose a 1–3 month horizon, and see a live Prophet forecast with segment-specific MAE/RMSE
- **Anomaly Report** — toggle between Isolation Forest, Rolling Z-Score, or their high-confidence overlap
- **Product Demand Segments** — live K-Means clustering with per-cluster stocking recommendations

### 8. Executive Business Report
A 2-page, non-technical report (`summary.pdf`) written for a Head of Supply Chain and CFO audience — covering the forecast, top anomalies, segmentation findings, and data-backed recommendations.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data handling | Pandas, NumPy |
| Time series | Statsmodels (SARIMA, seasonal decomposition, ADF test) |
| Forecasting | Facebook Prophet, XGBoost |
| ML / Clustering | Scikit-learn (Isolation Forest, K-Means, PCA) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |

---

## 🚀 Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/Gulshan-heap/superstore-demand-radar.git
cd superstore-demand-radar
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook analysis.ipynb
```

**4. Run the dashboard**
```bash
streamlit run app.py
```
Make sure `train.csv` is in the same folder as `app.py` — the app reads it directly on load.

---

## 📈 Model Performance Summary

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | $15,448.15 | $18,565.97 | 18.26% |
| **Prophet** | **$14,501.23** | $19,156.48 | **17.76%** |
| XGBoost | $22,817.62 | $26,383.53 | 26.99% |

**Prophet** is used as the production forecasting model based on the lowest average error and percentage error across backtesting. XGBoost consistently underestimates peak-season demand since tree-based models can't natively extrapolate beyond their training range.

---

## ⚠️ Known Limitations

- Forecasts are based entirely on historical patterns and cannot anticipate one-off shocks (supply disruptions, sudden policy changes, new competitors).
- XGBoost forecasts beyond the training window require recursive lag prediction, which compounds error over longer horizons.
- Anomaly detection thresholds (Isolation Forest contamination rate, Z-score window size) are configured based on this dataset's scale and may need retuning for other retailers or time spans.

---

## 📄 License

This project was built as part of an internship data science assignment. Dataset sourced from the [Kaggle Superstore Sales dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting).
