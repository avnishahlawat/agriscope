# 🌾 AgriScope — Agricultural Market Price Analyzer & Time-Series Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avnishahlawat/agriscope/blob/main/AgriScope_EDA_and_Modeling_Notebook.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/avnishahlawat/agriscope/blob/main/AgriScope_EDA_and_Modeling_Notebook.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview
**AgriScope** is an end-to-end agricultural market intelligence and time-series forecasting platform designed to solve wholesale mandi price volatility, regional market fragmentation, and distress selling across India. By fusing **1.11M+ daily transaction records from AGMARKNET** with **hyper-local historical weather data from the Open-Meteo API**, AgriScope provides leak-free recursive 1–30 day price forecasts, multi-mandi spatial price spread analytics, and automated decision-support via an interactive **Streamlit** web application.

---

## 👤 Academic Information
- **Learner Name:** Avnish Kumar
- **Roll Number:** 2400320100331
- **Institution:** ABES Engineering College, Ghaziabad
- **Project Supervisor:** Prof. Poonam Jaswani
- **Programme:** HCLTech Career Shaper Programme (III Year Industry Project, 2026–27)
- **Project ID & Track:** Project No. 85 / P_085 (Track 05: Time-Series Market Analytics)

---

## 🚀 Live Demo & Online Notebook Views
Teachers, evaluators, and recruiters can inspect all code, plots, and models directly online:

| Resource | Direct Link | Description |
| :--- | :--- | :--- |
| 📓 **Interactive Notebook (GitHub)** | [View on GitHub](AgriScope_EDA_and_Modeling_Notebook.ipynb) | Full EDA & modeling notebook with 77 outputs rendered in-browser |
| ⚡ **Fast Notebook Render (nbviewer)** | [Open in nbviewer](https://nbviewer.org/github/avnishahlawat/agriscope/blob/main/AgriScope_EDA_and_Modeling_Notebook.ipynb) | High-speed rendering with interactive tables and plots |
| 🚀 **Run in Google Colab** | [Launch in Colab](https://colab.research.google.com/github/avnishahlawat/agriscope/blob/main/AgriScope_EDA_and_Modeling_Notebook.ipynb) | Run models and experiments in the cloud with zero setup |
| 🌐 **Interactive Streamlit App** | `http://localhost:8501` / Deploy on Streamlit Cloud | Live multi-mandi forecasting and spatial spread dashboard |

---

## 📂 Repository Deliverables

```
agriscope/
├── AgriScope_EDA_and_Modeling_Notebook.ipynb  # Primary Jupyter Notebook with full EDA, code & plots
├── app.py                                    # Interactive Streamlit Web Application
├── requirements.txt                          # Python dependencies for local run & Cloud deploy
│
├── Project_Synopsis_AvnishKumar_2400320100331.docx # Official HCLTech Project Synopsis
├── agriscope_p_085_project_report.docx       # 10-Chapter Technical Project Report (Word)
├── AgriScope_EDA_and_Modeling_Notebook.docx  # EDA & Modeling Notebook Document with 5 figures
├── SRS_P085_agriscope.docx                   # System Requirements Specification Document
│
├── dataset.csv                               # Cleaned modeling dataset (5,015 rows, 20 columns)
├── data_dictionary.csv                       # Feature definitions, types, and domain bounds
├── weather_data.csv                          # Open-Meteo hyper-local weather records (4,745 rows)
├── agriscope_model_predictions_output.csv    # 56-day holdout predictions and error residuals
└── assets/                                   # Architectural diagrams and visualization figures
```

---

## 🏛️ System Architecture
The system follows a four-tier pipeline:
1. **Data Ingestion & Auditing Layer:** Ingests 1,118,899 raw AGMARKNET records and queries Open-Meteo REST API across 13 geographic coordinates. Enforces domain check: `Min Price ≤ Modal Price ≤ Max Price`.
2. **Data Engineering & Anti-Leakage Engine:** 15 high-density series (>80% continuity), daily median aggregation, controlled short-gap interpolation (≤3d), and strict `shift(1)` enforcement before calculating rolling windows (7d, 30d).
3. **Modeling & Benchmarking Engine:** Strict chronological 56-day holdout validation (20 June to 14 August 2025; no shuffling). Evaluates Seasonal Naive (Lag-7), SARIMA(1,1,1)×(1,1,1,7), and LightGBM Regressor.
4. **Presentation & Decision-Support Layer:** Interactive Streamlit dashboard with KPI metric cards, Plotly forecast trajectories, and spatial arbitrage radar.

---

## 📊 Academic Benchmark & Results

| Forecasting Model | Scope | MAE (₹/Qtl) | RMSE (₹/Qtl) | MAPE (%) | Core Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Seasonal Naive Baseline (Lag-7)** | Aggregate (15 Series) | ₹269.88 | ₹538.92 | 6.41% | Simple weekly persistence heuristic |
| **LightGBM (Price-Only Features)** | Aggregate (15 Series) | **₹166.22** | ₹317.32 | **4.27%** | **38.4% error reduction** over naive baseline |
| **LightGBM (Price + Weather) [DEPLOYED]** | Aggregate (15 Series) | ₹167.42 | **₹309.60** | 4.35% | **Weather Paradox:** -2.43% tail RMSE reduction during storms |
| **SARIMA (1,1,1)×(1,1,1,7)** | Brinjal – Asansol | ₹440.26 | ₹500.89 | 9.51% | High error due to linear inability to model supply shocks |

---

## 💻 How to Run Locally

### 1. Clone the repository:
```bash
git clone https://github.com/avnishahlawat/agriscope.git
cd agriscope
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard:
```bash
streamlit run app.py
```
The dashboard will open automatically at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud (Free Hosted Link)
1. Fork or push this repository to GitHub under your account (`https://github.com/avnishahlawat/agriscope`).
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository:** `avnishahlawat/agriscope`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**.
6. Within 2 minutes, your dashboard will be permanently live on a public URL (e.g. `https://agriscope.streamlit.app`) to share with your teachers, recruiters, and evaluators!

---

## 📜 Citation & References
1. Directorate of Marketing & Inspection (DMI), Ministry of Agriculture and Farmers Welfare, Govt. of India, [AGMARKNET Portal](https://agmarknet.gov.in).
2. Ke, G. et al. (2017). *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree."* NeurIPS 30, pp. 3146–3154.
3. Box, G. E. P. et al. (2015). *"Time Series Analysis: Forecasting and Control,"* 5th Ed., John Wiley & Sons.
4. Open-Meteo Global Meteorological Database, [Historical Weather API](https://open-meteo.com).
