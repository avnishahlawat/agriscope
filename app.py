import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriScope — Agricultural Market Price Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Scoped so it does not interfere with st.metric or st.dataframe)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2e7d32;
        border-bottom: 2px solid #c8e6c9;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .badge-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #43a047;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Loader
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "agriscope_processed_model_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Processed dataset '{file_path}' not found. Please run the notebook/pipeline first.")
        st.stop()
    df = pd.read_csv(file_path)
    df["Price Date"] = pd.to_datetime(df["Price Date"])
    return df

@st.cache_resource
def load_model():
    model_path = "lgb_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

data_df = load_data()
model = load_model()

# ---------------------------------------------------------
# Sidebar: User Controls
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🌾 AgriScope Controls")

commodities = sorted(data_df["Commodity"].unique().tolist())
selected_commodity = st.sidebar.selectbox("Select Commodity", commodities, index=0)

available_markets = sorted(
    data_df[data_df["Commodity"] == selected_commodity]["Market Name"].unique().tolist()
)
selected_market = st.sidebar.selectbox("Select Mandi (Market)", available_markets, index=0)

# Replace slider with number_input to prevent dynamic JS module import errors
forecast_days = st.sidebar.number_input(
    "Forecast Horizon (days)",
    min_value=1,
    max_value=30,
    value=7,
    step=1,
    help="Select how many future days to forecast using recursive LightGBM."
)

show_weather = st.sidebar.checkbox("Show Historical Weather Overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** AgriScope (P_085)  
**Dataset:** AGMARKNET & Open-Meteo  
**Models:** Seasonal Naive, LightGBM, SARIMA  
""")

# ---------------------------------------------------------
# Filter Historical Data for Selected Series
# ---------------------------------------------------------
series_df = data_df[
    (data_df["Commodity"] == selected_commodity) & 
    (data_df["Market Name"] == selected_market)
].sort_values("Price Date").copy()

if series_df.empty:
    st.warning("No data found for the selected Commodity and Market combination.")
    st.stop()

# ---------------------------------------------------------
# Header & Context
# ---------------------------------------------------------
st.markdown('<div class="main-title">🌾 AgriScope — Agricultural Market Price Analyzer</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Real-Time Mandi Analysis, Micro-Climate Correlation & ML Price Forecasting for <b>{selected_commodity}</b> at <b>{selected_market}</b></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 1: Key Performance Indicators (Native st.metric)
# ---------------------------------------------------------
latest_row = series_df.iloc[-1]
latest_price = latest_row["Modal_Price"]
latest_date = latest_row["Price Date"]

# Calculate 7-day change
if len(series_df) >= 8:
    prev_7_price = series_df.iloc[-8]["Modal_Price"]
    pct_change_7d = ((latest_price - prev_7_price) / prev_7_price) * 100
    delta_7d = f"{pct_change_7d:+.1f}% vs 7d ago"
else:
    delta_7d = None

mandi_min_price = series_df["Modal_Price"].min()
mandi_max_price = series_df["Modal_Price"].max()
price_spread = mandi_max_price - mandi_min_price

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Latest Modal Price",
        value=f"₹{latest_price:,.0f}/quintal",
        delta=delta_7d
    )

with kpi2:
    st.metric(
        label="Reporting Period",
        value=f"{len(series_df)} Days",
        delta=f"From {series_df['Price Date'].min().strftime('%d %b %Y')}"
    )

with kpi3:
    st.metric(
        label="Historical Price Range",
        value=f"₹{mandi_min_price:,.0f} – ₹{mandi_max_price:,.0f}",
        delta=f"Spread: ₹{price_spread:,.0f}"
    )

with kpi4:
    st.metric(
        label="Latest Record Date",
        value=latest_date.strftime("%d %b %Y"),
        delta="Official Mandi Log"
    )

# ---------------------------------------------------------
# Section 2: Historical Price Trend & Moving Averages
# ---------------------------------------------------------
st.markdown('<div class="section-header">📈 Historical Price Trend & Volatility Analysis</div>', unsafe_allow_html=True)

fig_hist = go.Figure()

fig_hist.add_trace(go.Scatter(
    x=series_df["Price Date"],
    y=series_df["Modal_Price"],
    mode="lines+markers",
    name="Daily Modal Price",
    line=dict(color="#2e7d32", width=2.5),
    marker=dict(size=4)
))

if "Rolling_Mean_7" in series_df.columns:
    fig_hist.add_trace(go.Scatter(
        x=series_df["Price Date"],
        y=series_df["Rolling_Mean_7"],
        mode="lines",
        name="7-Day Moving Average",
        line=dict(color="#1976d2", width=1.8, dash="dash")
    ))

if "Rolling_Mean_30" in series_df.columns:
    fig_hist.add_trace(go.Scatter(
        x=series_df["Price Date"],
        y=series_df["Rolling_Mean_30"],
        mode="lines",
        name="30-Day Moving Average",
        line=dict(color="#f57c00", width=1.8, dash="dot")
    ))

fig_hist.update_layout(
    title=f"Historical Price Movement — {selected_commodity} ({selected_market})",
    xaxis_title="Date",
    yaxis_title="Price (₹ / Quintal)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white"
)

st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# Section 3: Machine Learning Future Price Forecast
# ---------------------------------------------------------
st.markdown(f'<div class="section-header">🤖 ML Price Forecast ({forecast_days}-Day Recursive Horizon)</div>', unsafe_allow_html=True)

def generate_forecast(trained_model, hist_df, commodity, market, horizon):
    prices = hist_df["Modal_Price"].tolist()
    dates = hist_df["Price Date"].tolist()
    last_dt = dates[-1]
    
    forecast_list = []
    
    for step in range(1, horizon + 1):
        target_dt = last_dt + pd.Timedelta(days=step)
        
        lag_1 = prices[-1]
        lag_7 = prices[-7] if len(prices) >= 7 else prices[0]
        lag_14 = prices[-14] if len(prices) >= 14 else prices[0]
        lag_30 = prices[-30] if len(prices) >= 30 else prices[0]
        
        roll_7_m = float(np.mean(prices[-7:]))
        roll_7_s = float(np.std(prices[-7:]))
        roll_30_m = float(np.mean(prices[-30:]))
        roll_30_s = float(np.std(prices[-30:]))
        
        feat_row = {
            "Day_of_Week": [target_dt.dayofweek],
            "Month": [target_dt.month],
            "Week_of_Year": [int(target_dt.isocalendar().week)],
            "Lag_1": [lag_1],
            "Lag_7": [lag_7],
            "Lag_14": [lag_14],
            "Lag_30": [lag_30],
            "Rolling_Mean_7": [roll_7_m],
            "Rolling_Std_7": [roll_7_s],
            "Rolling_Mean_30": [roll_30_m],
            "Rolling_Std_30": [roll_30_s],
            "Commodity": pd.Series([commodity], dtype="category"),
            "Market Name": pd.Series([market], dtype="category"),
            "temperature_2m_mean": [hist_df["temperature_2m_mean"].iloc[-1] if "temperature_2m_mean" in hist_df else 28.0],
            "temperature_2m_max": [hist_df["temperature_2m_max"].iloc[-1] if "temperature_2m_max" in hist_df else 33.0],
            "temperature_2m_min": [hist_df["temperature_2m_min"].iloc[-1] if "temperature_2m_min" in hist_df else 23.0],
            "precipitation_sum": [hist_df["precipitation_sum"].iloc[-1] if "precipitation_sum" in hist_df else 0.0],
            "wind_speed_10m_max": [hist_df["wind_speed_10m_max"].iloc[-1] if "wind_speed_10m_max" in hist_df else 12.0]
        }
        
        X_df = pd.DataFrame(feat_row)
        
        if trained_model is not None:
            pred_p = float(trained_model.predict(X_df)[0])
        else:
            # Fallback autoregressive estimate if model binary missing
            pred_p = 0.6 * lag_1 + 0.3 * roll_7_m + 0.1 * lag_7
            
        prices.append(pred_p)
        forecast_list.append({
            "Date": target_dt.strftime("%Y-%m-%d"),
            "Forecasted Modal Price (₹/Quintal)": round(pred_p, 2),
            "Estimated Daily Volatility (₹)": round(roll_7_s, 2)
        })
        
    return pd.DataFrame(forecast_list)

forecast_table = generate_forecast(model, series_df, selected_commodity, selected_market, int(forecast_days))

# Forecast Metrics
f_start = forecast_table.iloc[0]["Forecasted Modal Price (₹/Quintal)"]
f_end = forecast_table.iloc[-1]["Forecasted Modal Price (₹/Quintal)"]
f_change = ((f_end - f_start) / f_start) * 100

f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Forecast Start Price", f"₹{f_start:,.2f}/quintal")
with f2:
    st.metric("Forecast Horizon End Price", f"₹{f_end:,.2f}/quintal")
with f3:
    st.metric("Projected Trend Change", f"{f_change:+.2f}%", delta=f"{f_change:+.2f}%")

# Combined Forecast Plot
hist_tail = series_df.tail(45)

fig_fore = go.Figure()

# Recent actuals
fig_fore.add_trace(go.Scatter(
    x=hist_tail["Price Date"],
    y=hist_tail["Modal_Price"],
    mode="lines+markers",
    name="Recent Actual Price",
    line=dict(color="#37474f", width=2.5)
))

# Forecast line
fig_fore.add_trace(go.Scatter(
    x=pd.to_datetime(forecast_table["Date"]),
    y=forecast_table["Forecasted Modal Price (₹/Quintal)"],
    mode="lines+markers",
    name="LightGBM Future Forecast",
    line=dict(color="#d32f2f", width=3, dash="dot"),
    marker=dict(symbol="star", size=8, color="#d32f2f")
))

fig_fore.update_layout(
    title=f"Recent Actuals & {forecast_days}-Day Future Forecast — {selected_commodity} ({selected_market})",
    xaxis_title="Date",
    yaxis_title="Price (₹ / Quintal)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_white"
)

col_fchart, col_ftable = st.columns([3, 2])
with col_fchart:
    st.plotly_chart(fig_fore, use_container_width=True)
with col_ftable:
    st.markdown("**📋 Forecasted Schedule:**")
    # Native st.dataframe without .to_html() to fix Bug B
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Section 4: Market-to-Market Comparison
# ---------------------------------------------------------
st.markdown(f'<div class="section-header">⚖️ Market-to-Market Comparison for {selected_commodity}</div>', unsafe_allow_html=True)

# Find latest date per market
comm_df = data_df[data_df["Commodity"] == selected_commodity].copy()
latest_comm_prices = comm_df.sort_values("Price Date").groupby("Market Name").last().reset_index()

lowest_mkt = latest_comm_prices.sort_values("Modal_Price").iloc[0]
highest_mkt = latest_comm_prices.sort_values("Modal_Price").iloc[-1]
mkt_spread = highest_mkt["Modal_Price"] - lowest_mkt["Modal_Price"]

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.metric("Lowest Price Mandi", f"{lowest_mkt['Market Name']}", delta=f"₹{lowest_mkt['Modal_Price']:,.0f}/quintal", delta_color="inverse")
with mc2:
    st.metric("Highest Price Mandi", f"{highest_mkt['Market Name']}", delta=f"₹{highest_mkt['Modal_Price']:,.0f}/quintal")
with mc3:
    st.metric("Inter-Mandi Spread", f"₹{mkt_spread:,.0f}/quintal", delta=f"{((mkt_spread)/lowest_mkt['Modal_Price'])*100:.1f}% variation")

fig_bar = px.bar(
    latest_comm_prices.sort_values("Modal_Price"),
    x="Market Name",
    y="Modal_Price",
    color="Modal_Price",
    color_continuous_scale="Viridis",
    text_auto=".0f",
    title=f"Latest Modal Price Comparison across Mandis for {selected_commodity}"
)
fig_bar.update_layout(
    xaxis_title="Market Name",
    yaxis_title="Modal Price (₹/Quintal)",
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# Section 5: Historical Weather & Climate Impact
# ---------------------------------------------------------
if show_weather and "temperature_2m_mean" in series_df.columns:
    st.markdown(f'<div class="section-header">🌦️ Historical Weather & Climate Patterns ({selected_market})</div>', unsafe_allow_html=True)
    
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.metric("Avg Daily Temperature", f"{series_df['temperature_2m_mean'].mean():.1f} °C")
    with w2:
        st.metric("Max Recorded Temp", f"{series_df['temperature_2m_max'].max():.1f} °C")
    with w3:
        st.metric("Total Precipitation", f"{series_df['precipitation_sum'].sum():.1f} mm")
    with w4:
        st.metric("Peak Daily Rain", f"{series_df['precipitation_sum'].max():.1f} mm")

    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(
        x=series_df["Price Date"],
        y=series_df["temperature_2m_mean"],
        name="Mean Temp (°C)",
        line=dict(color="#f57c00", width=2)
    ))
    fig_w.add_trace(go.Bar(
        x=series_df["Price Date"],
        y=series_df["precipitation_sum"],
        name="Precipitation (mm)",
        marker_color="#0288d1",
        yaxis="y2",
        opacity=0.6
    ))
    
    fig_w.update_layout(
        title=f"Temperature & Rainfall at {selected_market}",
        xaxis_title="Date",
        yaxis=dict(title="Temperature (°C)", side="left"),
        yaxis2=dict(title="Precipitation (mm)", side="right", overlaying="y"),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_w, use_container_width=True)

# ---------------------------------------------------------
# Section 6: Model Performance Benchmark & Methodology Card
# ---------------------------------------------------------
st.markdown('<div class="section-header">🔬 Academic Benchmark & Evaluation Summary</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-card">
<b>Evaluation Protocol:</b> Chronological 56-day holdout validation (20 June 2025 to 14 August 2025) across 15 commodity-market series. No random shuffling or future leakage permitted.
</div>
""", unsafe_allow_html=True)

perf_data = pd.DataFrame({
    "Forecasting Model": [
        "Seasonal Naive Baseline (Lag-7)",
        "LightGBM (Price-Only Features)",
        "LightGBM (Price + Weather)",
        "SARIMA (1,1,1)x(1,1,1,7) [Brinjal-Asansol]"
    ],
    "Scope": ["Aggregate (15 Series)", "Aggregate (15 Series)", "Aggregate (15 Series)", "Single Representative Series"],
    "MAE (₹)": [269.88, 166.22, 167.42, 440.26],
    "RMSE (₹)": [538.92, 317.32, 309.60, 500.89],
    "MAPE (%)": ["6.41%", "4.27%", "4.35%", "9.51%"],
    "Core Finding": [
        "Simple weekly seasonal persistence baseline",
        "Substantial improvement over naive rule via lag & rolling dynamics",
        "Slightly improves RMSE (-2.4%), but slight increase in MAE/MAPE (mixed effect)",
        "Higher error than ML due to linear limitation on abrupt vegetable volatility"
    ]
})

st.dataframe(perf_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("AgriScope System v2.0 | College Data Science Project P_085 | Built with Python, LightGBM, Streamlit & Plotly")
