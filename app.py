import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

st.set_page_config(page_title="Demand Forecasting", layout="wide")

# -----------------------------------------
# Synthetic Data Generator (2–3 years)
# -----------------------------------------
def generate_synthetic_data(days=900):
    rng = np.random.default_rng(42)
    base = 200 + 20 * np.sin(np.linspace(0, 12 * np.pi, days))
    noise = rng.normal(0, 8, days)
    trend = np.linspace(0, 40, days)
    data = base + noise + trend
    dates = pd.date_range(start="2020-01-01", periods=days)
    return pd.Series(data, index=dates)

# -----------------------------------------
# ARIMA/SARIMA Forecast
# -----------------------------------------
def arima_forecast(ts, steps=30):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)

    ts = ts.reset_index(drop=True)

    model = SARIMAX(ts, order=(2,1,2), seasonal_order=(1,1,1,7))
    res = model.fit(disp=False)
    f = res.forecast(steps=steps)

    dates = pd.date_range(start=ts.index[-1] + 1, periods=steps)
    return pd.Series(f, index=dates)

# -----------------------------------------
# Prophet Forecast
# -----------------------------------------
def prophet_forecast(ts, steps=30):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)

    df = pd.DataFrame({
        "ds": pd.date_range(start="2020-01-01", periods=len(ts)),
        "y": ts.values
    })

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    out = forecast.set_index("ds")["yhat"].tail(steps)
    return out

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.title("📦 Warehousing Demand Forecasting (Time Series)")
st.markdown("""
Predict the next 30 days of warehouse package volume using  
**ARIMA/SARIMA** and **Prophet** forecasting models.
""")

# --- Sidebar
st.sidebar.header("Options")
mode = st.sidebar.radio("Select Data Source:", ["Use Synthetic Data", "Upload CSV"])

# -----------------------------------------
# Load Time-Series Data
# -----------------------------------------
if mode == "Use Synthetic Data":
    ts = generate_synthetic_data()
    st.success("Synthetic dataset loaded successfully.")

else:
    file = st.sidebar.file_uploader("Upload a CSV with a single column of values")
    if file:
        df = pd.read_csv(file)
        col = df.columns[0]
        ts = df[col].dropna()
        ts.index = pd.date_range(start="2020-01-01", periods=len(ts))
        st.success("Custom dataset loaded successfully.")
    else:
        st.warning("Please upload a CSV.")
        st.stop()

# -----------------------------------------
# Show Historical Data
# -----------------------------------------
with st.expander("📈 View Historical Data"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ts.plot(ax=ax)
    ax.set_title("Historical Package Volume")
    st.pyplot(fig)

# -----------------------------------------
# Run Forecasting
# -----------------------------------------
if st.button("Run Forecast"):
    try:
        with st.spinner("Running ARIMA/SARIMA..."):
            arima_f = arima_forecast(ts)

        with st.spinner("Running Prophet..."):
            prophet_f = prophet_forecast(ts)

        # -----------------------------------------
        # Combine for Comparison
        # -----------------------------------------
        combined = pd.DataFrame({
            "ARIMA/SARIMA": arima_f,
            "Prophet": prophet_f
        })

        st.subheader("📊 Forecast Results (Next 30 Days)")
        st.dataframe(combined)

        # -----------------------------------------
        # Plot Forecasts
        # -----------------------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ts.tail(120).plot(ax=ax2, label="Recent History")
        arima_f.plot(ax=ax2, label="ARIMA/SARIMA Forecast")
        prophet_f.plot(ax=ax2, label="Prophet Forecast")
        ax2.legend()
        ax2.set_title("Demand Forecast Comparison")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
