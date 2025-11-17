import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import io

st.set_page_config(page_title="Demand Forecasting", layout="wide")

# ========================================================
# Custom Metrics (replacing sklearn)
# ========================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# ========================================================
# Synthetic Data Generator (2–3 years)
# ========================================================
def generate_synthetic_data(days=900):
    rng = np.random.default_rng(42)
    base = 200 + 20 * np.sin(np.linspace(0, 12 * np.pi, days))
    noise = rng.normal(0, 8, days)
    trend = np.linspace(0, 40, days)
    data = base + noise + trend
    dates = pd.date_range(start="2020-01-01", periods=days)
    return pd.Series(data, index=dates)

# ========================================================
# ARIMA/SARIMA Forecast
# ========================================================
def arima_forecast(ts, steps=30, p=2, d=1, q=2, sp=1, sd=1, sq=1):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)

    ts = ts.reset_index(drop=True)

    model = SARIMAX(
        ts,
        order=(p, d, q),
        seasonal_order=(sp, sd, sq, 7)
    )
    res = model.fit(disp=False)
    f = res.forecast(steps=steps)

    dates = pd.date_range(start=ts.index[-1] + 1, periods=steps)
    return pd.Series(f, index=dates)

# ========================================================
# Prophet Forecast
# ========================================================
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

# ========================================================
# LSTM Forecast
# ========================================================
def lstm_forecast(ts, steps=30, epochs=20, window=14):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)

    ts = ts.values.reshape(-1, 1)
    gen = TimeseriesGenerator(ts, ts, length=window, batch_size=16)

    model = Sequential([
        LSTM(32, activation="relu", return_sequences=False),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(gen, epochs=epochs, verbose=0)

    forecast = []
    current_window = ts[-window:].reshape(1, window, 1)

    for _ in range(steps):
        next_val = model.predict(current_window)[0][0]
        forecast.append(next_val)
        current_window = np.append(current_window[:, 1:, :], [[[next_val]]], axis=1)

    dates = pd.date_range(start=pd.Timestamp.today(), periods=steps)
    return pd.Series(forecast, index=dates)

# ========================================================
# Streamlit UI
# ========================================================
st.title("📦 Advanced Warehousing Demand Forecasting System")
st.markdown("Forecast the next 30 days using **ARIMA/SARIMA**, **Prophet**, and **LSTM**.")

# -----------------------------------------
# Sidebar
# -----------------------------------------
st.sidebar.header("⚙ Settings")

theme = st.sidebar.radio("Theme:", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

mode = st.sidebar.radio("Select Data Source:", ["Synthetic Data", "Upload CSV"])

forecast_horizon = st.sidebar.slider("Forecast Days", 7, 60, 30)

# ARIMA Params
st.sidebar.subheader("ARIMA/SARIMA Parameters")
p = st.sidebar.number_input("p", 0, 5, 2)
d = st.sidebar.number_input("d", 0, 2, 1)
q = st.sidebar.number_input("q", 0, 5, 2)
sp = st.sidebar.number_input("Seasonal p", 0, 5, 1)
sd = st.sidebar.number_input("Seasonal d", 0, 2, 1)
sq = st.sidebar.number_input("Seasonal q", 0, 5, 1)

# LSTM Params
st.sidebar.subheader("LSTM Parameters")
epochs = st.sidebar.slider("Epochs", 5, 100, 20)
window = st.sidebar.slider("Window Size", 5, 30, 14)

# -----------------------------------------
# Load Data
# -----------------------------------------
if mode == "Synthetic Data":
    ts = generate_synthetic_data()
    st.success("Synthetic dataset loaded.")
else:
    file = st.sidebar.file_uploader("Upload CSV", type='csv')
    if file:
        df = pd.read_csv(file)
        col = df.columns[0]
        ts = df[col].dropna()
        ts.index = pd.date_range(start="2020-01-01", periods=len(ts))
        st.success("CSV loaded successfully.")
    else:
        st.warning("Upload a CSV to continue.")
        st.stop()

# -----------------------------------------
# Historical Plot
# -----------------------------------------
with st.expander("📈 Historical Data"):
    fig, ax = plt.subplots(figsize=(12, 3))
    ts.plot(ax=ax)
    ax.set_title("Historical Package Volume")
    st.pyplot(fig)

# -----------------------------------------
# Run Forecast
# -----------------------------------------
if st.button("Run Forecast"):
    try:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("Running ARIMA/SARIMA...")
            arima_f = arima_forecast(
                ts, steps=forecast_horizon,
                p=p, d=d, q=q, sp=sp, sd=sd, sq=sq
            )

        with col2:
            st.info("Running Prophet...")
            prophet_f = prophet_forecast(ts, steps=forecast_horizon)

        with col3:
            st.info("Training LSTM...")
            lstm_f = lstm_forecast(
                ts, steps=forecast_horizon,
                epochs=epochs, window=window
            )

        # Combine results
        combined = pd.DataFrame({
            "ARIMA/SARIMA": arima_f.values,
            "Prophet": prophet_f.values,
            "LSTM": lstm_f.values
        }, index=arima_f.index)

        # -------------------------------------
        # Metrics (compare with last horizon)
        # -------------------------------------
        last_real = ts.tail(forecast_horizon)

        mape_vals = {}
        rmse_vals = {}

        for model_name, pred in combined.items():
            pred_aligned = pred[:len(last_real)]
            mape_vals[model_name] = mean_absolute_percentage_error(last_real, pred_aligned)
            rmse_vals[model_name] = np.sqrt(mean_squared_error(last_real, pred_aligned))

        st.subheader("📊 Model Performance Metrics")
        st.write("**MAPE (Lower = Better):**", mape_vals)
        st.write("**RMSE (Lower = Better):**", rmse_vals)

        # -------------------------------------
        # Forecast Plot
        # -------------------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ts.tail(120).plot(ax=ax2, label="Recent History")
        combined["ARIMA/SARIMA"].plot(ax=ax2, label="ARIMA")
        combined["Prophet"].plot(ax=ax2, label="Prophet")
        combined["LSTM"].plot(ax=ax2, label="LSTM")
        ax2.legend()
        ax2.set_title("Forecast Comparison")
        st.pyplot(fig2)

        # -------------------------------------
        # Download Button
        # -------------------------------------
        csv_buffer = io.StringIO()
        combined.to_csv(csv_buffer)
        st.download_button(
            "📥 Download Forecast CSV",
            csv_buffer.getvalue(),
            "forecast.csv",
            "text/csv"
        )

        st.dataframe(combined)

    except Exception as e:
        st.error(f"Error: {e}")
