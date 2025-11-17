# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import io
import warnings
warnings.filterwarnings("ignore")

# Modeling libraries (may not be installed in all environments)
HAS_PROPHET = False
HAS_TF = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    try:
        # older installation name
        from fbprophet import Prophet  # type: ignore
        HAS_PROPHET = True
    except Exception:
        HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except Exception:
    HAS_TF = False

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Warehousing Demand Forecasting", layout="wide")

# ---------------------------
# Helpers: synthetic dataset
# ---------------------------
@st.cache_data
def generate_synthetic_days(days=900, seed=42):
    """Generate realistic-ish daily package counts time series (~2.5 years by default)."""
    rng = np.random.default_rng(seed)
    start = pd.to_datetime("2022-01-01")
    dates = pd.date_range(start, periods=days, freq="D")
    # components
    trend = np.linspace(50, 120, days)                    # slowly increasing demand
    weekly = 10 * (np.sin(2 * np.pi * np.arange(days) / 7) + 1)  # weekly seasonality
    yearly = 20 * np.sin(2 * np.pi * np.arange(days) / 365)  # yearly seasonality
    noise = rng.normal(scale=8, size=days)
    spikes = np.zeros(days)
    # occasional spikes (promotions, events)
    spike_days = rng.choice(np.arange(30, days-30), size=max(3, days//300), replace=False)
    for sd in spike_days:
        spikes[sd:sd+3] += rng.integers(20, 60)  # multi-day spike
    counts = np.maximum(0, (trend + weekly + yearly + spikes + noise).round()).astype(int)
    df = pd.DataFrame({"ds": dates, "y": counts})
    return df

# ---------------------------
# IO: user CSV parser
# ---------------------------
def parse_uploaded_csv(uploaded_file):
    """
    Expect CSV with a column named 'ds' or 'date' and a column named 'y' or 'value' or 'count'.
    Tries to be flexible.
    """
    df = pd.read_csv(uploaded_file)
    cols = [c.lower() for c in df.columns]
    # find date column
    date_col = None
    for candidate in ("ds", "date", "timestamp"):
        if candidate in cols:
            date_col = df.columns[cols.index(candidate)]
            break
    if date_col is None:
        st.error("Could not find a date column in your CSV. Please include a 'date' or 'ds' column.")
        return None
    # find value column
    val_col = None
    for candidate in ("y", "value", "count", "packages", "volume"):
        if candidate in cols:
            val_col = df.columns[cols.index(candidate)]
            break
    if val_col is None:
        # fallback: use the next numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            val_col = numeric_cols[0]
        else:
            st.error("Could not find a numeric value column in CSV. Please include 'y' or 'count' column.")
            return None
    df2 = pd.DataFrame({"ds": pd.to_datetime(df[date_col]), "y": pd.to_numeric(df[val_col], errors="coerce")})
    df2 = df2.dropna().sort_values("ds").reset_index(drop=True)
    return df2

# ---------------------------
# Forecast helpers
# ---------------------------
def forecast_prophet(df, periods=30):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst

def sarima_grid_search(ts, p_range=(0,2), d_range=(0,1), q_range=(0,2), seasonal=(7,0,0,0)):
    # small grid search on p,d,q to avoid heavy compute
    best_aic = np.inf
    best_order = None
    best_result = None
    for p in range(p_range[0], p_range[1]+1):
        for d in range(d_range[0], d_range[1]+1):
            for q in range(q_range[0], q_range[1]+1):
                try:
                    model = SARIMAX(ts, order=(p,d,q), seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                        best_result = res
                except Exception:
                    continue
    return best_result, best_order

def forecast_sarima(df, periods=30, seasonal_period=7):
    ts = df.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
    # use small grid
    res, order = sarima_grid_search(ts.values, p_range=(0,2), d_range=(0,1), q_range=(0,2), seasonal=(1,0,1,seasonal_period))
    if res is None:
        raise RuntimeError("SARIMA failed to converge for available orders.")
    pred = res.get_forecast(steps=periods)
    index = pd.date_range(ts.index[-1] + timedelta(days=1), periods=periods, freq="D")
    fcst = pd.DataFrame({
        "ds": index,
        "yhat": pred.predicted_mean,
        "yhat_lower": pred.conf_int().iloc[:,0],
        "yhat_upper": pred.conf_int().iloc[:,1]
    })
    return fcst, order

# LSTM helper functions
def create_lstm_dataset(series, look_back=14):
    X, y = [], []
    for i in range(len(series)-look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def forecast_lstm(df, periods=30, epochs=40, look_back=14):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed in this environment.")
    ts = df.set_index("ds")["y"].asfreq("D").fillna(method="ffill").values.astype(float)
    X, y = create_lstm_dataset(ts, look_back=look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, input_shape=(look_back,1), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    # iterative forecasting
    recent = ts[-look_back:].tolist()
    preds = []
    for _ in range(periods):
        x_in = np.array(recent[-look_back:]).reshape((1, look_back, 1))
        p = model.predict(x_in, verbose=0)[0,0]
        preds.append(p)
        recent.append(p)
    index = pd.date_range(df["ds"].max() + timedelta(days=1), periods=periods, freq="D")
    fcst = pd.DataFrame({"ds": index, "yhat": np.array(preds)})
    fcst["yhat_lower"] = fcst["yhat"] - 1.96*np.std(preds)
    fcst["yhat_upper"] = fcst["yhat"] + 1.96*np.std(preds)
    return fcst

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📦 Warehousing Demand Forecasting — ARIMA / Prophet / LSTM")
st.markdown("Forecast next 30 days of daily package volume. Upload your CSV or use a generated dataset.")

# left column: controls
c1, c2 = st.columns([1,2])
with c1:
    st.header("Data")
    use_synth = st.checkbox("Use synthetic dataset (recommended)", value=True)
    uploaded = st.file_uploader("Or upload CSV (date + value)", type=["csv"])
    n_days = st.slider("Synthetic history length (days)", 365, 1100, 900, step=30)
    if uploaded:
        df = parse_uploaded_csv(uploaded)
        if df is None:
            st.stop()
    elif use_synth:
        df = generate_synthetic_days(n_days)
    else:
        st.info("Upload a CSV or enable synthetic data.")
        st.stop()

    st.markdown("---")
    st.header("Model")
    model_choice = st.selectbox("Select model", options=["SARIMA (statsmodels)", "Prophet (if available)", "LSTM (optional)"])
    st.write("Prophet installed:" , "✅" if HAS_PROPHET else "❌")
    st.write("TensorFlow installed:", "✅" if HAS_TF else "❌")

    periods = st.number_input("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)

    # SARIMA params (small)
    st.markdown("**SARIMA tuning (grid search, small)**")
    seasonal_period = st.selectbox("Seasonal period (days)", [7, 30, 365], index=0)
    pmax = st.slider("p max", 0, 3, 2)
    dmax = st.slider("d max", 0, 2, 1)
    qmax = st.slider("q max", 0, 3, 2)

    st.markdown("**LSTM options**")
    lstm_epochs = st.number_input("LSTM epochs", min_value=5, max_value=200, value=40, step=5)
    lstm_lookback = st.number_input("LSTM lookback (days)", min_value=5, max_value=60, value=14, step=1)
    st.markdown("---")
    run_button = st.button("Run forecasting")

# right column: visuals / results
with c2:
    st.subheader("Historical data")
    st.line_chart(data=df.rename(columns={"ds":"date"}).set_index("date")["y"])

    st.write(df.head(8))

    if run_button:
        with st.spinner("Training / forecasting..."):
            try:
                if model_choice.startswith("SARIMA"):
                    # run small grid search
                    ts = df.set_index("ds")["y"].asfreq("D").fillna(method="ffill")
                    res, order = sarima_grid_search(ts.values, p_range=(0,pmax), d_range=(0,dmax), q_range=(0,qmax),
                                                   seasonal=(1,0,1,seasonal_period))
                    if res is None:
                        st.error("SARIMA failed to find a convergent model with given grid.")
                        st.stop()
                    fcst_df, order_used = forecast_sarima(df, periods=periods, seasonal_period=seasonal_period)
                    st.success(f"SARIMA model fit. Order (p,d,q): {order_used}")
                    models_results = {"SARIMA": fcst_df}

                elif model_choice.startswith("Prophet"):
                    if not HAS_PROPHET:
                        st.error("Prophet is not installed in this environment.")
                        st.stop()
                    prophet_df = df.rename(columns={"ds":"ds", "y":"y"})[["ds","y"]]
                    fcst = forecast_prophet(prophet_df, periods=periods)
                    models_results = {"Prophet": fcst}

                else:  # LSTM
                    if not HAS_TF:
                        st.error("TensorFlow / Keras not installed. LSTM not available.")
                        st.stop()
                    fcst = forecast_lstm(df, periods=periods, epochs=int(lstm_epochs), look_back=int(lstm_lookback))
                    models_results = {"LSTM": fcst}

                # Plot results combined
                st.subheader("Forecast comparison")
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(df["ds"], df["y"], label="history", linewidth=2)
                colors = ["tab:orange","tab:green","tab:red","tab:purple"]
                for (name, fdf), col in zip(models_results.items(), colors):
                    ax.plot(fdf["ds"], fdf["yhat"], label=f"{name} forecast", linestyle="--", color=col)
                    ax.fill_between(fdf["ds"], fdf.get("yhat_lower", fdf["yhat"]-1),
                                    fdf.get("yhat_upper", fdf["yhat"]+1), alpha=0.15, color=col)
                ax.set_xlabel("Date")
                ax.set_ylabel("Daily package count")
                ax.legend()
                st.pyplot(fig)

                # Show numeric table for forecast
                st.subheader("Forecast (next days)")
                # concat all forecasts (if multiple models eventually)
                # Here models_results is single-model but keep general
                combined = None
                for name, fdf in models_results.items():
                    t = fdf[["ds","yhat"]].copy()
                    t = t.rename(columns={"yhat": f"yhat_{name}"})
                    if combined is None:
                        combined = t
                    else:
                        combined = combined.merge(t, on="ds", how="outer")
                st.dataframe(combined.set_index("ds").round(2))

                # Download button
                csv_buf = io.StringIO()
                combined.to_csv(csv_buf, index=False)
                st.download_button("Download forecast CSV", data=csv_buf.getvalue(), file_name="forecast.csv")
            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")

st.markdown("---")
st.markdown("**Notes & Tips**")
st.markdown("""
- The synthetic dataset is intended for experimentation; for production use provide a clean CSV with daily counts.
- SARIMA grid search here is intentionally small to keep computation reasonable in a Streamlit environment.
- Prophet and LSTM require extra packages; the app will inform you if they are not installed.
- For robust production forecasting, add cross-validation, hyperparameter tuning, and external regressors (holidays, promotions, weather).
""")
