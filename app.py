# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

# Optional packages that may not be installed
HAS_PROPHET = False
HAS_TF = False
HAS_PLOTLY = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # older name
        HAS_PROPHET = True
    except Exception:
        HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Warehousing Demand Forecasting", layout="wide")

# -----------------------------
# Custom metrics (robust)
# -----------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# Utilities
# -----------------------------
def ensure_series(s):
    """Return pandas Series indexed by DatetimeIndex; drop NaN/inf."""
    if isinstance(s, pd.DataFrame):
        # take first numeric column
        numeric = s.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("DataFrame contains no numeric columns.")
        s = numeric.iloc[:, 0]
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    s = s.astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.date_range(start="2020-01-01", periods=len(s), freq="D")
    return s

def generate_synthetic_days(days=900, seed=42):
    rng = np.random.default_rng(seed)
    # realistic-ish components
    trend = np.linspace(50, 150, days)
    weekly = 12 * np.sin(2 * np.pi * np.arange(days) / 7)
    yearly = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = rng.normal(0, 8, days)
    spikes = np.zeros(days)
    # a few spikes
    for sd in rng.choice(np.arange(60, days-60), size=max(3, days//300), replace=False):
        spikes[sd:sd+3] += rng.integers(30, 80)
    data = np.maximum(0, (trend + weekly + yearly + noise + spikes).round())
    dates = pd.date_range(start="2020-01-01", periods=days, freq="D")
    return pd.Series(data, index=dates)

# -----------------------------
# ARIMA / SARIMA (safe)
# -----------------------------
def fit_arima_safe(series, p=1, d=1, q=1, sp=0, sd=1, sq=1, m=7, steps=30):
    """Fit SARIMAX safely, return forecast Series with index"""
    s = ensure_series(series)
    # Clean series and check length
    if len(s) < max(20, m + 10):
        # too short, fallback: repeat last value
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        fallback = pd.Series([s.iloc[-1]] * steps, index=idx)
        return fallback, {"warning": "series too short for ARIMA; returned constant forecast"}
    # Try fitting; fallback parameters if errors occur
    try:
        model = SARIMAX(s, order=(p, d, q), seasonal_order=(sp, sd, sq, m),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, low_memory=True)
        pred = res.get_forecast(steps=steps)
        mean = pred.predicted_mean
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        # create DataFrame with predictions and conf intervals
        ci = pred.conf_int()
        out = pd.DataFrame({
            "yhat": mean.values,
            "yhat_lower": ci.iloc[:, 0].values,
            "yhat_upper": ci.iloc[:, 1].values
        }, index=idx)
        return out, {"order": (p, d, q), "seasonal_order": (sp, sd, sq, m)}
    except Exception as e:
        # fallback to very safe simple forecast (last value repeated)
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        fallback = pd.DataFrame({
            "yhat": [s.iloc[-1]] * steps,
            "yhat_lower": [s.iloc[-1]] * steps,
            "yhat_upper": [s.iloc[-1]] * steps
        }, index=idx)
        return fallback, {"warning": f"ARIMA failed: {str(e)}"}

# -----------------------------
# Prophet
# -----------------------------
def prophet_forecast_safe(series, steps=30):
    if not HAS_PROPHET:
        raise RuntimeError("Prophet not installed.")
    s = ensure_series(series)
    df = pd.DataFrame({"ds": s.index, "y": s.values})
    try:
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=steps)
        fc = m.predict(future)
        out = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(steps)
        return out
    except Exception as e:
        # fallback constant
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        fallback = pd.DataFrame({"yhat": [s.iloc[-1]] * steps, "yhat_lower":[s.iloc[-1]]*steps, "yhat_upper":[s.iloc[-1]]*steps}, index=idx)
        return fallback

# -----------------------------
# LSTM (optional, safe)
# -----------------------------
def lstm_forecast_safe(series, steps=30, epochs=20, window=14):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed.")
    s = ensure_series(series)
    # if too short, fallback
    if len(s) < window + 5:
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        fallback = pd.Series([s.iloc[-1]] * steps, index=idx)
        return pd.DataFrame({"yhat": fallback.values, "yhat_lower": fallback.values, "yhat_upper": fallback.values}, index=idx)
    arr = s.values.astype(float).reshape(-1, 1)
    gen = TimeseriesGenerator(arr, arr, length=window, batch_size=16)
    model = Sequential([LSTM(32, activation="relu", input_shape=(window, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(gen, epochs=epochs, verbose=0)
    forecast = []
    current = arr[-window:].reshape(1, window, 1)
    for _ in range(steps):
        p = model.predict(current, verbose=0)[0][0]
        forecast.append(float(p))
        current = np.append(current[:, 1:, :], [[[p]]], axis=1)
    idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    arr_fc = np.array(forecast)
    std = max(np.std(arr_fc), 0.0)
    df = pd.DataFrame({"yhat": arr_fc, "yhat_lower": arr_fc - 1.96*std, "yhat_upper": arr_fc + 1.96*std}, index=idx)
    return df

# -----------------------------
# Decomposition
# -----------------------------
def decompose_series(series, period=7):
    s = ensure_series(series)
    stl = STL(s, period=period, robust=True)
    res = stl.fit()
    comp = pd.DataFrame({"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}, index=s.index)
    return comp

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_forecasts_on_recent(history, forecasts_dict, horizon=30):
    """Compute MAPE and RMSE of each forecast against the last `horizon` days of history (if available)."""
    hist = ensure_series(history)
    recent = hist.tail(horizon)
    metrics = {}
    for k, df in forecasts_dict.items():
        preds = df["yhat"].values[:len(recent)]
        if len(preds) != len(recent) or len(preds) == 0:
            metrics[k] = {"mape": np.inf, "rmse": np.inf}
            continue
        mape = mean_absolute_percentage_error(recent.values, preds)
        rmse = np.sqrt(mean_squared_error(recent.values, preds))
        metrics[k] = {"mape": float(mape), "rmse": float(rmse)}
    # choose best by mape
    best = min(metrics.items(), key=lambda kv: kv[1]["mape"])[0]
    return metrics, best

# -----------------------------
# UI
# -----------------------------
st.title("📦 Warehousing Demand Forecasting — ARIMA/SARIMA / Prophet / LSTM")
st.markdown("Forecast the next days of package volume for a warehouse. Enter parameters manually or use Auto-safe defaults.")

# Sidebar: theme and data
with st.sidebar:
    st.header("Settings")
    theme = st.radio("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>body{background-color:#0b1221;color:#ddd}</style>", unsafe_allow_html=True)
    data_mode = st.radio("Data Source", ["Synthetic", "Upload CSV"])
    n_days = st.slider("Synthetic history length (days)", min_value=365, max_value=1500, value=900, step=30)
    forecast_horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)

    st.markdown("---")
    st.subheader("ARIMA / SARIMA (manual)")
    arima_auto = st.checkbox("Auto-safe ARIMA params (recommended)", value=True)
    if not arima_auto:
        p = st.number_input("p (AR order)", 0, 5, 1, help="Number of lag observations included in the model.")
        d = st.number_input("d (differencing)", 0, 2, 1, help="Number of times the raw observations are differenced.")
        q = st.number_input("q (MA order)", 0, 5, 1, help="Size of moving-average window.")
        sp = st.number_input("Seasonal P", 0, 3, 0, help="Seasonal AR order (uses seasonal period m).")
        sd = st.number_input("Seasonal D", 0, 2, 1, help="Seasonal differencing.")
        sq = st.number_input("Seasonal Q", 0, 3, 0, help="Seasonal MA order.")
        m = st.number_input("Seasonal period (m)", 1, 365, 7, help="Length of season (7 weekly, 30 monthly).")
    else:
        # safe defaults
        p, d, q, sp, sd, sq, m = 1, 1, 1, 0, 1, 0, 7

    st.markdown("---")
    st.subheader("LSTM (optional)")
    use_lstm = st.checkbox("Enable LSTM (TensorFlow required)", value=False)
    if use_lstm:
        lstm_epochs = st.number_input("LSTM epochs", 5, 200, 20)
        lstm_window = st.number_input("LSTM window (days)", 5, 60, 14)
    else:
        lstm_epochs, lstm_window = 20, 14

    st.markdown("---")
    st.subheader("Models to run")
    run_arima = st.checkbox("Run ARIMA/SARIMA", value=True)
    run_prophet = st.checkbox("Run Prophet", value=HAS_PROPHET)
    run_lstm = st.checkbox("Run LSTM", value=False if not HAS_TF else True)

    st.markdown("---")
    st.subheader("Glossary & Help")
    with st.expander("📘 Time-Series Glossary (p,d,q,P,D,Q,m)"):
        st.markdown("""
**p (AR order)**: number of past values used to predict current value.  
**d (differencing)**: how many times to difference data to remove trend.  
**q (MA order)**: number of past forecast errors used.  

**P (seasonal AR)**: AR for seasonal lag (e.g., 7 days ago).  
**D (seasonal differencing)**: remove seasonal trend.  
**Q (seasonal MA)**: MA for seasonal errors.  
**m (seasonal period)**: length of the season (7 = weekly, 30 ≈ monthly).

_Tooltips are available on inputs — try small values for p,d,q to avoid instability._
        """)

# Load data (main area)
if data_mode == "Synthetic":
    ts = generate_synthetic_days(n_days)
    st.success("Synthetic dataset loaded.")
else:
    uploaded = st.file_uploader("Upload CSV file (date optional, first numeric column used)", type=["csv"])
    if uploaded is None:
        st.warning("Please upload CSV or choose Synthetic data.")
        st.stop()
    df_up = pd.read_csv(uploaded)
    # pick first numeric column
    numeric_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.error("Uploaded CSV must contain at least one numeric column.")
        st.stop()
    col = numeric_cols[0]
    s = pd.Series(df_up[col].astype(float).values)
    s.index = pd.date_range(start="2020-01-01", periods=len(s), freq="D")
    ts = s
    st.success("CSV loaded.")

# Keep raw copy for plotting (so we never truncate history)
raw_series = ensure_series(ts)

# Layout: left controls, right visuals
col_left, col_right = st.columns([1, 2])

with col_right:
    st.subheader("Historical data (full)")
    if HAS_PLOTLY:
        fig = px.line(x=raw_series.index, y=raw_series.values, labels={"x":"Date","y":"Count"}, title="Full Historical Series")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(raw_series.index, raw_series.values, color="tab:blue")
        ax.set_title("Full Historical Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily package count")
        st.pyplot(fig)

    # Decomposition section (STL)
    with st.expander("Seasonal decomposition (STL)"):
        try:
            comp = decompose_series(raw_series, period=7)
            if HAS_PLOTLY:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend","Seasonal","Residual"))
                fig.add_trace(go.Scatter(x=comp.index, y=comp["trend"], name="Trend"), row=1, col=1)
                fig.add_trace(go.Scatter(x=comp.index, y=comp["seasonal"], name="Seasonal"), row=2, col=1)
                fig.add_trace(go.Scatter(x=comp.index, y=comp["resid"], name="Residual"), row=3, col=1)
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, axes = plt.subplots(3,1, figsize=(10,6), sharex=True)
                axes[0].plot(comp.index, comp["trend"]); axes[0].set_title("Trend")
                axes[1].plot(comp.index, comp["seasonal"]); axes[1].set_title("Seasonal")
                axes[2].plot(comp.index, comp["resid"]); axes[2].set_title("Residual")
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Decomposition error: {e}")

# Forecast button
if st.button("Run forecasting"):
    st.info("Running selected models...")

    forecasts = {}
    info_msgs = {}

    # ARIMA
    if run_arima:
        st.write("▶ Running ARIMA/SARIMA ...")
        try:
            out_arima, info = fit_arima_safe(raw_series, p=int(p), d=int(d), q=int(q), sp=int(sp), sd=int(sd), sq=int(sq), m=int(m), steps=int(forecast_horizon))
            forecasts["ARIMA"] = out_arima
            info_msgs["ARIMA"] = info
            st.success("ARIMA completed.")
        except Exception as e:
            st.error(f"ARIMA error: {e}")

    # Prophet
    if run_prophet:
        if not HAS_PROPHET:
            st.error("Prophet is not installed — cannot run Prophet.")
        else:
            st.write("▶ Running Prophet ...")
            try:
                out_prop = prophet_forecast_safe(raw_series, steps=int(forecast_horizon))
                forecasts["Prophet"] = out_prop
                st.success("Prophet completed.")
            except Exception as e:
                st.error(f"Prophet error: {e}")

    # LSTM
    if run_lstm and use_lstm:
        if not HAS_TF:
            st.error("TensorFlow not installed — LSTM unavailable.")
        else:
            st.write("▶ Running LSTM ...")
            try:
                out_lstm = lstm_forecast_safe(raw_series, steps=int(forecast_horizon), epochs=int(lstm_epochs), window=int(lstm_window))
                forecasts["LSTM"] = out_lstm
                st.success("LSTM completed.")
            except Exception as e:
                st.error(f"LSTM error: {e}")

    if len(forecasts) == 0:
        st.error("No forecasts produced. Check model selections and package availability.")
        st.stop()

    # Evaluate models on recent history (auto-select best)
    metrics, best_model = evaluate_forecasts_on_recent(raw_series, forecasts, horizon=min(30, len(raw_series)))
    st.subheader("Model metrics on recent history (lower is better)")
    st.table(pd.DataFrame(metrics).T)

    st.success(f"Auto-selected best model by MAPE: **{best_model}**")

    # Combine forecasts into a single DataFrame for plotting and download
    # Ensure common index = forecast horizon dates
    base_idx = list(forecasts.values())[0].index
    combined_df = pd.DataFrame(index=base_idx)
    for name, df in forecasts.items():
        combined_df[f"{name}_yhat"] = df["yhat"].values
        combined_df[f"{name}_lower"] = df["yhat_lower"].values
        combined_df[f"{name}_upper"] = df["yhat_upper"].values

    # Plot history + forecasts
    st.subheader("Forecast Comparison")
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_series.index, y=raw_series.values, name="History", line=dict(color="black")))
        colors = {"ARIMA":"orange","Prophet":"green","LSTM":"red"}
        for name, df in forecasts.items():
            fig.add_trace(go.Scatter(x=df.index, y=df["yhat"], name=f"{name} forecast", line=dict(color=colors.get(name,None))))
            fig.add_trace(go.Scatter(
                x=df.index.tolist() + df.index[::-1].tolist(),
                y=df["yhat_upper"].tolist() + df["yhat_lower"][::-1].tolist(),
                fill='toself', fillcolor='rgba(0,100,80,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False
            ))
        fig.update_layout(title="History + Forecasts", xaxis_title="Date", yaxis_title="Count", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(raw_series.index, raw_series.values, label="History", color="black")
        for name, df in forecasts.items():
            ax.plot(df.index, df["yhat"], label=f"{name} forecast")
            ax.fill_between(df.index, df["yhat_lower"], df["yhat_upper"], alpha=0.15)
        ax.legend()
        ax.set_title("History + Forecasts")
        st.pyplot(fig)

    # Show forecast table and download
    st.subheader("Forecast table (downloadable)")
    st.dataframe(combined_df.round(2))
    csv_buf = io.StringIO()
    combined_df.to_csv(csv_buf)
    st.download_button("Download forecasts CSV", csv_buf.getvalue(), file_name="forecasts.csv", mime="text/csv")

    # Detailed view of selected model
    st.subheader(f"Detailed view: {best_model}")
    best_df = forecasts[best_model]
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(raw_series.index[-365:], raw_series.values[-365:], label="History (last 365 days)")
    ax2.plot(best_df.index, best_df["yhat"], label=f"{best_model} forecast")
    ax2.fill_between(best_df.index, best_df["yhat_lower"], best_df["yhat_upper"], alpha=0.2)
    ax2.legend()
    st.pyplot(fig2)

    # Show any info messages from ARIMA
    if "ARIMA" in info_msgs:
        st.info(f"ARIMA info: {info_msgs['ARIMA']}")

st.markdown("---")
st.markdown("Notes: ARIMA/SARIMA can be sensitive to parameter choices and data length. Use auto-safe defaults if unsure. Prophet and LSTM are optional and require extra packages.")
