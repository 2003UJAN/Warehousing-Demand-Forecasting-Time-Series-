# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

# Optional packages (Prophet, TensorFlow, Plotly) — app handles absence gracefully
HAS_PROPHET = False
HAS_TF = False
HAS_PLOTLY = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # fallback name
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
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Demand Forecasting — Auto Select + Decomposition", layout="wide")

# -------------------------
# Utils / Generators
# -------------------------
def generate_synthetic_data(days=900):
    rng = np.random.default_rng(42)
    base = 200 + 20 * np.sin(np.linspace(0, 12 * np.pi, days))
    noise = rng.normal(0, 8, days)
    trend = np.linspace(0, 40, days)
    data = base + noise + trend
    dates = pd.date_range(start="2020-01-01", periods=days)
    return pd.Series(data, index=dates)

def ensure_series(ts):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)
    if isinstance(ts, pd.DataFrame):
        ts = ts.iloc[:, 0]
    ts = ts.dropna()
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.date_range(start="2020-01-01", periods=len(ts))
    return ts

# -------------------------
# Forecast functions
# -------------------------
def arima_forecast(ts, steps=30, p=2, d=1, q=2, sp=1, sd=1, sq=1):
    ts = ensure_series(ts)
    model = SARIMAX(ts, order=(p, d, q), seasonal_order=(sp, sd, sq, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    f = res.get_forecast(steps=steps)
    idx = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    mean = f.predicted_mean
    ci = f.conf_int()
    return pd.DataFrame({
        "ds": idx,
        "yhat": mean.values,
        "yhat_lower": ci.iloc[:, 0].values,
        "yhat_upper": ci.iloc[:, 1].values
    }).set_index("ds")

def prophet_forecast(ts, steps=30):
    if not HAS_PROPHET:
        raise RuntimeError("Prophet is not available in this environment.")
    ts = ensure_series(ts)
    df = pd.DataFrame({"ds": ts.index, "y": ts.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=steps)
    fc = m.predict(future)
    out = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(steps)
    out.index = pd.DatetimeIndex(out.index)
    return out

def lstm_forecast(ts, steps=30, epochs=20, window=14):
    if not HAS_TF:
        raise RuntimeError("TensorFlow/Keras is not available in this environment.")
    ts = ensure_series(ts)
    arr = ts.values.reshape(-1, 1)
    gen = TimeseriesGenerator(arr, arr, length=window, batch_size=16)
    model = Sequential([LSTM(32, activation="relu", input_shape=(window, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(gen, epochs=epochs, verbose=0)
    forecast = []
    current = arr[-window:].reshape(1, window, 1)
    for _ in range(steps):
        p = model.predict(current, verbose=0)[0][0]
        forecast.append(p)
        current = np.append(current[:, 1:, :], [[[p]]], axis=1)
    idx = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    s = pd.Series(forecast, index=idx)
    # crude uncertainty band
    std = np.std(forecast) if len(forecast) > 1 else 0.0
    return pd.DataFrame({"yhat": s.values, "yhat_lower": s.values - 1.96*std, "yhat_upper": s.values + 1.96*std}, index=idx)

# -------------------------
# Auto model selection
# -------------------------
def evaluate_models_on_recent(history, forecasts, horizon=30):
    """
    history: pd.Series (indexed by date)
    forecasts: dict{name: forecast_df(index=ds, columns yhat)}
    Compare forecasts against last `horizon` entries of history (if available)
    Returns dicts of MAPE/RMSE and chosen model name (min MAPE)
    """
    history = ensure_series(history)
    recent = history.tail(horizon)
    metrics = {}
    for name, fc in forecasts.items():
        # align: take first len(recent) predictions
        preds = fc["yhat"].values[:len(recent)]
        if len(preds) < 1 or len(preds) != len(recent):
            # can't compute metric reliably — mark as inf
            metrics[name] = {"mape": np.inf, "rmse": np.inf}
            continue
        mape = mean_absolute_percentage_error(recent.values, preds)
        rmse = np.sqrt(mean_squared_error(recent.values, preds))
        metrics[name] = {"mape": float(mape), "rmse": float(rmse)}
    # choose best by mape
    best = min(metrics.items(), key=lambda kv: kv[1]["mape"])[0]
    return metrics, best

# -------------------------
# Decomposition (STL)
# -------------------------
def decompose_series(ts, period=7):
    ts = ensure_series(ts)
    stl = STL(ts, period=period, robust=True)
    res = stl.fit()
    comp_df = pd.DataFrame({
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid
    }, index=ts.index)
    return comp_df

# -------------------------
# Streamlit UI
# -------------------------
st.title("📦 Advanced Demand Forecasting — Auto Model Select & Decomposition")
st.markdown("ARIMA/SARIMA, Prophet, LSTM | Auto-select best model | Interactive plots | Seasonal decomposition")

# Sidebar controls
st.sidebar.header("Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body { background-color: #0b1221; color: #ddd; }</style>", unsafe_allow_html=True)

data_mode = st.sidebar.radio("Data source", ["Synthetic", "Upload CSV"])
n_days = st.sidebar.slider("Synthetic history length (days)", 365, 1500, 900, step=30)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30)
model_choices = st.sidebar.multiselect("Run models:", ["ARIMA", "Prophet", "LSTM"], default=["ARIMA", "Prophet"])
p = st.sidebar.number_input("ARIMA p", 0, 5, 2)
d = st.sidebar.number_input("ARIMA d", 0, 2, 1)
q = st.sidebar.number_input("ARIMA q", 0, 5, 2)
sp = st.sidebar.number_input("Seasonal p", 0, 2, 1)
sd = st.sidebar.number_input("Seasonal d", 0, 1, 1)
sq = st.sidebar.number_input("Seasonal q", 0, 2, 1)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 5, 200, 20)
lstm_window = st.sidebar.number_input("LSTM window", 5, 60, 14)

st.sidebar.markdown("---")
st.sidebar.write("Package availability:")
st.sidebar.write(f"Prophet: {'✅' if HAS_PROPHET else '❌'}")
st.sidebar.write(f"TensorFlow: {'✅' if HAS_TF else '❌'}")
st.sidebar.write(f"Plotly: {'✅' if HAS_PLOTLY else '❌'}")

# Load data
if data_mode == "Synthetic":
    ts = generate_synthetic_data(n_days)
    st.success("Synthetic dataset loaded.")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (first numeric column used)", type=["csv"])
    if uploaded is not None:
        df_u = pd.read_csv(uploaded)
        # pick first numeric column
        numeric_cols = df_u.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.error("Uploaded CSV must contain at least one numeric column.")
            st.stop()
        col = numeric_cols[0]
        s = pd.Series(df_u[col].values)
        s.index = pd.date_range(start="2020-01-01", periods=len(s), freq="D")
        ts = s
        st.success("Uploaded CSV loaded.")
    else:
        st.info("Upload a CSV or switch to Synthetic data.")
        st.stop()

# Show history
with st.expander("Show historical data (last 200 days)"):
    hist_plot = ts.tail(200)
    if HAS_PLOTLY:
        fig = px.line(x=hist_plot.index, y=hist_plot.values, labels={"x":"Date", "y":"Count"}, title="History")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10,3))
        hist_plot.plot(ax=ax)
        ax.set_title("History")
        st.pyplot(fig)

# Decomposition
with st.expander("Seasonal decomposition (STL)"):
    try:
        comp = decompose_series(ts, period=7)
        if HAS_PLOTLY:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend","Seasonal","Residual"))
            fig.add_trace(go.Scatter(x=comp.index, y=comp["trend"], name="Trend"), row=1, col=1)
            fig.add_trace(go.Scatter(x=comp.index, y=comp["seasonal"], name="Seasonal"), row=2, col=1)
            fig.add_trace(go.Scatter(x=comp.index, y=comp["resid"], name="Residual"), row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, axes = plt.subplots(3,1, figsize=(10,6), sharex=True)
            comp["trend"].plot(ax=axes[0], title="Trend")
            comp["seasonal"].plot(ax=axes[1], title="Seasonal")
            comp["resid"].plot(ax=axes[2], title="Residual")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Decomposition error: {e}")

# Run forecasting
if st.button("Run all selected models"):
    forecasts = {}
    errors = {}
    st.info("Running models... this may take a moment.")
    # ARIMA
    if "ARIMA" in model_choices:
        try:
            arima_fc = arima_forecast(ts, steps=forecast_horizon, p=p, d=d, q=q, sp=sp, sd=sd, sq=sq)
            forecasts["ARIMA"] = arima_fc
            st.success("ARIMA done.")
        except Exception as e:
            st.error(f"ARIMA failed: {e}")

    # Prophet
    if "Prophet" in model_choices:
        if not HAS_PROPHET:
            st.error("Prophet not installed — skipping.")
        else:
            try:
                prop_fc = prophet_forecast(ts, steps=forecast_horizon)
                forecasts["Prophet"] = prop_fc
                st.success("Prophet done.")
            except Exception as e:
                st.error(f"Prophet failed: {e}")

    # LSTM
    if "LSTM" in model_choices:
        if not HAS_TF:
            st.error("TensorFlow not installed — skipping LSTM.")
        else:
            try:
                lstm_fc = lstm_forecast(ts, steps=forecast_horizon, epochs=int(lstm_epochs), window=int(lstm_window))
                forecasts["LSTM"] = lstm_fc
                st.success("LSTM done.")
            except Exception as e:
                st.error(f"LSTM failed: {e}")

    if len(forecasts) == 0:
        st.error("No forecasts produced. Check model availability and parameters.")
        st.stop()

    # Auto-select best model using recent history
    metrics, best = evaluate_models_on_recent(ts, forecasts, horizon=min(30, len(ts)))
    st.subheader("Model metrics (on recent history)")
    st.table(pd.DataFrame(metrics).T)

    st.success(f"Auto-selected best model: **{best}** (lowest MAPE)")

    # Combine and plot interactive chart (Plotly if available)
    combined_df = pd.DataFrame(index=next(iter(forecasts.values())).index)
    for name, dfc in forecasts.items():
        combined_df[f"{name}_yhat"] = dfc["yhat"].values
        combined_df[f"{name}_lower"] = dfc["yhat_lower"].values
        combined_df[f"{name}_upper"] = dfc["yhat_upper"].values

    # Plot history + forecasts
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="History", line=dict(color="black")))
        colors = {"ARIMA":"orange","Prophet":"green","LSTM":"red"}
        for name in forecasts:
            fig.add_trace(go.Scatter(x=forecasts[name].index, y=forecasts[name]["yhat"], name=f"{name} forecast", line=dict(color=colors.get(name, None))))
            fig.add_trace(go.Scatter(
                x=forecasts[name].index.tolist() + forecasts[name].index[::-1].tolist(),
                y=forecasts[name]["yhat_upper"].tolist() + forecasts[name]["yhat_lower"].tolist(),
                fill="toself", fillcolor="rgba(0,100,80,0.1)", line=dict(color="rgba(255,255,255,0)"), showlegend=False
            ))
        fig.update_layout(title="Forecast Comparison", xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12,5))
        ts.tail(120).plot(ax=ax, label="History")
        for name in forecasts:
            forecasts[name]["yhat"].plot(ax=ax, label=f"{name} forecast")
            ax.fill_between(forecasts[name].index, forecasts[name]["yhat_lower"], forecasts[name]["yhat_upper"], alpha=0.15)
        ax.legend()
        st.pyplot(fig)

    # Show numeric forecast table and download
    out = pd.DataFrame(index=combined_df.index)
    for name in forecasts:
        out[f"{name}_yhat"] = forecasts[name]["yhat"].values
        out[f"{name}_lower"] = forecasts[name]["yhat_lower"].values
        out[f"{name}_upper"] = forecasts[name]["yhat_upper"].values

    st.subheader("Forecast table (downloadable)")
    st.dataframe(out.round(2))
    csv_buf = io.StringIO()
    out.to_csv(csv_buf)
    st.download_button("Download forecasts CSV", csv_buf.getvalue(), file_name="forecasts.csv", mime="text/csv")

    # Show selected model forecast details
    st.subheader(f"Selected model: {best} — detailed view")
    fc_best = forecasts[best]
    st.line_chart(pd.concat([ts.tail(120), fc_best["yhat"]]))

st.markdown("---")
st.markdown("Notes: Auto-selection uses MAPE on recent history (if available). Seasonality decomposition uses STL with weekly period by default.")
