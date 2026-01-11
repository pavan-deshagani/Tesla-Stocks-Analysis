import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model

# ------------------------------
# Load artifacts
# ------------------------------
model = load_model("lstm_model.h5")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")
meta = joblib.load("meta.pkl")

WINDOW_SIZE = meta["window_size"]
FEATURES = meta["features"]

TARGET_FEATURE = "Close"
TARGET_IDX = FEATURES.index(TARGET_FEATURE)

st.set_page_config(page_title="LSTM Stock Price Predictor", layout="wide")
st.title("📈 LSTM Stock Price Forecast")

st.info(
    "This LSTM model is trained for **next-day prediction**. "
    "Multi-day forecasts are generated recursively and become less reliable "
    "as the horizon increases."
)

# ------------------------------
# Sidebar inputs
# ------------------------------
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")

time_unit = st.sidebar.selectbox(
    "Prediction Unit",
    ["Days", "Months", "Years"]
)

horizon_value = st.sidebar.slider(
    "Prediction Horizon",
    min_value=1,
    max_value=60 if time_unit == "Days" else 12,
    value=5
)

period = st.sidebar.selectbox("Historical Data Used", ["1y", "2y", "5y", "10y"])

# ------------------------------
# Convert horizon to trading days
# ------------------------------
if time_unit == "Days":
    future_days = horizon_value
elif time_unit == "Months":
    future_days = horizon_value * 21
else:
    future_days = horizon_value * 252

MAX_FORECAST_DAYS = WINDOW_SIZE * 2

if future_days > MAX_FORECAST_DAYS:
    st.warning(
        f"Forecast horizon capped to {MAX_FORECAST_DAYS} trading days "
        "based on model window size."
    )
    future_days = MAX_FORECAST_DAYS

# ------------------------------
# Fetch data
# ------------------------------
df = yf.download(ticker, period=period)

if df.empty:
    st.error("Invalid ticker or no data available.")
    st.stop()

# ------------------------------
# Feature engineering (MUST MATCH TRAINING)
# ------------------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["STD20"] = df["Close"].rolling(20).std()

df = df.dropna()

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    st.error(f"Missing features in data: {missing}")
    st.stop()

X = df[FEATURES]

if len(X) < WINDOW_SIZE:
    st.error(
        f"Not enough data after feature engineering. "
        f"Need at least {WINDOW_SIZE} rows, got {len(X)}."
    )
    st.stop()

# ------------------------------
# Scale & prepare window
# ------------------------------
X_scaled = x_scaler.transform(X)
current_window = np.expand_dims(X_scaled[-WINDOW_SIZE:], axis=0)

# ------------------------------
# Recursive LSTM prediction
# ------------------------------
future_predictions_scaled = []

for _ in range(future_days):
    pred_scaled = model.predict(current_window, verbose=0)
    future_predictions_scaled.append(pred_scaled[0, 0])

    # shift window
    current_window = np.roll(current_window, -1, axis=1)

    # update only target feature
    current_window[0, -1, TARGET_IDX] = pred_scaled[0, 0]

# Inverse scaling
future_predictions = y_scaler.inverse_transform(
    np.array(future_predictions_scaled).reshape(-1, 1)
)

# ------------------------------
# Display results
# ------------------------------
last_price = float(df["Close"].iloc[-1])
predicted_price = future_predictions[-1][0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Last Close Price", f"{last_price:.2f}")

with col2:
    st.metric(
        f"Predicted Price after {future_days} trading days",
        f"{predicted_price:.2f}"
    )

# ------------------------------
# Plot
# ------------------------------
st.subheader("📉 Forecasted Price Path")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(future_predictions, marker="o", label="Predicted")
ax.set_xlabel("Future Trading Days")
ax.set_ylabel("Predicted Price")
ax.legend()
st.pyplot(fig)

st.warning(
    "Multi-step forecasts reuse the same model recursively. "
    "Only the Close price is updated; other features are assumed stable."
)

st.caption(
    "This application is for educational purposes only. "
    "Model predictions are not financial advice."
)
