import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="RNN Stock Prediction & Profit Estimator",
    layout="wide"
)

st.title("RNN Stock Price Forecast & Profit Estimation")
st.write("Trained on long history. Forecasted cautiously.")

# ---------------------------------
# Load assets
# ---------------------------------
@st.cache_resource
def load_assets():
    model = load_model("rnn_model.h5")

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    X_test = np.load("X_test_seq.npy")
    y_test = np.load("y_test_seq.npy")

    return model, scaler_y, meta, X_test, y_test


model, scaler_y, meta, X_test_seq, y_test_seq = load_assets()

WINDOW_SIZE = meta["window_size"]
FEATURES = meta["features"]

TARGET_FEATURE = "Close"
TARGET_IDX = FEATURES.index(TARGET_FEATURE)

# ---------------------------------
# Sidebar inputs
# ---------------------------------
st.sidebar.header("Prediction Settings")

time_unit = st.sidebar.selectbox(
    "Prediction Unit",
    ["Days", "Months", "Years"]
)

horizon_value = st.sidebar.slider(
    "Prediction Horizon",
    min_value=1,
    max_value=60 if time_unit == "Days" else 12,
    value=1
)

investment = st.sidebar.number_input(
    "Investment Amount",
    min_value=1000,
    value=10000,
    step=1000
)

# ---------------------------------
# Convert to trading days
# ---------------------------------
if time_unit == "Days":
    future_days = horizon_value
elif time_unit == "Months":
    future_days = horizon_value * 21
else:
    future_days = horizon_value * 252

# Hard cap based on window size
MAX_FORECAST_DAYS = WINDOW_SIZE * 2

if future_days > MAX_FORECAST_DAYS:
    st.warning(
        f"Forecast horizon capped to {MAX_FORECAST_DAYS} trading days "
        "based on model window size."
    )
    future_days = MAX_FORECAST_DAYS

# ---------------------------------
# Prepare last window
# ---------------------------------
last_window = X_test_seq[-1:]      # shape: (1, window_size, features)
current_window = last_window.copy()

# ---------------------------------
# Recursive prediction (RNN)
# ---------------------------------
future_predictions_scaled = []

for _ in range(future_days):
    pred_scaled = model.predict(current_window, verbose=0)
    future_predictions_scaled.append(pred_scaled[0, 0])

    # shift window left
    current_window = np.roll(current_window, -1, axis=1)

    # update ONLY target feature
    current_window[0, -1, TARGET_IDX] = pred_scaled[0, 0]

# Inverse scale predictions
future_predictions = scaler_y.inverse_transform(
    np.array(future_predictions_scaled).reshape(-1, 1)
)

# ---------------------------------
# Profit calculation
# ---------------------------------
last_actual_price = scaler_y.inverse_transform(
    y_test_seq[-1].reshape(1, -1)
)[0][0]

predicted_price = future_predictions[-1][0]

price_change = predicted_price - last_actual_price
percentage_return = (price_change / last_actual_price) * 100
profit = (price_change / last_actual_price) * investment

# ---------------------------------
# Display metrics
# ---------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Last Actual Price", f"{last_actual_price:.2f}")
col2.metric("Predicted Price", f"{predicted_price:.2f}")
col3.metric("Expected Return (%)", f"{percentage_return:.2f}%")
col4.metric("Investment", f"{investment:.0f}")

st.subheader("Profit / Loss Estimation")

if profit >= 0:
    st.success(f"Expected Profit: ₹{profit:.2f}")
else:
    st.error(f"Expected Loss: ₹{abs(profit):.2f}")

# ---------------------------------
# Plot forecast
# ---------------------------------
st.subheader("Future Price Forecast")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(future_predictions, marker="o")
ax.set_xlabel("Future Trading Days")
ax.set_ylabel("Predicted Price")
ax.set_title("RNN Forecasted Prices")
st.pyplot(fig)

# ---------------------------------
# Decision signal
# ---------------------------------
st.subheader("Model Signal")

if percentage_return > 2:
    st.success("BUY 📈")
elif percentage_return < -2:
    st.error("SELL 📉")
else:
    st.warning("HOLD ⚖️")

st.warning(
    "This RNN uses recursive multi-step forecasting. "
    "Only the target feature is updated; other features "
    "are assumed to remain stable."
)

st.caption(
    "This application is for educational purposes only. "
    "Model predictions are not financial advice."
)
