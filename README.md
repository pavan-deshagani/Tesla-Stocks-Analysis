# 📈 Stock Price Prediction Using Deep Learning (RNN & LSTM)

## 📌 Project Overview
This project focuses on predicting stock prices using deep learning techniques, specifically **SimpleRNN** and **LSTM (Long Short-Term Memory)** models. Historical stock market data is used to learn time-series patterns and forecast future stock prices. The project also includes **interactive Streamlit applications** to visualize predictions and simulate short-term forecasts.

The implementation demonstrates how deep learning models can be applied to financial time-series data while maintaining realistic assumptions and constraints.

---

## 🎯 Problem Statement
To create a predictive deep learning model that forecasts the stock price of Tesla using previous stock data, by learning temporal dependencies in historical price movements and evaluating short-term future trends.

---

## 🧠 Models Used
- **SimpleRNN**  
  - Used as a baseline model for sequence learning.
  - Suitable for short-term dependencies.

- **LSTM (Long Short-Term Memory)**  
  - Designed to capture long-term dependencies in time-series data.
  - Provides more stable and reliable predictions compared to SimpleRNN.

Both models are trained using a **fixed window size** approach for next-day stock price prediction.

---

## 📊 Dataset Description
The historical stock data includes the following features:

- **Date** – Trading day reference  
- **Open** – Opening price of the stock  
- **High** – Highest price during the day  
- **Low** – Lowest price during the day  
- **Close** – Closing price of the stock  
- **Adj Close** – Adjusted closing price accounting for splits/dividends  
- **Volume** – Number of shares traded  

Additional technical indicators such as moving averages and rolling standard deviation were engineered for model training.

---

## 🧩 Project Structure
├── notebook.ipynb # Data preprocessing, model training, evaluation
├── app.py # Streamlit app for LSTM next-day prediction
├── app1.py # Streamlit app for RNN multi-step forecasting
├── lstm_model.h5 # Trained LSTM model
├── rnn_model.h5 # Trained SimpleRNN model
├── x_scaler.pkl # Scaler for input features
├── y_scaler.pkl # Scaler for target variable
├── scaler_y.pkl # Target scaler (RNN)
├── meta.pkl # Metadata (window size, feature list)
├── X_test_seq.npy # Test input sequences
├── y_test_seq.npy # Test target values
└── README.md # Project documentation




## 🔍 Explanation of Key Files

### 📓 `notebook.ipynb`
- Data collection and preprocessing
- Feature engineering and scaling
- Window-based sequence creation
- Training and evaluation of RNN and LSTM models
- Performance analysis using metrics like MSE and RMSE

---

### 🟢 `app.py` – LSTM Next-Day Prediction App
- Uses the trained **LSTM model**
- Predicts the **next trading day’s closing price**
- Fetches live historical data using `yfinance`
- Applies the same preprocessing and scaling used during training
- Displays:
  - Last closing price
  - Predicted next-day closing price
  - Price trend visualization

> ⚠️ This app performs **only one-step (next-day) prediction**, as per the training objective.

---

### 🔵 `app1.py` – RNN Forecasting & Profit Simulation App
- Uses the trained **SimpleRNN model**
- Performs **recursive multi-step forecasting**
- Supports prediction horizons in:
  - Days
  - Months
  - Years (internally converted to trading days)
- Maintains window size consistency during recursive prediction
- Estimates:
  - Future stock price
  - Expected return percentage
  - Profit or loss based on user investment
- Includes warnings about forecasting assumptions and limitations

---

## 🪟 Windowing Concept
- A fixed **window size** is used during training and inference.
- Each input sample consists of historical data over the window length.
- Multi-step predictions are generated using **recursive forecasting**, where each prediction is fed back into the window.
- Only the target feature (Close price) is updated during recursion, while other features are assumed stable.

---

## 📈 Streamlit Applications
Both applications provide an interactive interface for:
- Selecting prediction parameters
- Visualizing model outputs
- Understanding model behavior and limitations


python -m streamlit run app.py
python -m streamlit run app1.py

⚠️ Limitations
Stock markets are highly volatile and influenced by external factors.

Long-term forecasts become less reliable due to error accumulation.

Models assume historical patterns will repeat, which may not always hold true.

✅ Conclusion
This project demonstrates the practical use of deep learning models for stock price prediction using historical data. By combining RNN and LSTM architectures with proper time-series preprocessing and interactive deployment, the project highlights both the potential and the limitations of deep learning in financial forecasting.

📌 Disclaimer
This project is intended for educational and research purposes only.
The predictions generated by the models should not be considered financial advice.



### Final note (important)
This README is:
- ✅ Honest about model behavior  
- ✅ Aligned with your actual code  
- ✅ Safe for GitHub, viva, and interviews  
- ✅ Not over-claiming results  

If you want, next I can:
- Shorten this for **resume/GitHub preview**
- Add **screenshots section**
- Add **requirements.txt**
- Write a **project abstract** separately  

Just say what you want next.
