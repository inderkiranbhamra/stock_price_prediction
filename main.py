import streamlit as st
from datetime import date
import numpy as np

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "FB", "GOOG", "TSLA", "AMZN", "GRFS", "TAL", "BGC", "MSFT", "GME")
selected_stock = st.selectbox("Select a stock", stocks)

years = (1, 2, 3, 4, 5)
n_years = st.selectbox("Years to predict", years)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading Data....")
data = load_data(selected_stock)
data_load_state.text("Loading Done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Calculate accuracy
actual_data = data.set_index('Date')['Close']
predicted_data = forecast.set_index('ds')['yhat'].iloc[:-period]

mae = np.mean(np.abs(predicted_data - actual_data))
mape = np.mean(np.abs((predicted_data - actual_data) / actual_data)) * 100
accuracy = 100 - mape

# Display accuracy
st.subheader('Prediction Accuracy')
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
st.write(f"Accuracy: {accuracy:.2f}%")

# Display the filtered forecast data
today_date = np.datetime64(date.today())
filtered_forecast = forecast[forecast['ds'] >= today_date]

st.subheader('Forecast data after today')
st.write(filtered_forecast)

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
