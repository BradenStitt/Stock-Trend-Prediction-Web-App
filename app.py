import streamlit as sl
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

sl.title("Stock Prediction App")

# include all stock tickers in the S&P 500, as well as the S&P 500 itself, Dow Jones, and NASDAQ
stocks = ("SPY", "DJIA", "NDAQ", "AAPL", "MSFT", "GOOG", "AMZN", "FB", "TSLA", "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "MA", "HD", "VZ", "DIS", "BAC", "INTC", "CMCSA", "PFE", "ADBE", "NFLX", "NVDA", "PYPL", "CRM", "T", "TMO", "AVGO", "CSCO", "KO", "PEP", "ABT", "ORCL", "COST", "ACN", "MCD", "QCOM", "NEE", "MDLZ", "TXN", "LLY", "UNP", "WFC", "BMY", "DHR", "LIN", "IBM", "AMGN", "LOW", "SBUX", "UPS", "CVX", "MMM", "HON", "AMT", "AXP", "GILD", "GS", "BA", "CAT", "XOM", "MCK", "MDT", "WMT", "DOW", "INTU",
          "BKNG", "CHTR", "ZTS", "FIS", "MU", "CVS", "ANTM", "PLD", "BLK", "GPN", "USB", "ISRG", "TGT", "WBA", "NKE", "PNC", "SYK", "TMUS", "C", "LMT", "SPGI", "MS", "DUK", "SO", "RTX", "DE", "ADP", "MDLZ", "ZBH", "HUM", "CI", "EL", "ROST", "CME", "COP", "ETN", "EOG", "ICE", "EXC", "AEP", "AIG", "AMAT", "APH", "ADI", "ANSS", "ANTM", "AON", "APA", "AIV", "AAP", "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALXN", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AM")
selected_stocks = sl.selectbox("Select dataset for prediction", stocks)
number_years = sl.slider("Years of prediction:", 1, 5)
period = number_years * 365


@sl.cache  # Cache the data
def load_data(ticker):
    """Load the data"""
    # data needs to display date
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = sl.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

sl.subheader("Raw data")
sl.write(data.tail())


def plot_raw_data():
    """Plot raw data"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"],
                  y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    sl.plotly_chart(fig)


plot_raw_data()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

sl.subheader("Forecast data")
sl.write(forecast.tail())

sl.write("Forecast data")
figure1 = plot_plotly(model, forecast)
sl.plotly_chart(figure1)

sl.write("Forecast components")
figure2 = model.plot_components(forecast)
sl.write(figure2)
