import yfinance as yf
import streamlit as st

st.write("""
# Simple Stock Price App
Shown are the stock **closing price** and **volume** of Google
""")

# define the ticker symbol
tickerSymbol = 'GOOGL'
# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2020-1-1', end='2020-12-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)

st.write("""
## Volume Price
""")
st.bar_chart(tickerDf.Volume)
