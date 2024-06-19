import yfinance as yf
import streamlit as st

st.write("""
    # Simple Stock Price App

    Shown are the stock **closing price** and **volume** of Google!

""")

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
tickerDf = tickerData.history(start='2010-5-31', end='2023-5-31')

st.write("""
    ### Closing Price
""")
st.line_chart(tickerDf.Close)

st.write("""
    ### Volume Price
""")
st.line_chart(tickerDf.Volume)
