import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')



class Main():


    def __init__(self):
        pass

    
    def __new__(self):
        tickerSymbol = st.text_input('Please Enter a Valid Stock Ticker!', 'GOOGL')
        st.write(f"""

        # Simple Stock Price App

        Shown are the stock closing price and volume of {tickerSymbol}!

        """)


        tickerData = yf.Ticker(tickerSymbol)

        tickerDf = tickerData.history(period='id', start='2010-5-31', end='2020-5-31')

        st.line_chart(tickerDf.Close)
        st.line_chart(tickerDf.Volume)


if __name__ == '__main__':
    Main()