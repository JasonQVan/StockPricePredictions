#install the dependencies

import yfinance as yf
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


class check_ticker:
    def __init__(self, chosen_ticker):
        self.ticker = chosen_ticker

    def check_ticker(self):
        try:
            ticker_check = yf.Ticker(self.ticker)
            ticker_check.info
            print(1)
        except:
            print(f"{self.ticker} does not exist.")




#get stock data from yfinance
#ticker_symbol = 'TSLA'
#ticker_data = yf.Ticker(ticker_symbol)
#tickerDf = ticker_data.history(period='1d', start='2021-05-01', end=f'{date.today()}')
#tickerDf = tickerDf[['Open']]

ticker = input().upper()
print(ticker)
