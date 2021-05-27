# install the dependencies
import yfinance as yf
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,
from sklearn.model_selection import train_test_split


# check the ticker to see if exist
class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker

    def check_ticker(self):
            ticker_info = yf.Ticker(self.ticker)
            if len(ticker_info.info) == 1:
                return False
            else:
                return True

#pulls the necessary data
class GetData():
    def __init__(self, ticker):
        self.ticker = ticker
        self.today_date = date.today()
        self.begin_month = self.today_date.replace(day=1)

    def get_data(self):
        chosen_ticker = yf.Ticker(self.ticker)
        tickerData = chosen_ticker.history(period="1d", start=self.begin_month, end=self.today_date)
        print(tickerData[['Close']])

    def plot_data(self):
        chosen_ticker = yf.Ticker(self.ticker)
        tickerData = chosen_ticker.history(period="1d", start=self.begin_month, end=self.today_date)


ticker_code = input().upper()
tk = Ticker(ticker_code)
gd = GetData(ticker_code)

today_date = date.today()
begin_month = today_date.replace(month=1, day=1)
days = 5

ticker_data = yf.Ticker(ticker_code)
tickerData = ticker_data.history(period="1d", start=begin_month, end=today_date, output_format='pandas')
tickerData['Prediction'] = tickerData['Close'].shift(-1)
tickerData.dropna(inplace=True)
forecast_time = days

X = np.array(tickerData.drop(['Prediction'], 1))
Y = np.array(tickerData['Prediction'])
X = preprocessing.scale(X)
X_prediction = X[-forecast_time:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
clf = LinearRegression()
clf.fit(X_train, Y_train)
prediction = (clf.predict(X_prediction))

print(prediction)
