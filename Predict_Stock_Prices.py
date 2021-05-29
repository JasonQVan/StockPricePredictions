# install the dependencies
import yfinance as yf
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# check the ticker to see if exist
class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker

    def check_ticker(self):
        ticker_info = yf.Ticker(self.ticker)
            if not ticker_info.info:
                return False
            else:
                return True


# pulls the necessary data
class GetData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.today_date = date.today()
        self.begin_month = self.today_date.replace(month=1, day=1)

    def get_data(self):
        chosen_ticker = yf.Ticker(self.ticker)
        raw_ticker_data = chosen_ticker.history(period="1d", start=self.begin_month, end=self.today_date, output='pandas')
        return raw_ticker_data


# process the raw data
class ProcessData():
    def __init__(self, t_data):
        self.ticker_data = t_data

    def get_days(self, days):
        return days

    def prediction_column(self):
        ticker_data['Prediction'] = ticker_data['Close'].shift(-1)  # adds a column Predictions next to Close column
        ticker_data.dropna(inplace=True)  # removes any missing values
        return ticker_data

    def data_to_array(self):
        X = np.array(ticker_data.drop(['Prediction'], 1))
        Y = np.array(ticker_data['Prediction'])
        return X, Y


ticker_code = input("Enter ticker.").upper()
tk = Ticker(ticker_code)
gd = GetData(ticker_code)
ticker_data = gd.get_data()
pd = ProcessData(ticker_data)
try:
    forecast_time = pd.get_days(int(input('Enter number of days to forecast:')))
except ValueError:
    print("Enter number of days.")
ticker_data = pd.prediction_column()
x, y = pd.data_to_array()
x = preprocessing.scale(x)
x_prediction = x[-forecast_time:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
clf = LinearRegression()
clf.fit(x_train, y_train)
prediction = (clf.predict(x_prediction))
print(prediction)
