import sys
import math

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        data = pd.read_csv(sys.argv[1])
    else:
        print("Provide the csv with historical data for analysis")
        sys.exit()

    # Drop NaN rows
    data = data.dropna()

    # Reset index so that there are no missing indexes from the removed rows
    data = data.reset_index(drop=True)

    data['Date'] = pd.to_datetime(data['Date'])

    # Differente between the dates in days
    data['Date_delta'] = (data['Date'] - data['Date'].min()
                          ) / np.timedelta64(1, 'D')

    data['Variation'] = data['Close'].sub(data['Open'])

    print("Correlation between data")
    print(data[['Date_delta', 'Close', 'Open', 'High', 'Low', 'Volume']].corr())

    # x1=data.Date
    # y1=data.Close
    # fig = px.scatter(data, x='Date', y='Close')
    # fig.show()

    maxDate = data['Date_delta'].max()

    features = ['Open', 'High', 'Low', 'Volume', 'Close']

    X = data[features]
    Y = data[['Close']]

    # use last day's values to predict next day's close
    X = X.drop([0])
    X = X.reset_index()

    # remove last item so they have same size
    Y = Y.drop([Y.size - 1])  

    model = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=0)

    model.fit(x_train, y_train)

    print("model score:")
    print(model.score(x_test, y_test))

    print('coefficients:')
    print(model.coef_)
