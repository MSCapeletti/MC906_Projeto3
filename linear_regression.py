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

    if len(sys.argv) >= 3:
        data = pd.read_csv(sys.argv[1])
        dateToPredict = datetime.fromisoformat(
            sys.argv[2])  # Format must be 'YYYY-MM-DD'
    else:
        print("Provide the csv with historical data for analysis and a date to predict the closing value")
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

    X = data[['Open', 'High', 'Low', 'Volume', 'Close']]
    X.pop(0)  # use last day's values to predict next day's close

    Y = data[['Close']].pop()  # remove last item so they have same size

    model = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=0)

    model.fit(x_train, y_train)

    # print("Coefficients and constant found:")
    # print(model.coef_, model.intercept_)

    # print("Real values for the period provided, predicted values for the same period:")
    # print(y_v, model.predict(x_v))

    print("model score:")
    print(model.score(x_test, y_test))

    # dates_deltaToPredict = np.arange(maxDate+1, ((dateToPredict - data['Date'].min()) / np.timedelta64(1, 'D'))+1 )

    # dates_deltaToPredict = [int(i) for i in dates_deltaToPredict]

    # datesToPredict = []

    # for date_delta in dates_deltaToPredict:

    #     date = data['Date'].min() + np.timedelta64(int(date_delta), 'D')
    #     datesToPredict.append(date)

    # for item in dates_deltaToPredict:
    #     print(model.predict([[item]]))
