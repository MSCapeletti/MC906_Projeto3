import sys
import math

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime
import plotly.graph_objects as go


def flatten_list(dataframe):
    flat_list = []
    for sublist in dataframe.values:
        for item in sublist:
            flat_list.append(item)
    return flat_list


# Esse programa tenta predizer o valor de fechamento da ação na data fornecida, baseando-se na data e os valores de abertura anteriores e previstos
# através de uma regressão linear múltipla

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

    print("Correlation between Date and Closing values:")
    print(data[['Date_delta', 'Open', 'Close']].corr())

    maxDate = data['Date_delta'].max()

    x_v = data[['Date_delta', 'Open']]
    y_v = data[['Close']]

    model = LinearRegression()
    model.fit(x_v, y_v)

    # print("Coefficients and constant found:")
    # print(model.coef_, model.intercept_)

    # print("Real values for the period provided, predicted values for the same period:")
    # print(y_v, model.predict(x_v))

    print("model score:")
    print(round(model.score(x_v, y_v), 5))

    dates_deltaToPredict = np.arange(maxDate+1, ((dateToPredict - data['Date'].min()) / np.timedelta64(1, 'D'))+1)

    dates_deltaToPredict = [int(i) for i in dates_deltaToPredict]

    datesToPredict = []

    for date_delta in dates_deltaToPredict:
        date = data['Date'].min() + np.timedelta64(int(date_delta), 'D')
        datesToPredict.append(date)


    print("Closing values predicted for the dates:")

    # Previsão do valor de fechamento para as datas após o último dia dos dados disponíveis,
    # para valor de abertura do primeiro dia da previsão utiliza-se o valor de fechamento do último dia dos dados disponíveis
    # para valor de abertura dos demais dias após isso utiliza-se o valor previsto de fechamento do dia anterior
    openValue = data.loc[len(data['Open'])-1, 'Open']

    i = 0
    closing_values = []
    df2 = pd.DataFrame(columns=['Date_delta'])
    df3 = pd.DataFrame(columns=['Close'])
    while i < len(dates_deltaToPredict):

        dd = dates_deltaToPredict[i]
        date = datesToPredict[i]
        closingValue = model.predict([[dd, openValue]])[0][0]
        closing_values.append(closingValue)
        print("Date: "+date.strftime('%Y-%m-%d')+", closing value: "+ str(round(closingValue, 2)))
        df2 = df2.append({'Date_delta': dates_deltaToPredict[i]}, ignore_index=True)
        df3 = df3.append({'Close': closingValue}, ignore_index=True)
        openValue = closingValue

        i = i + 1

    dates = x_v[['Date_delta']]
    dates = dates.append(df2, ignore_index=True)
    close_v = data[['Close']]
    close_v = close_v.append(df3, ignore_index=True)


    dates = flatten_list(dates)
    flat_list = []
    for sublist in close_v.values:
        for item in sublist:
            flat_list.append(item)
    close_v = flat_list

    # Graph showing the close stock values predicted
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=dates_deltaToPredict, y=closing_values, mode='lines', name='close'))
    # fig.show()

    # Graph correlating the Open and Close stock alues
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data['Open'], y=data['Close'], mode='lines', name='close'))
    # fig.show()

    # Generating a comparative graph between the prices we have and future prices our model predicted
    # fig = go.Figure()
    # fig.add_trace((go.Scatter(x=dates, y=data['Close'], mode='lines', name='real_data')))
    # fig.add_trace((go.Scatter(x=df2['Date_delta'], y=closing_values, mode='lines', name='future_data')))
    # fig.show()


    # px.scatter(data, x='Open', y='Close', )
