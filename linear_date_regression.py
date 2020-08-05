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


# Esse programa tenta predizer o valor de fechamento da ação na data fornecida, baseando-se apenas nesses 2 parâmetros
# através de uma regressão linear simples, pois para valores futuros de uma ação é improvável que alguém possua informações como
# data de abertura, alta, baixa, média e volume

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
    correlation = data[['Date_delta', 'Close']].corr()
    print(correlation)
    # graph showing the correlation got
    # fig = px.scatter(correlation, x='Date_delta', y='Close')
    # fig.show()

    # x1=data.Date
    # y1=data.Close
    # fig = px.scatter(data, x='Date', y='Close')
    # fig.show()

    maxDate = data['Date_delta'].max()

    x_v = data[['Date_delta']]
    y_v = data[['Close']]

    model = LinearRegression()
    model.fit(x_v, y_v)

    # print("Coefficients and constant found:")
    # print(model.coef_, model.intercept_)

    print("Real values for the period provided, predicted values for the same period:")
    print(y_v, model.predict(x_v))
    # for i in model.predict(x_v):
    #     i
    predictions = model.predict(x_v).tolist()
    # ToDo: improve this chunk of code
    # data from predict is flatten into a list for the y axis
    flat_list = []
    for sublist in predictions:
        for item in sublist:
            flat_list.append(item)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=y_v['Close'], mode='lines', name='real_values', ))
    fig.update_layout(
        xaxis_title="Dates",
        yaxis_title="Close Values",
    )
    fig.add_trace(go.Scatter(x=data['Date'], y=flat_list, mode='lines', name='predictions'))
    fig.show()

    print("model score:")
    print(round(model.score(x_v, y_v), 5))

    dates_deltaToPredict = np.arange(
        maxDate+1, ((dateToPredict - data['Date'].min()) / np.timedelta64(1, 'D'))+1)

    dates_deltaToPredict = [int(i) for i in dates_deltaToPredict]

    datesToPredict = []

    for date_delta in dates_deltaToPredict:

        date = data['Date'].min() + np.timedelta64(int(date_delta), 'D')
        datesToPredict.append(date)



    print("Closing values predicted for the dates:")

    # Previsão do valor de fechamento para as datas após o último dia dos dados disponíveis

    closing_values = []
    i = 0
    while i < len(dates_deltaToPredict):

        dd = dates_deltaToPredict[i]
        date = datesToPredict[i]

        closingValue = model.predict([[dd]])[0][0]
        closing_values.append(closingValue)
        print("Date: "+date.strftime('%Y-%m-%d')+", closing value: "+ str(round(closingValue, 2)))

        i = i + 1

    # Graph showing the predicted closing stock values on dates requested
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=dates_deltaToPredict, y=closing_values, mode='lines', name='close'))
    # fig.show()
