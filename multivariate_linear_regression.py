import sys
import math

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime
import visuals_and_graphics as vag
from process import process


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

    trainData = data[data['Date_delta'] <= 0.8*maxDate]
    validationData = data[data['Date_delta'] > 0.8*maxDate]

    X = trainData[features]
    Y = trainData['Close']
    # use last day's values to predict next day's close
    # remove first day from target
    Y = Y.drop([0])
    Y = Y.reset_index(drop=True)

    # remove the last day from input
    X = X.drop([X.shape[0] - 1])
    X.reset_index(drop=True)

    model = LinearRegression()
    model.fit(X, Y)

    predicted_data = pd.DataFrame(columns=['Date', 'Close'])
    closingValues = model.predict(validationData[features])

    for idx, value in enumerate(closingValues):
        if idx + 1 < validationData.shape[0]:
            date = validationData['Date'].values[idx + 1]
        else:
            date = validationData['Date'].values[idx] + np.timedelta64(1, 'D') # predict next day
        predicted_data = predicted_data.append({'Date': date, 'Close': value}, ignore_index=True)

    # Grafico com os Close históricos e previstos para as mesmas datas
    vag.fig_real_predicted_values(
        data, predicted_data, 'Comparacao entre previsão e real, previsao feita com os dados historicos anteriores')
