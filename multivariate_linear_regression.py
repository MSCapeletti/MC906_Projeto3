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

def adjustDateForModel(X, Y):
    # use last day's values to predict next day's close
    # remove first day from target
    Y = Y.drop([0])
    Y = Y.reset_index(drop=True)

    # remove the last day from input
    X = X.drop([X.shape[0] - 1])
    X = X.reset_index(drop=True)
    return X, Y

def fittedModel(trainData, features, target):
    model = LinearRegression()

    X = trainData[features]
    Y = trainData[target]

    X, Y = adjustDateForModel(X, Y)

    model.fit(X, Y)

    return model

def predictNextDates(lastDayData, dates):
    featuresModels = dict()
    for target in features:
        newModel = fittedModel(trainData, features, target)
        featuresModels[target] = newModel

    predictedData = pd.DataFrame(columns=data.columns)
    predictedData = predictedData.append(lastDayData)
    idx = 1
    while predictedData.shape[0] < dates.shape[0]:
        lastDayFeatures = predictedData[features].tail(1)
        open_value = featuresModels['Open'].predict(lastDayFeatures)
        close_value = featuresModels['Close'].predict(lastDayFeatures)
        low_value = featuresModels['Low'].predict(lastDayFeatures)
        high_value = featuresModels['High'].predict(lastDayFeatures)
        volume_value = featuresModels['Volume'].predict(lastDayFeatures)
        date = dates[idx]

        predictedData = predictedData.append({
            'Date': date,
            'Open': open_value[0],
            'Close': close_value[0],
            'Low': low_value[0],
            'High': high_value[0],
            'Volume': volume_value[0]
        }, ignore_index=True)

        idx = idx + 1

    predictedData = predictedData.drop([0])

    return predictedData

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

    maxDate = data['Date_delta'].max()

    features = ['Open', 'High', 'Low', 'Volume', 'Close']

    trainData = data[data['Date_delta'] <= 0.8*maxDate]
    validationData = data[data['Date_delta'] > 0.8*maxDate]

    X = trainData[features]
    Y = trainData['Close']
    
    X, Y = adjustDateForModel(X, Y)

    model = fittedModel(trainData, features, 'Close')

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
        validationData, predicted_data, 'Comparação entre o histórico e a predição realizada pelo modelo.')
    

    # -------------------------------------------------------------------------------------------
    # try a prediction of more days by predicting each value conseuctively
    # -------------------------------------------------------------------------------------------
    predicted_data = predictNextDates(validationData.head(1), validationData['Date'].values)
    vag.fig_real_predicted_values(
        validationData, predicted_data, 'Comparação entre o histórico e a predição realizada pelo modelo.')

    future_dates = np.array(validationData['Date'].tail(1).values)
    for i in range(20):
        future_dates = np.append(future_dates, future_dates[i] + np.timedelta64(1, 'D'))

    predicted_data = predictNextDates(validationData.tail(1), future_dates)
    vag.fig_real_predicted_values(
        validationData, predicted_data, 'Comparação entre o histórico e a predição realizada pelo modelo.')
