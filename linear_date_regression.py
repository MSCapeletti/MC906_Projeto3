import sys
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import date
from datetime import datetime
import visuals_and_graphics as vag
from process import process

# Esse programa tenta predizer o valor de fechamento da ação na data fornecida, baseando-se apenas nesses 2 parâmetros
# através de uma regressão linear simples, pois para valores futuros de uma ação é improvável que alguém possua informações como
# data de abertura, alta, baixa, média e volume

data, dates_deltaToPredict, datesToPredict = process()

print("Correlation between Date and Closing values:")
print(data[['Date_delta', 'Close']].corr())

x_v = data[['Date_delta']]
y_v = data[['Close']]

# modelo para a predição dos valores futuros
model = LinearRegression()
model.fit(x_v, y_v)

# print("Real values for the period provided, predicted values for the same period:")
# print(y_v, model.predict(x_v))

print("model score:")
print(round(model.score(x_v, y_v), 5))

# Previsão do valor de fechamento para as datas após o último dia dos dados disponíveis
i = 0
df2 = pd.DataFrame(columns=['Date_delta', 'Close'])
while i < len(dates_deltaToPredict):

    dd = dates_deltaToPredict[i]
    closingValue = model.predict([[dd]])[0][0]
    date = datesToPredict[i]
    df2 = df2.append({'Date': date, 'Close': closingValue}, ignore_index=True)

    i = i + 1

# Graph correlating the Open and Close stock values
vag.fig_sct_open_close(data)

# Generating a comparative graph between the prices we have and future prices our model predicted
vag.fig_real_predicted_values(data, df2, 'Regressao Linear Simples usando Date')


maxDate = data['Date_delta'].max()
trainData = data[data['Date_delta'] <= 0.8*maxDate]
validationData = data[data['Date_delta'] > 0.8*maxDate]
x_t = trainData[['Date_delta']]
y_t = trainData[['Close']]

# modelo para comparação dos dados previstos com os reais
trainModel = LinearRegression()
trainModel.fit(x_t, y_t)

df3 = pd.DataFrame(columns=['Date_delta', 'Close'])
for dd in validationData['Date_delta']:
    date = validationData[validationData['Date_delta'] == dd]['Date'].values[0]
    closingValue = trainModel.predict([[dd]])[0][0]
    df3 = df3.append({'Date': date, 'Close': closingValue}, ignore_index=True)

# Grafico com os Close históricos e previstos para as mesmas datas
vag.fig_real_predicted_values(data, df3, 'Comparacao entre previsão e real, previsao feita com os dados historicos anteriores')



