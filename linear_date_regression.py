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

df3 = data.copy()
df3['Close'] = pd.DataFrame(model.predict(df3[['Date_delta']]).tolist(), columns=['Close'])

# Graph correlating the Open and Close stock values
vag.fig_sct_open_close(data)

# Grafico com os Close históricos e previstos para as mesmas datas
vag.fig_real_predicted_values(data, df3, 'Regressao Linear Simples usando Date')

# Generating a comparative graph between the prices we have and future prices our model predicted
vag.fig_real_predicted_values(data, df2, 'Regressao Linear Simples usando Date')

