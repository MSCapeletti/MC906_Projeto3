import sys
from datetime import datetime
import pandas as pd
import numpy as np

def process():
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

    maxDate = data['Date_delta'].max()

    # Diferença em dias em relação ao primeiro dia disponivel no csv, começando a partir do ultimo dia disponivel no csv
    dates_deltaToPredict = np.arange(maxDate+1, ((dateToPredict - data['Date'].min()) / np.timedelta64(1, 'D'))+1)
    # Conversão dos itens da lista para int
    dates_deltaToPredict = [int(i) for i in dates_deltaToPredict]

    # Lista com as datas correspondentes aos dates_delta
    datesToPredict = []
    for date_delta in dates_deltaToPredict:
        date = data['Date'].min() + np.timedelta64(int(date_delta), 'D')
        datesToPredict.append(date)

    return data, dates_deltaToPredict, datesToPredict