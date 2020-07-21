import sys
import math

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


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

    data['Date_delta'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')

    data['Variation'] = data['Close'].sub(data['Open'])

    #print(data.head())

    # x1=data.Date
    # y1=data.Close
    # fig = px.scatter(data, x='Date', y='Close')
    # fig.show()

    x_v = data[['Open', 'High', 'Low', 'Date_delta']]
    y_v = data[['Close']]

    model = LinearRegression()
    model.fit(x_v, y_v)

    print("Coefficients and constant found:")
    print(model.coef_, model.intercept_)
