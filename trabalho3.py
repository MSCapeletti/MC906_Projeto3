import sys
import math

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

    data['Variation'] = data['Close'].sub(data['Open'])

    #print(data.head())

    x1=data.Date
    y1=data.Close
    fig = px.scatter(data, x='Date', y='Close')
    fig.show()