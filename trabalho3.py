import pandas as pd
import sys


if __name__ == '__main__':
    
    if len(sys.argv) >= 2:
        data = pd.read_csv(sys.argv[1])
    else:
        print("Provide the csv with historical data for analysis")
        sys.exit()

    print(data.head())