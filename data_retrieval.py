import pandas as pd
import datetime
import json
from IPython.display import clear_output
import pandas_datareader as pdr
import numpy as np


### Pulling stock list from Virtus LifeSci Biotech ETF

stocks = pd.read_excel('positions_bbc.xls', index_col=0, skiprows=1).Ticker.values

#### Creating dictionary and JSON

dataframe_dict= {}
for i in stocks: 

    df = pdr.get_data_yahoo(i, start='2009-01-01')
    # Change inde to a date  string
    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    
    dataframe_dict[i] = df.to_dict('index')
    
    with open(f'{i}.json', 'w') as f:
        json.dump(df.to_dict('index'), f)

with open('stock_info.json', 'w') as f:
    json.dump(dataframe_dict, f)

### Creating a dictionary and JSON for Benchmark index and ETF

benchmark_dict = {}
# SP500(broader market)
df = pdr.get_data_yahoo('SPY', start='2009-01-01')
df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
benchmark_dict['SPY'] = df.to_dict('index')

# Vanguard Health Care Index Fund (Healthcare Industry)
df = pdr.get_data_yahoo('VHT', start='2009-01-01')
df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
benchmark_dict['VHT'] = df.to_dict('index')

#SPDR S&P Pharmaceuticals ETF (Pharmaceutical Drug Industry)
df = pdr.get_data_yahoo('XPH', start='2009-01-01')
df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
benchmark_dict['XPH'] = df.to_dict('index')



with open('benchmark.json', 'w') as fp:
    json.dump(benchmark_dict, fp)