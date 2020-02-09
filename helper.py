import pandas as pd
import pandas_datareader as pdr
import numpy as np



stocks = ['ITCI', 'AXSM', 'CVM', 'KRTX', 'APLS', 'PRVB', 'CRTX', 'EPZM',
    'CRBP', 'CCXI', 'TGTX', 'IMGN', 'FATE', 'AKBA', 'DNLI', 'RIGL', 'PRNB', 
    'ARVN', 'ZYME', 'MRNA', 'CYTK', 'FTSV', 'ASND', 'XBIT', 'ALEC', 'ORTX', 
    'ADVM', 'MGNX', 'AKRO', 'ESPR', 'BHVN', 'YMAB', 'RGNX', 'MYOK', 'TPTX', 
    'ATNX', 'BLUE', 'AGEN', 'AVRO', 'DCPH', 'CTMX', 'SRNE', 'ARDX', 'BCRX', 
    'RETA', 'KOD', 'DTIL','RARX', 'AUTL', 'GTHX', 'CARA', 'KDMN', 'XNCR', 
    'ACHN', 'ARNA', 'RCKT', 'TBIO', 'VYGR', 'SGMO', 'ODT', 'ANAB', 'ATRA', 
    'CNST', 'GERN', 'BPMC', 'ALLO', 'BBIO', 'FGEN', 'PGNX', 'MGTX', 'NXTC',
    'IMMU', 'ZIOP', 'CRSP', 'IOVA', 'VKTX', 'EIDX', 'MYOV', 'AMRS',
    'KRYS', 'KURA', 'MDGL', 'UBX', 'TCDA', 'QURE', 'MRTX', 'ASMB',
    'GLYC', 'RYTM', 'FIXX', 'DRNA', 'ARWR', 'ALLK', 'GOSS', 'WVE']

def stock_dict_to_dataframe(data):
    """
    Converting to dictionary to pandas dataframe.
    """
    
    data_in_list = []
    for stock in stocks:
        stock_dict_list = data[stock]
        for index, dict_ in enumerate(stock_dict_list):
            dict_['Ticker'] = stock
            data_in_list.append(dict_)
    return pd.DataFrame(data_in_list)    



def pulling_yahoo_stock_dict(stocks=stocks):
    """
    Pulling Yahoo Finance data for selected stocks and coverting it to dictionary.
    """
    dataframe_dict= {}
    for stock in stocks: 
        df = pdr.get_data_yahoo(stock, start='2000-01-01')
    # Change inde to a date  string
        df['Date'] = df.index.strftime("%Y-%m-%d %H:%M:%S")  
        dataframe_dict[stock] = df.to_dict('records')
    return dataframe_dict


def create_target(data, period = [1, 2], low_column ='Low_1', high_column='Max_1'):
    #Next Day Max/Min Value (% Difference from Closing Price)
    
    for stock in stocks:
        
        stock_dict_list = data[stock]
        num_days = len(stock_dict_list)
        for index, dict_ in enumerate(stock_dict_list):
            _open = dict_['Open']
            if index+period[1] >= num_days:
                _low = None
                _max = None
            else:
                _low = np.min([dict_['Low'] for dict_ in stock_dict_list[index+period[0]:index+period[1]]])/ _open -1
                _max = np.max([dict_['High'] for dict_ in stock_dict_list[index+period[0]:index+period[1]]])/ _open -1
            stock_dict_list[index][low_column] = _low
            stock_dict_list[index][high_column] = _max
        data[stock] = stock_dict_list
       
    return data

def removing_nulls_from_list(data, lower, higher):
    for stock in stocks:
        stock_dict_list = data[stock]
        transfer_list = []
        for index, dict_ in enumerate(stock_dict_list):
            transfer_list.append(dict_)
        data[stock] = transfer_list[lower:higher]
    return data

def min_max_scale(stock_dict_list, input_columns = ['High', 'Low', 'Open', 'Close', 'Adj Close']):
    for index, dict_ in enumerate(stock_dict_list):
        for key in input_columns:
            max_value = np.array([dict_[key] for dict_ in stock_dict_list]).max()+ 0.05
            min_value = np.array([dict_[key] for dict_ in stock_dict_list]).min()- 0.05
            stock_dict_list[index][key] = (dict_[key]-min_value)/ max_value
    return stock_dict_list

def binning_percent_change(stock_dict_list, columns, bins):
    for index, dict_ in enumerate(stock_dict_list):
        for column in columns:
                
                stock_dict_list[index][column+'_under_bin'] = 1 if dict_[column] < bins[0]-.05 else 0
                stock_dict_list[index][column+'_over_bin'] = 1 if dict_[column] >= bins[-1]+0.05 else 0        
                for i in bins:
                    stock_dict_list[index][column+'_'+str(i)] = 1 if (dict_[column]<(i+ 0.05)) & (dict_[column]>= i- 0.05) else 0
    return stock_dict_list