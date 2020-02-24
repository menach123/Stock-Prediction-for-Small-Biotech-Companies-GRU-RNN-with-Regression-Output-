import pandas as pd
import datetime
import numpy as np
import pandas_datareader as pdr
import plotly.figure_factory as ff
from math import ceil, floor
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from keras import models
from keras import layers
from keras import optimizers 

class Prediction(): 

    def __init__(self):
        pass 

    def pull_stock_info(self, new_stock='SGMO'):
        
        """
        Pulling a dateframe of the daily stock pricing for the last 1001 days for the inputed stock. 
        """
        self.stock = new_stock
        
        #Pulling stock info
        self.intial_dataframe = pdr.get_data_yahoo(self.stock, start='2000-01-01').iloc[-1001:]
        self.open_ = self.intial_dataframe['Open']
        self.close_ = self.intial_dataframe['Close']
        self.low_ = self.intial_dataframe['Low']
        self.high_ = self.intial_dataframe['High']
        self.timestamps = self.intial_dataframe.index
        self.stock_dict_list  = self.intial_dataframe.to_dict('records')
        pass

    def enter_info(self, timestamps, stock_dict_list):
        """
        Overide information
        """

        self.open_ = np.array([i['Open'] for i in self.stock_dict_list])
        self.close_ = np.array([i['Close'] for i in self.stock_dict_list])
        self.low_ = np.array([i['Low'] for i in self.stock_dict_list])
        self.high_ = np.array([i['High'] for i in self.stock_dict_list])
        self.timestamps = timestamps
        self.stock_dict_list  = stock_dict_list
        pass

    def process_stock_info(self, low_range = [-12, 5], high_range = [-5, 14]):
        """

        """
        num_days = len(self.stock_dict_list)
        #Ranges for probabilities distribution
        self.low_range= low_range 
        self.low_values = np.array(range(low_range[0],low_range[1]))/100
        self.high_range = high_range 
        self.high_values = np.array(range(high_range[0],high_range[1]))/100
        period = [1, 2]

        self.current_close = round(self.stock_dict_list[-1]['Close'], 2)
        self.current_open = round(self.stock_dict_list[-1]['Open'], 2)
        self.previous_open = round(self.stock_dict_list[-2]['Open'], 2)

        #Creating additional input data, "%_Change", and  target values, "Low_1" and "High_1"
        for index, dict_ in enumerate(self.stock_dict_list):
            _open = dict_['Open']
            if index>0:
                percent_change = (self.stock_dict_list[index-1]['Open']-_open)/ self.stock_dict_list[index-1]['Open']
                self.stock_dict_list[index]['%_Change'] = percent_change
            if index+period[0] >= num_days:
                _low = None
                _max = None
                real_low = None
                real_max = None
            else:
                _low = np.min([dict_['Low'] for dict_ in self.stock_dict_list[index+period[0]:index+period[1]]])/ _open -1
                real_low = _low
                if _low <= low_range[0]/100:
                    _low = low_range[0]/100
                if _low >= (low_range[1]-1)/100:
                    _low = (low_range[1]-1)/100

                _max = np.max([dict_['High'] for dict_ in self.stock_dict_list[index+period[0]:index+period[1]]])/ _open -1
                real_max = _max
                if _max <= high_range[0]/100:
                    _max = high_range[0]/100
                if _max >= (high_range[1]-1)/100:
                    _max = (high_range[1]-1)/100
                _low = round(_low,2)
                _max = round(_max, 2)
            self.stock_dict_list[index]['Low_1'] = _low
            self.stock_dict_list[index]['High_1'] = _max
            self.stock_dict_list[index]['Real_Low_1'] = real_low
            self.stock_dict_list[index]['Real_High_1'] = real_max
         
        pass

    def scale_price_feature(self):
        """
        Dividing price values by the max price values
        """
        
        price_columns = ['High', 'Low', 'Open', 'Close', 'Adj Close']
        self.data_length = len(price_columns)+ 2
        list_of_prices = []
        for index, dict_ in enumerate(self.stock_dict_list):
            list_of_prices.append([dict_[column] for column in price_columns])     
        max_ = np.max(list_of_prices)
        for index, dict_ in enumerate(self.stock_dict_list):
            for column in price_columns:
                self.stock_dict_list[index][column] = self.stock_dict_list[index][column]/max_
        return self.stock_dict_list
    
    def scale_volume_feature(self):
        """
        Dividing volume by the max volume
        """
        max_ = np.max([dict_['Volume'] for index, dict_ in enumerate(self.stock_dict_list)])
        for index, dict_ in enumerate(self.stock_dict_list):
            self.stock_dict_list[index]['Volume'] = self.stock_dict_list[index]['Volume']/ max_
        pass
        
    def create_model_input(self):
        """
        Create model inputs and target arrays (Next Day Low, Next Day High)to verify model.
        """
        
        
        self.scale_price_feature()
            
            
        # Dividing volume by the max volume
        self.scale_volume_feature()
        
        #Formatting the Input Data
        self.model_input = []
        for i in self.stock_dict_list[1:]:
            row =np.array([i['High'], i['Low'], i['Open'], i['Close'], i['Adj Close'], i['Volume'], i['%_Change']])
            self.model_input.append(row)
        
        self.data_length  = len(self.model_input[0])
        self.model_input = np.array(self.model_input).reshape(len(self.model_input), 1, self.data_length)
        pass
        
    def instantiate_models(self, data_length=7, target_length=7):
        """
        Instantiating high and low models for prediction.
        """
        self.low_filename = f'Models\Model_GRU_Low_1{self.stock}.h5'                        
        self.high_filename = f'Models\Model_GRU_High_1{self.stock}.h5'
        model = models.Sequential()
        model.add(layers.GRU(25, return_sequences=True, input_shape=(1,self.data_length)))
        model.add(layers.Dropout(0.5))
        model.add(layers.GRU(50, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation = "linear"))
        
        model.load_weights(self.low_filename)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        self.low_model = model
        model = models.Sequential()
        model.add(layers.GRU(25, return_sequences=True, input_shape=(1,self.data_length)))
        model.add(layers.Dropout(0.5))
        model.add(layers.GRU(50, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation = "linear"))
        model.load_weights(self.high_filename)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        self.high_model = model
        
        pass

    def generate_prediction(self):
        """
        Generate predicts
        """
        self.low_output = self.low_model.predict(self.model_input[:-1]).reshape(len(self.model_input[:-1]))
        self.today_low = self.low_output[-1]
        self.low_output = self.low_model.predict(self.model_input).reshape(len(self.model_input))
        self.tommorrow_low = self.low_output[-1]
        
        self.high_output = self.high_model.predict(self.model_input[:-1]).reshape(len(self.model_input[:-1]))
        self.today_high = self.high_output[-1]
        self.high_output = self.high_model.predict(self.model_input).reshape(len(self.model_input))
        self.tommorrow_high = self.high_output[-1]
        pass

    def load_prediction(self):
        """
        Store prediction and calculates and stores residuals. 
                """
        for name, output in zip(['Low_1', 'High_1'], [self.low_output, self.high_output]):
            for i, j in enumerate(output):
                transfer_dict =  self.stock_dict_list[i+1]
                self.stock_dict_list[i+1][f'Predict_{name}'] = j
                self.stock_dict_list[i+1][f'Round_Predict_{name}'] = round(j, 2)
                if (i+2) < len(self.stock_dict_list):
                    self.stock_dict_list[i+1][f'Residual_{name}'] =  transfer_dict[name]-j
        pass

    def distribution(self, column, predicted, centered):
        """
        Make normal distribution using the median and standard deviation of the residuals and the predicted model value
        Input-  residual_column - Column name in stock_dict_list for the residuals to be used.
                predicted - Model predicted values. 
                    Common values : tommorrow_low, today_low, tommorrow_high, and today_high
        """
        if column == 'Low_1':
            adj = 0.03
        else:
            adj = -0.03

        residuals = [i[f'Residual_{column}'] for i in self.stock_dict_list[1:-1] if i[f'Round_Predict_{column}'] == round(predicted, 2)]
        residuals_unadjusted = [i[f'Residual_{column}'] for i in self.stock_dict_list[1:-1]]
        
         
        distribution_ = np.random.normal(loc=np.median(residuals)+ predicted+ 1, scale=np.std(residuals)/2, size=100000)* centered
        
        return distribution_

    def prediction_process(self):
        """
        
        """
        self.generate_prediction()
        self.load_prediction()
         

        self.previous_low_simulation = self.distribution('Low_1', self.today_low, self.previous_open)
        self.low_simulation = self.distribution('Low_1', self.tommorrow_low, self.current_open)
        self.previous_high_simulation = self.distribution('High_1', self.today_high, self.previous_open)
        self.high_simulation = self.distribution('High_1', self.tommorrow_high, self.current_open)

    def create_KDE(self):
        """
        
        """
        hist_data = [self.low_simulation, self.high_simulation]

        group_labels = ['Next Day Low', 'Next Day High']
        colors = ['#FC0505', '#028700']

        
        # Create distplot with curve_type set to 'normal'
        self.distplot_tommorrow = ff.create_distplot(hist_data, group_labels, bin_size=0.01, show_hist=False, colors=colors, histnorm='probability', show_rug=False)
        today = self.timestamps[-1]
        tommorrow = datetime.timedelta(days=1)+ today
        today = today.strftime("%m/%d/%Y")
        tommorrow = tommorrow.strftime("%m/%d/%Y")
        self.distplot_tommorrow.layout.title = f'Prediction Distributions for {self.stock} {tommorrow}'
        not_names = ['Open', 'Current']
        y = [i.y for i in self.distplot_tommorrow['data'] if i.name not in not_names]
        y = np.max([i.max() for i in y if type(i) == np.ndarray])*1.05

        self.distplot_tommorrow.layout.yaxis.range = [0,y]
        
        top = int(50*(90*self.current_open// 50+ 1))
        bottom = int(50*(110*self.current_open// 50))
        ticks = np.array([i/100 for i in range(top, bottom+50, 50)])


        self.distplot_tommorrow.layout.xaxis.tickvals = ticks
        self.distplot_tommorrow.layout.xaxis.ticktext = ['$'+str(i)+'0' for i in ticks]
        self.distplot_tommorrow.layout.xaxis.range = [ticks[0],ticks[-1]]
        self.distplot_tommorrow.layout.xaxis.ticks = 'outside'
        self.distplot_tommorrow.layout.xaxis.title = 'Stock Price $'

        self.distplot_tommorrow.add_trace(go.Scatter(
                                                        x=[self.current_open, self.current_close],
                                                        y=[self.distplot_tommorrow.layout.yaxis.range[1]*.175, self.distplot_tommorrow.layout.yaxis.range[1]*.275],
                                                        text=[f"Open ${self.current_open}",
                                                            f"Close ${self.current_close}"],
                                                        mode="text" ,
                                                        legendgroup = 'Price'
                                                    ))

        self.distplot_tommorrow.add_shape(
                            # Line Vertical
                            go.layout.Shape(
                                type="line",
                                x0=self.current_open,
                                y0=0,
                                x1=self.current_open,
                                y1=self.distplot_tommorrow.layout.yaxis.range[1]*.15,
                                line=dict(
                                    color="RoyalBlue",
                                    width=3)))
        self.distplot_tommorrow.add_shape(
                            # Line Vertical
                            go.layout.Shape(
                                type="line",
                                x0=self.current_close,
                                y0=0,
                                x1=self.current_close,
                                y1=self.distplot_tommorrow.layout.yaxis.range[1]*.25,
                                line=dict(
                                    color="LightSeaGreen",
                                    width=3)))
        quantiles = [0.025, 0.159, 0.329, 0.415, 0.5, 0.585, 0.671, 0.841, 0.975]
        buy_points = [np.quantile(self.distplot_tommorrow.data[0]['x'], i, interpolation='lower') for i in quantiles[:6]]
        
        indexs = [i for i, j in enumerate(self.distplot_tommorrow.data[0]['x']) if j in buy_points]
        buy_points = [round(i,2) for i in buy_points]
        y = [self.distplot_tommorrow.data[0]['y'][i] for i in indexs]

        self.distplot_tommorrow.add_trace(go.Scatter(
                                                            x=buy_points,
                                                            y=y,
                                                            name="Buy Points",
                                                            mode='markers',
                                                            marker=dict(
                                                            color='#87fc83',
                                                            size=2,
                                                            line=dict(
                                                                color='#013b05',
                                                                width=5
                                                            ))))
        
        high_points = [np.quantile(self.distplot_tommorrow.data[1]['x'], i, interpolation='higher') for i in quantiles[3:]]
        
        indexs = [i for i, j in enumerate(self.distplot_tommorrow.data[1]['x']) if j in high_points ]
        high_points = [round(i,2) for i in high_points]
        y = [self.distplot_tommorrow.data[1]['y'][i] for i in indexs]
        self.distplot_tommorrow.add_trace(go.Scatter(
                                                            x=high_points,
                                                            y=y,
                                                            name="Sell Points",
                                                            mode='markers',
                                                            marker=dict(
                                                            color='#fa000c',
                                                            size=2,
                                                            line=dict(
                                                                color='#013b05',
                                                                width=5
                                                            ))))

        hist_data = [self.previous_low_simulation, self.previous_high_simulation]

        group_labels = ['Today Low', 'Today High']
        colors = ['#750407', '#165702']

        
        # Create distplot with curve_type set to 'normal'
        self.distplot_today = ff.create_distplot(hist_data, group_labels, bin_size=0.01, show_hist=False, colors=colors, histnorm='probability', show_rug=False)
        self.distplot_today.layout.title = f'Prediction Distributions for {self.stock} {today}'
        not_names = ['Open', 'Current']
        y = [i.y for i in self.distplot_today['data'] if i.name not in not_names]
        y = np.max([i.max() for i in y if type(i) == np.ndarray])*1.05

        self.distplot_today.layout.yaxis.range = [0,y]

        top = int(50*(90*self.current_open// 50+ 1))
        bottom = int(50*(110*self.current_open// 50))
        ticks = np.array([i/100 for i in range(top, bottom+50, 50)])


        self.distplot_today.layout.xaxis.tickvals = ticks
        self.distplot_today.layout.xaxis.ticktext = ['$'+str(i)+'0' for i in ticks]
        self.distplot_today.layout.xaxis.range = [ticks[0],ticks[-1]]
        self.distplot_today.layout.xaxis.ticks = 'outside'
        self.distplot_today.layout.xaxis.title = 'Stock Price $'

        self.distplot_today.add_trace(go.Scatter(
                                                        x=[self.current_open, self.current_close],
                                                        y=[self.distplot_tommorrow.layout.yaxis.range[1]*.175, self.distplot_tommorrow.layout.yaxis.range[1]*.275],
                                                        text=[f"Open ${self.current_open}",
                                                            f"Close ${self.current_close}"],
                                                        mode="text",
                                                        legendgroup = 'Price'
                                                    ))

        self.distplot_today.add_shape(
                            # Line Vertical
                            go.layout.Shape(
                                type="line",
                                x0=self.current_open,
                                y0=0,
                                x1=self.current_open,
                                y1=self.distplot_tommorrow.layout.yaxis.range[1]*.15,
                                line=dict(
                                    color="RoyalBlue",
                                    width=3)))
        self.distplot_today.add_shape(
                            # Line Vertical
                            go.layout.Shape(
                                type="line",
                                x0=self.current_close,
                                y0=0,
                                x1=self.current_close,
                                y1=self.distplot_tommorrow.layout.yaxis.range[1]*.25,
                                line=dict(
                                    color="LightSeaGreen",
                                    width=3)))

        buy_points = [np.quantile(self.distplot_today.data[0]['x'], i, interpolation='lower') for i in quantiles[:6]]
        
        indexs = [i for i, j in enumerate(self.distplot_today.data[0]['x']) if j in buy_points]
        buy_points = [round(i,2) for i in buy_points]
        y = [self.distplot_today.data[0]['y'][i] for i in indexs]

        self.distplot_today.add_trace(go.Scatter(
                                                            x=buy_points,
                                                            y=y,
                                                            name="Buy Points",
                                                            mode='markers',
                                                            marker=dict(
                                                            color='#87fc83',
                                                            size=2,
                                                            line=dict(
                                                                color='#013b05',
                                                                width=5
                                                            ))))
         
        high_points = [np.quantile(self.distplot_today.data[1]['x'], i, interpolation='higher') for i in quantiles[3:]]
        
        indexs = [i for i, j in enumerate(self.distplot_today.data[1]['x']) if j in high_points ]
        high_points = [round(i,2) for i in high_points]
        y = [self.distplot_today.data[1]['y'][i] for i in indexs]
        

        self.distplot_today.add_trace(go.Scatter(
                                                            x=high_points,
                                                            y=y,
                                                            name="Sell Points",
                                                            mode='markers',
                                                            marker=dict(
                                                            color='#fa000c',
                                                            size=2,
                                                            line=dict(
                                                                color='#013b05',
                                                                width=5
                                                            ))))
                            
        pass

    def predict_plot(self):
        X = self.timestamps[1:][-99:]

        
        open_ = [i for i in self.open_[-60:]]
        high_ = [i for i in self.high_[-60:]]
        low_ = [i for i in self.low_[-60:]]
        close_ = [i for i in self.close_[-60:]]
        pred_low = [i['Predict_Low_1']* j +j for i, j in zip(self.stock_dict_list[-61:-1], open_)]
        low_miss = [i['Predict_Low_1']* j +j for i, j in zip(self.stock_dict_list[-61:-1], open_)]
        pred_high = [i['Predict_High_1']* j +j for i, j in zip(self.stock_dict_list[-61:-1], open_)]
        high_miss = [i['Predict_High_1']* j +j for i, j in zip(self.stock_dict_list[-61:-1], open_)]
        for i in range(0,60):
            if pred_low[i] < low_[i]:
                low_miss[i] = None
        for i in range(0,60):
            if pred_high[i] < high_[i]:
                high_miss[i] = None

        self.time_series = go.Figure(data=[go.Candlestick(x=X,
                open=open_,
                high=high_,
                low=low_,
                close=close_)])
        self.time_series.layout.title = f'Prediction {self.stock} Time Chart'

        self.time_series.add_trace(go.Scatter(x=X, 
                                                    y=pred_high,
                                                    name="High Prediction",
                                                    line_color='#87fc83',
                                                    opacity=0.8))
        self.time_series.add_trace(go.Scatter(
                                                    x=X,
                                                    y=pred_low,
                                                    name="Low Prediction",
                                                    line_color='#fc8383',
                                                    opacity=0.8))

        self.time_series.add_trace(go.Scatter(
                                                    x=X,
                                                    y=high_miss,
                                                    name="High Misses",
                                                    mode='markers',
                                                    marker=dict(
                                                    color='#87fc83',
                                                    size=2,
                                                    line=dict(
                                                        color='#013b05',
                                                        width=5
                                                    ))))
        self.time_series.add_trace(go.Scatter(
                                                    x=X,
                                                    y=low_miss,
                                                    name="Low Misses",
                                                    mode='markers',
                                                    marker=dict(
                                                    color='#fc8383',
                                                    size=2,
                                                    line=dict(
                                                        color='#520104',
                                                        width=5
                                                    ))))
        
        pass

    def model_training(self, low_file = 'Models\Model_GRU_Low_1BASE.h5', high_file = 'Models\Model_GRU_High_1BASE.h5'):
        """

        """
        
        x= []                
        data = []
        target = []
        target_h = []
        dict_ = {}
        for i in self.stock_dict_list[1:]:
            row =np.array([i['High'], i['Low'], i['Open'], i['Close'], i['Adj Close'], i['Volume'], i['%_Change']]) #, i['Moving10'], i['Moving30']])
            data_length = len(row)
            x.append(row)
            row = np.array(i['Low_1'])
            target.append(row)
            row = np.array(i['High_1'])
            target_h.append(row)
        dict_['Data'] = np.array(x).reshape(len(x),1,data_length)[1:-1]
        dict_['Target'] = np.array(target)[1:-1]
        dict_['Target_High'] =np.array(target_h)[1:-1]
        processed_data = dict_

    
        length_ = len(processed_data['Data'])
        length_test = len(processed_data['Data'])//8
        test_start = length_- 100
        test_end = length_test+ test_start
        processed_data['data_1'] = processed_data['Data'][:test_start]
        
        processed_data['data_test'] = processed_data['Data'][test_start:]
        
        processed_data['target_1'] = processed_data['Target'][:test_start]
        processed_data['target_test'] = processed_data['Target'][test_start:]
        processed_data['target_high'] = processed_data['Target_High'][:test_start]
        processed_data['target_high_test'] = processed_data['Target_High'][test_start:]

        model = models.Sequential()
        model.add(layers.GRU(25, return_sequences=True, input_shape=(1,data_length)))
        model.add(layers.Dropout(0.5))
        model.add(layers.GRU(50, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation = "linear"))
        model.compile(loss='mse', optimizer='adam')

        model.load_weights(low_file)
       
            
        epochs = 10
        loss = []
        val_loss = []   
        acc = []
        val_acc = []
        rank = []
        check = 0
        for run in range(1,200):
            
            (feature_train, label_train, feature_test, label_test) = (processed_data['data_1'],processed_data['target_1'], 
                                                                            processed_data['data_test'],
                                                                    processed_data['target_test']) 
            model.fit(feature_train, label_train, epochs=epochs, 
                    validation_data = (feature_test, label_test), verbose=1)
            dict_ = model.history.history

            for i in dict_['loss']:
                loss.append(i)


            for i in dict_['val_loss']:
                val_loss.append(i)

            
            predict = model.predict(processed_data['Data'][-100:])
            true = processed_data['Target'][-100:]
            residual = true- predict
            transfer_dict = {'Epochs':(run*epochs),
                            'Residual STD': np.std(residual),
                            'Residual Median': np.median(residual),
                                'r2': r2_score(true, predict)}

            
            rank.append(transfer_dict)
            if check <= np.abs(transfer_dict['r2']):
                filepath = f'Models\Model_GRU_Low_1{self.stock}.h5'
                model.save_weights(filepath)
                print(transfer_dict)
                check = np.abs(transfer_dict['r2'])
                                                    
        

        model.load_weights(high_file)


        epochs = 10
        lossh = []
        val_lossh = []   
        acch = []
        val_acch = []
        rankh = []
        check = 0

        for run in range(1,200):
            
            (feature_train, label_train, feature_test, label_test) = (processed_data['data_1'],
                                                                    processed_data['target_high'], 
                                                                    processed_data['data_test'],
                                                                    processed_data['target_high_test']) 
            model.fit(feature_train, label_train, batch_size=25, epochs=epochs, 
                    validation_data = (feature_test, label_test), verbose=1)
            dict_ = model.history.history

            for i in dict_['loss']:
                lossh.append(i)

            for i in dict_['val_loss']:
                val_lossh.append(i)


            
            predict = model.predict(processed_data['Data'][-100:])
            true = processed_data['Target_High'][-100:]
            residual = true- predict
            transfer_dict = {'Epochs':(run*epochs),
                            'Residual STD': np.std(residual),
                            'Residual Median': np.median(residual),
                                'r2': r2_score(true, predict)}
            

            
            rankh.append(transfer_dict)
            if check <= np.abs(transfer_dict['r2']):
                filepath = f'Models\Model_GRU_High_1{self.stock}.h5'
                model.save_weights(filepath)
                print(transfer_dict)
                check = transfer_dict['r2']
        pass


