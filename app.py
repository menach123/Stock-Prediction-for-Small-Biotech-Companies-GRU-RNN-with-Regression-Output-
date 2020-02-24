import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data_retrieval import Prediction

def string_generate(string, plotly_figure):
    for i in plotly_figure:
        i = str(i)
        if len(i) == 3:
            i+= '0'
        string +=f'   $'+i+','
    return string


dict_ = {}
predict = Prediction()


for stock in ['SNA','NUE','T','EXAS', 'CRL','AXSM', 'IMMU', 'SGMO']:

    predict.pull_stock_info(new_stock=stock)
    predict.process_stock_info()
    predict.create_model_input()
    predict.instantiate_models()
    
    predict.prediction_process()
    predict.create_KDE()
    predict.predict_plot()
    dict_[stock] = [predict.distplot_today, predict.distplot_tommorrow, predict.time_series]
from components.NameFields import first_name_components, last_name_components


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

today_buy = string_generate('BUY -', predict.distplot_today.data[3]['x'])

today_sell = string_generate('SELL -', predict.distplot_today.data[4]['x'])
 
tommorrow_buy = string_generate('BUY -', predict.distplot_tommorrow.data[3]['x'])
 
tommorrow_sell = string_generate('SELL -', predict.distplot_tommorrow.data[4]['x'])

top_markdown_text = '''
# Stock Prediction Dashboard
-----
'''





##############
# App Layout Here
##############
app.layout = html.Div([
    dcc.Dropdown(id='stock_drop',
                    options=[
                        {'label': 'Sangamo Therapeutics, Inc. (SGMO)', 'value': 'SGMO'},
                        {'label': 'AT&T Inc. (T)', 'value': 'T'},
                        {'label': 'Snap-on Incorporated (SNA)', 'value': 'SNA'},
                        {'label': 'Nucor Corporation (NUE)', 'value': 'NUE'},
                        {'label': 'Axsome Therapeutics, Inc. (AXSM)', 'value': 'AXSM'},
                         {'label': 'Charles River Laboratories International, Inc. (CRL)', 'value': 'CRL'},
                        {'label': 'Immunomedics, Inc. (IMMU)', 'value': 'IMMU'}, 
                        {'label': 'Exact Sciences Corporation (EXAS)', 'value': 'EXAS'}],
                    value='SGMO'),
    dcc.Graph(
        id='today',
        figure=predict.distplot_today),
    
    dcc.Markdown(id='today_buy', children=today_buy),
    dcc.Markdown(id='today_sell',children=today_sell),
    dcc.Graph(
        id='tommorrow',
        figure=predict.distplot_tommorrow),
        
    dcc.Markdown(id= 'tommorrow_buy', children=tommorrow_buy),
    dcc.Markdown(id= 'tommorrow_sell', children=tommorrow_sell),
    dcc.Graph(
        id='time_chart',
        figure=predict.time_series)])
    


@app.callback(
    [Output('today', 'figure'), Output('tommorrow', 'figure'), Output('time_chart', 'figure'), 
    Output(component_id='today_buy', component_property='children'), Output(component_id='today_sell', component_property='children'), 
    Output(component_id='tommorrow_buy', component_property='children'),Output(component_id='tommorrow_sell', component_property='children')],
    [Input('stock_drop', 'value')])
def update_output(value):
    today_buy = string_generate('BUY -', dict_[value][0].data[3]['x'])

    today_sell = string_generate('SELL -', dict_[value][0].data[4]['x'])
    
    tommorrow_buy = string_generate('BUY -', dict_[value][1].data[3]['x'])
    
    tommorrow_sell = string_generate('SELL -', dict_[value][1].data[4]['x'])


    return dict_[value][0], dict_[value][1], dict_[value][2], today_buy, today_sell, tommorrow_buy, tommorrow_sell




# @app.callback(
#     Output(http://127.0.0.1:8050/='basic-interactions', component_property='figure'),
#     [Input(component_id='first_name_field', component_property='value')]
# )
# def update_basic_interactions(n_points):
#     n_points = int(n_points)
#     x = np.random.randint(0, 10, n_points)
#     y = np.random.randint(0, 10, n_points)
#     figure_component = dcc.Graph(
#         id='basic-interactions',
#         figure={
#             'data': [
#                 {
#                     'x': x,
#                     'y': y,
#                     'name': 'Trace 1',
#                     'mode': 'markers',
#                     'marker': {'size': 12}
#                 }
#             ],
#             'layout': {
#                 'clickmode': 'event+select'
#             }
#         }
#     )
#     return figure_component

if __name__ == '__main__':
    app.run_server(debug=True)
