# -*- coding: utf-8 -*-
"""
@author: Ahmad Mojiri
"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import os
from assets.component_model import SolarResource, WindSource_windlab
from assets.optimisation import Optimise
from assets.plotting import prep_results_to_print, prep_results_to_plot,LCOH2
import pdb

colors = {'background': 'whitesmoke',
          'text': 'black'}

locations = [{'label':'Burnie 1', 'value':'Burnie 1'},
             {'label':'Burnie 2', 'value':'Burnie 2'},
             {'label':'Burnie 3','value':'Burnie 3'},
             {'label':'Burnie 4','value':'Burnie 4'},
             {'label':'Gladstone 1','value':'Gladstone 1'},
             {'label':'Gladstone 2','value':'Gladstone 2'},
             {'label':'Gladstone 3','value':'Gladstone 3'},
             {'label':'Pilbara 1','value':'Pilbara 1'},
             {'label':'Pilbara 2','value':'Pilbara 2'},
             {'label':'Pilbara 3','value':'Pilbara 3'},
             {'label':'Pilbara 4','value':'Pilbara 4'},
             {'label':'Pinjara 1','value':'Pinjara 1'},
             {'label':'Pinjara 2','value':'Pinjara 2'},
             {'label':'Pinjara 3','value':'Pinjara 3'},
             {'label':'Pinjara 4','value':'Pinjara 4'},
             {'label':'Upper Spencer Gulf 1','value':'Upper Spencer Gulf 1'},
             {'label':'Upper Spencer Gulf 2','value':'Upper Spencer Gulf 2'},
             {'label':'Upper Spencer Gulf 3','value':'Upper Spencer Gulf 3'},
             {'label':'Upper Spencer Gulf 4','value':'Upper Spencer Gulf 4'},
             {'label':'Darwin','value':'Darwin'}]

results_var = [{'label':'PV output [kW]', 'value':'pv_pout'},
               {'label':'Wind output [kW]', 'value':'wind_pout'},
               {'label':'Curtailed power [kW]', 'value':'curtail_p'},
               {'label':'Battery input power [kW]', 'value':'bat_pin'},
               {'label':'Battery output power [kW]', 'value':'bat_pout'},
               {'label':'Electrolyser input power [kW]', 'value':'el_pin'},
               {'label':'Compressor 1 H2 flow [kg/s]', 'value':'comp1_hflow'},
               {'label':'Compressor 2 H2 flow [kg/s]', 'value':'comp2_hflow'},
               {'label':'Compressor 2 input power [kg/s]', 'value':'comp2_pin'},
               {'label':'Reduced H2 output [kg/s]', 'value':'res_hout'},
               {'label':'Pipe H2 output [kg/s]', 'value':'pipe_storage_hout'},
               {'label':'UG H2 output [kg/s]', 'value':'ug_storage_hout'},
               {'label':'UG storage level [kg]', 'value':'ug_storage_level'},
               {'label':'Pipe storage level [kg]', 'value':'pipe_storage_level'},
               {'label':'Shutdown allowance level [kg]', 'value':'reserve_h_level'},
               {'label':'Battery charge level [kWh]', 'value':'bat_e'},
               {'label':'Load [kg/s]', 'value':'LOAD'},]

storage_types = [{'label':'Salt Cavern','value':'Salt Cavern'},
                 {'label':'Lined Rock Cavern','value':'Lined Rock'}]

Layout= {'xaxis':{'showline':True, 'linewidth':1.25,
                  'mirror':True,'linecolor':'black',
                  'nticks':20, 'tickangle':30,
                  'gridcolor':'LightPink'},
         'yaxis':{'showline':True, 'linewidth':1.25,
                  'mirror':True,'gridcolor':'LightPink'},
         'legend':{'yanchor':"bottom", 'y':1,
                   'xanchor':"left",'x':0.1,'orientation':'h'},
                   'plot_bgcolor': colors['background'],
           'title':'',
           'font': {'color': colors['text']}  }

layout_LCOH2 = {'xaxis':{'showline':True, 'linewidth':1.25,
                  'mirror':True,'linecolor':'black',
                  'gridcolor':'LightPink', 'showticklabels':False},
                'yaxis':{'showline':True, 'linewidth':1.25,
                  'mirror':True,'gridcolor':'LightPink',
                  'title': 'USD per kg of H<sub>2</sub>'},
                'legend':{'yanchor':"bottom", 'y':0,
                   'xanchor':"left",'x':1.1,'orientation':'v'},
                   'plot_bgcolor': colors['background'],
               'title':'',
               'marker_line_color':'black',
               'barmode':'stack',
               'font': {'color': colors['text']}  }



data_to_plot = None
RESULTS = None

app = dash.Dash(__name__,requests_pathname_prefix='/app/')

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server=app.server



app.layout = html.Div(
    [
        html.Div(
            # header
            [dbc.Row(html.H2("Green H2"), style={'color': '#3B8B76', 'padding': 3, 'text-align': 'center'}),
             dbc.Row(html.H2("Australian National University"),
                     style={'color': '#3B8B76', 'padding': 3, 'text-align': 'center'}),
             ],
            style={'padding-bottom': '20px', 'padding-top': '20px',
                   'border-bottom': '3px solid #3B8B76', 'margin-right': '5%', 'margin-left': '5%'}),
        # section 1
        html.Div([

            html.Div(children=[
                dbc.Row([
                    dbc.Label("Electrolyser efficiency:", width=3, style={'padding-right': 2}),
                    dbc.Col(dcc.Input(id="EL_ETA", type="number",
                                       value=0.7,
                                       min=0.5,
                                       max=1.0,
                                       step=0.1,
                                       style={}), width=2,
                    ),
                    
#Location 
                    
                    dbc.Col(width=2),
                    # dbc.Label("None has been selected!", width=3, style={'padding-right': 2}),
                   dbc.Col( dcc.Loading(
                                id="Lolcation_Status",
                                type="default",
                                children= html.Div('None has been selected!',
                                                   id='location_status',
                                                   style={'textAlign': 'right'}),
                                style={'padding-right': 2}), width=3),
                    dbc.Col(
                        dcc.Dropdown( id='location_selector',
                                    options=locations,
                                    multi=False,
                                    searchable=True,
                                    placeholder='Select a location!'
                                    ),width={'size':2, 'offset':0}
                    )
                    ,
                ],
                    
#Battery charging efficiency                      
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Battery charging efficiency:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.95", id="examddple-password-row"
                            ), width=2,
                        ),
                        dbc.Col(width=2),
                        dbc.Label("UG Storage type:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dcc.Dropdown(['Salt', 'Rock'])
                        )
                        ,
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Battery discharging efficiency:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.95"
                            ), width=2,
                        ),

                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Unit cost of PV:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Unit cost of Wind:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        dbc.Col(width=3),

                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Unit cost of Electrolyser:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kW", width=2, style={'padding-right': 2}),

                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("UG storage max capacity:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("kg of H2", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Unit cost of pipe storage:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kg of H2", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Pipe storage min capacity:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("kg of H2", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Battery energy unit price:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Battery power unit price:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Capcaity factor:", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("Percent[%]", width=2, style={'padding-right': 2}),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row(
                    [
                        dbc.Label("Load", width=3, style={'padding-right': 2}),
                        dbc.Col(
                            dbc.Input(
                                type="number", placeholder="0.7"
                            ), width=2,
                        ),
                        dbc.Label("kgH2/s", width=2, style={'padding-right': 2}),
                        dbc.Col(width=3),
                        dbc.Col(dbc.Button(
                            "Optimise!", id="example-button", className="me-1", n_clicks=0
                        ), ),
                    ],
                    className="",
                    style={'padding': 3}
                ),

            ], style={'font-size': '13px', 'padding-right': '2%', 'margin': 10, 'flex': 1,
                      'border-right': '1px dotted black'}),

            html.Div(children=[
                dbc.Row(dbc.Col(width=12),
                        style={'height': '70pt'}),
                dbc.Row(
                    [
                        dbc.Col(
                            width=2,
                        ),
                        dbc.Col(
                            # dbc.Label("Result:", width=2, style={'padding-right': 2}),
                            dcc.Textarea(
                                title='',
                                id='textarea-example',
                                value='Result....',
                                style={'width': '85%', 'height': 300},
                            ),
                            width=10,
                        ),
                    ],
                    className="",
                    style={'padding': 3}
                ),
                dbc.Row([
                    dbc.Col(dbc.Button(
                        "Export!", id="export", className="me-1", n_clicks=0,
                    ), width=11, style={'text-align': 'right'}),

                ]),
                dbc.Row(dbc.Col(width=12),
                        style={'height': '20pt'}),

            ], style={'font-size': '13px', 'padding': 5, 'margin': 10, 'flex': 1,
                      }),

        ], style={'display': 'flex', 'flex-direction': 'row', 'padding-bottom': '20px', 'padding-top': '20px',
                  'border-bottom': '3px solid #3B8B76', 'margin-right': '5%', 'margin-left': '5%'}),
        # section 2
        html.Div([
            html.Div(children=[dbc.Row(
                [
                    dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(['PV output', 'Winf output'], 'Choose a variable',
                                         style={}, multi=True), width=7,

                            style={'text-align': ''}),

                        dbc.Col(dbc.Button(
                            "Download Data!", id="download", className="me-1", n_clicks=0,
                        ), width=5, style={}),

                    ]),
                ],
                className="",
                style={'padding-top': '20px', }
            ), ], style={'font-size': '13px', 'padding': 5, 'margin': 10, 'flex': 1,
                         }),
            html.Div(children=[dbc.Row(
                dbc.Col(dcc.Graph(
                    figure={
                        'data': [
                            {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                        ],
                        'layout': {
                            'title': 'Dash Data Visualization'
                        }
                    }
                )),

            ), ], style={'font-size': '13px', 'padding': 5, 'margin': 10, 'flex': 1,
                         }),
        ], style={'display': 'flex', 'flex-direction': 'row', 'border-bottom': '3px solid #3B8B76',
                  'margin-right': '5%',
                  'margin-left': '5%'}),
        # section 3
        html.Div([
            html.Div(children=[dbc.Row(
                [
                    dbc.Row(
                        [
                            dbc.Label("Interest rate:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("Percent [%]", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("Lifetime:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("Years", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("FOM PV:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("Years", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("FOM Wind:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("FOM Electrolyser:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("VOM PV:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("FOM PV:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("VOM Wind:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kW", width=2, style={'padding-right': 2}),
                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        [
                            dbc.Label("VOM Electrolyser:", width=3, style={'padding-right': 2}),
                            dbc.Col(
                                dbc.Input(
                                    type="number", placeholder="0.7"
                                ), width=2,
                            ),
                            dbc.Label("USD/kg of H2", width=2, style={'padding-right': 2}),
                            dbc.Col(width=3),

                        ],
                        className="",
                        style={'padding': 3}
                    ),
                    dbc.Row(
                        style={'height': '30px'},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(width=4),
                            dbc.Col(dbc.Button(
                                "Calculate LCOH2!", id="ebutton", className="me-1", n_clicks=0
                            ), ),
                        ], style={'padding': 3},
                    )
                ],
                className="",
                style={'padding-top': '20px', }
            ), ], style={'font-size': '13px', 'padding': 5, 'margin': 10, 'flex': 1,
                         }),
            html.Div(children=[

                dbc.Row(

                    dbc.Col(dcc.Graph(
                        figure={
                            'data': [
                                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                            ],
                            'layout': {
                                'title': ' Data Visualization'
                            }
                        }
                    )),

                ), ], style={'font-size': '13px', 'padding': 5, 'margin': 10, 'flex': 1,
                             }),
        ], style={'display': 'flex', 'flex-direction': 'row', 'border-bottom': '3px solid #3B8B76',
                  'margin-right': '5%',
                  'margin-left': '5%'}),

        ###

        # Footer
        html.Div([
            dbc.Row(html.Div("The work has been supported by the Heavy Industry"
                             " Low-carbon Transition Cooperative Research Centre whose activities are"
                             " funded by the Australian Governments Cooperative Research Centre Program."),
                    style={'text-align': 'left', 'padding': 3, }),
            dbc.Row([html.Div("For more information please contact:"),
                     html.Br(),
                     html.Div("Dr Ahmad Mojiri (ahmad.mojiri@anu.edu.au)"),
                     html.Br(),
                     html.Div("A/Professor Joe Coventry (joe.coventry@anu.edu.au)"), ],
                    style={'text-align': 'left', 'padding': 3, }),
        ],
            style={'padding-bottom': '20px', 'margin-right': '5%', 'margin-left': '5%'}),
    ])






              

#-------------------------------------
@app.callback(
            [Output('OPTIMISATION_TEXT','children'),
             Output('Output', 'value')],
            [Input('Optimise','n_clicks'),
             State('EL_ETA', 'value'),
             State('BAT_ETA_IN', 'value'),
             State('BAT_ETA_OUT', 'value'),
             State('C_PV', 'value'),
             State('C_WIND', 'value'),
             State('C_EL', 'value'),
             State('UG_STORAGE_CAPA_MAX', 'value'),
             State('C_PIPE_STORAGE', 'value'),
             State('PIPE_STORAGE_CAPA_MIN', 'value'),
             State('C_BAT_ENERGY', 'value'),
             State('C_BAT_POWER', 'value'),
             State('LOAD', 'value'),
             State('CF', 'value'),
             State('location_selector','value'),
             State('storage_selector','value')],
             prevent_initial_call=True
            )
def optimise(click, el_eta, bat_eta_in, bat_eta_out, c_pv, c_wind, c_el,
             ug_storage_capa_max, c_pipe_stprage, pipe_storage_capa_min,
             c_bat_energy, c_bat_power, load, cf, location, storage_type):
    # try:
    #     SolarResource(location)
    # except:
    #     text = 'Error: Choose a location!'
    #     results_to_print = 'None'
    # else:
    SolarResource(location)
    WindSource_windlab(location)
    simparams = dict(EL_ETA = el_eta,       
                     BAT_ETA_in = bat_eta_in, 
                     BAT_ETA_out = bat_eta_out,  
                     C_PV = c_pv,          
                     C_WIND = c_wind,        
                     C_EL = c_el,          
                     UG_STORAGE_CAPA_MAX = ug_storage_capa_max, 
                     C_PIPE_STORAGE = c_pipe_stprage,
                     PIPE_STORAGE_CAPA_MIN = pipe_storage_capa_min, 
                     C_BAT_ENERGY = c_bat_energy,        
                     C_BAT_POWER = c_bat_power,        
                     ) 
    text = 'Completed!'
    results = Optimise(load, cf, storage_type, simparams)
    global RESULTS
    RESULTS = prep_results_to_print(results,simparams)
    
    OUTPUT = str(RESULTS).replace(', ',',\n ').replace('{', '').replace('}', '').replace("'",'').replace(',','')
    
    global data_to_plot
    data_to_plot = prep_results_to_plot(results, simparams, location)
    SolarResource(location)    
    return ([text,OUTPUT])





#Choose the location and update weather data
@app.callback(
            [Output('location_status','children')],
            [Input('location_selector','value')],
            prevent_initial_call=True
             )
def update_weather_data(location):
    try:
        SolarResource(location)
    except:
        text = '%s has been selected'%(location)
    else:
        SolarResource(location)
        WindSource_windlab(location)
        text = 'Weather data updated!'
    return([text])
            


#Export the results
@app.callback(Output("download_csv", "data"),
              Input("export", "n_clicks"),
              prevent_initial_call=True,
              )
def export(n_clicks):
    data = pd.DataFrame(RESULTS,index=[0])
    return dcc.send_data_frame(data.to_csv, "Results.csv")




#Download Timeseries
@app.callback(Output("download_data_csv", "data"),
              Input("download", "n_clicks"),
              State('variable_selector','value'),
              prevent_initial_call=True,
              )
def download(n_clicks,variables):
    cols = ['time'] + variables
    data = data_to_plot[cols]
    return dcc.send_data_frame(data.to_csv, "Timeseries.csv")




#plot the selected variables
@app.callback(
            [Output('GraphI', 'figure')],
            [Input('variable_selector','value')],
            prevent_initial_call=True
             )
def update_graph(variables):
      plot_data=[]
      chart={'data': plot_data}            
      for col in variables:
          plot_data = plot_data + [go.Scatter(x=data_to_plot.time,
                                              y=data_to_plot[col],
                                              name=col)]
          layoutI = Layout.copy()
          chart={'data': plot_data,
                 'layout': layoutI }
      
      return([chart])


#Calulcate LCOH2
@app.callback(
            [Output('graph_LCOH2', 'figure')],
            [Input('LCOH2','n_clicks'),
             State('interest_rate', 'value'),
             State('lifetime', 'value'),
             State('pv_fom', 'value'),
             State('wind_fom', 'value'),
             State('elec_fom', 'value'),
             State('pv_vom', 'value'),
             State('wind_vom', 'value'),
             State('elec_vom', 'value'),
             ],
            prevent_initial_call=True
             )
def update_graph_LCOH2(click,i, life,
                       pv_fom, wind_fom, elec_fom,
                       pv_vom, wind_vom, elec_vom):
    
    
    
    LCOH2_items = LCOH2(RESULTS, data_to_plot, i, life,
                          pv_fom, wind_fom, elec_fom,
                          pv_vom, wind_vom, elec_vom)
    
    
    
    plot_data_LCOH2=[]
    for col in LCOH2_items.columns.tolist():
        plot_data_LCOH2 = plot_data_LCOH2 + [go.Bar(x=[1],
                                            y = LCOH2_items[col],
                                            name=col)]
    layout_LCOH2_I = layout_LCOH2.copy()
    layout_LCOH2_I['title'] = 'LCOH2 = %0.2f USD/kg'%(LCOH2_items.sum(axis=1).values[0])
    LCOH2_chart={'data': plot_data_LCOH2,
                 'layout': layout_LCOH2_I }
      
    return([LCOH2_chart])            


if __name__ == '__main__':
     app.run_server(debug=False,host='0.0.0.0')