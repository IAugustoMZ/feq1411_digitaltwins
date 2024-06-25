import json
import dash
import joblib
import numpy as np
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

N_DATA = 1000
P_ANOM = 0.05

sidebar = html.Div(
    [
        html.Hr(),
        html.H2('FEQ1411 - Digital Twins e Simulação', style={'color': '#7FDBFF'}),
        html.Hr(),
        dbc.Button('Execute', id='play-button', n_clicks=0, color="primary", className='mx-auto', size='lg'),  # Making the button bigger
        dcc.Store(id='memory-output', storage_type='session'),
        dcc.Interval(id='interval-component', interval=1000, max_intervals=0),
        dbc.Switch(
            id='robust-toggle', 
            label='Robust Simulation Correction', 
            value=False,
            style={'margin': '20px'}
        )
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#282b30",
        "color": "#7FDBFF",
    },
)

# Define the layout of the main content
content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(id='card1', children=[html.H5('Feed Temperature (deg. C): '), html.P(id='last-y1')], color="primary", className="mb-3"), width=3),
                dbc.Col(dbc.Card(id='card2', children=[html.H5('Feed Pressure (bar): '), html.P(id='last-y2')], color="primary", className="mb-3"), width=3),
                dbc.Col(dbc.Card(id='card3', children=[html.H5('Anomaly Probability:  '), html.P(id='last-y3')], color="warning", className="mb-3"), width=3),
                dbc.Col(dbc.Card(id='card4', children=[html.H5('Avg. Simulation Error (kJ/kg):  '), html.P(id='last-y4')], color="warning", className="mb-3"), width=3)
            ],
            style={'margin': '0'}
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id='example-graph1', config={'displayModeBar': False}), className="m-3")
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='example-graph4', config={'displayModeBar': False}), className="mb-3", width=6),
                dbc.Col(dcc.Graph(id='example-graph5', config={'displayModeBar': False}), className="mb-3", width=6)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='example-graph2', config={'displayModeBar': False}), className="mb-3", width=6),
                dbc.Col(dcc.Graph(id='example-graph3', config={'displayModeBar': False}), className="mb-3", width=6),
            ]
        ),
    ],
    style={"margin-left": "18rem", "padding": "2rem 1rem", 'backgroundColor': '#282b30', 'color': '#7FDBFF'},
)

app.layout = dbc.Container(
    fluid=True, style={'backgroundColor': '#282b30', 'color': '#7FDBFF'}, children=[
    sidebar, content])

@app.callback(
    Output('memory-output', 'data'),
    [Input('play-button', 'n_clicks'),
     Input('robust-toggle', 'value')]
)
def load_data(n_clicks, robust_value):
    if n_clicks > 0:
        df = pd.read_csv('./data/data_with_anomaly.csv')
        data = pd.DataFrame(np.repeat(df.head(1), N_DATA, axis=0),
                            columns=df.columns)

        data['specific_energy'] = (data['e_cond'] + data['e_boiler']) / data['styr_flow']
        data.drop(columns=['e_cond', 'e_boiler', 'styr_flow'], inplace=True)
        data['t_in'] = data['t_in'] + np.random.normal(0, 0.1, len(data))
        data['p_in'] = data['p_in'] / 100000 + np.random.normal(0, 0.01, len(data))

        # randomly select 5 % of rows indexs to be the temperature anomalies
        anomalous_indices = np.random.choice(data.index, int(len(data) * P_ANOM), replace=False)
        data.loc[anomalous_indices, 't_in'] += np.random.normal(100, 100, len(anomalous_indices))

        # predict the specific energy
        data['specific_energy_sensor'] = joblib.load('./code/models/specific_energy_model.pkl').predict(data[['t_in']])

        if robust_value:
            cumulated_mean = 0
            for i, row in data.iterrows():
                if i in anomalous_indices:
                    cumulated_mean += 0
                else:
                    cumulated_mean += row['specific_energy_sensor'] - row['specific_energy']

                # make the correction
                data.loc[i, 'mean_error'] = cumulated_mean / (i + 1)
                
            data['specific_energy'] = data['specific_energy'] + data['mean_error']
        else:
            # calculate the error between the sensor and the simulation
            data['error'] = data['specific_energy_sensor'] - data['specific_energy']

            # calculate the rolling cumulative mean of the error
            data['mean_error'] = data['error'].cumsum() / (data.index + 1)

            # correct the simulation with the error
            data['specific_energy'] = data['specific_energy'] + data['mean_error']

        # load the SOM to detect anomalies
        som = joblib.load('./code/models/som.pkl')

        # load the scaler
        scaler = joblib.load('./code/models/scaler.pkl')

        # scale the data
        x = scaler.transform(data[['t_in', 'specific_energy']])

        # get winning neurons for each sample
        winning = [som.winner(x_i) for x_i in x]

        # load the json file with the anomaly probabilities
        with open('./data/prob_table.json', 'r') as f:
            anomaly_probabilities = json.load(f)

        # get the anomaly probabilities for each sample
        data['anomaly_probability'] = [anomaly_probabilities['mean'].get(str(w), 0) for w in winning]

        return data.to_dict('records')
    else:
        return []

@app.callback(
    Output('interval-component', 'max_intervals'),
    [Input('memory-output', 'data')]
)
def set_max_intervals(data):
    return len(data)

@app.callback(
    Output('example-graph1', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('memory-output', 'data')]
)
def update_graph1(n, data):
    if data:
        df = pd.DataFrame(data[:n])

        
        return {
            'data': [
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['specific_energy_sensor'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5},  # Making the plot transparent
                    name='Specific Energy (kJ / kg)',  # Naming the series
                    legendgroup='specific_energy',  # Grouping for legend
                    showlegend=True  # Show legend on the plot
                ),
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['specific_energy'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5},  # Making the plot transparent
                    name='Specific Energy Simulation (kJ / kg)',  # Naming the series
                    legendgroup='specific_energy',  # Grouping for legend
                    showlegend=True  # Show legend on the plot
                )
            ],
            'layout': go.Layout(
                yaxis={'title': 'Specific Energy (kJ / kg)'},  # Title for Y Axis
                title='Specific Energy - Distillation Tower T-101',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                yaxis={'title': 'Specific Energy (kJ / kg)'},  # Title for Y Axis
                title='Specific Energy - Distillation Tower T-101',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }

@app.callback(
    Output('example-graph5', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('memory-output', 'data')]
)
def update_graph5(n, data):
    if data:
        df = pd.DataFrame(data[:n])

        
        return {
            'data': [
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['mean_error'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5},  # Making the plot transparent
                    name='Specific Energy (kJ / kg)',  # Naming the series
                    legendgroup='specific_energy',  # Grouping for legend
                    showlegend=False  # Show legend on the plot
                )
            ],
            'layout': go.Layout(
                yaxis={'title': 'Simulation Error (kJ / kg)'},  # Title for Y Axis
                title='Specific Energy - Simulation Error',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                yaxis={'title': 'Specific Energy (kJ / kg)'},  # Title for Y Axis
                title='Specific Energy - Distillation Tower T-101',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    
@app.callback(
    Output('example-graph4', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('memory-output', 'data')]
)
def update_graph4(n, data):
    if data:
        df = pd.DataFrame(data[:n])
        
        return {
            'data': [
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['specific_energy_sensor'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5},  # Making the plot transparent
                    name='Specific Energy (kJ / kg)',  # Naming the series
                    legendgroup='specific_energy',  # Grouping for legend
                    showlegend=False  # Show legend on the plot
                ),
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['specific_energy'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5},  # Making the plot transparent
                    name='Specific Energy Simulation (kJ / kg)',  # Naming the series
                    legendgroup='specific_energy',  # Grouping for legend
                    showlegend=False  # Show legend on the plot
                )
            ],
            'layout': go.Layout(
                yaxis={
                    'title': 'Specific Energy (kJ / kg)',
                    'range': [3615, 3623]
                },  # Title for Y Axis
                title='Specific Energy - Distillation Tower T-101 - Approved Region',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'},
                shapes=[  # Adding a green horizontal span area
                    {
                        'type': 'rect',
                        'xref': 'paper',
                        'yref': 'y',
                        'x0': 0,
                        'y0': 3618,  # Start of the span area on y-axis
                        'x1': N_DATA,
                        'y1': 3620,  # End of the span area on y-axis
                        'fillcolor': 'rgba(0,255,0,0.1)',  # Green with alpha 0.1
                        'layer': 'below',
                        'line_width': 0,  # No border line
                    }
                ]
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                yaxis={
                    'title': 'Specific Energy (kJ / kg)'
                },  # Title for Y Axis
                title='Specific Energy - Distillation Tower T-101 - Approved Region',
                margin={'l': 100, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'},
                shapes=[  # Adding a green horizontal span area
                    {
                        'type': 'rect',
                        'xref': 'paper',
                        'yref': 'y',
                        'x0': 0,
                        'y0': 0,  # Start of the span area on y-axis
                        'x1': N_DATA,
                        'y1': N_DATA,  # End of the span area on y-axis
                        'fillcolor': 'rgba(0,255,0,0.1)',  # Green with alpha 0.1
                        'layer': 'below',
                        'line_width': 0,  # No border line
                    }
                ]
            )
        }
    
@app.callback(
    Output('example-graph2', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('memory-output', 'data')]
)
def update_graph2(n, data):
    if data:
        df = pd.DataFrame(data[:n])
        return {
            'data': [
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['t_in'] - 273.15,  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5}  # Making the plot transparent
                )
            ],
            'layout': go.Layout(
                title='Feed Temperature (deg. C)',
                margin={'l': 50, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                title='Feed Temperature (deg. C)',
                margin={'l': 50, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }

@app.callback(
    Output('example-graph3', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('memory-output', 'data')]
)
def update_grap3(n, data):
    if data:
        df = pd.DataFrame(data[:n])
        return {
            'data': [
                go.Scatter(
                    x=df.index,  # replace 'x' with your actual column name
                    y=df['p_in'],  # replace 'y' with your actual column name
                    mode='lines+markers',
                    marker={'opacity': 0.5}  # Making the plot transparent
                )
            ],
            'layout': go.Layout(
                title='Feed Pressure (bar)',
                margin={'l': 50, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(
                title='Feed Pressure (bar)',
                margin={'l': 50, 'r': 0, 't': 30, 'b': 20},
                plot_bgcolor='#282b30',
                paper_bgcolor='#282b30',
                font={'color': '#7FDBFF'}
            )
        }
    
@app.callback(
[Output('last-y1', 'children'),
    Output('last-y2', 'children'),
    Output('last-y3', 'children'),
    Output('last-y4', 'children')],
[Input('interval-component', 'n_intervals')],
[State('memory-output', 'data')]
)
def update_card_values(n, data):
    if data:
        df = pd.DataFrame(data[:n])

        t_in = df['t_in'].tail(1).tolist()
        p_in = df['p_in'].tail(1).tolist()
        prob = df['anomaly_probability'].tail(1).tolist()
        erro_medio = df['mean_error'].tail(1).tolist()
        
        return round(t_in[0] - 273.15, 2), round(p_in[0], 2), f'{round(prob[0] * 100, 2)} %', round(erro_medio[0], 2)
    else:
        return None, None, None, None

if __name__ == '__main__':
    app.run_server(debug=True)
