import copy
import dash
from dash import html, Dash, dcc, Output, Input, callback, MATCH, State, ALL
import pickle
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import metrics
from weights import get_ensemble_names_weights, weights_plot, slider_section

dash.register_page(__name__, path='/')

with open('pages/autosklearn_reg.pkl', 'rb') as file:
    model = pickle.load(file)

ensemble_model = model
data = pd.read_csv('pages/SalaryData.csv')
X = data.iloc[:, :5]
y = data[['Salary']]

models_name, weights = get_ensemble_names_weights(ensemble_model)
original_weights = copy.deepcopy(weights)

layout = html.Div(children=[
    html.H1(children='This is our Home page'),
    html.Div([slider_section(model_name, i) for i, model_name in enumerate(models_name)], style={'color': 'white'}),
    dcc.Graph(figure=weights_plot(model, X, y, models_name, weights), className="plot", id='my-graph'),
    html.Button('Reset weights', id='update-button', n_clicks=0)
])


@callback(
Output('my-graph', 'figure', allow_duplicate=True),
    Input({"type": "part_add", "index": ALL}, 'value'),
    prevent_initial_call=True
)
def display_output(values):
    sum_slider_values = sum(values)
    weights = [value / sum_slider_values for value in values]
    return weights_plot(ensemble_model, X, y, models_name, weights)


@callback(
    Output('my-graph', 'figure', allow_duplicate=True),
    Input('update-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_graph(n_clicks):
    if n_clicks is None:
        return dash.no_update
    return weights_plot(ensemble_model, X, y, models_name, original_weights)
