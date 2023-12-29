from dash import html
import dash_bootstrap_components as dbc


navigation_row = html.Div([
    dbc.Row([
        dbc.Col([html.Button('Metrics', id="metrics", className="button_1")], width=2),
        dbc.Col([html.Button('Compatimetrics', id="compatimetrics", className="button_1")], width=2),
        dbc.Col([html.Button('Weights', id="weights", className="button_1")], width=2),
        dbc.Col([html.Button('XAI', id="xai", className="button_1")], width=2),
    ], justify="center"),
], className="navigation_buttons")

