import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc


dash.register_page(__name__)

layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("1. Train model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/train.png", height="70px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([
        dbc.Col(html.H1("2. Save model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/save.png", height="70px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([
        dbc.Col(html.H1("3. Upload data file"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/upload.png", height="70px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([
        dbc.Col(html.H1("4. Select column"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/select.png", height="70px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([
        dbc.Col(html.H1("5. Upload model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/upload.png", height="70px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([
        dbc.Col(html.H1("6. Analise metrics and plots"), align="center", width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/analise.png", height="70px"), align="center", width='auto'),
    ], justify="start"),

], className="instruction")


