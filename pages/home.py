import dash
from dash import html


dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='This is our Home page'),

    html.Div(children='''
        There will be instructions!
    '''),

])
