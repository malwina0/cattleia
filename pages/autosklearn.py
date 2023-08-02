import dash
from dash import html, dcc, callback, Input, Output, State, dash_table

import base64
import datetime
import io
import plotly.graph_objs as go


import sys
sys.path.append("..")
import metrics



import pandas as pd


dash.register_page(__name__)

layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])




