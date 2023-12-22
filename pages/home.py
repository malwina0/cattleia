import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
from callbacks import *
from callbacks.about_us import about_us

sys.path.append("..")

dash.register_page(__name__, path='/')

# page layout
layout = html.Div([
    dcc.Store(id='csv_data', data=[], storage_type='memory'),
    dcc.Store(id='y_label_column', data=[], storage_type='memory'),
    dcc.Store(id='metrics_plots', data=[], storage_type='memory'),
    dcc.Store(id='compatimetric_plots', data=[], storage_type='memory'),
    dcc.Store(id='weight_plots', data=[], storage_type='memory'),
    dcc.Store(id='predictions', data={}, storage_type='memory'),
    dcc.Store(id='model_names', data=[], storage_type='memory'),
    dcc.Store(id='task', data=[], storage_type='memory'),
    dcc.Store(id='proba_predictions', data=[], storage_type='memory'),
    dcc.Store(id='weights_list', data=[], storage_type='memory'),
    # side menu
    html.Div([
        dbc.Container([
            html.Br(),
            dbc.Button("Instruction", id="instruction-button", className='page-button'),
            dbc.Button('About us', id='about-us-button', className='page-button', style={'display': 'none'}),
            html.Hr(),
            html.H5("Upload csv data", className="sidepanel_text"),
            dcc.Upload(
                id='upload_csv_data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="upload_data",
                multiple=True
            ),
            html.Div(id='select_y_label_column'),
            html.Div(id='upload_model_section'),
            html.Div(id="switch_annotation")
        ], className="px-3 sidepanel")
    ], id="side_menu_div"),
    # plots
    html.Div([
        dcc.Loading(id="loading-1", type="default", children=html.Div(about_us, id="plots"), className="spin"),
    ], id="plots_div"),
])
