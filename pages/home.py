import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import sys
from callbacks import *

sys.path.append("..")

dash.register_page(__name__, path='/')

about_us = html.Div([
    html.H2("What is cattleia?", className="about_us_main"),
    html.H2("""
        The cattleia tool, through tables and visualizations, 
        allows you to look at the metrics of the models to assess their 
        contribution to the prediction of the built committee. 
        Also, it introduces compatimetrics, which enable analysis of 
        model similarity. Our application support model ensembles 
        created by automatic machine learning packages available in Python, 
        such as Auto-sklearn, AutoGluon, and FLAML.
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
    html.H2("About us", className="about_us_main"),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/dominik.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Dominik Kędzierski", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/malwina.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Malwina Wojewoda", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/jakub.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Jakub Piwko", className="about_us_img")], align="center")
        ], width=4),
    ], style={"max-width": "70%", "height": "auto"}),
    html.Br(),
    html.H2("""
        We are Data Science students at Warsaw University of Technology, 
        who are passionate about data analysis and machine learning. 
        Through our work and projects, we aim to develop skills in the area of data analysis, 
        creating predictive models and extracting valuable insights from numerical information. 
        Our mission is to develop skills in this field and share our knowledge with others. 
        Cattleia is our project created as a Bachelor thesis.  
        Our project co-ordinator and supervisor is Anna Kozak
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
])

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
            dcc.Link(html.H5("Instruction"), href="/instruction", className="sidepanel_text"),
            dbc.Button("Change Content", id="link-button", color="link", style={"textDecoration": "underline"}),
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
