import dash
from dash import html, dcc, Output, Input, callback, State, ALL
import dash_bootstrap_components as dbc
import sys
import metrics
import shutil
import pandas as pd
from utils import parse_data

from weights import slider_section, get_ensemble_names_weights, weights_plot, weights_metrics_table

sys.path.append("..")

dash.register_page(__name__,  path='/')

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
            dbc.Row([html.Img(src="assets/dominik.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Dominik Kędzierski", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/malwina.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Malwina Wojewoda", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/jakub.png",
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
        Our project co-ordinator and supervisor is Anna Kozak.
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
    ])


# page layout
layout = html.Div([
    dcc.Store(id='csv_data', data=[], storage_type='memory'),
    dcc.Store(id='y_label_column', data=[], storage_type='memory'),
    dcc.Store(id='ensemble_model', data=[], storage_type='memory'),
    dcc.Store(id='library', data='', storage_type='memory'),
    dcc.Store(id='model_names', data=[], storage_type='memory'),
    dcc.Store(id='weights', data=[], storage_type='memory'),
    # side menu
    html.Div([
        dbc.Container([
            dcc.Link(html.H5("Instruction"), href="/instruction", className="sidepanel_text"),
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
        ], className="px-3 sidepanel")
    ], id="side_menu_div"),
    # plots
    html.Div([
        dcc.Loading(id="loading-1", type="default", children=html.Div(about_us, id="plots"), className="spin")
    ], id="plots_div")
])


# part responsible for adding csv file
@callback(
    [Output('csv_data', 'data'),
    Output('select_y_label_column', 'children')],
    [Input('upload_csv_data', 'contents'),
    State('upload_csv_data', 'filename')]
)
def update_output(contents, filename):
    data = []
    children = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        data = df.to_dict()

        children = html.Div([
            html.P(filename, className="sidepanel_text"),
            html.Hr(),
            html.H5("Select target colum", className="sidepanel_text"),
            dcc.Dropdown(id='column_select', className="dropdown-class",
                         options=[{'label': x, 'value': x} for x in df.columns]),
            html.Hr(),
        ])

    return data, children


# part responsible for choosing target column
@callback(
    [Output('y_label_column', 'data'),
        Output('upload_model_section', 'children')],
    Input('column_select', 'value')
)
def select_columns(value):
    children = html.Div([
        html.H5("Upload model", className="sidepanel_text"),
        dcc.Upload(
            id='upload_model',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            className="upload_data",
            multiple=True
        ),
    ])
    data = {'name': value}

    return data, children

# @callback(Output('ensemble_model', 'data'),
#           Input('upload_model', 'contents'),
#           State('upload_model', 'filename'),
#           )
# def update_model_0(contents, filename):
#     if contents:
#         contents = contents[0]
#         filename = filename[0]
#
#         # delete folder if it already is, only for autogluon
#         try:
#             shutil.rmtree('./uploaded_model')
#         except FileNotFoundError:
#             pass
#
#         ensemble_model, library = parse_data(contents, filename)
#         print(library)
#         model_names, weights = get_ensemble_names_weights(ensemble_model)
#         original_weights = copy.deepcopy(weights)
#         return [ensemble_model]


# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('plots', 'children'),
)
def update_model(contents, filename, df, column, about_us):
    children = about_us
    if contents:
        contents = contents[0]
        filename = filename[0]

        # delete folder if it already is, only for autogluon
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        model, library = parse_data(contents, filename)
        models_name, weights = get_ensemble_names_weights(model)

        df = pd.DataFrame.from_dict(df)
        df = df.dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        if y.squeeze().nunique() > 10:
            task = "regression"
        else:
            task = "classification"
        if task == "regression":
            plot_component = [
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=metrics.mse_plot(model, X, y, library=library), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=metrics.mape_plot(model, X, y, library=library), className="plot")],
                            width=6),
                ]),
                dcc.Graph(figure=metrics.mae_plot(model, X, y, library=library), className="plot"),
                dcc.Graph(figure=metrics.correlation_plot(model, X, library=library, task=task, y=y),
                          className="plot"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot"),
            ]
        else:
            plot_component = [
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=metrics.accuracy_plot(model, X, y, library=library),
                        className="plot")], width=6),
                    dbc.Col([dcc.Graph(figure=metrics.precision_plot(model, X, y, library=library),
                        className="plot")], width=6),
                ]),
                dbc.Row([
                    dbc.Col(
                        [dcc.Graph(figure=metrics.recall_plot(model, X, y, library=library), className="plot")],
                        width=6),
                    dbc.Col(
                        [dcc.Graph(figure=metrics.f1_score_plot(model, X, y, library=library), className="plot")],
                        width=6),
                ]),
                dcc.Graph(figure=metrics.correlation_plot(model, X, library=library, task=task, y=y),
                          className="plot"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot"),
                dbc.Row([
                    dbc.Col([
                        html.Div([slider_section(model_name, weights[i], i) for i, model_name in enumerate(models_name)],
                         style={'color': 'white'})
                    ], width=8),
                    dbc.Col([weights_metrics_table()], width=4)
                ]),
                dcc.Graph(figure=weights_plot(model, X, y, models_name, weights), className="plot", id='my-graph'),
            ]

        for plot in metrics.permutation_feature_importance_all(model, X, y, library=library, task=task):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        for plot in metrics.partial_dependence_plots(model, X, library=library, autogluon_task=task):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        # It may be necessary to keep the model for the code with weights,
        # for now we remove the model after making charts
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        children = html.Div(plot_component)

    return children

@callback(
Output('my-graph', 'figure', allow_duplicate=True),
    Input({"type": "part_add", "index": ALL}, 'value'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    prevent_initial_call=True
)
def display_output(values, contents, filename, df, column):
    if contents:
        contents = contents[0]
        filename = filename[0]

        # delete folder if it already is, only for autogluon
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        model, library = parse_data(contents, filename)
        models_name, weights = get_ensemble_names_weights(model)

        df = pd.DataFrame.from_dict(df)
        df = df.dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        sum_slider_values = sum(values)

        weights = [value / sum_slider_values for value in values]
        return weights_plot(model, X, y, models_name, weights)

# callback responsible for moving the menu
dash.clientside_callback(
    """
    function() {
        var menu = document.getElementById('side_menu_div');
        var content = document.getElementById('plots_div');

        menu.style.left = '0';
        content.style.left = '250px';

        window.addEventListener('scroll', function() {
            var currentScrollY = window.scrollY;
            console.log(currentScrollY);
            if (currentScrollY > 50) {
                menu.style.left = '-250px';
                content.style.marginLeft = '0';
            } else {
                menu.style.left = '0';
                content.style.marginLeft = '250px';
            }
        });
    }
    """,
    Output('side_menu_div', 'style'),
    Output('plots_div', 'style'),
    Input('upload_csv_data', 'contents'),
    prevent_initial_call=False
)
