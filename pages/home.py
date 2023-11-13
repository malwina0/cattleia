import dash
from dash import html, dcc, callback, Input, Output, State
import base64
import io
import dash_bootstrap_components as dbc
import sys
import zipfile
import numpy as np
import compatimetrics_plots
import metrics
import shutil
import pandas as pd
from utils import get_predictions_from_model, get_task_from_model
from autogluon.tabular import TabularPredictor

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
        Our project co-ordinator and supervisor is Anna Kozak
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
])

# page layout
layout = html.Div([
    dcc.Store(id='csv_data', data=[], storage_type='memory'),
    dcc.Store(id='y_label_column', data=[], storage_type='memory'),
    dcc.Store(id='predictions', data=[], storage_type='memory'),
    dcc.Store(id='model_names', data=[], storage_type='memory'),
    dcc.Store(id='task', data=[], storage_type='memory'),
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
        dcc.Loading(id="loading-1", type="default", children=html.Div(about_us, id="plots"), className="spin"),
    ], id="plots_div"),
    html.Div(id='model_selection'),
    html.Div(id='compatimetrics_container', children=html.Div(id='compatimetrics_plots'))
])


# data loading function
def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df
        elif "pkl" in filename:
            model = pd.read_pickle(io.BytesIO(decoded))
            if "<class 'flaml" in str(model.__class__).split("."):
                library = "Flaml"
            else:
                library = "AutoSklearn"
            return model, library
        elif "zip" in filename:
            with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
                zip_ref.extractall('./uploaded_model')

            model = TabularPredictor.load('./uploaded_model', require_py_version_match=False)
            library = "AutoGluon"
            return model, library

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])


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
    try:
        shutil.rmtree('./uploaded_model')
    except FileNotFoundError:
        pass

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


# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Output('model_names', 'data'),
    Output('predictions', 'data'),
    Output('task', 'data'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('plots', 'children'),
)
def update_model(contents, filename, df, column, about_us):
    model_names = []
    predictions = []
    task = []
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

        df = pd.DataFrame.from_dict(df)
        df = df.dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        task = get_task_from_model(model, y, library)
        predictions = get_predictions_from_model(model, X, y, library, task)
        model_names = list(predictions.keys())

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
            ]

        # for plot in metrics.permutation_feature_importance_all(model, X, y, library=library, task=task):
        #     plot_component.append(dcc.Graph(figure=plot, className="plot"))
        #
        # for plot in metrics.partial_dependence_plots(model, X, library=library, autogluon_task=task):
        #     plot_component.append(dcc.Graph(figure=plot, className="plot"))

        # It may be necessary to keep the model for the code with weights,
        # for now we remove the model after making charts
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass
        children = html.Div(plot_component)

    return children, model_names, predictions, task


@callback(
    Output('model_selection', 'children'),
    Input('model_names', 'data')
)
def update_model_selector(model_names):
    children = []
    if model_names:
        title = html.H2("Compatimetrics", className="compatimetrics_title", style={'color': 'white'})
        dropdown = dcc.Dropdown(id='model_select', className="dropdown-class",
                                options=[{'label': x, 'value': x} for x in model_names],
                                value=model_names[0], clearable=False)
        children = html.Div([title, dropdown])
    return children


@callback(
    Output('compatimetrics_plots', 'children'),
    State('predictions', 'data'),
    Input('model_select', 'value'),
    State('task', 'data'),
    State('csv_data', 'data'),
    State('y_label_column', 'data')
)
def update_compatimetrics_plot(predictions, model_to_compare, task, df, column):
    children = []
    df = pd.DataFrame.from_dict(df)
    df = df.dropna()
    y = df.iloc[:, df.columns == column["name"]]
    y = y.squeeze()
    if model_to_compare:
        if task == 'classification':
            children = [dbc.Row([
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                   className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                   className="plot")],
                        width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.disagreement_ratio_plot(predictions, y, model_to_compare),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjunctive_metrics_plot(predictions, y, model_to_compare),
                                       className="plot")],
                            width=6),
                ]),
                dbc.Row(
                    [dcc.Graph(figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                               className='plot')
                ]),
                dbc.Row(
                    [dcc.Graph(figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                               className='plot')
                ]),
            ]
        elif task == 'regression':
            children = [dbc.Row([
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_matrix(predictions), className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_matrix(predictions), className="plot")],
                        width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.ar_matrix(predictions, y), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.sdr_matrix(predictions, y), className="plot"), ],
                            width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_comparison(predictions, model_to_compare), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_comparison(predictions, model_to_compare), className="plot")],
                            width=6),
                ]),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_rmse_plot(predictions, y, model_to_compare),
                        className='plot')
                     ]),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_distribution(predictions, model_to_compare), className="plot")
                ]),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_boxplot(predictions, y, model_to_compare), className="plot")
                ])
            ]
        else:
            children.append(html.H2('MULTIKLASOWEJ JESZCZE NIE WŁANCZASZ JESZCZE'))
    return children


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
