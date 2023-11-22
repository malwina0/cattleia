import dash
from dash import html, dcc, Output, Input, callback, State, ALL, dash_table
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
from utils import get_predictions_from_model, get_task_from_model, parse_data
from autogluon.tabular import TabularPredictor
import dash_daq as daq
from weights import slider_section, get_ensemble_names_weights, \
    tbl_metrics, tbl_metrics_adj_ensemble, calculate_metrics, calculate_metrics_adj_ensemble

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
    dcc.Store(id='metrics_plots', data=[], storage_type='memory'),
    dcc.Store(id='compatimetric_plots', data=[], storage_type='memory'),
    dcc.Store(id='weight_pots', data=[], storage_type='memory'),
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
            html.Div(id="switch_annotation")
        ], className="px-3 sidepanel")
    ], id="side_menu_div"),
    # plots
    html.Div([
        dcc.Loading(id="loading-1", type="default", children=html.Div(about_us, id="plots"), className="spin"),
    ], id="plots_div"),
    html.Div(id='model_selection'),
    html.Div(id='compatimetrics_container', children=html.Div(id='compatimetrics_plots'))
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
     Output('upload_model_section', 'children'),
     Output('switch_annotation', 'children')],
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
    switch = html.Div([
        html.Hr(),
        html.H5("Annotation OFF", className="sidepanel_text", id="switch_text"),
        daq.ToggleSwitch(
            id='my-toggle-switch',
            value=False,
            color="#0072ef"
        )
    ])

    data = {'name': value}

    return data, children, switch


# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Output('metrics_plots', 'data'),
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
    task = []
    predictions = []
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
        models_name, weights = get_ensemble_names_weights(model, library)

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
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=metrics.rmse_plot(model, X, y, library=library), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=metrics.r_2_plot(model, X, y, library=library), className="plot")],
                            width=6),
                ]),
                dcc.Graph(figure=metrics.mae_plot(model, X, y, library=library), className="plot"),
                dcc.Graph(figure=metrics.correlation_plot(model, X, library=library, task=task, y=y),
                          className="plot"),
                html.H2("asdasdasdasdasdaaaaaaaaaaaaaaaaaaasadasda", className="about_us_str", id="ann_1"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot")
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
                html.H2("""
                    Partial Dependence isolate one specific feature's effect on the model's output while maintaining 
                    all other features at fixed values. It capturing how the model's output changes as the chosen 
                    feature varies. 
                    """, className="annotation_str", id="ann_1"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot")
            ]
        if library != "Flaml":
            plot_component.append(
                dbc.Row([
                    dbc.Col([
                        html.Div([], style={'height': '31px'}),  # placeholder to show metrics in the same line
                        html.Div(
                            [slider_section(model_name, weights[i], i) for i, model_name in enumerate(models_name)],
                            style={'color': 'white'})
                    ], width=7),
                    dbc.Col([tbl_metrics(model, X, y, task, library, weights)
                             ], width=4)
                ])
            )
            plot_component.append(
                dbc.Row([
                    dbc.Col([tbl_metrics_adj_ensemble(model, X, y, task, library, weights)], width=4)
                ], justify="center")
            )

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

        plot_component.insert(0, html.Div([
            dbc.Row([
                dbc.Col([html.Button('Weights', id="weights", className="button_1")], width=2),
                dbc.Col([html.Button('Metrics', id="metrics", className="button_1")], width=2),
                dbc.Col([html.Button('Compatimetrics', id="compatimetrics", className="button_1")],width=2),
            ], justify="center"),
        ], style={"display": "block", "position": "sticky"}))

        children = html.Div(plot_component)

    return children, children, model_names, predictions, task


@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('weights', 'n_clicks'),
    State('weight_pots', 'data'),
    State('metrics_plots', 'data'),
    prevent_initial_call=True
)
def show_metrics(n_clicks, data, metrics_plots):
    if n_clicks >= 1:
        return data
    return metrics_plots

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('metrics', 'n_clicks'),
    State('metrics_plots', 'data'),
    prevent_initial_call=True
)
def show_metrics(n_clicks, data):
    return data



@callback(
    [Output('ann_1', 'style'),
     Output('switch_text', 'children')],
    Input('my-toggle-switch', 'value')
)
def update_output(value):
    if value:
        return {}, "Annotation ON"
    else:
        return {"display": "none"}, "Annotation OFF"


@callback(
    Output('model_selection', 'children'),
    Input('model_names', 'data')
)
def update_model_selector(model_names):
    children = []
    if model_names:
        title = html.H4("Choose model for compatimetrics analysis", className="compatimetrics_title", style={'color': 'white'})
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
            children = [dbc.Row([
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                   className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                   className="plot")],
                        width=6),
            ]), dbc.Row([
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                   className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                   className="plot")],
                        width=6),
            ]), dbc.Row([
                dbc.Col([dcc.Graph(
                    figure=compatimetrics_plots.conjunctive_precision_multiclass_plot(predictions, y, model_to_compare),
                    className="plot")],
                    width=6),
                dbc.Col([dcc.Graph(
                    figure=compatimetrics_plots.conjunctive_recall_multiclass_plot(predictions, y, model_to_compare),
                    className="plot")],
                    width=6),
            ]), dbc.Row(
                [dcc.Graph(
                    figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                    className='plot')
                ]), dbc.Row(
                [dcc.Graph(
                    figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                    className='plot')
                ])
            ]
    return children


@callback(
    Output('metrics-table', 'data', allow_duplicate=True),
    Output('adj_weights-table', 'data'),
    Input({"type": "weight_slider", "index": ALL}, 'value'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('task', 'data'),
    prevent_initial_call=True
)
def display_output(values, contents, filename, df, column, task):
    if contents:
        contents = contents[0]
        filename = filename[0]

        ensemble_model, library = parse_data(contents, filename)
        df = pd.DataFrame.from_dict(df).dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]].squeeze()

        sum_slider_values = sum(values)
        weights = [round((value / sum_slider_values), 2) for value in values]

        df = calculate_metrics(ensemble_model, X, y, task, library, weights)
        df_adj = calculate_metrics_adj_ensemble(ensemble_model, X, y, task, library, weights)

        return df.to_dict('records'), df_adj.to_dict('records')


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
