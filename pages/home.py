import dash
from dash import html, dcc, Output, Input, callback, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_daq as daq
import shutil
import pandas as pd
import sys
from components import compatimetrics_plots, metrics
import shutil
import pandas as pd
from utils.utils import get_predictions_from_model, get_task_from_model, parse_data, parse_model, \
    get_probability_pred_from_model, get_ensemble_weights
import dash_daq as daq
from components.weights import slider_section, \
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
        if ".csv" in filename:
            df = parse_data(contents)
            data = df.to_dict()
            # Creating the dropdown menu with full labels displayed as tooltips
            options = [{'label': x[:20] + '...' if len(x) > 20 else x, 'value': x, 'title': x} for x in df.columns]

            children = html.Div([
                html.P(filename, className="sidepanel_text"),
                html.Hr(),
                html.H5("Select target colum", className="sidepanel_text"),
                dcc.Dropdown(
                    id='column_select',
                    className="dropdown-class",
                    options=options
                ),
                html.Hr(),
            ])
        else:
            children = html.Div(["Please provide the file in .csv format."], style={"color": "white"})

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
        html.H5("Annotation", className="sidepanel_text", id="switch_text"),
        daq.ToggleSwitch(
            id='my-toggle-switch',
            value=True,
            color="#0072ef"
        )
    ])

    data = {'name': value}

    return data, children, switch


# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Output('metrics_plots', 'data'),
    Output('weight_plots', 'data'),
    Output('model_names', 'data'),
    Output('predictions', 'data'),
    Output('task', 'data'),
    Output('proba_predictions', 'data'),
    Output('weights_list', 'data'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('plots', 'children'),
)
def update_model(contents, filename, df, column, about_us):
    model_names, weights, task, predictions, proba_predictions, weights_plots = ([] for _ in range(6))
    children = about_us
    if contents:
        contents = contents[0]
        filename = filename[0]

        # delete folder if it already is, only for autogluon
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        if ('.pkl' in filename) or ('.zip' in filename):
            model, library = parse_model(contents, filename)
            weights = get_ensemble_weights(model, library)

            df = pd.DataFrame.from_dict(df)
            df = df.dropna()
            X = df.iloc[:, df.columns != column["name"]]
            y = df.iloc[:, df.columns == column["name"]]
            y = y.squeeze()

            task = get_task_from_model(model, y, library)
            predictions = get_predictions_from_model(model, X, y, library, task)
            model_names = list(predictions.keys())
            base_models = model_names[1:len(model_names)]
            if task == "regression":
                metrics_plots = [
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=metrics.mse_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=metrics.mape_plot(predictions, y), className="plot")],
                                width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=metrics.rmse_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=metrics.r_2_plot(predictions, y), className="plot")],
                                width=6),
                    ]),
                    dcc.Graph(figure=metrics.mae_plot(predictions, y), className="plot"),
                ]
            else:
                proba_predictions = get_probability_pred_from_model(model, X, library)
                metrics_plots = [
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=metrics.accuracy_plot(predictions, y),
                                           className="plot")], width=6),
                        dbc.Col([dcc.Graph(figure=metrics.precision_plot(predictions, y),
                                           className="plot")], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            [dcc.Graph(figure=metrics.recall_plot(predictions, y), className="plot")],
                            width=6),
                        dbc.Col(
                            [dcc.Graph(figure=metrics.f1_score_plot(predictions, y), className="plot")],
                            width=6),
                    ]),
                ]

            metrics_plots += [
                dcc.Graph(figure=metrics.correlation_plot(predictions, task=task, y=y),
                          className="plot"),
                html.H2("""
                    Prediction compare plot shows the differences between model predictions and true values. 
                    The x-axis shows observations and the y-axis shows models. For classification, the color on the 
                    plot indicates whether a given prediction is correct, while for regression tasks the percentage 
                    difference between the true and predicted value is shown.
                    """,
                        className="annotation_str", id="ann_0"),
                dcc.Graph(figure=metrics.prediction_compare_plot(predictions, y, task=task),
                          className="plot")]

            weights_plots = []
            if library != "Flaml":
                weights_plots.append(
                    html.H2(
                        html.P(["""
                        Below on the left, sliders are available for modifying the weight value assigned to each model
                         within the ensemble. Initially, these values are set to the elected by the AutoML package. 
                         To revert to these default values, click the "Reset weights" button.""", html.Br(),
                        """On the right, a table displays weight values alongside task-specific metrics for each 
                        individual model. Editing the cells within the "Weights" column allows for direct modification 
                        of a model's weight. Any adjustments made will proportionally modify other weights to ensure 
                        their sum remains at 1.""", html.Br(),
                        """Below in the table, metrics for both the ensemble model's original weights (set by AutoML) 
                        and custom weights (set manually) are presented side by side. Changes in metrics are visually 
                        indicated: cells turn green for improved metrics and red if the changes result in inferior 
                        performance.
                        """]),
                    className="annotation_str", id="ann_2")
                )
                weights_plots.append(html.Br())
                weights_plots.append(
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.Div([], style={'height': '31px'}),  # placeholder to show metrics in the same line
                                html.Div(
                                    [slider_section(model_name, weights[i], i) for i, model_name in
                                     enumerate(base_models)],
                                    style={'color': 'white'}),
                                html.Button('Reset weights', id='update-weights-button', n_clicks=0)
                            ], width=7),
                            dbc.Col([tbl_metrics(predictions, y, task, weights),
                                     html.Div(id='weight-update-info', style={'color': 'white'})
                                     ], width=4)
                        ]),
                        dbc.Row([
                            dbc.Col([tbl_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights)],
                                    width=4)
                        ], justify="center")
                    ], className="weight-analysis-col")
                )

            for plot in metrics.permutation_feature_importance_all(model, X, y, library=library, task=task):
                metrics_plots.append(dcc.Graph(figure=plot, className="plot"))

            metrics_plots.append(
                html.H2("""
                    Partial Dependence isolate one specific feature's effect on the model's output while maintaining 
                    all other features at fixed values. It capturing how the model's output changes as the chosen 
                    feature varies. When the number of observations is large, in order to speed up the generation of 
                    graphs, only a subset of the data is used for calculations.
                    """,
                        className="annotation_str", id="ann_1")
            )

            if len(X) < 2000:
                for plot in metrics.partial_dependence_plots(model, X, library=library, autogluon_task=task):
                    metrics_plots.append(dcc.Graph(figure=plot, className="plot"))
            else:
                for plot in metrics.partial_dependence_plots(model, X.sample(2000), library=library,
                                                             autogluon_task=task):
                    metrics_plots.append(dcc.Graph(figure=plot, className="plot"))

            # It may be necessary to keep the model for the code with weights,
            # for now we remove the model after making charts
            try:
                shutil.rmtree('./uploaded_model')
            except FileNotFoundError:
                pass

            metrics_plots.insert(0, html.Div([
                dbc.Row([
                    dbc.Col([html.Button('Weights', id="weights", className="button_1")], width=2),
                    dbc.Col([html.Button('Metrics', id="metrics", className="button_1")], width=2),
                    dbc.Col([html.Button('Compatimetrics', id="compatimetrics", className="button_1")], width=2),
                ], justify="center"),
            ], style={"display": "block", "position": "sticky"}))
            weights_plots.insert(0, html.Div([
                dbc.Row([
                    dbc.Col([html.Button('Weights', id="weights", className="button_1")], width=2),
                    dbc.Col([html.Button('Metrics', id="metrics", className="button_1")], width=2),
                    dbc.Col([html.Button('Compatimetrics', id="compatimetrics", className="button_1")], width=2),
                ], justify="center"),
            ], style={"display": "block", "position": "sticky"}))

            weights_plots = html.Div(weights_plots)
            children = html.Div(metrics_plots)
        else:
            children = html.Div(["Please provide the file in .pkl or .zip format."], style={"color": "white"})

    return children, children, weights_plots, model_names, predictions, task, proba_predictions, weights


# callbacks for buttons to change plots categories
@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('weights', 'n_clicks'),
    State('weight_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_weights(n_clicks, data, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        return data
    return children


@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('metrics', 'n_clicks'),
    State('metrics_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_metrics(n_clicks, data, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        return data
    return children


@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('compatimetrics', 'n_clicks'),
    State('compatimetric_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_compatimetrics(n_clicks, data, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        return data
    return children


# callback display compatimetric part
@callback(
    Output('compatimetric_plots', 'data', allow_duplicate=True),
    Input('model_names', 'data'),
    prevent_initial_call=True
)
def update_model_selector(model_names):
    if len(model_names) > 0:
        model_names.pop(0)
    children = []
    if model_names:
        title = html.H4("Choose model for compatimetrics analysis:", className="compatimetrics_title",
                        style={'color': 'white'})
        dropdown = dcc.Dropdown(id='model_select', className="dropdown-class",
                                options=[{'label': x, 'value': x} for x in model_names],
                                value=model_names[0], clearable=False)
        elements = [title, dropdown]
        elements.insert(0, html.Div([
            dbc.Row([
                dbc.Col([html.Button('Weights', id="weights", className="button_1")], width=2),
                dbc.Col([html.Button('Metrics', id="metrics", className="button_1")], width=2),
                dbc.Col([html.Button('Compatimetrics', id="compatimetrics", className="button_1")], width=2),
            ], justify="center"),
        ], style={"display": "block", "position": "sticky"}))
        elements.append(html.Div(id='compatimetrics_container', children=html.Div(id='compatimetrics_plots')))
        children = html.Div(elements)
    return children


# callback to update compatimetric plots
@callback(
    Output('compatimetrics_plots', 'children'),
    State('predictions', 'data'),
    Input('model_select', 'value'),
    State('task', 'data'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
)
def update_compatimetrics_plot(predictions, model_to_compare, task, df, column):
    children = []
    df = pd.DataFrame.from_dict(df)
    df = df.dropna()
    y = df.iloc[:, df.columns == column["name"]]
    y = y.squeeze()
    if model_to_compare:
        if task == 'classification':
            children = [
                html.H3("""
                        Matrices below show how similar two classifiers are by calculating percentage of observations
                        that two models predicted the same in case of uniformity, and differently in case of incompatibility
                       """,
                        className="annotation_str", id="ann_comp_6"),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                       className="plot")],
                            width=6),
                    html.H3("""
                    Matrix below on the right shows value of Average Collective Score which is a metric that 
                    sums number of doubly correct predictions and number of disagreements with coefficient 0.5 and
                    then dividing it by number of observations. It measures joined performance with consideration
                    of double correct prediction and disagreements.
                   """,
                            className="annotation_str", id="ann_comp_7"),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                ]),
                dbc.Row([dbc.Col([
                    html.H3("""Disagreement ratio presented on plot below on the left is measuring how many 
                        observations were predicted differently by two models regarding to the class of the record. 
                        It can show which class was more difficult to predict when joining models.""",
                        className="annotation_str", id="ann_comp_9"),], width=6),
                    dbc.Col([html.H3("""Conjunctive metrics are analogous to standard evaluation metrics,
                        but instead of comparing target variable with one prediction vector, we use two prediction vectors 
                        at the same time. Simply we mark prediction as correct, if two models predicted it correctly. 
                        Thus, conjunctive accuracy, presented on matrix above, precision and recall, showed together below, 
                        are good indicators of joined model performance as they measure the same ratios as original
                        metrics. Worth mentioning - conjunctive recall is generally lower and conjunctive precision 
                        is generally higher, which is related to their definition.""",
                        className="annotation_str", id="ann_comp_8"),
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.disagreement_ratio_plot(predictions, y, model_to_compare),
                        className="plot")],
                        width=6),
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_metrics_plot(predictions, y, model_to_compare),
                        className="plot")],
                        width=6),
                ]),
                html.H3("""
                       Plot below is showing ratio of predictions on different level of correctness. Doubly correct
                       prediction occurs when two models predicted observation right, disagreement when one of models
                       is missing, and doubly incorrect when two models labeled wrong class.
                       """,
                        className="annotation_str", id="ann_comp_10"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                        className='plot')
                     ]),
                html.H3("""
                        On the plot below one can observe the progess of incresing average collective score 
                        through the whole data set. This plot can be helpful when searching for areas of data set
                        where prediction was less effective. 
                       """,
                        className="annotation_str", id="ann_comp_11"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                        className='plot')
                     ]),
            ]
        elif task == 'regression':
            children = [dbc.Row([
                html.H3("""
                                Matrices below show the distance between two prediction vectros obtained from base models. 
                                MSE calculates mean of squared distance between vectors. RMSE is a root of MSE. The bigger the values,
                                the less similar are two models.
                                """,
                        className="annotation_str", id="ann_comp_1"),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_matrix(predictions), className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_matrix(predictions), className="plot")],
                        width=6),
                ]),
                html.H3("""
                    Matrices below show ratio of agreement and strong disagreement between two models. Agreement ratio 
                    calculates the percentage of observations that two models predicted closer than fiftieth part of 
                    standard deviation of target variable. On the other hand, disagreement ratio calculates the percantage
                    of observations witch have prediction difference bigger than standard deviation of target variable
                    """,
                        className="annotation_str", id="ann_comp_2"),

                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.ar_matrix(predictions, y), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.sdr_matrix(predictions, y), className="plot"), ],
                            width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_comparison(predictions, model_to_compare),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_comparison(predictions, model_to_compare),
                                       className="plot")],
                            width=6),
                ]),
                html.H3("""
                        Conjunctive RMSE is calculated based on mean of two prediction vectors. On this plot 
                        score of RMSE of prediction of chosen model is compared to predictions joined with other models
                        in ensemble. 
                        """,
                        className="annotation_str", id="ann_comp_3"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_rmse_plot(predictions, y, model_to_compare),
                        className='plot')
                    ]),
                html.H3("""
                        Plot below shows actual difference of predictions between chosen model 
                        and other models in ensemble through the whole data set.
                        """,
                        className="annotation_str", id="ann_comp_4"),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_distribution(predictions, model_to_compare),
                              className="plot")
                ]),
                html.H3("""
                            Plot below shows distribution of absolute prediction differences of chosen model and 
                            other models in ensemble. Pink dashed lines outline thresholds of agreement (lower line)
                            and strong disagreement (higher line), which help decide which models are closer
                            prediction-wise. 
                            """,
                        className="annotation_str", id="ann_comp_5"),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_boxplot(predictions, y, model_to_compare),
                              className="plot")
                ])
            ]
        else:
            children = [
                html.H3("""
                        Matrices below show how similar two classifiers are by calculating percentage of observations
                        that two models predicted the same in case of uniformity, and differently in case of incompatibility
                       """,
                        className="annotation_str", id="ann_comp_12"),
                dbc.Row([
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
                ]),
                html.H3("""
                        Conjunctive metrics are analogous to standard evaluation metrics, but instead of comparing target
                        variable with one prediction vector, we use two prediction vectors at the same time. Simply we
                        mark prediction as correct, if two models predicted it correctly. In case of multiclass 
                        classification we additionally distinguish weighted and macro versions of recall and precision.
                        Thus, conjunctive accuracy, presented on matrix above on the right, precision and recall, showed 
                        below, are good indicators of joined model performance as they measure the same ratios as original
                        metrics. Worth mentioning - conjunctive recall is generally lower and conjunctive precision 
                        is generally higher, which is related to their definition. 
                                   """,
                        className="annotation_str", id="ann_comp_14"),
                dbc.Row([
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_precision_multiclass_plot(predictions, y,
                                                                                          model_to_compare),
                        className="plot")],
                        width=6),
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_recall_multiclass_plot(predictions, y,
                                                                                       model_to_compare),
                        className="plot")],
                        width=6),
                ]),
                html.H3("""
                     Plot below is showing ratio of predictions on different level of correctness. Doubly correct
                     prediction occurs when two models predicted observation right, disagreement when one of models
                     is missing, and doubly incorrect when two models labeled wrong class.
                     """,
                        className="annotation_str", id="ann_comp_15"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                        className='plot')
                    ]),
                html.H3("""
                      On the plot below one can observe the progress of increasing average collective score 
                      through the whole data set. This plot can be helpful when searching for areas of data set
                      where prediction was less effective. 
                     """,
                        className="annotation_str", id="ann_comp_16"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                        className='plot')
                    ])
            ]
    return children


# callback to reset weights values to default
@callback(
    Output({"type": "weight_slider", "index": ALL}, 'value', allow_duplicate=True),
    Input('update-weights-button', 'n_clicks'),
    State('weights_list', 'data'),
    prevent_initial_call=True
)
def reset_sliders(n_clicks, values):
    if n_clicks > 0:
        return values
    else:
        raise PreventUpdate


@callback(
    Output({"type": "weight_slider", "index": ALL}, 'value', allow_duplicate=True),
    Output('weight-update-info', 'children'),
    Output('metrics-table', 'data'),
    Input('metrics-table', 'data'),
    prevent_initial_call=True
)
def update_slider_with_table_weights(table_data):
    is_updated = [isinstance(row['Weight'], str) for row in table_data]
    index = is_updated.index(True) if True in is_updated else None
    updated_table_data = table_data.copy()

    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    if index is not None:
        non_float_values = [row['Weight'] for row in table_data if not is_float(row['Weight'])]
        if not non_float_values:
            values = [float(row['Weight']) for row in table_data]
            updated_value = values[index]
            if updated_value > 1:
                info = html.Div([f"Please ensure values are within the range of [0, 1]. The provided value of "
                                 f"{updated_value} has been adjusted to 1."])
                updated_value = 1
            elif updated_value < 0:
                info = html.Div([f"Please ensure values are within the range of [0, 1]. The provided value of "
                                 f"{updated_value} has been adjusted to 0."])
                updated_value = 0
            else:
                info = None
        else:
            info = html.Div(
                [f"A numeric value is expected, but a different value was provided. Therefore, it has been set to 0."])
            updated_value = 0

        other_values = [float(row['Weight']) for idx, row in enumerate(table_data) if idx != index]
        sum_values = sum(value for i, value in enumerate(other_values))
        weights_adj = [round(((1 - updated_value) * value / sum_values), 2) for value in other_values]
        weights_adj.insert(index, updated_value)
        for i, row in enumerate(updated_table_data):
            if i < len(other_values) + 1:
                row['Weight'] = weights_adj[i]
    else:
        weights_adj = [float(row['Weight']) for row in table_data]
        info = None
        updated_table_data = table_data.copy()

    return weights_adj, info, updated_table_data


# callback to changing model weights
@callback(
    Output('metrics-table', 'data', allow_duplicate=True),
    Output('adj_weights-table', 'data'),
    Input({"type": "weight_slider", "index": ALL}, 'value'),
    Input('upload_model', 'contents'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('task', 'data'),
    State('predictions', 'data'),
    State('proba_predictions', 'data'),
    prevent_initial_call=True
)
def display_output(values, contents, df, column, task, predictions, proba_predictions):
    if contents:
        df = pd.DataFrame.from_dict(df).dropna()
        y = df.iloc[:, df.columns == column["name"]].squeeze()
        sum_slider_values = sum(values)
        weights_adj = [round((value / sum_slider_values), 2) for value in values]
        df = calculate_metrics(predictions, y, task, weights_adj)
        df_adj = calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights_adj)
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


# callbacks to display annotations
@callback(
    Output('ann_1', 'style'),
    Output('ann_0', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}, {}
    else:
        return {"display": "none"}, {"display": "none"}


@callback(
    Output('ann_2', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}
    else:
        return {"display": "none"}


@callback(
    Output('ann_comp_1', 'style'),
    Output('ann_comp_2', 'style'),
    Output('ann_comp_3', 'style'),
    Output('ann_comp_4', 'style'),
    Output('ann_comp_5', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 5 * [{}]
    else:
        return 5 * [{"display": "none"}]


@callback(
    Output('ann_comp_6', 'style'),
    Output('ann_comp_7', 'style'),
    Output('ann_comp_8', 'style'),
    Output('ann_comp_9', 'style'),
    Output('ann_comp_10', 'style'),
    Output('ann_comp_11', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 6 * [{}]
    else:
        return {"display": "none"}


@callback(
    Output('ann_comp_12', 'style'),
    Output('ann_comp_13', 'style'),
    Output('ann_comp_14', 'style'),
    Output('ann_comp_15', 'style'),
    Output('ann_comp_16', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 5 * [{}]
    else:
        return 5 * [{"display": "none"}]
