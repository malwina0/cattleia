from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
from components import metrics
import shutil
import pandas as pd
from utils.utils import get_predictions_from_model, get_task_from_model, parse_model, get_ensemble_weights,\
    get_probability_pred_from_model
from components.weights import slider_section, tbl_metrics, tbl_metrics_adj_ensemble
from components.navigation import navigation_row

# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Output('metrics_plots', 'data'),
    Output('weight_plots', 'data'),
    Output('xai_plots', 'data'),
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
    model_names, weights, task, predictions, proba_predictions, weights_plots, xai_plots = ([] for _ in range(7))
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
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=metrics.mae_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=metrics.correlation_plot(predictions, task=task, y=y),
                                  className="plot")], width=6),
                    ])
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
                    dcc.Graph(figure=metrics.correlation_plot(predictions, task=task, y=y),
                              className="plot"),
                ]

            metrics_plots += [
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
            else:
                weights_plots.append(
                    dbc.Row([
                        html.Div(["""FLAML library does not incorporate the use of weights in its ensemble creation 
                        process."""],
                        className='page-text'
                        )
                    ],
                    className='plot'
                    )
                )

            for plot in metrics.permutation_feature_importance_all(model, X, y, library=library, task=task):
                xai_plots.append(dcc.Graph(figure=plot, className="plot"))

            xai_plots.append(
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
                    xai_plots.append(dcc.Graph(figure=plot, className="plot"))
            else:
                for plot in metrics.partial_dependence_plots(model, X.sample(2000), library=library,
                                                             autogluon_task=task):
                    xai_plots.append(dcc.Graph(figure=plot, className="plot"))

            # It may be necessary to keep the model for the code with weights,
            # for now we remove the model after making charts
            try:
                shutil.rmtree('./uploaded_model')
            except FileNotFoundError:
                pass

            metrics_plots.insert(0, navigation_row)
            metrics_plots.insert(1, html.Div([], className="navigation_placeholder"))

            weights_plots.insert(0, navigation_row)
            weights_plots.insert(1, html.Div([], className="navigation_placeholder"))
            weights_plots = html.Div(weights_plots)

            children = html.Div(metrics_plots, style={"position":"relative", "overflow": "auto"})
        else:
            children = html.Div(["Please provide the file in .pkl or .zip format."], style={"color": "white"})

    return children, children, weights_plots, xai_plots, model_names, predictions, task, proba_predictions, weights