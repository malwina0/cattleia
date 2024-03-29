from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import shutil
import pandas as pd

from components.annotations import ann_metrics_prediction_compare, ann_weights_sliders, ann_weights_metrics, \
    ann_weights_ensemble, ann_xai_feature_importance, ann_xai_partial_dep
from components.metrics import mse_plot, mape_plot, rmse_plot, r_2_plot, mae_plot, correlation_plot, accuracy_plot, \
    precision_plot, recall_plot, f1_score_plot, prediction_compare_plot
from components.navigation import navigation_row
from components.weights import slider_section, tbl_metrics, tbl_metrics_adj_ensemble
from components.xai import prepare_feature_importance, feature_importance_plot, partial_dependence_plots
from utils.utils import get_predictions_from_model, get_task_from_model, parse_model, get_ensemble_weights, \
    get_probability_pred_from_model

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
    Output('pd_plots_dict', 'data'),
    Output('selected_model_name', 'children'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('plots', 'children'),
)
def update_model(contents, filename, df, column, about_us):
    model_names, weights, task, predictions, proba_predictions, weights_plots, xai_plots = ([] for _ in range(7))
    children = about_us
    pd_plots_dict = {}
    filename_info = html.Div()
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

            # metrics component
            if task == "regression":
                metrics_plots = [
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=mse_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=mape_plot(predictions, y), className="plot")],
                                width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=rmse_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=r_2_plot(predictions, y), className="plot")],
                                width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=mae_plot(predictions, y), className="plot")],
                                width=6),
                        dbc.Col([dcc.Graph(figure=correlation_plot(predictions, task=task, y=y),
                                  className="plot")], width=6),
                    ])
                ]
            else:
                proba_predictions = get_probability_pred_from_model(model, X, library)
                metrics_plots = [
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=accuracy_plot(predictions, y),
                                           className="plot")], width=6),
                        dbc.Col([dcc.Graph(figure=precision_plot(predictions, y),
                                           className="plot")], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            [dcc.Graph(figure=recall_plot(predictions, y), className="plot")],
                            width=6),
                        dbc.Col(
                            [dcc.Graph(figure=f1_score_plot(predictions, y), className="plot")],
                            width=6),
                    ]),
                    dcc.Graph(figure=correlation_plot(predictions, task=task, y=y),
                              className="plot"),
                ]

            metrics_plots += [dbc.Row([
                html.H3(['Prediction compare matrix'], className='annotation-title'),
                ann_metrics_prediction_compare,
                dcc.Graph(figure=prediction_compare_plot(predictions, y, task=task))
                ], className="custom-caption")
            ]


            metrics_plots.insert(0, navigation_row)
            metrics_plots.insert(1, html.Div([], className="navigation_placeholder"))

            # weights component
            weights_plots = []
            if library != "FLAML":
                weights_plots.append(
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([ann_weights_sliders], width=7),
                            dbc.Col([ann_weights_metrics], width=4)
                        ]),
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
                            dbc.Col([ann_weights_ensemble], width=4),
                            dbc.Col([tbl_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights)],
                                    width=4)
                        ], justify="left"),
                        dbc.Row([dbc.Col(width=4), dbc.Col(id="weights-color-info", width=4)], justify='left')
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

            # XAI component
            fi_df = prepare_feature_importance(model, X, y, library=library, task=task)
            fi_plot = feature_importance_plot(fi_df)
            xai_plots.append(
                dbc.Row([
                    html.H3(['Feature Importance across models'], className='annotation-title'),
                    ann_xai_feature_importance,
                    dcc.Graph(figure=fi_plot, className="plot")
                ], className = "custom-caption")
            )

            if len(X) < 2000:
                pd_plots_dict = partial_dependence_plots(model, X, library=library, task=task)
            else:
                pd_plots_dict = partial_dependence_plots(model, X.sample(2000), library=library, task=task)
            pd_variables = list(pd_plots_dict.keys())

            if len(pd_variables) > 0:
                dropdown = dcc.Dropdown(id='variable_select_dropdown', className="dropdown-class",
                                        options=[{'label': x, 'value': x} for x in pd_variables],
                                        value=pd_variables[0], clearable=False)
                selected_variable_plot = pd_plots_dict.get(pd_variables[0])
                xai_plots.append(
                    dbc.Row([
                        html.H3(['Partial Dependence across models'], className='annotation-title'),
                        html.H5("Choose a column to display its partial dependence plot:", className='annotation-title'),
                        dropdown,
                        ann_xai_partial_dep,
                        dcc.Graph(
                            figure=selected_variable_plot,
                            className="plot",
                            id='partial_dependence_plot'
                        )
                    ], className="custom-caption")
                )

            else:
                xai_plots.append(
                    dbc.Row([
                        html.Div(["""It was not possible to make a partial dependence chart for any variable."""],
                                 className='page-text'
                                 )
                    ],className='plot')
                )


            # It may be necessary to keep the model for the code with weights,
            # for now we remove the model after making charts
            try:
                shutil.rmtree('./uploaded_model')
            except FileNotFoundError:
                pass

            children = html.Div(metrics_plots, style={"position":"relative", "overflow": "auto"})
        else:
            children = html.Div(["Please provide the file in .pkl or .zip format."], style={"color": "white"})

        filename_info = html.Div([f"Loaded file: {filename},", html.Br(),
                                  f"library: {library}."], style={"color": "white"})

    return children, children, weights_plots, xai_plots, model_names, predictions, task, proba_predictions, weights, \
        pd_plots_dict, filename_info
