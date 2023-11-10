import copy
import dash
from dash import html, Dash, dcc, Output, Input, callback, MATCH, State, ALL, dash_table
import dash_bootstrap_components as dbc
import pickle
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error
import metrics
import numpy as np


def get_ensemble_names_weights(ensemble_model, library):
    weights = []
    model_names = []

    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            model_names.append(str(type(model._final_estimator.choice)).split('.')[-1][:-2])
            weights.append(weight)
    elif library == "AutoGluon":
        model_names = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info']['base_model_names']
        # TODO
        # test if needed and handle cases when there is more layers (not 'S1F1')
        weights = list(ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['children_info']['S1F1']['model_weights'].values())

    return model_names, weights


def slider_section(model_name, weight, i):
    return html.Div([
        dbc.Row([
                dbc.Col(
                    html.Div(f'{model_name}', style={'textAlign': 'right', 'paddingRight': '0px', 'color': 'white'}),
                    width=3
                ),
                dbc.Col(
                    dcc.Slider(
                        id={"type": "weight_slider", "index": i},
                        min=0,
                        max=1,
                        step=0.01,
                        value=weight,
                        tooltip={"placement": "right", "always_visible": False},
                        updatemode='drag',
                        persistence=True,
                        persistence_type='session',
                        marks=None,
                        className='weight-slider',
                    ),
                    width=9
                )
        ], style={'height': '31px'})
    ], style={'display': 'inline'})

def calculate_metrics_regression(ensemble_model, X, y, library, weights):
    mape = []
    mae = []
    mse = []
    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, model.predict(X)))))
            mae.append(float('%.*g' % (3, mean_absolute_error(y, model.predict(X)))))
            mse.append(round(mean_squared_error(y, model.predict(X))))
    elif library == "AutoGluon":
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info']['base_model_names']:
            ensemble_model.set_model_best(model_name)
            mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, ensemble_model.predict(X)))))
            mae.append(float('%.*g' % (3, mean_absolute_error(y, ensemble_model.predict(X)))))
            mse.append(round(mean_squared_error(y, ensemble_model.predict(X))))
        ensemble_model.set_model_best(final_model)

    data = {
        'weight': weights,
        'MAPE': mape,
        'MAE': mae,
        'MSE': mse
    }
    return pd.DataFrame(data)

def tbl_metrics_regression(ensemble_model, X, y, library, weights):
    df = calculate_metrics_regression(ensemble_model, X, y, library, weights)
    return dash_table.DataTable(
                            data=df.to_dict('records'),  # convert DataFrame to format compatible with dash
                            columns=[
                                {'name': col, 'id': col} for col in df.columns
                            ],
                            style_table={'backgroundColor': '#2c2f38', 'border': '2px solid #2c2f38'},
                            style_cell={
                                'textAlign': 'center',
                                'color': 'white',
                                'border': '2px solid #2c2f38',
                                'backgroundColor': '#1e1e1e',
                                'height': '30px'
                            },
                            id='metrics-table'
                        )


def calculate_metrics_regression_adj_ensemble(ensemble_model, X, y, library, weights):
    mse = round(mean_squared_error(y, ensemble_model.predict(X)))
    mae = float('%.*g' % (3, mean_absolute_error(y, ensemble_model.predict(X))))
    mape = float('%.*g' % (3, mean_absolute_percentage_error(y, ensemble_model.predict(X))))

    predictions = []

    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            predictions.append(model.predict(X).tolist())
    elif library == "AutoGluon":
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info']['base_model_names']:
            ensemble_model.set_model_best(model_name)
            predictions.append(ensemble_model.predict(X).tolist())
        ensemble_model.set_model_best(final_model)

    y_adj = np.sum(np.array(predictions).T * weights, axis=1)
    mse_adj = round(mean_squared_error(y, y_adj))
    mae_adj = float('%.*g' % (3, mean_absolute_error(y, y_adj)))
    mape_adj = float('%.*g' % (3, mean_absolute_percentage_error(y, y_adj)))
    df_metrics = pd.DataFrame({
        'metric': ['MSE', 'MAE', 'MAPE'],
        'Original model': [mse, mae, mape],
        'Adjusted model': [mse_adj, mae_adj, mape_adj]
    })
    return df_metrics


def tbl_metrics_regression_adj_ensemble(ensemble_model, X, y, library, weights):
    df = calculate_metrics_regression_adj_ensemble(ensemble_model, X, y, library, weights)
    return dash_table.DataTable(
                            data=df.to_dict('records'), # convert DataFrame to format compatible with dash
                            columns=[
                                {'name': col, 'id': col} for col in df.columns
                            ],
                            style_table={'backgroundColor': '#2c2f38', 'border': '2px solid #2c2f38'},
                            style_cell={
                                'textAlign': 'center',
                                'color': 'white',
                                'border': '2px solid #2c2f38',
                                'backgroundColor': '#1e1e1e',
                                'height': '30px'
                            },
                            id='adj_weights-table'
                        )


# def calculate_classification_metrics(ensemble_model, X, y):
#     accuracy = accuracy_score(y, ensemble_model.predict(X))
#     precision = precision_score(y, ensemble_model.predict(X), average='micro')
#     recall = recall_score(y, ensemble_model.predict(X), average='micro')
#     f1 = f1_score(y, ensemble_model.predict(X), average='micro')
#
#     predictions =[]
#     class_names = list(y.unique())
#     class_names.sort()
#     for weight, model in ensemble_model.get_models_with_weights():
#         pass
#         # TODO
#         # prediction = model.predict(X)
#         # print(prediction)
#         # predictions.append([class_names[idx] for idx in prediction])
#         # print(predictions)
#
#     df_metrics = pd.DataFrame({
#         '': ['Accuracy', 'Precision', 'Recall', 'F1 score'],
#         'Original model': [accuracy, precision, recall, f1],
#         'Adjusted model': [accuracy, precision, recall, f1]
#     })

