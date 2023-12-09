from dash import html, Dash, dcc, Output, Input, callback, MATCH, State, ALL, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error

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
        ], style={'height': '30px'})
    ], style={'display': 'inline'})


def calculate_metrics(predictions, y, task, weights):
    if task == "regression":
        mape, mae, mse = ([] for _ in range(3))
        for model_name, prediction in predictions.items():
            if model_name != 'Ensemble':
                mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, prediction))))
                mae.append(float('%.*g' % (3, mean_absolute_error(y, prediction))))
                mse.append(round(mean_squared_error(y, prediction)))
        data = {
            'weight': weights,
            'MAPE': mape,
            'MAE': mae,
            'MSE': mse
        }
    else:
        accuracy, precision, recall, f1 = ([] for _ in range(4))
        for model_name, prediction in predictions.items():
            if model_name != 'Ensemble':
                accuracy.append(round(accuracy_score(y, prediction), 2))
                precision.append(round(precision_score(y, prediction, average='macro'), 2))
                recall.append(round(recall_score(y, prediction, average='macro'), 2))
                f1.append(round(f1_score(y, prediction, average='macro'), 2))
        data = {
            'weight': weights,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1 score': f1
        }

    return pd.DataFrame(data)


def tbl_metrics(predictions, y, task, weights):
    df = calculate_metrics(predictions, y, task, weights)
    return dash_table.DataTable(
        data=df.to_dict('records'),  # convert DataFrame to format compatible with dash
        columns=[
            {'name': col, 'id': col} for col in df.columns
        ],
        style_table={'backgroundColor': '#2c2f38', 'border': '1px solid #2c2f38'},
        style_cell={
            'textAlign': 'center',
            'color': 'white',
            'border': '1px solid #2c2f38',
            'backgroundColor': '#1e1e1e',
            'height': '30px'
        },
        id='metrics-table'
    )


def calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights):
    if task == "regression":
        mse = round(mean_squared_error(y, predictions['Ensemble']))
        mae = float('%.*g' % (3, mean_absolute_error(y, predictions['Ensemble'])))
        mape = float('%.*g' % (3, mean_absolute_percentage_error(y, predictions['Ensemble'])))
        predictions = list(dict((name, predictions[name]) for name in predictions if name != 'Ensemble').values())
        y_adj = np.sum(np.array(predictions).T * weights, axis=1)
        mse_adj = round(mean_squared_error(y, y_adj))
        mae_adj = float('%.*g' % (3, mean_absolute_error(y, y_adj)))
        mape_adj = float('%.*g' % (3, mean_absolute_percentage_error(y, y_adj)))
        df_metrics = pd.DataFrame({
            'metric': ['MSE', 'MAE', 'MAPE'],
            'Original model': [mse, mae, mape],
            'Adjusted model': [mse_adj, mae_adj, mape_adj]
        })
    else:
        accuracy = round(accuracy_score(y, predictions['Ensemble']), 2)
        precision = round(precision_score(y, predictions['Ensemble'], average='macro'), 2)
        recall = round(recall_score(y, predictions['Ensemble'], average='macro'), 2)
        f1 = round(f1_score(y, predictions['Ensemble'], average='macro'), 2)

        y_proba_adj = [
            [sum(w * x for x, w in zip(elements, weights)) for elements in zip(*rows)]
            for rows in zip(*proba_predictions)
        ]
        y_adj = np.argmax(np.array(y_proba_adj), axis=1)
        class_names = list(y.unique())
        class_names.sort()
        y_class_adj = [class_names[idx] for idx in y_adj]

        accuracy_adj = round(accuracy_score(y, y_class_adj), 2)
        precision_adj = round(precision_score(y, y_class_adj, average='macro'), 2)
        recall_adj = round(recall_score(y, y_class_adj, average='macro'), 2)
        f1_adj = round(f1_score(y, y_class_adj, average='macro'), 2)

        df_metrics = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1 score'],
            'Original model': [accuracy, precision, recall, f1],
            'Adjusted model': [accuracy_adj, precision_adj, recall_adj, f1_adj]
        })
    return df_metrics


def tbl_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights):
    df = calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights)
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
        id='adj_weights-table'
    )