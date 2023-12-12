from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error, r2_score

def slider_section(model_name, weight, i):
    """Creates an HTML Div containing a slider and model name.

    Parameters
    ----------
    model_name : str
        The name of the model to display.
    weight : float
        The initial value for the slider representing the weight of the model.
    i : int
        Index used to identify the slider.

    Returns
    -------
    dash_html_components.Div
        An HTML Div element containing a row with two columns:
        - The first column displays the model name.
        - The second column contains a Dash Slider component for adjusting the weight.
    """
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
                    persistence_type='memory',
                    marks=None,
                    className='weight-slider',
                ),
                width=9
            )
        ], style={'height': '30.3px'})
    ], style={'display': 'inline'})


def calculate_metrics(predictions, y, task, weights):
    """Calculate evaluation metrics for individual models and their weighted ensemble.

    Parameters
    ----------
    predictions : dict
        Dictionary containing predictions where keys represent model names and values are predictions.
    y : array-like of shape (n_samples,)
        True target values.
    task : {'regression', 'classification'}
        Task type, 'regression' for regression problems and 'classification' for classification problems.
    weights : array-like of shape (n_estimators,)
        Weights applied to individual predictions in the ensemble.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing evaluation metrics for each individual model and the weighted ensemble.

    Notes
    -----
    For regression tasks, the following metrics are calculated for individual models:
    - MAPE (Mean Absolute Percentage Error)
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - R squared (Coefficient of Determination)

    For classification tasks, the following metrics are calculated for individual models:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    The DataFrame includes columns for weights assigned to models and respective metrics.
    """
    if task == "regression":
        mape, mae, mse, rmse, r_squared = ([] for _ in range(5))
        for model_name, prediction in predictions.items():
            if model_name != 'Ensemble':
                mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, prediction))))
                mae.append(float('%.*g' % (3, mean_absolute_error(y, prediction))))
                mse.append(round(mean_squared_error(y, prediction)))
                rmse.append(round(mean_squared_error(y, prediction, squared=False)))
                r_squared.append(float('%.*g' % (3, r2_score(y, prediction))))
        data = {
            'Weight': weights,
            'MAPE': mape,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R squared': r_squared
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
            'Weight': weights,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1
        }

    return pd.DataFrame(data)


def tbl_metrics(predictions, y, task, weights):
    """Create a Dash DataTable displaying evaluation metrics for models such as MAPE, MAE, MSE, RMSE and R squared for
    regression tasks, or accuracy, precision, recall, and F1 score for classification tasks. It uses the
    `calculate_metrics` function to compute these metrics based on provided predictions, true labels, and weights.

    Parameters
    ----------
    predictions : dict
        A dictionary containing model names as keys and their corresponding predictions as values.
    y : array-like
        The true labels or values.
    task : str
        The type of task, either "regression" or "classification".
    weights : array-like
        The weights associated with the models.

    Returns
    -------
    dash_table.DataTable
        A Dash DataTable containing evaluation metrics for each model.
    """
    df = calculate_metrics(predictions, y, task, weights)
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[
            {'name': col, 'id': col, 'editable': True if col == 'Weight' else False} for col in df.columns
        ],
        style_table={
            'backgroundColor': '#3a3e4a',
            'border': '3px solid #1e1e1e',
        },
        style_cell={
            'textAlign': 'center',
            'color': 'white',
            'border': '1.5px solid #1e1e1e',
            'backgroundColor': '#3a3e4a',
            'height': '30px'
        },
        style_header={
            'fontWeight': 'bold',
            'color': '#ffc4f7'
        },
        id='metrics-table'
    )


def calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights):
    """Calculate adjusted evaluation metrics for an ensemble model. For regression tasks,
    it computes MSE, MAE, and MAPE for both the original and adjusted ensemble predictions based on
    provided predictions, true labels, and weights. For classification tasks, it computes accuracy,
    precision, recall, and F1 score for both the original and adjusted ensemble predictions.

    Parameters
    ----------
    predictions : dict
        A dictionary containing model names as keys and their corresponding predictions as values.

    proba_predictions : list
        A list containing probability predictions.

    y : array-like
        The true labels or values.

    task : str
        The type of task, either "regression" or "classification".

    weights : array-like
        The weights associated with the models.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing calculated evaluation metrics for the ensemble model, both original and adjusted.
    """
    if task == "regression":
        mse = round(mean_squared_error(y, predictions['Ensemble']))
        mae = float('%.*g' % (3, mean_absolute_error(y, predictions['Ensemble'])))
        mape = float('%.*g' % (3, mean_absolute_percentage_error(y, predictions['Ensemble'])))
        rmse = round(mean_squared_error(y, predictions['Ensemble'], squared=False))
        r_squared = float('%.*g' % (3, r2_score(y, predictions['Ensemble'])))

        predictions = list(dict((name, predictions[name]) for name in predictions if name != 'Ensemble').values())
        y_adj = np.sum(np.array(predictions).T * weights, axis=1)

        mse_adj = round(mean_squared_error(y, y_adj))
        mae_adj = float('%.*g' % (3, mean_absolute_error(y, y_adj)))
        mape_adj = float('%.*g' % (3, mean_absolute_percentage_error(y, y_adj)))
        rmse_adj = round(mean_squared_error(y, y_adj, squared=False))
        r_squared_adj = float('%.*g' % (3, r2_score(y, y_adj)))

        df_metrics = pd.DataFrame({
            'Metric': ['MSE', 'MAE', 'MAPE', 'RMSE', 'R squared'],
            'Original model': [mse, mae, mape, rmse, r_squared],
            'Adjusted model': [mse_adj, mae_adj, mape_adj, rmse_adj, r_squared_adj]
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
    """
    Create a Dash DataTable displaying adjusted evaluation metrics for an ensemble model. It uses
    the `calculate_metrics_adj_ensemble` function to compute these metrics based on provided predictions,
    probability predictions, true labels, and weights.

    Parameters
    ----------
    predictions : dict
        A dictionary containing model names as keys and their corresponding predictions as values.
    proba_predictions : list
        A list containing probability predictions.
    y : array-like
        The true labels or values.
    task : str
        The type of task, either "regression" or "classification".
    weights : array-like
        The weights associated with the models.

    Returns
    -------
    dash_table.DataTable
        A Dash DataTable containing adjusted evaluation metrics for the ensemble model. The table contains conditional
        styling highlighting differences between original and adjusted models.
    """
    df = calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights)
    style_data_conditional = []
    if task == 'regression':
        style_data_conditional.extend([
            {
                'if': {
                    'filter_query': '{Original model} < {Adjusted model}',
                    'column_id': 'Adjusted model',
                    'row_index': [0, 1, 2, 3]
                },
                'backgroundColor': '#662f2f',
            },
            {
                'if': {
                    'filter_query': '{Original model} > {Adjusted model}',
                    'column_id': 'Adjusted model',
                    'row_index': [0, 1, 2, 3]
                },
                'backgroundColor': '#2b5c35',
            },
            {
                'if': {
                    'filter_query': '{Original model} < {Adjusted model}',
                    'column_id': 'Adjusted model',
                    'row_index': 4
                },
                'backgroundColor': '#2b5c35',
            },
            {
                'if': {
                    'filter_query': '{Original model} > {Adjusted model}',
                    'column_id': 'Adjusted model',
                    'row_index': 4
                },
                'backgroundColor': '#662f2f',
            }
        ])
    else:
        style_data_conditional.extend([
            {
                'if': {
                    'filter_query': '{Original model} > {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'backgroundColor': '#662f2f',
            },
            {
                'if': {
                    'filter_query': '{Original model} < {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'backgroundColor': '#2b5c35',
            }
        ])
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[
            {'name': col, 'id': col} for col in df.columns
        ],
        style_table={
            'backgroundColor': '#3a3e4a',
            'border': '3px solid #1e1e1e',
            'marginTop': '10px'
        },
        style_cell={
            'textAlign': 'center',
            'color': 'white',
            'border': '1.5px solid #1e1e1e',
            'backgroundColor': '#3a3e4a',
            'height': '30px'
        },
        style_header={
            'fontWeight': 'bold',
            'color': '#ffc4f7'
        },
        style_data_conditional=style_data_conditional,
        id='adj_weights-table'
    )