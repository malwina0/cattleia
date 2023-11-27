from dash import html, Dash, dcc, Output, Input, callback, MATCH, State, ALL, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error


def get_ensemble_names_weights(ensemble_model, library):
    weights, model_names = ([] for _ in range(2))

    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            model_names.append(str(type(model._final_estimator.choice)).split('.')[-1][:-2])
            weights.append(weight)
    elif library == "AutoGluon":
        model_names = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']
        # TODO: test if needed and handle cases when there is more layers (not 'S1F1')
        weights = list(ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['children_info']['S1F1'][
                           'model_weights'].values())

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
        ], style={'height': '30px'})
    ], style={'display': 'inline'})


def calculate_metrics(ensemble_model, X, y, task, library, weights):
    if task == "regression":
        mape, mae, mse = ([] for _ in range(3))
        if library == "AutoSklearn":
            for weight, model in ensemble_model.get_models_with_weights():
                mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, model.predict(X)))))
                mae.append(float('%.*g' % (3, mean_absolute_error(y, model.predict(X)))))
                mse.append(round(mean_squared_error(y, model.predict(X))))
        elif library == "AutoGluon":
            final_model = ensemble_model.get_model_best()
            for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
                'base_model_names']:
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
    else:
        accuracy, precision, recall, f1 = ([] for _ in range(4))
        if library == "AutoSklearn":
            class_names = list(y.unique())
            class_names.sort()
            for weight, model in ensemble_model.get_models_with_weights():
                prediction = model.predict(X)
                prediction_class = [class_names[idx] for idx in prediction]
                accuracy.append(round(accuracy_score(y, prediction_class), 2))
                precision.append(round(precision_score(y, prediction_class, average='micro'), 2))
                recall.append(round(recall_score(y, prediction_class, average='micro'), 2))
                f1.append(round(f1_score(y, prediction_class, average='micro'), 2))
        elif library == "AutoGluon":
            final_model = ensemble_model.get_model_best()
            for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
                'base_model_names']:
                ensemble_model.set_model_best(model_name)
                accuracy.append(round(accuracy_score(y, ensemble_model.predict(X)), 2))
                precision.append(round(precision_score(y, ensemble_model.predict(X), average='micro'), 2))
                recall.append(round(recall_score(y, ensemble_model.predict(X), average='micro'), 2))
                f1.append(round(f1_score(y, ensemble_model.predict(X), average='micro'), 2))
            ensemble_model.set_model_best(final_model)
        data = {
            'weight': weights,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1 score': f1
        }

    return pd.DataFrame(data)


def tbl_metrics(ensemble_model, X, y, task, library, weights):
    df = calculate_metrics(ensemble_model, X, y, task, library, weights)
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


def calculate_metrics_adj_ensemble(ensemble_model, X, y, task, library, weights):
    if task == "regression":
        mse = round(mean_squared_error(y, ensemble_model.predict(X)))
        mae = float('%.*g' % (3, mean_absolute_error(y, ensemble_model.predict(X))))
        mape = float('%.*g' % (3, mean_absolute_percentage_error(y, ensemble_model.predict(X))))
        predictions = []

        if library == "AutoSklearn":
            for weight, model in ensemble_model.get_models_with_weights():
                predictions.append(model.predict(X).tolist())
        elif library == "AutoGluon":
            final_model = ensemble_model.get_model_best()
            for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
                'base_model_names']:
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
    else:
        accuracy = round(accuracy_score(y, ensemble_model.predict(X)), 2)
        precision = round(precision_score(y, ensemble_model.predict(X), average='micro'), 2)
        recall = round(recall_score(y, ensemble_model.predict(X), average='micro'), 2)
        f1 = round(f1_score(y, ensemble_model.predict(X), average='micro'), 2)
        proba_predictions = []
        if library == "AutoSklearn":
            for weight, model in ensemble_model.get_models_with_weights():
                prediction = model.predict_proba(X)
                proba_predictions.append(prediction.tolist())
        elif library == "AutoGluon":
            final_model = ensemble_model.get_model_best()
            for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
                'base_model_names']:
                ensemble_model.set_model_best(model_name)
                proba_predictions.append(ensemble_model.predict_proba(X).values.tolist())
            ensemble_model.set_model_best(final_model)

        y_proba_adj = [
            [sum(w * x for x, w in zip(elements, weights)) for elements in zip(*rows)]
            for rows in zip(*proba_predictions)
        ]
        y_adj = np.argmax(np.array(y_proba_adj), axis=1)
        class_names = list(y.unique())
        class_names.sort()
        y_class_adj = [class_names[idx] for idx in y_adj]

        accuracy_adj = round(accuracy_score(y, y_class_adj), 2)
        precision_adj = round(precision_score(y, y_class_adj, average='micro'), 2)
        recall_adj = round(recall_score(y, y_class_adj, average='micro'), 2)
        f1_adj = round(f1_score(y, y_class_adj, average='micro'), 2)

        df_metrics = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1 score'],
            'Original model': [accuracy, precision, recall, f1],
            'Adjusted model': [accuracy_adj, precision_adj, recall_adj, f1_adj]
        })
    return df_metrics


# def set_cell_colors(row):
#     if row['Original model'] > row['Adjusted model']:
#         return {'backgroundColor': 'green', 'color': 'black'}
#     elif row['Original model'] < row['Adjusted model']:
#         return {'backgroundColor': 'yellow', 'color': 'black'}
#     else:
#         return {'backgroundColor': '', 'color': 'black'}


def tbl_metrics_adj_ensemble(ensemble_model, X, y, task, library, weights):
    df = calculate_metrics_adj_ensemble(ensemble_model, X, y, task, library, weights)
    style_data_conditional = []
    if task == 'regression':
        style_data_conditional.extend([
            {
                'if': {
                    'filter_query': '{Original model} > {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'color': 'black',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'backgroundColor': 'green',
            },
            {
                'if': {
                    'filter_query': '{Original model} < {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'color': 'black',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'backgroundColor': 'yellow',
            }
        ])
    else:
        style_data_conditional.extend([
            {
                'if': {
                    'filter_query': '{Original model} > {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'color': 'black',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'backgroundColor': 'blue',
            },
            {
                'if': {
                    'filter_query': '{Original model} < {Adjusted model}',
                    'column_id': 'Adjusted model'
                },
                'color': 'black',
                'fontWeight': 'bold',
                'textDecoration': 'underline',
                'backgroundColor': 'red',
            }
        ])
    return dash_table.DataTable(
        data=df.to_dict('records'),  # Konwersja DataFrame do formatu obsługiwanego przez dash
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
        style_data_conditional=style_data_conditional,
        id='adj_weights-table'
    )
