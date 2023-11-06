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


def get_ensemble_names_weights(ensemble_model):
    weights = []
    model_names = []
    for weight, model in ensemble_model.get_models_with_weights():
        model_names.append(str(type(model._final_estimator.choice)).split('.')[-1][:-2])
        weights.append(weight)
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


def weights_plot(ensemble_model, X, y, models_name, weights):
    fig = metrics.empty_fig()
    for i in range(len(models_name)):
        fig.add_trace(go.Bar(x=[i], y=[weights[i]], name=models_name[i]))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="Ensemble weights",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(models_name))),
            ticktext=models_name
        ),
        showlegend=False
    )
    return fig

def calculate_classification_metrics(ensemble_model, X, y):
    accuracy = accuracy_score(y, ensemble_model.predict(X))
    precision = precision_score(y, ensemble_model.predict(X), average='micro')
    recall = recall_score(y, ensemble_model.predict(X), average='micro')
    f1 = f1_score(y, ensemble_model.predict(X), average='micro')

    predictions =[]
    class_names = list(y.unique())
    class_names.sort()
    for weight, model in ensemble_model.get_models_with_weights():
        pass
        # TODO
        # prediction = model.predict(X)
        # print(prediction)
        # predictions.append([class_names[idx] for idx in prediction])
        # print(predictions)

    df_metrics = pd.DataFrame({
        '': ['Accuracy', 'Precision', 'Recall', 'F1 score'],
        'Original model': [accuracy, precision, recall, f1],
        'Adjusted model': [accuracy, precision, recall, f1]
    })

def calculate_regression_metrics(ensemble_model, X, y):
    accuracy = accuracy_score(y, ensemble_model.predict(X))
    precision = precision_score(y, ensemble_model.predict(X), average='micro')
    recall = recall_score(y, ensemble_model.predict(X), average='micro')
    f1 = f1_score(y, ensemble_model.predict(X), average='micro')

    predictions =[]
    class_names = list(y.unique())
    class_names.sort()
    for weight, model in ensemble_model.get_models_with_weights():
        pass
        # TODO
        # prediction = model.predict(X)
        # print(prediction)
        # predictions.append([class_names[idx] for idx in prediction])
        # print(predictions)

    df_metrics = pd.DataFrame({
        '': ['Accuracy', 'Precision', 'Recall', 'F1 score'],
        'Original model': [accuracy, precision, recall, f1],
        'Adjusted model': [accuracy, precision, recall, f1]
    })

def metrics_table(ensemble_model, X, y, weights):
    # weights = []
    mape = []
    mae = []
    mse = []
    for weight, model in ensemble_model.get_models_with_weights():
        # weights.append(weight)
        mape.append(float('%.*g' % (3, mean_absolute_percentage_error(y, model.predict(X)))))
        mae.append(float('%.*g' % (3, mean_absolute_error(y, model.predict(X)))))
        mse.append(float('%.*g' % (3, mean_squared_error(y, model.predict(X)))))

    data = {
        'weight': weights,
        'MAPE': mape,
        'MAE': mae,
        'MSE': mse
    }
    df = pd.DataFrame(data)
    return dash_table.DataTable(
                            data=df.to_dict('records'),  # Konwersja DataFrame na format zgodny z Dash
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

