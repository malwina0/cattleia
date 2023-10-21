import copy
import dash
from dash import html, Dash, dcc, Output, Input, callback, MATCH, State, ALL
import dash_bootstrap_components as dbc
import pickle
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

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
        dbc.Row(
            [
                dbc.Col(
                    html.Div(f'{model_name}', style={'textAlign': 'right', 'paddingRight': '0px', 'color': 'white'}),
                    width=3
                ),
                dbc.Col(
                    dcc.Slider(
                        id={"type": "part_add", "index": i},
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
                    width=10
                )
        ])
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


def weights_metrics_table():
    return html.Table([
            # Table Header
            html.Tr([
                html.Th('Column 1'),
                html.Th('Column 2'),
                html.Th('Column 3')
            ]),

            # Table Rows (5 empty rows)
            html.Tr([
                html.Td(''),
                html.Td(''),
                html.Td('')
            ]),
            html.Tr([
                html.Td(''),
                html.Td(''),
                html.Td('')
            ]),
            html.Tr([
                html.Td(''),
                html.Td(''),
                html.Td('')
            ]),
            html.Tr([
                html.Td(''),
                html.Td(''),
                html.Td('')
            ]),
            html.Tr([
                html.Td(''),
                html.Td(''),
                html.Td('')
            ])
        ])

