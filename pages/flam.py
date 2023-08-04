import dash
from dash import html, dcc, callback, Input, Output, State
import base64
import io
import dash_bootstrap_components as dbc
import sys
import metrics
import pandas as pd
sys.path.append("..")


dash.register_page(__name__)

# wygląd strony
layout = html.Div([
    dcc.Store(id='csv_data', data=[], storage_type='memory'),
    dcc.Store(id='y_label_column', data=[], storage_type='memory'),
    dbc.Row([
        dbc.Col([
            dbc.Container([
                html.H5("Select data"),
                dcc.Upload(
                    id='upload_csv_data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=True
                ),
                html.Div(id='select_y_label_column'),
                html.Div(id='upload_model_section'),
            ], className="px-3 sidepanel")
        ], width=2),
        dbc.Col([
            dcc.Loading(id="loading-1",type="default",children=html.Div(id="plots"),className="spinxD")
            #dbc.Spinner(html.Div(id='plots'), spinner_class_name="spinxD"),
        ], width=10)
    ])
])


# funkcja odpowiedzialna za prawidlowe wczytanie danych
def parse_data_model_flaml(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "pkl" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_pickle(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


# czesc odpowiedzialna za dodanie pliku csv
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
        df = parse_data_model_flaml(contents, filename)
        data = df.to_dict()

        children = html.Div([
            html.H5(filename),
            html.P("Inset X axis data"),
            dcc.Dropdown(id='column_select',
                         options=[{'label': x, 'value': x} for x in df.columns]),
            html.Hr(),
        ])

    return data, children


# Wybranie kolumny
@callback(
    [Output('y_label_column', 'data'),
        Output('upload_model_section', 'children')],
    Input('column_select', 'value')
)
def select_kolumns(value):
    children = html.Div([
        html.H5(value),
        html.H5("Select model"),
        dcc.Upload(
            id='upload_model',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=True
        ),
        html.Hr(),
    ])
    data = {'name': value}

    return data, children


# wybranie modelu a nastepnie narysowanie wykresow
@callback(
    Output('plots', 'children'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
)
def update_model(contents, filename, df, column):
    children = []
    print(column)
    print(df)
    if contents:
        contents = contents[0]
        filename = filename[0]
        model = parse_data_model_flaml(contents, filename)
        print(type(model))

        df = pd.DataFrame.from_dict(df)
        df = df.dropna()

        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        print(X)
        print(y)

        plot_component = [
            dbc.Row([
                dbc.Col([dcc.Graph(figure=metrics.mse_plot(model, X, y), className="plot")], width=6),
                dbc.Col([dcc.Graph(figure=metrics.mape_plot(model, X, y), className="plot")], width=6),
            ]),

            #dcc.Graph(figure=metrics.mse_plot(model, X, y), className="plot"),
            #dcc.Graph(figure=metrics.mape_plot(model, X, y)),
            dcc.Graph(figure=metrics.mae_plot(model, X, y), className="plot"),
            dcc.Graph(figure=metrics.correlation_plot(model, X, y, task="regression"), className="plot"),
            dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, "Flaml", "regression"), className="plot"),
        ]
        for plot in metrics.permutation_feature_importance_all(model, X, y, task="regression"):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        for plot in metrics.partial_dependence_plots(model, X):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        children = html.Div(plot_component)

    return children
