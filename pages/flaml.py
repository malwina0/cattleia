import dash
from dash import html, dcc, callback, Input, Output, State
import base64
import io
import sys
sys.path.append("..")
import metrics
import pandas as pd


dash.register_page(__name__)


layout = html.Div([
    html.H5("Select data"),
    dcc.Upload(
        id='upload-data',
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
    html.Div(id='output-data-upload1'),

    dcc.Store(id='store-data1', data=[], storage_type='memory'),
    dcc.Store(id='column_name', data=[], storage_type='memory'),

    html.Div(id='output-data-upload2'),

    html.Div(id='plots'),
])





def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df

#dodanie pliku csv
@callback(
    [Output('store-data1', 'data'),
    Output('output-data-upload1', 'children')],
    [Input('upload-data', 'contents'),
    State('upload-data', 'filename'),]
)
def update_output(contents, filename):
    data = []
    children = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        data = df.to_dict()

        children = html.Div([
            html.H5(filename),
            html.P("Inset X axis data"),
            dcc.Dropdown(id='column_select',
                         options=[{'label': x, 'value': x} for x in df.columns]),


            #dash_table.DataTable(
            #    df.to_dict('records'),
            #    [{'name': i, 'id': i} for i in df.columns]
            #),

            html.Hr(),  # horizontal line
        ])

    return data, children


## Wybranie kolumny
@callback(
    [Output('column_name', 'data'),
    Output('output-data-upload2', 'children')],
    Input('column_select', 'value'),
    State('store-data1', 'data')
)
def select_kolumns(value, tmp):
    children = html.Div([
        html.H5(value),
        html.H5("Select model"),
        dcc.Upload(
            id='upload-data3',
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
            # Allow multiple files to be uploaded
            multiple=True
        ),

        html.Hr(),  # horizontal line
    ])
    data = {'name' : ''}
    data['name'] = value
    return data, children












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



#wybranie modelu
@callback(
    Output('plots', 'children'),
    Input('upload-data3', 'contents'),
    State('upload-data3', 'filename'),
    State('store-data1', 'data'),
    State('column_name', 'data'),
)
def update_model(contents, filename, df, column):
    children = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        model = parse_data_model_flaml(contents, filename)
        df = pd.DataFrame.from_dict(df)
        df = df.dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        print(X)
        print(y)
        print(type(y))
        print(y)
        children = html.Div([
            html.H5("sad"),
            html.H5(column["name"]),
            #regresja
            dcc.Graph(figure=metrics.mse_plot(model, X, y)),
            dcc.Graph(figure=metrics.mape_plot(model, X, y)),
            dcc.Graph(figure=metrics.mae_plot(model, X, y)),
            dcc.Graph(figure=metrics.correlation_plot(model, X, y, task="regression")),
            dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, "Flaml", "regression")),

        ])

    return children
