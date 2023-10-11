import dash
from dash import html, dcc, callback, Input, Output, State
import base64
import io
import dash_bootstrap_components as dbc
import sys
import zipfile
import metrics
import shutil
import pandas as pd
from autogluon.tabular import TabularPredictor
sys.path.append("..")

dash.register_page(__name__,  path='/')

# page layout
layout = html.Div([
    dcc.Store(id='csv_data', data=[], storage_type='memory'),
    dcc.Store(id='y_label_column', data=[], storage_type='memory'),
    # side menu
    html.Div([
        dbc.Container([
            dcc.Link(html.H5("Instruction"), href="/instruction", className="sidepanel_text"),
            html.Hr(),
            html.H5("Upload csv data", className="sidepanel_text"),
            dcc.Upload(
                id='upload_csv_data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="upload_data",
                multiple=True
            ),
            html.Div(id='select_y_label_column'),
            html.Div(id='upload_model_section'),
        ], className="px-3 sidepanel")
    ], id="side_menu_div"),
    # plots
    html.Div([
        dcc.Loading(id="loading-1", type="default", children=html.Div(id="plots"), className="spin")
    ], id="plots_div")
])


# data loading function
def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df
        elif "pkl" in filename:
            model = pd.read_pickle(io.BytesIO(decoded))
            if "<class 'flaml" in str(model.__class__).split("."):
                library = "Flaml"
            else:
                library = "AutoSklearn"
            return model, library
        elif "zip" in filename:
            with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
                zip_ref.extractall('./uploaded_model')

            model = TabularPredictor.load('./uploaded_model', require_py_version_match=False)
            library = "AutoGluon"
            return model, library

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])


# part responsible for adding csv file
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
        df = parse_data(contents, filename)
        data = df.to_dict()

        children = html.Div([
            html.P(filename, className="sidepanel_text"),
            html.Hr(),
            html.H5("Select target colum", className="sidepanel_text"),
            dcc.Dropdown(id='column_select', className="dropdown-class",
                         options=[{'label': x, 'value': x} for x in df.columns]),
            html.Hr(),
        ])

    return data, children


# part responsible for choosing target column
@callback(
    [Output('y_label_column', 'data'),
        Output('upload_model_section', 'children')],
    Input('column_select', 'value')
)
def select_columns(value):
    children = html.Div([
        html.H5("Upload model", className="sidepanel_text"),
        dcc.Upload(
            id='upload_model',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            className="upload_data",
            multiple=True
        ),
    ])
    data = {'name': value}

    return data, children


# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children'),
    Input('upload_model', 'contents'),
    State('upload_model', 'filename'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
)
def update_model(contents, filename, df, column):
    children = []
    if contents:
        contents = contents[0]
        filename = filename[0]

        # delete folder if it already is, only for autogluon
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        model, library = parse_data(contents, filename)

        df = pd.DataFrame.from_dict(df)
        df = df.dropna()
        X = df.iloc[:, df.columns != column["name"]]
        y = df.iloc[:, df.columns == column["name"]]
        y = y.squeeze()

        if y.squeeze().nunique() > 10:
            task = "regression"
        else:
            task = "classification"

        if task == "regression":
            plot_component = [
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=metrics.mse_plot(model, X, y, library=library), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=metrics.mape_plot(model, X, y, library=library), className="plot")],
                            width=6),
                ]),
                dcc.Graph(figure=metrics.mae_plot(model, X, y, library=library), className="plot"),
                dcc.Graph(figure=metrics.correlation_plot(model, X, library=library, task=task, y=y),
                          className="plot"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot"),
            ]
        else:
            plot_component = [
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=metrics.accuracy_plot(model, X, y, library=library),
                        className="plot")], width=6),
                    dbc.Col([dcc.Graph(figure=metrics.precision_plot(model, X, y, library=library),
                        className="plot")], width=6),
                ]),
                dbc.Row([
                    dbc.Col(
                        [dcc.Graph(figure=metrics.recall_plot(model, X, y, library=library), className="plot")],
                        width=6),
                    dbc.Col(
                        [dcc.Graph(figure=metrics.f1_score_plot(model, X, y, library=library), className="plot")],
                        width=6),
                ]),
                dcc.Graph(figure=metrics.correlation_plot(model, X, library=library, task=task, y=y),
                          className="plot"),
                dcc.Graph(figure=metrics.prediction_compare_plot(model, X, y, library=library, task=task),
                          className="plot"),
            ]

        for plot in metrics.permutation_feature_importance_all(model, X, y, library=library, task=task):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        for plot in metrics.partial_dependence_plots(model, X, library=library, autogluon_task=task):
            plot_component.append(dcc.Graph(figure=plot, className="plot"))

        # It may be necessary to keep the model for the code with weights,
        # for now we remove the model after making charts
        try:
            shutil.rmtree('./uploaded_model')
        except FileNotFoundError:
            pass

        children = html.Div(plot_component)

    return children


# callback responsible for moving the menu
dash.clientside_callback(
    """
    function() {
        var menu = document.getElementById('side_menu_div');
        var content = document.getElementById('plots_div');

        menu.style.left = '0';
        content.style.left = '250px';

        window.addEventListener('scroll', function() {
            var currentScrollY = window.scrollY;
            console.log(currentScrollY);
            if (currentScrollY > 50) {
                menu.style.left = '-250px';
                content.style.marginLeft = '0';
            } else {
                menu.style.left = '0';
                content.style.marginLeft = '250px';
            }
        });
    }
    """,
    Output('side_menu_div', 'style'),
    Output('plots_div', 'style'),
    Input('upload_csv_data', 'contents'),
    prevent_initial_call=False
)
