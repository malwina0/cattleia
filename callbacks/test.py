from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
from components import metrics
import shutil
import pandas as pd
from utils.utils import get_predictions_from_model, get_task_from_model, parse_model, get_ensemble_weights,\
    get_probability_pred_from_model
from components.weights import slider_section, tbl_metrics, tbl_metrics_adj_ensemble
# part responsible for adding model and showing plots
@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('link-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_metrics(n_clicks):
    children = []
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        children = html.Div([
            dbc.Row([
                dbc.Col(html.H2("1. Train model"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/train.png", height="30px", className="instruction-img"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("Train the model using Autosklearn, Autogluon or Flaml.", className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/train_model.png",
                              style={"max-width": "40%", "height": "auto"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H2("2. Save model  "), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/save.png", height="30px", className="instruction-img"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("""For Flaml and Autosklern save model using pickle.dump method, 
                             for AutoGluon pack the folder containing all the files into a zip archive.""",
                             className="instruction_str", style={"max-width": "70%", "height": "auto"})]),
            html.Br(),
            dbc.Row([html.Img(src="assets/images/save_fl_as.png",
                              style={"max-width": "40%", "height": "auto"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([html.Img(src="assets/images/save_ag.png",
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H1("3. Upload data file"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/upload.png", height="60px"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("Upload a csv file with data that model was trained.", className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/csv_upload.png",
                              style={"max-width": "30%", "height": "auto"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H1("4. Select column"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/select.png", height="50px"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("Select the column that is the target of the model.", className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/select_csv.png",
                              style={"max-width": "25%", "height": "auto"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H1("5. Upload model"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/upload.png", height="60px"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("Upload saved model, .pkl file for Flaml and Autosklearn, .zip file for AutoGluon.",
                             className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/upload_model.png",
                              style={"max-width": "29%", "height": "auto"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H1("6. Analise metrics and plots"), align="center", width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/analise.png", height="50px"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.H2("Analyse plots and tables created by cattleia to better understand the ensemble model.",
                             className="instruction_str")]),
            html.Br(),
            html.Br(),
            html.Br(),
        ], className="instruction")
    return children