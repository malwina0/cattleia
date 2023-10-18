import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc


dash.register_page(__name__)

layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("1. Train model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/train.png", height="60px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("Train the model using Autosklearn, Autogluon or Flaml.", className="instruction_str")]),
    dbc.Row([html.Img(src="assets/train_model.png",
                      style={"max-width": "40%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1("2. Save model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/save.png", height="50px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("""For Flaml and Autosklern save model using pickle.dump method, 
                     for AutoGluon pack the folder containing all the files into a zip archive.""",
                     className="instruction_str")]),
    html.Br(),
    dbc.Row([html.Img(src="assets/save_fl_as.png",
                      style={"max-width": "40%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([html.Img(src="assets/save_ag.png",
                      style={"max-width": "40%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1("3. Upload data file"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/upload.png", height="60px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("Upload a csv file with data that model was trained.", className="instruction_str")]),
    dbc.Row([html.Img(src="assets/csv_upload.png",
                      style={"max-width": "30%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1("4. Select column"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/select.png", height="50px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("Select the column that is the target of the model.", className="instruction_str")]),
    dbc.Row([html.Img(src="assets/select_csv.png",
                      style={"max-width": "25%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1("5. Upload model"), width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/upload.png", height="60px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("Upload saved model, .pkl file for Flaml and Autosklearn, .zip file for AutoGluon.",
                     className="instruction_str")]),
    dbc.Row([html.Img(src="assets/upload_model.png",
                      style={"max-width": "29%", "height": "auto"},
                      className="instruction_str")
             ]),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1("6. Analise metrics and plots"), align="center", width='auto', className="instruction_main"),
        dbc.Col(html.Img(src="assets/analise.png", height="50px"), align="center", width='auto'),
    ], justify="start"),
    dbc.Row([html.H2("Analyze plots and tables created by cattleia to better understand the ensemble model.",
                     className="instruction_str")]),
    html.Br(),
    html.Br(),
    html.Br(),
], className="instruction")


