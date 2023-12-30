from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Output('about-us-button', 'style', allow_duplicate=True),
    Output('instruction-button', 'style', allow_duplicate=True),
    Input('instruction-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_instruction(n_clicks):
    children = []
    if n_clicks is None:
        return children, {'display': 'none'}
    if n_clicks >= 1:
        children = html.Div([
            dbc.Row([
                dbc.Col(html.H3("1. Train model"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/train.png", height="30px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div("Train the model using FLAML, Auto-sklearn or AutoGluon.", className="instruction_str")]),
            dbc.Row([dcc.Markdown("""
                                        ```python
                                        # FLAML
                                        from flaml import AutoML
                                        flaml_model - AutoML()
                                        flaml_model.fit(X_train, y_train, task="regression", ensemble=True)

                                        # Auto-sklearn
                                        from autosklearn.classificationimport AutoSklearnClassifier
                                        autosklearn_model = AutoSklearnClassifier()
                                        autosklearn_model.fit(X_train, y_train)

                                        # AutoGluon
                                        from autogluon.tabular import TabularPredictor
                                        autogluon_model = TabularPredictor(label='class', path='gluon_models')
                                        autogluon_model.fit(train_data=train_data, time_limit=180)
                                        ```
                                       """)], className="instruction-code"),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H3("2. Save model"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/save.png", height="30px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div("""For FLAML and Auto-sklearn save model using pickle.dump method.""",
                             className="instruction_str")]),
            dbc.Row([dcc.Markdown("""
                            ```python
                            # FLAML
                            import pickle
                            with open("flaml_model.pkl", "wb") as f:
                                pickle.dump(flaml_model, f, pickle.HIGHEST_PROTOCOL)
                            
                            # Auto-sklearn
                            import pickle
                            with open("autosklearn_model.pkl", "wb") as f:
                                pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
                            ```
                           """)], className="instruction-code"),
            dbc.Row([html.Div("""For AutoGluon pack all the files into a zip archive.""",
                              className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/save_ag.png",
                              className="instruction_img")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H3("3. Upload data file"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/upload.png", height="40px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div("Upload a csv file with data that model was trained.", className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/csv_upload.png",
                              style={"max-width": "30%"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H3("4. Select column"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/select.png", height="40px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div("Select the column that is the target of the model.", className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/select_csv.png",
                              style={"max-width": "25%"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H3("5. Upload model"), width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/upload.png", height="40px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div("Upload saved model, .pkl file for FLAML and Auto-sklearn, .zip file for AutoGluon.",
                             className="instruction_str")]),
            dbc.Row([html.Img(src="assets/images/upload_model.png",
                              style={"max-width": "29%"},
                              className="instruction_str")
                     ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.H3("6. Analise metrics and plots"), align="center", width='auto', className="instruction_main"),
                dbc.Col(html.Img(src="assets/images/analise.png", height="30px", className="instruction-icon"), align="center", width='auto'),
            ], justify="start"),
            dbc.Row([html.Div(["Analyse plots and tables created by ", html.I("cattleia"), " to better understand the ensemble model."],
                             className="instruction_str")]),
            html.Br(),
            html.Br(),
            html.Br(),
        ], className="instruction")
    return children, {'display': 'block'}, {'display': 'none'}