from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc


about_us = html.Div([
    html.H2("What is cattleia?", className="about_us_main"),
    html.H2("""
        The cattleia tool, through tables and visualizations, 
        allows you to look at the metrics of the models to assess their 
        contribution to the prediction of the built committee. 
        Also, it introduces compatimetrics, which enable analysis of 
        model similarity. Our application support model ensembles 
        created by automatic machine learning packages available in Python, 
        such as Auto-sklearn, AutoGluon, and FLAML.
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
    html.H2("About us", className="about_us_main"),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/dominik.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Dominik Kędzierski", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/malwina.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Malwina Wojewoda", className="about_us_img")], align="center")
        ], width=4),
        dbc.Col([
            dbc.Row([html.Img(src="assets/images/jakub.png",
                              style={"max-width": "50%", "height": "auto"},
                              className="about_us_img")]),
            dbc.Row([html.H2("Jakub Piwko", className="about_us_img")], align="center")
        ], width=4),
    ], style={"max-width": "70%", "height": "auto"}),
    html.Br(),
    html.H2("""
        We are Data Science students at Warsaw University of Technology, 
        who are passionate about data analysis and machine learning. 
        Through our work and projects, we aim to develop skills in the area of data analysis, 
        creating predictive models and extracting valuable insights from numerical information. 
        Our mission is to develop skills in this field and share our knowledge with others. 
        Cattleia is our project created as a Bachelor thesis.  
        Our project co-ordinator and supervisor is Anna Kozak.
    """, className="about_us_str", style={"max-width": "70%", "height": "auto"}),
])

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Output('instruction-button', 'style', allow_duplicate=True),
    Output('about-us-button', 'style', allow_duplicate=True),
    Input('about-us-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_about_us(n_clicks):
    children = []
    if n_clicks is None:
        return children, {'display': 'none'}
    if n_clicks >= 1:
        children = about_us
    return children, {'display': 'block' }, {'display': 'none'}