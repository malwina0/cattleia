from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc

from components.annotations import ann_comp_overview
from components.navigation import navigation_row


# callback display compatimetrics part
@callback(
    Output('compatimetric_plots', 'data', allow_duplicate=True),
    Input('model_names', 'data'),
    prevent_initial_call=True
)
def update_model_selector(model_names):
    if len(model_names) > 0:
        model_names.pop(0)
    children = []
    if model_names:
        compatimetrics_elements = [
            navigation_row,
            html.Div([], className="navigation_placeholder"),
            dbc.Row([
                ann_comp_overview,
                html.H5("Choose a model for compatimetrics analysis:", className="annotation-title"),
                dcc.Dropdown(id='model_select', className="dropdown-class",
                    options=[{'label': x, 'value': x} for x in model_names],
                    value=model_names[0], clearable=False)
            ], className="custom-caption"),
            html.Div(id='compatimetrics_container', children=html.Div(id='compatimetrics_plots'))
        ]
        children = html.Div(compatimetrics_elements)
    return children