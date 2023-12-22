from dash import html, dcc, Output, Input, callback
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
        title = html.H4("Choose model for compatimetrics analysis:", className="compatimetrics_title")
        dropdown = dcc.Dropdown(id='model_select', className="dropdown-class",
                                options=[{'label': x, 'value': x} for x in model_names],
                                value=model_names[0], clearable=False)
        elements = [title, dropdown]
        elements.insert(0, navigation_row)
        elements.insert(1, html.Div([], className="navigation_placeholder"))
        elements.append(html.Div(id='compatimetrics_container', children=html.Div(id='compatimetrics_plots')))
        children = html.Div(elements)
    return children