from dash import Output, Input, callback, State, html
import sys

from components.navigation import navigation_row

sys.path.append("..")

# callbacks for buttons to change plots categories
@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('weights', 'n_clicks'),
    State('weight_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_weights(n_clicks, weights_plots, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        weights_plots.insert(0, navigation_row)
        weights_plots.insert(1, html.Div([], className="navigation_placeholder"))
        weights_plots = html.Div(weights_plots)
        return weights_plots
    return children