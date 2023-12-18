from dash import Output, Input, callback, State
import sys
sys.path.append("..")

# callbacks for buttons to change plots categories
@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('weights', 'n_clicks'),
    State('weight_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_weights(n_clicks, data, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        return data
    return children