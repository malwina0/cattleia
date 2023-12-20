from dash import Output, Input, callback, State, ALL
from dash.exceptions import PreventUpdate
import sys
sys.path.append("..")

# callback to reset weights values to default
@callback(
    Output({"type": "weight_slider", "index": ALL}, 'value', allow_duplicate=True),
    Input('update-weights-button', 'n_clicks'),
    State('weights_list', 'data'),
    prevent_initial_call=True
)
def reset_sliders(n_clicks, values):
    if n_clicks > 0:
        return values
    else:
        raise PreventUpdate