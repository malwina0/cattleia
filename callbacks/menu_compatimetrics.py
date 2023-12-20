from dash import Output, Input, callback, State
import sys
sys.path.append("..")

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('compatimetrics', 'n_clicks'),
    State('compatimetric_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_compatimetrics(n_clicks, data, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        return data
    return children