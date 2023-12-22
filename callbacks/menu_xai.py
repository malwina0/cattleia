from dash import Output, Input, callback, State
from components.navigation import navigation_row

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input('xai', 'n_clicks'),
    State('xai_plots', 'data'),
    State('plots', 'children'),
    prevent_initial_call=True
)
def show_metrics(n_clicks, xai_plots, children):
    if n_clicks is None:
        return children
    if n_clicks >= 1:
        xai_plots.insert(0, navigation_row)
        return xai_plots
    return children