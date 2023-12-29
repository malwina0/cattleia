from dash import html, dcc, Output, Input, callback, State

@callback(
    Output('partial_dependence_plot', 'figure'),
    Input('variable_select_dropdown', 'value'),
    State('pd_plots_dict', 'data')
)
def update_graph(selected_value, pd_plots_dict):
    selected_variable_plot = pd_plots_dict.get(selected_value)
    return selected_variable_plot