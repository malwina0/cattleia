from dash import html, Output, Input, callback, ALL
import sys
sys.path.append("..")

@callback(
    Output({"type": "weight_slider", "index": ALL}, 'value', allow_duplicate=True),
    Output('weight-update-info', 'children'),
    Output('metrics-table', 'data'),
    Input('metrics-table', 'data'),
    prevent_initial_call=True
)
def update_slider_with_table_weights(table_data):
    is_updated = [isinstance(row['Weight'], str) for row in table_data]
    index = is_updated.index(True) if True in is_updated else None
    updated_table_data = table_data.copy()

    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    if index is not None:
        non_float_values = [row['Weight'] for row in table_data if not is_float(row['Weight'])]
        if not non_float_values:
            values = [float(row['Weight']) for row in table_data]
            updated_value = values[index]
            if updated_value > 1:
                info = html.Div([f"Please ensure values are within the range of [0, 1]. The provided value of "
                                 f"{updated_value} has been adjusted to 1."])
                updated_value = 1
            elif updated_value < 0:
                info = html.Div([f"Please ensure values are within the range of [0, 1]. The provided value of "
                                 f"{updated_value} has been adjusted to 0."])
                updated_value = 0
            else:
                info = None
        else:
            info = html.Div(
                [f"A numeric value is expected, but a different value was provided. Therefore, it has been set to 0."])
            updated_value = 0

        other_values = [float(row['Weight']) for idx, row in enumerate(table_data) if idx != index]
        sum_values = sum(value for i, value in enumerate(other_values))
        weights_adj = [round(((1 - updated_value) * value / sum_values), 2) for value in other_values]
        weights_adj.insert(index, updated_value)
        for i, row in enumerate(updated_table_data):
            if i < len(other_values) + 1:
                row['Weight'] = weights_adj[i]
    else:
        weights_adj = [float(row['Weight']) for row in table_data]
        info = None
        updated_table_data = table_data.copy()

    return weights_adj, info, updated_table_data