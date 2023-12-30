from dash import Output, Input, callback, State, ALL, html
import sys
import pandas as pd
from components.weights import calculate_metrics, calculate_metrics_adj_ensemble
sys.path.append("..")

# callback to changing model weights
@callback(
    Output('metrics-table', 'data', allow_duplicate=True),
    Output('adj_weights-table', 'data'),
    Output('weights-color-info', 'children'),
    Input({"type": "weight_slider", "index": ALL}, 'value'),
    Input('upload_model', 'contents'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
    State('task', 'data'),
    State('predictions', 'data'),
    State('proba_predictions', 'data'),
    prevent_initial_call=True
)
def display_output(values, contents, df, column, task, predictions, proba_predictions):
    if contents:
        df = pd.DataFrame.from_dict(df).dropna()
        y = df.iloc[:, df.columns == column["name"]].squeeze()
        sum_slider_values = sum(values)
        weights_adj = [round((value / sum_slider_values), 2) for value in values]
        df = calculate_metrics(predictions, y, task, weights_adj)
        df_adj = calculate_metrics_adj_ensemble(predictions, proba_predictions, y, task, weights_adj)
        info = html.Div([
            html.Div(
                "Green highlight means ",
                style={'display': 'inline', 'color': 'white'}
            ),
            html.Span(
                "improvement",
                style={'color': '#baf883', 'font-weight': 'bold'}
            ),
            html.Div(
                " after changing weights,",
                style={'display': 'inline', 'color': 'white'}
            ),
            html.Br(),
            html.Div(
                "and red means ",
                style={'display': 'inline', 'color': 'white'}
            ),
            html.Span(
                "deterioration",
                style={'color': '#fa6e6e', 'font-weight': 'bold'}
            ),
            html.Div(
                ".",
                style={'display': 'inline', 'color': 'white'}
            )
        ])
        return df.to_dict('records'), df_adj.to_dict('records'), info