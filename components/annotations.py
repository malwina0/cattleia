from dash import html

ann_metrics_prediction_compare = html.Div([
        "The matrix compares model predictions with actual values for each observation.",
        html.Ul([
            html.Li([html.Strong("Classification "), html.A("task: color indicates prediction correctness.")]),
            html.Li([html.Strong("Regression "),  html.A("task: color indicated percentage difference between predicted and actual value.")])
        ])
    ], className="annotation_str", id="ann_metrics_prediction_compare")