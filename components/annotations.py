from dash import html

ann_metrics_prediction_compare = html.Div([
        "The matrix compares model predictions with actual values for each observation.",
        html.Ul([
            html.Li([html.Strong("Classification "), html.A("task: color indicates prediction correctness.")]),
            html.Li([html.Strong("Regression "),  html.A("task: color indicates percentage difference between predicted and actual value.")])
        ])
    ], className="annotation_str", id="ann_metrics_prediction_compare")

ann_weights_metrics = html.Div([
    html.H4(['Individual model metrics'], className='annotation-title'),
    html.P(["A table displays weight values alongside task-specific metrics for each individual model."]),
    html.Ul([
        html.Li([
            html.Strong("Modifying"), html.A(" values in the "),
            html.Strong("'Weights' column"), html.A(" directly alters a model's weight.")]
        ),
        html.Li([
            html.A("Any adjustments made will proportionally modify other weights to ensure their "),
            html.Strong("sum remains at 1.")]
        )
    ])
], className="annotation_str", id="ann_weights_metrics")

ann_weights_sliders = html.Div([
    html.H4(['Sliders'], className='annotation-title'),
    html.P(["Sliders are provided foo adjusting the weight assigned to each model within the ensemble."]),
    html.Ul([
        html.Li([html.A("Initially, these values are pre-selected by the AutoML package.")]),
        html.Li([html.A("To restore these default values, click the 'Reset weights' button.")])
    ])
], className="annotation_str", id="ann_weights_sliders")

ann_weights_ensemble = html.Div([
    html.H4(['Ensemble Model Metrics'], className='annotation-title'),
    html.A(["In the table, metrics for both the "]),
    html.Strong(["ensemble model's original weights"]),
    html.A([" (set by AutoML) and "]),
    html.Strong(["custom weights"]),
    html.A([" (set manually) are presented side by side."])
], className="annotation_str", id="ann_weights_ensemble")

