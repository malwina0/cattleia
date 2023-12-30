from dash import Output, Input, callback

@callback(
    Output('ann_metrics_prediction_compare', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}
    else:
        return {"display": "none"}

@callback(
    Output('ann_xai_partial_dep', 'style'),
    Output('ann_xai_feature_importance', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 2 * [{}]
    else:
        return 2 * [{"display": "none"}]


@callback(
    Output('ann_weights_sliders', 'style'),
    Output('ann_weights_ensemble', 'style'),
    Output('ann_weights_metrics', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 3 * [{}]
    else:
        return 3 * [{"display": "none"}]


@callback(Output('ann_comp_overview', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}
    else:
        return {"display": "none"}

@callback(Output('ann_comp_uniformity', 'style'),
    Output('ann_comp_incompatibility', 'style'),
    Output('ann_comp_acs', 'style'),
    Output('ann_comp_conj_acc', 'style'),
    Output('ann_comp_pred_corr', 'style'),
    Output('ann_comp_collective', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 6 * [{}]
    else:
        return 6 * [{"display": "none"}]

@callback(
    Output('ann_comp_dis_ratio', 'style'),
    Output('ann_comp_conj_metrics', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 2 * [{}]
    else:
        return 2 * [{"display": "none"}]

@callback(
    Output('ann_comp_conj_precision', 'style'),
    Output('ann_comp_conj_recall', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 2 * [{}]
    else:
        return 2 * [{"display": "none"}]


@callback(Output('ann_comp_msd', 'style'),
    Output('ann_comp_rmsd', 'style'),
    Output('ann_comp_ar', 'style'),
    Output('ann_comp_sdr', 'style'),
    Output('ann_comp_conj_rmse', 'style'),
    Output('ann_comp_diff_dist', 'style'),
    Output('ann_comp_diff_boxplot', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 7 * [{}]
    else:
        return 7 * [{"display": "none"}]