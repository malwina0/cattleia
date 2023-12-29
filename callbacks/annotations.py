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

# @callback(
#     Output('ann_comp_1', 'style'),
#     Output('ann_comp_2', 'style'),
#     Output('ann_comp_3', 'style'),
#     Output('ann_comp_4', 'style'),
#     Output('ann_comp_5', 'style'),
#     Input('my-toggle-switch', 'value'),
# )
# def update_output(value):
#     if value:
#         return 5 * [{}]
#     else:
#         return 5 * [{"display": "none"}]

#
# @callback(
#     Output('ann_comp_6', 'style'),
#     Output('ann_comp_7', 'style'),
#     Output('ann_comp_8', 'style'),
#     Output('ann_comp_9', 'style'),
#     Output('ann_comp_10', 'style'),
#     Output('ann_comp_11', 'style'),
#     Input('my-toggle-switch', 'value'),
# )
# def update_output(value):
#     if value:
#         return 6 * [{}]
#     else:
#         return {"display": "none"}
#
#
# @callback(
#     Output('ann_comp_12', 'style'),
#     Output('ann_comp_13', 'style'),
#     Output('ann_comp_14', 'style'),
#     Output('ann_comp_15', 'style'),
#     Output('ann_comp_16', 'style'),
#     Input('my-toggle-switch', 'value'),
# )
# def update_output(value):
#     if value:
#         return 5 * [{}]
#     else:
#         return 5 * [{"display": "none"}]
