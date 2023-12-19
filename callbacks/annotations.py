from dash import Output, Input, callback

@callback(
    Output('ann_1', 'style'),
    Output('ann_0', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}, {}
    else:
        return {"display": "none"}, {"display": "none"}


@callback(
    Output('ann_2', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return {}
    else:
        return {"display": "none"}


@callback(
    Output('ann_comp_1', 'style'),
    Output('ann_comp_2', 'style'),
    Output('ann_comp_3', 'style'),
    Output('ann_comp_4', 'style'),
    Output('ann_comp_5', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 5 * [{}]
    else:
        return 5 * [{"display": "none"}]


@callback(
    Output('ann_comp_6', 'style'),
    Output('ann_comp_7', 'style'),
    Output('ann_comp_8', 'style'),
    Output('ann_comp_9', 'style'),
    Output('ann_comp_10', 'style'),
    Output('ann_comp_11', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 6 * [{}]
    else:
        return {"display": "none"}


@callback(
    Output('ann_comp_12', 'style'),
    Output('ann_comp_13', 'style'),
    Output('ann_comp_14', 'style'),
    Output('ann_comp_15', 'style'),
    Output('ann_comp_16', 'style'),
    Input('my-toggle-switch', 'value'),
)
def update_output(value):
    if value:
        return 5 * [{}]
    else:
        return 5 * [{"display": "none"}]