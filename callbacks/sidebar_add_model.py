import dash_daq as daq
from dash import html, dcc, Output, Input, callback
import sys
sys.path.append("..")

@callback(
    [Output('y_label_column', 'data'),
     Output('upload_model_section', 'children'),
     Output('switch_annotation', 'children')],
    Input('column_select', 'value')
)
def add_model(value):
    children = html.Div([
        html.H5("Upload model", className="sidepanel_text"),
        dcc.Upload(
            id='upload_model',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            className="upload_data",
            multiple=True
        ),
    ])
    switch = html.Div([
        html.Hr(),
        html.H5("Annotation", className="sidepanel_text", id="switch_text"),
        daq.ToggleSwitch(
            id='my-toggle-switch',
            value=True,
            color="#0072ef"
        )
    ])

    data = {'name': value}

    return data, children, switch