from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
from utils.utils import parse_data


@callback(
    Output('csv_data', 'data'),
    Output('select_y_label_column', 'children'),
    Output('plots', 'children', allow_duplicate=True),
    Input('upload_csv_data', 'contents'),
    State('upload_csv_data', 'filename'),
    prevent_initial_call=True
)
def add_csv_file(contents, filename):
    data = []
    children = []
    if contents:
        contents = contents[0]
        filename = filename[0]
        if ".csv" in filename:
            df = parse_data(contents)
            data = df.to_dict()
            # Creating the dropdown menu with full labels displayed as tooltips
            options = [{'label': x[:20] + '...' if len(x) > 20 else x, 'value': x, 'title': x} for x in df.columns]

            children = html.Div([
                html.P(filename, className="sidepanel_text"),
                html.Hr(),
                html.H5("Select target colum", className="sidepanel_text"),
                dcc.Dropdown(
                    id='column_select',
                    className="dropdown-class",
                    options=options
                ),
                html.Hr(),
            ])
        else:
            children = html.Div(["Please provide the file in .csv format."], style={"color": "white"})

    info_row = dbc.Row([
        html.Div(["""Please select the target column and upload the model."""],
                 className='page-text'
                 )],
        className='plot'
    )

    return data, children, info_row