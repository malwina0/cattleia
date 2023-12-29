from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = Dash("CATTLEIA",
           use_pages=True,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css'])

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/images/cattleia.png", height="80px"), ),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/",
                        style={"textDecoration": "none"},
                    ),
                ], width='auto'),
                dbc.Col(html.Img(src="assets/images/title.png", height="80px"), align="center", className="tittle",
                        width='auto'),
                dbc.Col([
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/images/github.png", height="70px"), ),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="https://github.com/malwina0/cattleia",
                        style={"textDecoration": "none", },
                    ),
                ], width='auto'),
            ], justify="between"),
        ], className="custom_nav_bar", style={"display": "block"}
    ),
    color='rgba(38,38,38, 0)',
    dark=True,
    sticky='top',
)

app.layout = html.Div([
    html.Link(
        href='https://fonts.cdnfonts.com/css/product-sans',
        rel='stylesheet'
    ),
    navbar,
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=False)
