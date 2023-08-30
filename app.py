from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc


app = Dash("CATTLEIA",
           use_pages=True,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/cattleia.png", height="40px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Home", className="nav_bar_text"), width="auto"),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/",
                        style={"textDecoration": "none"},
                    ),
                ], width="auto"),
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/flaml_logo.png", height="40px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Flaml", className="nav_bar_text"), width="auto"),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/flam",
                        style={"textDecoration": "none"},
                    ),
                ], width="auto"),
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/autosklearn_logo.png", height="40px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("AutoSklearn", className="nav_bar_text"), width="auto"),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/autosklearn",
                        style={"textDecoration": "none"},
                    ),
                ], width="auto"),
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/autogluon_logo.png", height="40px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("AutoGluon",  className="nav_bar_text"), width="auto"),
                            ],
                            align="center",
                            className="g-0",

                        ),
                        href="/autogluon",
                        style={"textDecoration": "none", },

                    ),
                ], width="auto",),
            ]),
        ], className="custom_nav_bar"
    ),
    color='rgba(38,38,38, 0)',
    dark=True,
    sticky='top',
)


app.layout = html.Div([
    navbar,
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)
