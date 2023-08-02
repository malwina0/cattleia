from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc


app = Dash("CATTLEIA", use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    dcc.Link(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="assets/cattleia.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Home"), width="auto"),
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
                                dbc.Col(html.Img(src="assets/flaml_logo.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Flaml"), width="auto"),
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
                                dbc.Col(html.Img(src="assets/autosklearn_logo.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("Flaml", className="test"), width="auto"),
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
                                dbc.Col(html.Img(src="assets/autogluon_logo.png", height="30px"), width="auto"),
                                dbc.Col(dbc.NavbarBrand("AutoGluon",  style={"font-size": "50px", }), width="auto"),
                            ],
                            align="center",
                            #align="left",
                            className="g-0",
                        ),
                        href="/autogluon",
                        style={"textDecoration": "none", },

                    ),
                ], width="auto"),
            ]),
        ]
    ),
    color='rgba(38,38,38)',
    dark=True,
)


app.layout = html.Div([

    navbar,
	dash.page_container
])

if __name__ == '__main__':
	app.run(debug=True)
