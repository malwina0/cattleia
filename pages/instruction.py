import dash
from dash import html, dcc, callback, Input, Output, State


dash.register_page(__name__)

layout = html.Div([
    html.Div(id='menu', children=[
        # Tutaj dodaj elementy Twojego menu
        html.H1("Menu"),
        html.H1("Pozycja 1"),
        html.H1("Pozycja 2"),
        html.H1("Pozycja 3"),
    ]),
    html.Div(id='content', children=[
        # Tutaj dodaj zawartość swojej strony
        html.H1("Zawartość strony"),
        html.H1("Aliquam erat volutpat. Sed vulputate nunc eu libero dictum, nec facilisis justo dapibus."),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),
        html.H1("Etiam vel quam vel ipsum iaculis sollicitudi"),
        html.H1("Zawartość strony"),

    ]),
    dcc.Interval(id='interval-component', interval=3000)
])

dash.clientside_callback(
    """
    function() {
        //var menu = document.getElementById('menu');
        //var content = document.getElementById('content');

        // Początkowo menu jest widoczne
        menu.style.left = '0';
        content.style.left = '250px';

        window.addEventListener('scroll', function() {
            var currentScrollY = window.scrollY;
            console.log(currentScrollY);
            if (currentScrollY > 50) {
                // Przewijanie w dół - chowaj menu
                menu.style.left = '-250px';
                content.style.marginLeft = '0';
            } else {
                // Przewijanie w górę - wysuwaj menu
                menu.style.left = '0';
                content.style.marginLeft = '250px';
            }
        });
    }
    """,
    Output('menu', 'style'),
    Output('content', 'style'),
    Input('menu', 'n_clicks'),
    prevent_initial_call=False
)
