general_layout = {
    'plot_bgcolor':'rgba(44,47,56,255)',
    'paper_bgcolor':'rgba(44,47,56,255)',
    'font_color':"rgba(225, 225, 225, 255)",
    'font_size':16,
    'title_font_color':"rgba(225, 225, 225, 255)",
    'title_font_size':18
}

matrix_layout = {
    'plot_bgcolor':'rgba(44,47,56,255)',
    'paper_bgcolor':'rgba(44,47,56,255)',
    'font_color':"rgba(225, 225, 225, 255)",
    'font_size':16,
    'title_font_color':"rgba(225, 225, 225, 255)",
    'title_font_size':28,
    'xaxis_title_standoff':300,
    'yaxis_ticklen':39,
    'coloraxis_colorbar_title_text':''
}

custom_scale_continuous = [
    [0, 'rgb(248, 213, 240)'],
    [0.25, 'rgb(255, 174, 244)'],
    [0.5, 'rgb(234, 71, 201)'],
    [0.75, 'rgb(120, 5, 97)'],
    [1, 'rgb(81, 2, 65)']
]

def ensemble_color(model_name, color_map):
    if model_name == 'Ensemble':
        color_map.append('#f5e9ab')
    else:
        color_map.append('#ffaef4')
    return color_map