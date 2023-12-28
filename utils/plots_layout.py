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

def ensemble_color(model_name, color_map):
    if model_name == 'Ensemble':
        color_map.append('#f5e9ab')
    else:
        color_map.append('#ffaef4')
    return color_map