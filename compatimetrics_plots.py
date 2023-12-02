import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from metrics import empty_fig
from compatimetrics import mean_squared_difference, \
    root_mean_squared_difference, strong_disagreement_ratio, agreement_ratio, conjunctive_rmse, \
    uniformity, disagreement_ratio, disagreement_postive_ratio, correctness_counter, \
    conjunctive_accuracy, conjunctive_precission, conjunctive_recall, average_collective_score, \
    macro_conjunctive_precission, weighted_conjunctive_precission, macro_conjunctive_recall, \
    weighted_conjunctive_recall

general_layout = {'plot_bgcolor':'rgba(44,47,56,255)',
        'paper_bgcolor':'rgba(44,47,56,255)',
        'font_color':"rgba(225, 225, 225, 255)",
        'font_size':16,
        'title_font_color':"rgba(225, 225, 225, 255)",
        'title_font_size':18}

matrix_layout = {'plot_bgcolor':'rgba(44,47,56,255)',
        'paper_bgcolor':'rgba(44,47,56,255)',
        'font_color':"rgba(225, 225, 225, 255)",
        'font_size':16,
        'title_font_color':"rgba(225, 225, 225, 255)",
        'title_font_size':28,
        'xaxis_title_standoff':300,
        'yaxis_ticklen':39}

def msd_matrix(predictions):
    """Mean Squared Difference matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    msd_matrix(predictions)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(mean_squared_difference(predictions[models[i]], predictions[models[j]]))
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']])
    fig.update_layout(matrix_layout,
                      title='Mean Squared Error')
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def rmsd_matrix(predictions):
    """Root Mean Squared Difference matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    rmsd_matrix(predictions)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(root_mean_squared_difference(predictions[models[i]],
                                                                  predictions[models[j]]))
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']])
    fig.update_layout(matrix_layout,
                      title="Root Mean Squared Error")
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def sdr_matrix(predictions, y):
    """Strong Disagreement Ratio matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    sdr_matrix(predictions, y)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(strong_disagreement_ratio(predictions[models[i]],
                                                               predictions[models[j]], y), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0,
                    zmax=1)
    fig.update_layout(matrix_layout,
        title="Strong Disagreement Ratio",
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def ar_matrix(predictions, y):
    """Agreement Ratio matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    ar_matrix(predictions, y)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(agreement_ratio(predictions[models[i]],
                                                     predictions[models[j]], y), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0,
                    zmax=1
                    )
    fig.update_layout(matrix_layout,
        title="Agreement Ratio"
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def msd_comparison(predictions, model_to_compare):
    """Mean Squared Distance bar plot of chosen model compared with other models in ensemble.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    msd_comparison(predictions, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    MSD = []
    for model in models:
        MSD.append(mean_squared_difference(predictions[model], compare_prediction))
    fig = empty_fig()
    for i in range(len(MSD)):
        fig.add_trace(go.Bar(x=[i], y=[MSD[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="Mean Squared Difference <br> of " + model_to_compare + " model",
        font_size=17,
        title_font_size=20,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(MSD))),
            ticktext=models,
            title_font_size=17,
        ),
        showlegend=False
    )

    return fig


def rmsd_comparison(predictions, model_to_compare):
    """Root Mean Squared Distance bar plot of chosen model compared with other models in ensemble.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    rmsd_comparison(predictions, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    RMSD = []
    for model in models:
        RMSD.append(root_mean_squared_difference(predictions[model], compare_prediction))
    fig = empty_fig()
    for i in range(len(RMSD)):
        fig.add_trace(go.Bar(x=[i], y=[RMSD[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title={'text': "Root Mean Squared Difference <br> of " + model_to_compare + " model"},
        font_size=17,
        title_font_size=20,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(RMSD))),
            ticktext=models,
            title_font_size=17,
        ),
        showlegend=False
    )

    return fig

def conjunctive_rmse_plot(predictions, y, model_to_compare):
    """Bar chart showing values of chosen model RMSE and Conjunctive RMSE between
    this and other models.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
        true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    conjunctive_rmse_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    color_map = {}
    RMSE = [root_mean_squared_difference(compare_prediction, y)]
    names = [model_to_compare]
    color_map[model_to_compare] = 'purple'
    for model in models:
        RMSE.append(conjunctive_rmse(predictions[model], compare_prediction, y))
        names.append("with " + model)
        color_map["with " + model] = 'rgba(0,114,239,255)'
    fig = empty_fig()
    for i in range(len(RMSE)):
        fig.add_trace(go.Bar(x=[RMSE[i]], y=[names[i]], orientation='h', marker_color=color_map[names[i]]))
    fig.add_vline(x=RMSE[0], line_width=3, line_dash='dash', line_color='magenta')
    fig.update_layout(
        title="Comparison of " + model_to_compare + " RMSE and joined models conjunctive RMSE",
        font_size=17,
        title_font_size=20,
        showlegend=False
    )

    return fig

def difference_distribution(predictions, model_to_compare):
    """Line chart displaying difference of predictions between chosen model
        and other models on the whole set.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    difference_distribution(predictions, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_models = np.array(predictions[model_to_compare])
    models.remove(model_to_compare)
    difference = []
    for model in models:
        difference.append(compare_models - np.array(predictions[model]))
    fig = empty_fig()
    for i in range(len(models)):
        fig.add_traces(go.Scatter(x=list(range(len(compare_models))), y=difference[i], mode='lines', name=models[i]))
    fig.update_layout(
        title="Difference of prediction between " + model_to_compare + " and other models",
        font_size=17,
        title_font_size=20,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        legend=dict(orientation="h")
    )

    return fig


def difference_boxplot(predictions, y, model_to_compare):
    """Boxplot of absolute difference between prediction of chosen model
        and other models with lines indicating Agreement Ratio
        and Strong Disagreement Ratio

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
        true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    difference_boxplot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_models = predictions[model_to_compare]
    models.remove(model_to_compare)
    n_observation = len(compare_models)
    difference = []
    model_name = []
    for model in models:
        difference = difference + list(np.abs(np.array(compare_models) - np.array(predictions[model])))
        model_name = model_name + [model] * n_observation
    df = pd.DataFrame(list(zip(difference, model_name)),
                      columns=['difference', 'model_name'])
    fig = px.box(df, x="model_name", y="difference")
    standard_deviation = np.std(y)
    fig.add_hline(y=standard_deviation, line_width=3, line_dash='dash', line_color='lightpink')
    fig.add_hline(y=standard_deviation / 50, line_width=3, line_dash='dash', line_color='lightpink')
    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(general_layout,
        title={'text': "Distribution of absolute difference between " + model_to_compare + " prediction <br> and " +
                       "other models predictions with tresholds of agreement and strong disagreement"},
        autosize=True,
        height=800
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
    )
    fig.update_xaxes(title='')
    return fig


def uniformity_matrix(predictions):
    """Uniformity matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    uniformity_matrix(predictions)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = round(uniformity(predictions[models[i]], predictions[models[j]]), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0, zmax=1)
    fig.update_layout(matrix_layout,
        title="Uniformity"
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def incompatibility_matrix(predictions):
    """Incompatibility matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    incompatibilty_matrix(predictions)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(disagreement_ratio(predictions[models[i]], predictions[models[j]]), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0, zmax=1)
    fig.update_layout(matrix_layout,
        title="Incompatibility"
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def acs_matrix(predictions, y):
    """Average Collective Score matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    acs_matrix(predictions, y)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(average_collective_score(predictions[models[i]], predictions[models[j]], y)[0], 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0, zmax=1)
    fig.update_layout(matrix_layout,
        title="Average Collective Score"
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def conjuntive_accuracy_matrix(predictions, y):
    """Conjunctive accuracy matrix of every pair of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    conjunctive_accuracy_matrix(predictions, y)
    """
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(conjunctive_accuracy(predictions[models[i]], predictions[models[j]], y), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']],
                    zmin=0, zmax=1)
    fig.update_layout(matrix_layout,
        title="Conjunctive Accuracy"
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig


def disagreement_ratio_plot(predictions, y, model_to_compare):
    """Disagreement Raio bar plot with class division of chosen model
        and other models.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    disagreement_ratio_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    classes = np.unique(y)
    ratio = []
    model_name = []
    class_name = []
    for model in models:
        ratio = ratio + [disagreement_postive_ratio(compare_prediction, predictions[model], y, classes[1]),
                         disagreement_postive_ratio(compare_prediction, predictions[model], y, classes[0])]
        model_name = model_name + [model] * 2
        class_name = class_name + [classes[1], classes[0]]
    df = pd.DataFrame(list(zip(ratio, model_name, class_name)),
                      columns=['ratio', 'model_name', 'class_name'])
    df['class_name'] = df['class_name'].astype(str)
    fig = px.bar(df, x='model_name', y='ratio', color='class_name', barmode='group')
    fig.update_layout(general_layout,
        title='Disagreement ratio in each class of ' + model_to_compare + " <br> model and other models",
        legend_title="Class name:"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Disagreement Ratio'
    )
    fig.update_xaxes(title='')
    return fig


def conjunctive_metrics_plot(predictions, y, model_to_compare):
    """Bar plot with conjunctive accuracy, conjunctive precision
    and conjucntive recall of chosen model with other models

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    conjunctive_metrics_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    value = []
    model_name = []
    metric_name = []
    for model in models:
        value = value + [conjunctive_accuracy(compare_prediction, predictions[model], y),
                         conjunctive_precission(compare_prediction, predictions[model], y),
                         conjunctive_recall(compare_prediction, predictions[model], y)]
        model_name = model_name + [model] * 3
        metric_name = metric_name + ['accuracy', 'precision', 'recall']
    df = pd.DataFrame(list(zip(value, model_name, metric_name)),
                      columns=['value', 'model_name', 'metric_name'])
    fig = px.bar(df, x='model_name', y='value', color='metric_name', barmode='group')
    fig.update_layout(general_layout,
        title='Conjunctive metrics values of ' + model_to_compare + "<br> model and other models",
        legend_title="Metric"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Metric value'
    )
    fig.data[0].marker.color = 'rgba(0,114,239,255)'
    fig.data[1].marker.color = 'purple'
    fig.data[2].marker.color = '#321e8a'
    fig.update_xaxes(title='')
    return fig


def prediction_correctness_plot(predictions, y, model_to_compare):
    """Bar plot showing ratio of observation predicted on different level
        of correctness of chosen model and other models

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    prediction_correctness_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    value = []
    model_name = []
    correctness = []
    for model in models:
        value = value + list(correctness_counter(compare_prediction, predictions[model], y))
        model_name = model_name + [model] * 3
        correctness = correctness + ['doubly correct', 'disagreement', 'doubly incorrect']
    df = pd.DataFrame(list(zip(value, model_name, correctness)),
                      columns=['value', 'model_name', 'correctness'])
    fig = px.bar(df, x='value', y='model_name', color='correctness')
    fig.update_layout(general_layout,
        title='Prediction correctness ratio of ' + model_to_compare + " model and other models.",
        legend_title="Correctness",
        # legend = dict(orientation="h")
    )
    fig.update_xaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Percentage'
    )
    fig.data[0].marker.color = '#168c65'
    fig.data[1].marker.color = '#2d2d87'
    fig.data[2].marker.color = '#a8324e'
    fig.update_yaxes(title='')
    return fig


def collective_cummulative_score_plot(predictions, y, model_to_compare):
    """Line chart showing progress of average collective score through
        whole set of chosen model and other models.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    collective_cummulative_score_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_models = predictions[model_to_compare]
    models.remove(model_to_compare)
    score = []
    for model in models:
        score.append(np.cumsum(average_collective_score(compare_models, predictions[model], y)[1]) / len(y))
    fig = empty_fig()
    for i in range(len(models)):
        fig.add_traces(go.Scatter(x=list(range(len(compare_models))), y=score[i], mode='lines', name=models[i]))
    fig.update_layout(
        title="Cummulative Collective Score of " + model_to_compare + " model",
        font_size=17,
        title_font_size=20,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
    )

    return fig

def conjunctive_precision_multiclass_plot(predictions, y, model_to_compare):
    """Bar plot showing weighted and macro conjunctive precision
        value for chosen model and remaining models.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    conjunctive_precision_multiclass_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    value = []
    model_name = []
    metric_type = []
    for model in models:
        value = value + [macro_conjunctive_precission(compare_prediction, predictions[model], y),
                         weighted_conjunctive_precission(compare_prediction, predictions[model], y)]
        model_name = model_name + [model] * 2
        metric_type = metric_type + ['macro', 'weighted']
    df = pd.DataFrame(list(zip(value, model_name, metric_type)),
                      columns=['value', 'model_name', 'metric_type'])
    fig = px.bar(df, x='model_name', y='value', color='metric_type', barmode='group')
    fig.update_layout(general_layout,
        title='Conjunctive precision values of ' + model_to_compare + "<br> model and other models",
        legend_title="Metric type"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Metric type'
    )
    fig.data[0].marker.color = 'rgba(0,114,239,255)'
    fig.data[1].marker.color = 'purple'
    fig.update_xaxes(title='')
    return fig

def conjunctive_recall_multiclass_plot(predictions, y, model_to_compare):
    """Bar plot showing weighted and macro conjunctive recall
        value for chosen model and remaining models.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y: list, numpy.array, pandas.series
    true values of predicted variable

    model_to_compare: string
        name of model to compare with other models

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    conjunctive_recall_multiclass_plot(predictions, y, model_to_compare)
    """
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    value = []
    model_name = []
    metric_type = []
    for model in models:
        value = value + [macro_conjunctive_recall(compare_prediction, predictions[model], y),
                         weighted_conjunctive_recall(compare_prediction, predictions[model], y)]
        model_name = model_name + [model] * 2
        metric_type = metric_type + ['macro', 'weighted']
    df = pd.DataFrame(list(zip(value, model_name, metric_type)),
                      columns=['value', 'model_name', 'metric_type'])
    fig = px.bar(df, x='model_name', y='value', color='metric_type', barmode='group')
    fig.update_layout(general_layout,
        title='Conjunctive recall values of ' + model_to_compare + "<br> model and other models",
        legend_title="Metric type"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Metric type'
    )
    fig.data[0].marker.color = 'rgba(0,114,239,255)'
    fig.data[1].marker.color = 'purple'
    fig.update_xaxes(title='')
    return fig