import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from metrics import empty_fig
from compatimetrics import mean_squared_difference, \
      root_mean_squared_difference, strong_disagreement_ratio, agreement_ratio, \
    uniformity, disagreement_ratio, disagreement_postive_ratio, correctness_counter, \
    conjunctive_accuracy, conjunctive_precission, conjunctive_recall, average_collective_score


def get_base_model_names(ensemble_model, library):
    model_names = []
    if library == "Flaml":
        ensemble_models = ensemble_model.model.estimators_
        for model in ensemble_models:
            model_names.append(type(model).__name__)
    elif library == "AutoGluon":
        model_names = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info']['base_model_names']
    else:
        for weight, model in ensemble_model.get_models_with_weights():
            model_names.append(str(type(model._final_estimator.choice)).split('.')[-1][:-2])
    return model_names



def get_predictions_from_model(ensemble_model, X, y, library, task):
    predictions={}
    if library == "Flaml":
        ensemble_models = ensemble_model.model.estimators_
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)
        if task == 'classification':
            for model in ensemble_models:
                y_pred_class = model.predict(X_transform)
                y_pred_class_name = ensemble_model._label_transformer.inverse_transform(y_pred_class)
                predictions[type(model).__name__] = y_pred_class_name
        else:
            for model in ensemble_models:
                predictions[type(model).__name__] = model.predict(X_transform)

    elif library == "AutoGluon":
        ensemble_models = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_models:
            ensemble_model.set_model_best(model_name)
            predictions[model_name] = ensemble_model.predict(X)
        ensemble_model.set_model_best(final_model)

    elif library == "AutoSklearn":
        if task == 'classification':
            class_names = list(y.unique())
            class_names.sort()
            for weight, model in ensemble_model.get_models_with_weights():
                prediction = model.predict(X)
                prediction_class = [class_names[idx] for idx in prediction]
                predictions[str(type(model._final_estimator.choice)).split('.')[-1][:-2]] = prediction_class
        else:
            for weight, model in ensemble_model.get_models_with_weights():
                predictions[str(type(model._final_estimator.choice)).split('.')[-1][:-2]] = model.predict(X)

    return predictions


def MSD_matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
    models = list(predictions.keys())
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = round(mean_squared_difference(predictions[models[i]], predictions[models[j]]), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors)
    fig.update_layout(
        title="Mean Squared Difference",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def RMSD_matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
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
                                                                  predictions[models[j]]), 2)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors)
    fig.update_layout(
        title="Root Mean Squared Difference",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def SDR_matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
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
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors)
    fig.update_layout(
        title="Strong Disagreement Ratio",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def AR_matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
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
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors)
    fig.update_layout(
        title="Agreement Ratio",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def MSD_comparison(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
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
        title="Mean Squared Difference of " + model_to_compare + " model",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(MSD))),
            ticktext=models
        ),
        showlegend=False
    )

    return fig

def RMSD_comparison(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
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
        title="Root Mean Squared Difference of " + model_to_compare + " model",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(RMSD))),
            ticktext=models
        ),
        showlegend=False
    )

    return fig

def Difference_Distribution(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
    models = list(predictions.keys())
    models.pop(0)
    compare_models = predictions[model_to_compare]
    models.remove(model_to_compare)
    difference = []
    for model in models:
        difference.append(compare_models - predictions[model])
    fig = empty_fig()
    for i in range(len(models)):
        fig.add_traces(go.Scatter(x=list(range(len(compare_models))), y=difference[i], mode='lines', name = models[i]))
    fig.update_layout(
        title="Difference of prediction between " + model_to_compare + " and other models",
        xaxis=dict(
            showgrid = False,
            zeroline = False,
            visible = False,
        ),
        legend = dict(orientation="h")
    )

    return fig

def Difference_Box(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
    models = list(predictions.keys())
    models.pop(0)
    compare_models = predictions[model_to_compare]
    models.remove(model_to_compare)
    n_observation = len(compare_models)
    difference = []
    model_name = []
    for model in models:
        difference = difference + list(compare_models - predictions[model])
        model_name = model_name + [model]*n_observation
    df = pd.DataFrame(list(zip(difference, model_name)),
               columns =['difference', 'model_name'])
    fig = px.box(df, x="model_name", y="difference")
    standard_deviation = np.std(y)
    fig.add_hline(y = standard_deviation, line_width=3, line_dash='dash')
    fig.add_hline(y = standard_deviation/50, line_width=3, line_dash='dash')
    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        autosize=True,
        height=800,
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
    )
    fig.update_xaxes(title='')
    return fig

def Uniformity_Matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 1
            else:
                matrix[i, j] = uniformity(predictions[models[i]], predictions[models[j]])
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors, zmin=0, zmax=1)
    fig.update_layout(
        title="Uniformity",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def Incompatibility_Matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = disagreement_ratio(predictions[models[i]], predictions[models[j]])
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors, zmin=0, zmax=1)
    fig.update_layout(
        title="Incompatibility",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def Average_Collective_Matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = average_collective_score(predictions[models[i]], predictions[models[j]], y)[0]
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors, zmin=0, zmax=1)
    fig.update_layout(
        title="Average Collective Score",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def Conjuntive_Accuracy_Matrix(ensemble_model, X, y, library, task):
    predictions = get_predictions_from_model(ensemble_model, X, y, library, task)
    models = list(predictions.keys())
    models.pop(0)
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = conjunctive_accuracy(predictions[models[i]], predictions[models[j]], y)
    matrix = pd.DataFrame(matrix, index=models, columns=models)
    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=custom_colors, zmin=0, zmax=1)
    fig.update_layout(
        title="Conjunctive Accuracy",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    fig.update_xaxes(tickangle=30)
    fig.update_xaxes(tickfont_size=10)
    fig.update_traces(textfont_size=13, textfont_color="rgba(255, 255, 255, 255)")

    return fig

def Disagreement_Ratio_Plot(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
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
        model_name = model_name + [model]*2
        class_name = class_name + [classes[1], classes[0]]
    df = pd.DataFrame(list(zip(ratio, model_name, class_name)),
               columns =['ratio', 'model_name', 'class_name'])
    df['class_name'] = df['class_name'].astype(str)
    fig = px.bar(df, x='model_name', y='ratio', color='class_name', barmode='group')
    fig.update_layout(
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        legend_title="Class name"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Disagreement Ratio'
    )
    fig.update_xaxes(title='')
    return fig

def Conjunctive_Metrics_Plot(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
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
        model_name = model_name + [model]*3
        metric_name = metric_name + ['accuracy', 'precision', 'recall']
    df = pd.DataFrame(list(zip(value, model_name, metric_name)),
               columns =['value', 'model_name', 'metric_name'])
    fig = px.bar(df, x='model_name', y='value', color='metric_name', barmode='group')
    fig.update_layout(
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        legend_title="Metric"
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Metric value'
    )
    fig.update_xaxes(title='')
    return fig

def Prediction_Correctness_Plot(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
    models = list(predictions.keys())
    models.pop(0)
    compare_prediction = predictions[model_to_compare]
    models.remove(model_to_compare)
    value = []
    model_name = []
    correctness = []
    for model in models:
        value = value + list(correctness_counter(compare_prediction, predictions[model], y))
        model_name = model_name + [model]*3
        correctness = correctness + ['doubly correct', 'disagreement', 'doubly incorrect']
    df = pd.DataFrame(list(zip(value, model_name, correctness)),
               columns =['value', 'model_name', 'correctness'])
    fig = px.bar(df, x='value', y='model_name', color='correctness')
    fig.update_layout(
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        legend_title="Correctness",
        legend = dict(orientation="h")
    )
    fig.update_xaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3,
        title='Percentage'
    )
    fig.update_yaxes(title='')
    return fig

def Collective_Cummulative_Score_Plot(ensemble_model, X, y, library, model_to_compare):
    predictions = get_predictions_from_model(ensemble_model, X, y, library)
    models = list(predictions.keys())
    models.pop(0)
    compare_models = predictions[model_to_compare]
    models.remove(model_to_compare)
    score = []
    for model in models:
        score.append(np.cumsum(average_collective_score(compare_models, predictions[model], y)[1])/len(y))
    fig = empty_fig()
    for i in range(len(models)):
        fig.add_traces(go.Scatter(x=list(range(len(compare_models))), y=score[i], mode='lines', name = models[i]))
    fig.update_layout(
        title="Cummulative Collective Score of " + model_to_compare + " model",
        xaxis=dict(
            showgrid = False,
            zeroline = False,
            visible = False,
        ),
        legend = dict(orientation="h")
    )

    return fig