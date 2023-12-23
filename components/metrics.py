import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, accuracy_score, \
    precision_score, recall_score, f1_score, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency


def empty_fig():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=20,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
    )
    fig.update_yaxes(
        gridcolor='rgba(51,54,61,255)',
        gridwidth=3
    )
    return fig


def accuracy_plot(predictions, y):
    """Accuracy metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    accuracy = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        accuracy.append(accuracy_score(y, prediction))

    fig = empty_fig()
    for i in range(len(accuracy)):
        fig.add_trace(go.Bar(x=[accuracy[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="Accuracy metrics values",
        showlegend=False
    )

    return fig


def precision_plot(predictions, y):
    """Precision metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    precision = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        precision.append(precision_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(precision)):
        fig.add_trace(go.Bar(x=[precision[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="Precision metrics values",
        showlegend=False
    )

    return fig


def recall_plot(predictions, y):
    """Recall metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    recall = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        recall.append(recall_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(recall)):
        fig.add_trace(go.Bar(x=[recall[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="Recall metrics values",
        showlegend=False
    )

    return fig


def f1_score_plot(predictions, y):
    """F1 metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    f1 = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        f1.append(f1_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(f1)):
        fig.add_trace(go.Bar(x=[f1[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="F score metric values",
        showlegend=False
    )

    return fig


def mape_plot(predictions, y):
    """MAPE metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    mape = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        mape.append(mean_absolute_percentage_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mape)):
        fig.add_trace(go.Bar(x=[mape[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="MAPE metric values",
        showlegend=False
    )

    return fig


def mae_plot(predictions, y):
    """MAE metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    mae = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        mae.append(mean_absolute_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mae)):
        fig.add_trace(go.Bar(x=[mae[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="MAE metric values",
        showlegend=False
    )

    return fig


def mse_plot(predictions, y):
    """MSE metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    mse = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        mse.append(mean_squared_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mse)):
        fig.add_trace(go.Bar(x=[mse[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="MSE metric values",
        showlegend=False
    )

    return fig


def rmse_plot(predictions, y):
    """RMSE metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    rmse = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        rmse.append(mean_squared_error(y, prediction, squared=False))

    fig = empty_fig()
    for i in range(len(rmse)):
        fig.add_trace(go.Bar(x=[rmse[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="RMSE metric values",
        showlegend=False
    )

    return fig


def r_2_plot(predictions, y):
    """R^2 metrics plot of individual models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    r2 = []
    models_name = []
    color_map = []
    for model_name, prediction in predictions.items():
        if model_name == 'Ensemble':
            color_map.append('lightblue')
        else:
            color_map.append('rgba(0,114,239,255)')
        models_name.append(model_name)
        r2.append(r2_score(y, prediction))

    fig = empty_fig()
    for i in range(len(r2)):
        fig.add_trace(go.Bar(x=[r2[i]], y=[models_name[i]], orientation='h', marker_color=color_map[i]))

    fig.update_layout(
        title="R^2 metric values",
        showlegend=False
    )

    return fig


def permutation_feature_importance_all(ensemble_model, X, y, library="Flaml", task="regression"):
    """Permutation feature importance plots of individual models from the ensemble model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    task : {'regression', 'classification'}
            string that specifies task type

    Returns
    -------
    plots : list of plotly.graph_objs._figure.Figure object
        plotly plot list
    """
    if library == "Flaml":
        plots = [permutation_feature_importance(ensemble_model, X, y, 'Ensemble')]

        ensemble_models = ensemble_model.model.estimators_
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)
        for model in ensemble_models:
            if task == "regression":
                plots.append(permutation_feature_importance(model, X_transform, y, type(model).__name__))
            if task == "classification" or task == 'multiclass':
                plots.append(permutation_feature_importance(model,
                                                            X_transform,
                                                            ensemble_model._label_transformer.transform(y),
                                                            type(model).__name__))
    elif library == "AutoGluon":
        if task == "regression":
            autogluon_task = "regression"
        else:
            autogluon_task = "classification"
        plots = [permutation_feature_importance(ensemble_model, X, y, 'Ensemble', autogluon_task)]

        ensemble_models = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_models:
            ensemble_model.set_model_best(model_name)
            plots.append(permutation_feature_importance(ensemble_model, X, y, model_name, autogluon_task))
        ensemble_model.set_model_best(final_model)
    elif library == "AutoSklearn":
        plots = [permutation_feature_importance(ensemble_model, X, y, 'Ensemble')]
        if task == "classification" or "multiclass":
            class_name = y.unique()
            class_index = {name: idx for idx, name in enumerate(class_name)}
            y_class_index = [class_index[y_elem] for y_elem in y]

        for weight, model in ensemble_model.get_models_with_weights():
            model_name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
            if task == "classification" or task == "multiclass":
                plots.append(permutation_feature_importance(model, X, y_class_index, model_name, task))
            else:
                plots.append(permutation_feature_importance(model, X, y, model_name, task))

    return plots


def permutation_feature_importance(model, X, y, name, task=False):
    """Permutation feature importance plot of individual models from the ensemble model.

    Parameters
    ----------
    model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    name : string that specifies the model name

    task : {'regression', 'classification', False}
            If library is not autogluon then it should be False. Otherwise it should be
            "regression" or "classification" depends on the task.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    if task == False:
        r = permutation_importance(model, X, y)
    elif task == "regression":
        r = permutation_importance(model, X, y, scoring='r2')
    elif task == "classification" or task == "multiclass":
        r = permutation_importance(model, X, y, scoring='accuracy')
    importance = r.importances_mean

    fig = empty_fig()
    for i in range(len(importance)):
        fig.add_trace(go.Bar(x=[i], y=[importance[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title=name + " model feature importance",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(importance))),
            ticktext=X.columns
        ),
        yaxis_range=[0, 1],
        showlegend=False
    )
    return fig


def correlation_plot(predictions, task="regression", y=None):
    """Prediction correlation plot of models from the ensemble model.

    Parameters
    ----------
    predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    task : {'regression', 'classification'}
            string that specifies the model task

    y : dataframe
        needed only for  AutoSklearn

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    predict_data = pd.DataFrame(predictions)
    if task == "regression":
        corr_matrix = predict_data.corr().round(2)
    if task == "classification" or task == "multiclass":
        variables = predict_data.columns
        n_variables = len(variables)
        correlation_matrix = np.zeros((n_variables, n_variables))
        for i in range(n_variables):
            for j in range(n_variables):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    contingency_table = pd.crosstab(predict_data[variables[i]], predict_data[variables[j]])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n = np.sum(contingency_table.values)
                    phi_c = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
                    correlation_matrix[i, j] = round(phi_c, 2)
        corr_matrix = pd.DataFrame(correlation_matrix, index=variables, columns=variables)

    custom_colors = ['rgba(242,26,155,255)',
                     'rgba(254,113,0,255)',
                     'rgba(255,168,0,255)',
                     'rgba(125,179,67,255)',
                     'rgba(3,169,245,255)']
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'purple']])
    fig.update_layout(
        title="Predictions models correlation",
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


def prediction_compare_plot(predictions, y, task="regression"):
    """Prediction compare plot of models from the ensemble model.
        For classification plot show if prediction of model is correct or incorrect.
        For regression it shows the difference between the prediction and the true
        value expressed as a percentage.

    Parameters
    ----------
   predictions: dictionary with predictions of ensemble component models
        of form {'model_name': 'prediction_vector'}

    y : target variable vector

    task : {'regression', 'classification'}
            string that specifies the model task

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot
    """
    plot_value = {}
    if task == "regression":
        for name, pred in predictions.items():
            plot_value[name] = [(i - j) / (j + 0.0000001) * 100 for i, j in zip(pred, y)]
    if task == "classification" or task == "multiclass":
        for name, pred in predictions.items():
            plot_value[name] = [1 if i == j else 0 for i, j in zip(pred, y)]

    plot_value = pd.DataFrame(plot_value).T

    if task == "regression":
        discrete_nonuniform = [[0, 'rgb(242,26,155)'],
                               [0.25, 'rgb(242,26,155)'],
                               [0.25, 'rgb(255,168,0)'],
                               [0.45, 'rgb(255,168,0)'],
                               [0.45, 'rgb(125,179,67)'],
                               [0.55, 'rgb(125,179,67)'],
                               [0.55, 'rgb(3,169,245)'],
                               [0.75, 'rgb(3,169,245)'],
                               [0.75, 'rgb(93,53,175)'],
                               [1, 'rgb(93,53,175)']]
    if task == "classification" or task == "multiclass":
        discrete_nonuniform = [[0, 'rgb(242,26,155)'],
                               [0.5, 'rgb(242,26,155)'],
                               [0.5, 'rgb(125,179,67)'],
                               [1, 'rgb(125,179,67)'],
                               ]

    fig = px.imshow(plot_value, text_auto=False, color_continuous_scale=discrete_nonuniform)
    fig.update_layout(
        title="Models predictions compare",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=15,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )
    if task == "regression":
        fig.update_layout(coloraxis=dict(cmin=-100, cmax=100))
    if task == "classification" or task == "multiclass":
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=150, r=200, t=100, b=20))
        fig.add_annotation(dict(font=dict(color='rgba(225, 225, 225, 255)', size=20),
                                x=1.055, y=1.02, showarrow=False,
                                text="Correct", textangle=0, xanchor='left',
                                xref="paper", yref="paper"))
        fig.add_annotation(dict(x=1.05, y=1, xref='paper',
                                yref='paper', showarrow=False,
                                width=10, height=10,
                                bgcolor='rgb(125,179,67)'))
        fig.add_annotation(dict(font=dict(color='rgba(225, 225, 225, 255)', size=20),
                                x=1.055, y=0.92, showarrow=False,
                                text="Incorrect", textangle=0, xanchor='left',
                                xref="paper", yref="paper"))
        fig.add_annotation(dict(x=1.05, y=0.9, xref='paper',
                                yref='paper', showarrow=False,
                                width=10, height=10,
                                bgcolor='rgb(242,26,155)'))

    return fig
