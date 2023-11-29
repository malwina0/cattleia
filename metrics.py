import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence


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
    """Accuracy metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    accuracy_plot(model_class, X_class, y_class)
    """
    accuracy = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        accuracy.append(accuracy_score(y, prediction))

    fig = empty_fig()
    for i in range(len(accuracy)):
        fig.add_trace(go.Bar(x=[i], y=[accuracy[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="Accuracy metrics values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(accuracy))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def precision_plot(predictions, y):
    """Precision metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    precision_plot(model_class, X_class, y_class)
    """
    precision = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        precision.append(precision_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(precision)):
        fig.add_trace(go.Bar(x=[i], y=[precision[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="Precision metrics values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(precision))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def recall_plot(predictions, y):
    """Recall metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    recall_plot(model_class, X_class, y_class)
    """
    recall = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        recall.append(recall_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(recall)):
        fig.add_trace(go.Bar(x=[i], y=[recall[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="Recall metrics values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(recall))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def f1_score_plot(predictions, y):
    """F1 metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    f1_score_plot(model_class, X_class, y_class)
    """
    f1 = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        f1.append(f1_score(y, prediction, average='macro'))

    fig = empty_fig()
    for i in range(len(f1)):
        fig.add_trace(go.Bar(x=[i], y=[f1[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="F score metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(f1))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def mape_plot(predictions, y):
    """MAPE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mape_plot(model_reg, X_reg, y_reg)
    """
    mape = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        mape.append(mean_absolute_percentage_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mape)):
        fig.add_trace(go.Bar(x=[i], y=[mape[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="MAPE metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(mape))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def mae_plot(predictions, y):
    """MAE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mae_plot(model_reg, X_reg, y_reg)
    """
    mae = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        mae.append(mean_absolute_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mae)):
        fig.add_trace(go.Bar(x=[i], y=[mae[i]], name=""))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="MAE metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(mae))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def mse_plot(predictions, y):
    """MSE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mse_plot(model_reg, X_reg, y_reg)
    """
    mse = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        mse.append(mean_squared_error(y, prediction))

    fig = empty_fig()
    for i in range(len(mse)):
        fig.add_trace(go.Bar(x=[i], y=[mse[i]], name=models_name[i]))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="MSE metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(mse))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def rmse_plot(predictions, y):
    """RMSE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    rmse_plot(model_reg, X_reg, y_reg)
    """
    rmse = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        rmse.append(mean_squared_error(y, prediction, squared=False))

    fig = empty_fig()
    for i in range(len(rmse)):
        fig.add_trace(go.Bar(x=[i], y=[rmse[i]], name=models_name[i]))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="RMSE metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(rmse))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def r_2_plot(predictions, y):
    """R^2 metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    r_2_plot(model_reg, X_reg, y_reg)
    """
    r2 = []
    models_name = []
    for model_name, prediction in predictions.items():
        models_name.append(model_name)
        r2.append(r2_score(y, prediction))

    fig = empty_fig()
    for i in range(len(r2)):
        fig.add_trace(go.Bar(x=[i], y=[r2[i]], name=models_name[i]))

    fig.update_traces(marker=dict(color='rgba(0,114,239,255)'))
    fig.update_layout(
        title="R^2 metric values",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(r2))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig


def permutation_feature_importance_all(ensemble_model, X, y, library="Flaml", task="regression"):
    """Permutation feature importance plots of individual models from the ensemled model.

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

    Examples
    --------
    permutation_feature_importance_all(model_reg, X_reg, y_reg, task="regression")
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
    """Permutation feature importance plot of individual models from the ensemled model.

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

    Examples
    --------
    permutation_feature_importance(model, X_transform, y, type(model).__name__)
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
    """Prediction correlation plot of models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    task : {'regression', 'classification'}
            string that specifies the model task

    y : dataframe
        needed only for  AutoSklearn

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    correlation_plot(model_reg, X_reg, y_reg, task="regression")
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
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=custom_colors)
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
    """Prediction compare plot of models from the ensemled model.
        For classification plot show if prediction of model is correct or incorrect.
        For regression it shows the difference between the prediction and the true
        value expressed as a percentage.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    task : {'regression', 'classification'}
            string that specifies the model task

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    prediction_compare_plot(model_reg, X_reg, y_reg, library="Flaml", task="regression")
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


def partial_dependence_plots(ensemble_model, X, library="Flaml", autogluon_task=False):
    """Permutation feature importance plot of individual models from the ensemled model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    autogluon_task : {False, 'classification', 'regression'}
            If library is not autogluon then it shoud be False. Otherwise it shoud be
            "regression" or "classification" depends on the task  .

    Returns
    -------
    fig : list of plotly.graph_objs._figure.Figure
        list of plotly plot

    Examples
    --------
    partial_dependence_plots(model_reg, X_reg, y_reg)
    """
    if library == "Flaml":
        ensemble_models = ensemble_model.model.estimators_
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)

        model_name = {}
        columns = []
        values = {}

        for i in range(X.shape[1]):
            try:
                values[X.columns[i]] = [partial_dependence(ensemble_model, X, [i])['average'][0]]
                model_name[X.columns[i]] = ['Ensemble']
                columns.append(X.columns[i])
            except TypeError:
                pass

        for model in ensemble_models:
            for i in range(X_transform.shape[1]):
                try:
                    values[X_transform.columns[i]].append(
                        partial_dependence(model._model, model._preprocess(X_transform), [i])['average'][0])
                    model_name[X_transform.columns[i]].append(type(model).__name__)
                except Exception:
                    pass
    if library == "AutoGluon":
        ensemble_models = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']

        model_name = {}
        columns = []
        values = {}

        if autogluon_task == "classification":
            ensemble_model._estimator_type = "classifier"
        else:
            ensemble_model._estimator_type = "regressor"
        ensemble_model.classes_ = ["a"]

        for i in range(X.shape[1]):
            try:
                values[X.columns[i]] = [partial_dependence(ensemble_model, X, [i])['average'][0]]
                model_name[X.columns[i]] = ['Ensemble']
                columns.append(X.columns[i])
            except TypeError:
                pass

        final_model = ensemble_model.get_model_best()
        for model in ensemble_models:
            ensemble_model.set_model_best(model)
            for i in range(X.shape[1]):
                try:
                    values[X.columns[i]].append(partial_dependence(ensemble_model, X, [i])['average'][0])
                    model_name[X.columns[i]].append(model)
                except Exception:
                    pass
        ensemble_model.set_model_best(final_model)
    elif library == "AutoSklearn":
        model_name = {}
        columns = []
        values = {}

        for i in range(X.shape[1]):
            try:
                values[X.columns[i]] = [partial_dependence(ensemble_model, X, [i])['average'][0]]
                model_name[X.columns[i]] = ['Ensemble']
                columns.append(X.columns[i])
            except TypeError:
                pass

        for weight, model in ensemble_model.get_models_with_weights():
            for i in range(X.shape[1]):
                try:
                    if autogluon_task == "classification":
                        values[X.columns[i]].append(partial_dependence_custom(model, X, [i])['average'][0])
                    else:
                        values[X.columns[i]].append(partial_dependence(model, X, [i])['average'][0])
                    name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
                    # ta czesc odpowada za to że nazwy modeli moga się powtarzać w autosklearnie
                    if name in model_name[X.columns[i]]:
                        number = 1
                        new_name = f"{name}_{number}"
                        while new_name in model_name[X.columns[i]]:
                            number += 1
                            new_name = f"{name}_{number}"

                        model_name[X.columns[i]].append(new_name)
                    else:
                        model_name[X.columns[i]].append(name)
                except Exception:
                    pass

    plots = []
    for variable in columns:
        plot_x_value = sorted(X[variable].unique())
        plots.append(partial_dependence_line_plot(values[variable], plot_x_value, model_name[variable], variable))

    return plots


def partial_dependence_line_plot(y_values, x_values, model_names, name):
    """partial dependence one plot

    Parameters
    ----------
    y_values, x_values : dataframe
        x data.

    model_names : list
        model names.

    name: String
        name of variable

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    partial_dependence_line_plot(values[variable], plot_x_value, model_name[variable], variable)
    """
    fig = empty_fig()
    fig.update_layout(
        title=f"{name} variable partial dependence plot",
        plot_bgcolor='rgba(44,47,56,255)',
        paper_bgcolor='rgba(44,47,56,255)',
        font_color="rgba(225, 225, 225, 255)",
        font_size=15,
        title_font_color="rgba(225, 225, 225, 255)",
        title_font_size=25,
        xaxis_title_standoff=300,
        yaxis_ticklen=39,
    )

    for line_value, model_name in zip(y_values, model_names):
        fig.add_trace(go.Scatter(x=x_values, y=line_value, mode='lines', name=model_name))
    return fig


# Code below is modified code from scikit-learn library to operate AutoSklearn partial dependence
from collections.abc import Iterable
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    BaseHistGradientBoosting,)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch
from sklearn.utils import _safe_assign
from sklearn.utils import check_array
from sklearn.utils.extmath import cartesian
from sklearn.utils import _safe_indexing
from sklearn.utils import _determine_key_type
from sklearn.utils import _get_column_indices
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.exceptions import NotFittedError
from scipy.stats.mstats import mquantiles
from scipy import sparse


def _partial_dependence_brute(est, grid, features, X, response_method):

    predictions = []
    averaged_predictions = []

    # define the prediction_method (predict, predict_proba, decision_function).
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, "predict_proba", None)
        decision_function = getattr(est, "decision_function", None)
        if response_method == "auto":
            # try predict_proba, then decision_function if it doesn't exist
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = (
                predict_proba
                if response_method == "predict_proba"
                else decision_function
            )
        if prediction_method is None:
            if response_method == "auto":
                raise ValueError(
                    "The estimator has no predict_proba and no "
                    "decision_function method."
                )
            elif response_method == "predict_proba":
                raise ValueError("The estimator has no predict_proba method.")
            else:
                raise ValueError("The estimator has no decision_function method.")

    X_eval = X.copy()
    for new_values in grid:
        for i, variable in enumerate(features):
            _safe_assign(X_eval, new_values[i], column_indexer=variable)

        try:
            # Note: predictions is of shape
            # (n_points,) for non-multioutput regressors
            # (n_points, n_tasks) for multioutput regressors
            # (n_points, 1) for the regressors in cross_decomposition (I think)
            # (n_points, 2) for binary classification
            # (n_points, n_classes) for multiclass classification
            pred = prediction_method(X_eval)

            predictions.append(pred)
            # average over samples
            averaged_predictions.append(np.mean(pred, axis=0))
        except NotFittedError as e:
            raise ValueError("'estimator' parameter must be a fitted estimator") from e

    n_samples = X.shape[0]

    # reshape to (n_targets, n_instances, n_points) where n_targets is:
    # - 1 for non-multioutput regression and binary classification (shape is
    #   already correct in those cases)
    # - n_tasks for multi-output regression
    # - n_classes for multiclass classification.
    predictions = np.array(predictions).T
    if is_regressor(est) and predictions.ndim == 2:
        # non-multioutput regression, shape is (n_instances, n_points,)
        predictions = predictions.reshape(n_samples, -1)
    elif is_classifier(est) and predictions.shape[0] == 2:
        # Binary classification, shape is (2, n_instances, n_points).
        # we output the effect of **positive** class
        predictions = predictions[1]
        predictions = predictions.reshape(n_samples, -1)

    # reshape averaged_predictions to (n_targets, n_points) where n_targets is:
    # - 1 for non-multioutput regression and binary classification (shape is
    #   already correct in those cases)
    # - n_tasks for multi-output regression
    # - n_classes for multiclass classification.
    averaged_predictions = np.array(averaged_predictions).T
    if is_regressor(est) and averaged_predictions.ndim == 1:
        # non-multioutput regression, shape is (n_points,)
        averaged_predictions = averaged_predictions.reshape(1, -1)
    elif is_classifier(est) and averaged_predictions.shape[0] == 2:
        # Binary classification, shape is (2, n_points).
        # we output the effect of **positive** class
        averaged_predictions = averaged_predictions[1]
        averaged_predictions = averaged_predictions.reshape(1, -1)

    return averaged_predictions, predictions


def _grid_from_X(X, percentiles, is_categorical, grid_resolution):
    """Generate a grid of points based on the percentiles of X.

    The grid is a cartesian product between the columns of ``values``. The
    ith column of ``values`` consists in ``grid_resolution`` equally-spaced
    points between the percentiles of the jth column of X.

    If ``grid_resolution`` is bigger than the number of unique values in the
    j-th column of X or if the feature is a categorical feature (by inspecting
    `is_categorical`) , then those unique values will be used instead.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_target_features)
        The data.

    percentiles : tuple of float
        The percentiles which are used to construct the extreme values of
        the grid. Must be in [0, 1].

    is_categorical : list of bool
        For each feature, tells whether it is categorical or not. If a feature
        is categorical, then the values used will be the unique ones
        (i.e. categories) instead of the percentiles.

    grid_resolution : int
        The number of equally spaced points to be placed on the grid for each
        feature.

    Returns
    -------
    grid : ndarray of shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= grid_resolution ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``grid_resolution``, or the number of
        unique values in ``X[:, j]``, whichever is smaller.
    """
    if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
        raise ValueError("'percentiles' must be a sequence of 2 elements.")
    if not all(0 <= x <= 1 for x in percentiles):
        raise ValueError("'percentiles' values must be in [0, 1].")
    if percentiles[0] >= percentiles[1]:
        raise ValueError("percentiles[0] must be strictly less than percentiles[1].")

    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")

    values = []
    # TODO: we should handle missing values (i.e. `np.nan`) specifically and store them
    # in a different Bunch attribute.
    for feature, is_cat in enumerate(is_categorical):
        try:
            uniques = np.unique(_safe_indexing(X, feature, axis=1))
        except TypeError as exc:
            # `np.unique` will fail in the presence of `np.nan` and `str` categories
            # due to sorting. Temporary, we reraise an error explaining the problem.
            raise ValueError(
                f"The column #{feature} contains mixed data types. Finding unique "
                "categories fail due to sorting. It usually means that the column "
                "contains `np.nan` values together with `str` categories. Such use "
                "case is not yet supported in scikit-learn."
            ) from exc
        if is_cat or uniques.shape[0] < grid_resolution:
            # Use the unique values either because:
            # - feature has low resolution use unique values
            # - feature is categorical
            axis = uniques
        else:
            # create axis based on percentiles and grid resolution
            emp_percentiles = mquantiles(
                _safe_indexing(X, feature, axis=1), prob=percentiles, axis=0
            )
            if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                raise ValueError(
                    "percentiles are too close to each other, "
                    "unable to build the grid. Please choose percentiles "
                    "that are further apart."
                )
            axis = np.linspace(
                emp_percentiles[0],
                emp_percentiles[1],
                num=grid_resolution,
                endpoint=True,
            )
        values.append(axis)

    return cartesian(values), values


def _partial_dependence_recursion(est, grid, features):
    averaged_predictions = est._compute_partial_dependence_recursion(grid, features)
    if averaged_predictions.ndim == 1:
        # reshape to (1, n_points) for consistency with
        # _partial_dependence_brute
        averaged_predictions = averaged_predictions.reshape(1, -1)

    return averaged_predictions


def partial_dependence_custom(
    estimator,
    X,
    features,
    *,
    categorical_features=None,
    feature_names=None,
    response_method="auto",
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    method="auto",
    kind="average",
):
    """Partial dependence of ``features``.

    Partial dependence of a feature (or a set of features) corresponds to
    the average response of an estimator for each possible value of the
    feature.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like or dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the partial dependence will be evaluated), and
        also to generate values for the complement features when the
        `method` is 'brute'.

    features : array-like of {int, str}
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.

    categorical_features : array-like of shape (n_features,) or shape \
            (n_categorical_features,), dtype={bool, int, str}, default=None
        Indicates the categorical features.

        - `None`: no feature will be considered categorical;
        - boolean array-like: boolean mask of shape `(n_features,)`
            indicating which features are categorical. Thus, this array has
            the same shape has `X.shape[1]`;
        - integer or string array-like: integer indices or strings
            indicating categorical features.

        .. versionadded:: 1.2

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.

        .. versionadded:: 1.2

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.

    percentiles : tuple of float, default=(0.05, 0.95)
        The lower and upper percentile used to create the extreme values
        for the grid. Must be in [0, 1].

    grid_resolution : int, default=100
        The number of equally spaced points on the grid, for each target
        feature.

    method : {'auto', 'recursion', 'brute'}, default='auto'
        The method used to calculate the averaged predictions:

        - `'recursion'` is only supported for some tree-based estimators
          (namely
          :class:`~sklearn.ensemble.GradientBoostingClassifier`,
          :class:`~sklearn.ensemble.GradientBoostingRegressor`,
          :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,
          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,
          :class:`~sklearn.tree.DecisionTreeRegressor`,
          :class:`~sklearn.ensemble.RandomForestRegressor`,
          ) when `kind='average'`.
          This is more efficient in terms of speed.
          With this method, the target response of a
          classifier is always the decision function, not the predicted
          probabilities. Since the `'recursion'` method implicitly computes
          the average of the Individual Conditional Expectation (ICE) by
          design, it is not compatible with ICE and thus `kind` must be
          `'average'`.

        - `'brute'` is supported for any estimator, but is more
          computationally intensive.

        - `'auto'`: the `'recursion'` is used for estimators that support it,
          and `'brute'` is used otherwise.

        Please see :ref:`this note <pdp_method_differences>` for
        differences between the `'brute'` and `'recursion'` method.

    kind : {'average', 'individual', 'both'}, default='average'
        Whether to return the partial dependence averaged across all the
        samples in the dataset or one value per sample or both.
        See Returns below.

        Note that the fast `method='recursion'` option is only available for
        `kind='average'`. Computing individual dependencies requires using the
        slower `method='brute'` option.

    Returns
    -------
    predictions : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])
    """

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not (hasattr(X, "__array__") or sparse.issparse(X)):
        X = check_array(X, force_all_finite="allow-nan", dtype=object)

    accepted_responses = ("auto", "predict_proba", "decision_function")
    if response_method not in accepted_responses:
        raise ValueError(
            "response_method {} is invalid. Accepted response_method names "
            "are {}.".format(response_method, ", ".join(accepted_responses))
        )

    if is_regressor(estimator) and response_method != "auto":
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )

    accepted_methods = ("brute", "recursion", "auto")
    if method not in accepted_methods:
        raise ValueError(
            "method {} is invalid. Accepted method names are {}.".format(
                method, ", ".join(accepted_methods)
            )
        )

    if kind != "average":
        if method == "recursion":
            raise ValueError(
                "The 'recursion' method only applies when 'kind' is set to 'average'"
            )
        method = "brute"

    if method == "auto":
        if isinstance(estimator, BaseGradientBoosting) and estimator.init is None:
            method = "recursion"
        elif isinstance(
            estimator,
            (BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor),
        ):
            method = "recursion"
        else:
            method = "brute"

    if method == "recursion":
        if not isinstance(
            estimator,
            (
                BaseGradientBoosting,
                BaseHistGradientBoosting,
                DecisionTreeRegressor,
                RandomForestRegressor,
            ),
        ):
            supported_classes_recursion = (
                "GradientBoostingClassifier",
                "GradientBoostingRegressor",
                "HistGradientBoostingClassifier",
                "HistGradientBoostingRegressor",
                "HistGradientBoostingRegressor",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
            )
            raise ValueError(
                "Only the following estimators support the 'recursion' "
                "method: {}. Try using method='brute'.".format(
                    ", ".join(supported_classes_recursion)
                )
            )
        if response_method == "auto":
            response_method = "decision_function"

        if response_method != "decision_function":
            raise ValueError(
                "With the 'recursion' method, the response_method must be "
                "'decision_function'. Got {}.".format(response_method)
            )

    if _determine_key_type(features, accept_slice=False) == "int":
        # _get_column_indices() supports negative indexing. Here, we limit
        # the indexing to be positive. The upper bound will be checked
        # by _get_column_indices()
        if np.any(np.less(features, 0)):
            raise ValueError("all features must be in [0, {}]".format(X.shape[1] - 1))

    features_indices = np.asarray(
        _get_column_indices(X, features), dtype=np.int32, order="C"
    ).ravel()

    feature_names = _check_feature_names(X, feature_names)

    n_features = X.shape[1]
    if categorical_features is None:
        is_categorical = [False] * len(features_indices)
    else:
        categorical_features = np.array(categorical_features, copy=False)
        if categorical_features.dtype.kind == "b":
            # categorical features provided as a list of boolean
            if categorical_features.size != n_features:
                raise ValueError(
                    "When `categorical_features` is a boolean array-like, "
                    "the array should be of shape (n_features,). Got "
                    f"{categorical_features.size} elements while `X` contains "
                    f"{n_features} features."
                )
            is_categorical = [categorical_features[idx] for idx in features_indices]
        elif categorical_features.dtype.kind in ("i", "O", "U"):
            # categorical features provided as a list of indices or feature names
            categorical_features_idx = [
                _get_feature_index(cat, feature_names=feature_names)
                for cat in categorical_features
            ]
            is_categorical = [
                idx in categorical_features_idx for idx in features_indices
            ]
        else:
            raise ValueError(
                "Expected `categorical_features` to be an array-like of boolean,"
                f" integer, or string. Got {categorical_features.dtype} instead."
            )

    grid, values = _grid_from_X(
        _safe_indexing(X, features_indices, axis=1),
        percentiles,
        is_categorical,
        grid_resolution,
    )

    if method == "brute":
        averaged_predictions, predictions = _partial_dependence_brute(
            estimator, grid, features_indices, X, response_method
        )

        # reshape predictions to
        # (n_outputs, n_instances, n_values_feature_0, n_values_feature_1, ...)
        predictions = predictions.reshape(
            -1, X.shape[0], *[val.shape[0] for val in values]
        )
    else:
        averaged_predictions = _partial_dependence_recursion(
            estimator, grid, features_indices
        )

    # reshape averaged_predictions to
    # (n_outputs, n_values_feature_0, n_values_feature_1, ...)
    averaged_predictions = averaged_predictions.reshape(
        -1, *[val.shape[0] for val in values]
    )

    if kind == "average":
        return Bunch(average=averaged_predictions, values=values)
    elif kind == "individual":
        return Bunch(individual=predictions, values=values)
    else:  # kind='both'
        return Bunch(
            average=averaged_predictions,
            individual=predictions,
            values=values,
        )
