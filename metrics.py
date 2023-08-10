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


def accuracy_plot(ensembled_model, X, y, library="Flaml"):
    """Accuracy metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    accuracy_plot(model_class, X_class, y_class)
    """

    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        accuracy = [accuracy_score(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            y_pred_class = model.predict(X_transform)
            y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
            accuracy.append(accuracy_score(y, y_pred_class_name))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        accuracy = [accuracy_score(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            accuracy.append(accuracy_score(y, ensembled_model.predict(X)))
        ensembled_model.set_model_best(final_model)

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


def precision_plot(ensembled_model, X, y, library="Flaml"):
    """Precision metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    precision_plot(model_class, X_class, y_class)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        precision = [precision_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            y_pred_class = model.predict(X_transform)
            y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
            precision.append(precision_score(y, y_pred_class_name, average='micro'))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        precision = [precision_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            precision.append(precision_score(y, ensembled_model.predict(X), average='micro'))
        ensembled_model.set_model_best(final_model)

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


def recall_plot(ensembled_model, X, y, library="Flaml"):
    """Recall metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    recall_plot(model_class, X_class, y_class)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_

        recall = [recall_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            y_pred_class = model.predict(X_transform)
            y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
            recall.append(recall_score(y, y_pred_class_name, average='micro'))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        recall = [recall_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            recall.append(recall_score(y, ensembled_model.predict(X), average='micro'))
        ensembled_model.set_model_best(final_model)

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


def f1_score_plot(ensembled_model, X, y, library="Flaml"):
    """F1 metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    f1_score_plot(model_class, X_class, y_class)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        f1 = [f1_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            y_pred_class = model.predict(X_transform)
            y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
            f1.append(f1_score(y, y_pred_class_name, average='micro'))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        f1 = [f1_score(y, ensembled_model.predict(X), average='micro')]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            f1.append(f1_score(y, ensembled_model.predict(X), average='micro'))
        ensembled_model.set_model_best(final_model)

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


def mape_plot(ensembled_model, X, y, library="Flaml"):
    """MAPE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mape_plot(model_reg, X_reg, y_reg)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        mape = [mean_absolute_percentage_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            mape.append(mean_absolute_percentage_error(y, model.predict(X_transform)))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        mape = [mean_absolute_percentage_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            mape.append(mean_absolute_percentage_error(y, ensembled_model.predict(X)))
        ensembled_model.set_model_best(final_model)

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


def mae_plot(ensembled_model, X, y, library="Flaml"):
    """MAE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mae_plot(model_reg, X_reg, y_reg)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        mae = [mean_absolute_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            mae.append(mean_absolute_error(y, model.predict(X_transform)))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        mae = [mean_absolute_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            mae.append(mean_absolute_error(y, ensembled_model.predict(X)))
        ensembled_model.set_model_best(final_model)

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


def mse_plot(ensembled_model, X, y, library="Flaml"):
    """MSE metrics plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    mse_plot(model_reg, X_reg, y_reg)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        mse = [mean_squared_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            mse.append(mean_squared_error(y, model.predict(X_transform)))
            models_name.append(type(model).__name__)
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        mse = [mean_squared_error(y, ensembled_model.predict(X))]
        models_name = ['Ensemble']

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            models_name.append(model_name)
            ensembled_model.set_model_best(model_name)
            mse.append(mean_squared_error(y, ensembled_model.predict(X)))
        ensembled_model.set_model_best(final_model)

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


def permutation_feature_importance_all(ensembled_model, X, y, library="Flaml", task="regression"):
    """Permutation feature importance plots of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    task : string
        task type, "regression" or "classification"

    Returns
    -------
    plots : list of plotly.graph_objs._figure.Figure object
        plotly plot list

    Examples
    --------
    permutation_feature_importance_all(model_reg, X_reg, y_reg, task="regression")
    """
    if library == "Flaml":
        plots = [permutation_feature_importance(ensembled_model, X, y, 'Ensemble')]

        ensemble_models = ensembled_model.model.estimators_
        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            if task == "regression":
                plots.append(permutation_feature_importance(model, X_transform, y, type(model).__name__))
            if task == "classification":
                plots.append(permutation_feature_importance(model,
                                                            X_transform,
                                                            ensembled_model._label_transformer.transform(y),
                                                            type(model).__name__))
    if library == "AutoGluon":
        if task == "regression":
            autogluon_task = "regression"
        else:
            autogluon_task = "classification"
        plots = [permutation_feature_importance(ensembled_model, X, y, 'Ensemble', autogluon_task)]

        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            ensembled_model.set_model_best(model_name)
            plots.append(permutation_feature_importance(ensembled_model, X, y, model_name, autogluon_task))
        ensembled_model.set_model_best(final_model)

    return plots


def permutation_feature_importance(model, X, y, name, autogluon_task=False):
    """Permutation feature importance plot of individual models from the ensemled model.

    Parameters
    ----------
    model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    name : string
        model name

    autogluon_task : boolean or string
        If library is not autogluon then it shoud be False. Otherwise it shoud be
        "regression" or "classification" depend to the task.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    permutation_feature_importance(model, X_transform, y, type(model).__name__)
    """
    if autogluon_task == False:
        r = permutation_importance(model, X, y)
    elif autogluon_task == "regression":
        r = permutation_importance(model, X, y, scoring='r2')
    elif autogluon_task == "classification":
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


def correlation_plot(ensembled_model, X, library="Flaml", task="regression"):
    """Prediction correlation plot of models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    task : string
        task type, "regression" or "classification"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    correlation_plot(model_reg, X_reg, y_reg, task="regression")
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        predict_data = {'Ensemble': ensembled_model.predict(X)}

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            if task == "regression":
                predict_data[type(model).__name__] = model.predict(X_transform)
            if task == "classification":
                y_pred_class = model.predict(X_transform)
                y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
                predict_data[type(model).__name__] = y_pred_class_name
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        predict_data = {'Ensemble': ensembled_model.predict(X)}

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            ensembled_model.set_model_best(model_name)
            predict_data[model_name] = ensembled_model.predict(X)
        ensembled_model.set_model_best(final_model)

    predict_data = pd.DataFrame(predict_data)
    if task == "regression":
        corr_matrix = predict_data.corr().round(2)
    if task == "classification":
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


def prediction_compare_plot(ensembled_model, X, y, library="Flaml", task="regression"):
    """Prediction compare plot of models from the ensemled model.
        For classification plot show if prediction of model is correct or incorrect.
        For regression it shows the difference between the prediction and the true
        value expressed as a percentage.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data

    y : dataframe
        y data

    library : string
        model library "Flaml", "AutoGluon" or "AutoSklearn"

    task : string
        task type, "regression" or "classification"

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        plotly plot

    Examples
    --------
    prediction_compare_plot(model_reg, X_reg, y_reg, library="Flaml", task="regression")
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        predict_data = {'Ensemble': ensembled_model.predict(X)}

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            if task == "regression":
                predict_data[type(model).__name__] = model.predict(X_transform)
            if task == "classification":
                y_pred_class = model.predict(X_transform)

                y_pred_class_name = ensembled_model._label_transformer.inverse_transform(y_pred_class)
                predict_data[type(model).__name__] = y_pred_class_name
    if library == "AutoGluon":
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']
        predict_data = {'Ensemble': ensembled_model.predict(X)}

        final_model = ensembled_model.get_model_best()
        for model_name in ensemble_models:
            ensembled_model.set_model_best(model_name)
            predict_data[model_name] = ensembled_model.predict(X)
        ensembled_model.set_model_best(final_model)

    plot_value = {}
    if task == "regression":
        for name, pred in predict_data.items():
            plot_value[name] = [(i - j) / (j + 0.0000001) * 100 for i, j in zip(pred, y)]
    if task == "classification":
        for name, pred in predict_data.items():
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
    if task == "classification":
        discrete_nonuniform = [[0, 'rgb(242,26,155)'],
                               [0.5, 'rgb(242,26,155)'],
                               [0.5, 'rgb(125,179,67)'],
                               [1, 'rgb(125,179,67)'],
                               ]

    fig = px.imshow(plot_value, text_auto=True, color_continuous_scale=discrete_nonuniform)
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
    if task == "classification":
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


def partial_dependence_plots(ensembled_model, X, library="Flaml", autogluon_task=False):
    """Permutation feature importance plot of individual models from the ensemled model.

    Parameters
    ----------
    ensembled_model : Flaml, AutoGluon or AutoSklearn ensembled model.

    X : dataframe
        x data.

    library : string
        library name, "Flaml", "AutoGluon" or "AutoSklearn"

    autogluon_task : boolean or string
        If library is not autogluon then it shoud be False. Otherwise it shoud be
        "regression" or "classification" depend to the task  .

    Returns
    -------
    fig : list of plotly.graph_objs._figure.Figure
        list of plotly plot

    Examples
    --------
    partial_dependence_plots(model_reg, X_reg, y_reg)
    """
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)

        model_name = {}
        columns = []
        values = {}

        for i in range(X.shape[1]):
            try:
                values[X.columns[i]] = [partial_dependence(ensembled_model, X, [i])['average'][0]]
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
        ensemble_models = ensembled_model.info()['model_info'][ensembled_model.get_model_best()]['stacker_info'][
            'base_model_names']

        model_name = {}
        columns = []
        values = {}

        if autogluon_task == "classification":
            ensembled_model._estimator_type = "classifier"
        else:
            ensembled_model._estimator_type = "regressor"
        ensembled_model.classes_ = ["a"]

        for i in range(X.shape[1]):
            try:
                values[X.columns[i]] = [partial_dependence(ensembled_model, X, [i])['average'][0]]
                model_name[X.columns[i]] = ['Ensemble']
                columns.append(X.columns[i])
            except TypeError:
                pass

        final_model = ensembled_model.get_model_best()
        for model in ensemble_models:
            ensembled_model.set_model_best(model)
            for i in range(X.shape[1]):
                try:
                    values[X.columns[i]].append(partial_dependence(ensembled_model, X, [i])['average'][0])
                    model_name[X.columns[i]].append(model)
                except Exception:
                    pass
        ensembled_model.set_model_best(final_model)

    plots = []
    for variable in columns:
        plot_x_value = sorted(X[variable].unique())
        plots.append(partial_dependence_line_plot(values[variable], plot_x_value, model_name[variable], variable))

    return plots


def partial_dependence_line_plot(y_values, x_values, model_names, name):
    """partial dependence one plot

    Parameters
    ----------
    y_values : dataframe
        x data.

    x_values : dataframe
        y data.

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
