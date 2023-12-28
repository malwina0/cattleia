from typing import Iterable
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
from scipy import sparse
from scipy.stats._mstats_basic import mquantiles
from sklearn.base import is_regressor, is_classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import BaseHistGradientBoosting
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import _safe_assign, _safe_indexing, check_array, _determine_key_type, _get_column_indices, Bunch
from sklearn.utils.extmath import cartesian

from components.metrics import empty_fig
from utils.plots_layout import matrix_layout


def partial_dependence_plots(ensemble_model, X, library="FLAML", task="regression"):
    """Genetarte partial dependence plots for features based on the provided ensemble model.

    Parameters
    ----------
    ensemble_model : FLAML, AutoGluon or Auto-sklearn ensemble model.

    X : pandas.DataFrame
       The input data containing features for which partial dependence plots will be generated.

    library : str, optional
       The library used for the ensemble model. Default is "FLAML".

    task : {'classification', 'regression', 'multiclass'}
        'classification', 'regression' or 'multiclass', depends on the task.

    Returns
    -------
    dict
       A dictionary containing partial dependence plots for each feature in the input data.
       The keys represent feature names, and the values are the corresponding plots.
    """
    model_name = {}
    columns = []
    values = {}

    if library == "FLAML":
        ensemble_models = ensemble_model.model.estimators_
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)
        update_partial_dependence_info(X, ensemble_model, values, model_name, columns)

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

        if task == "classification":
            ensemble_model._estimator_type = "classifier"
        else:
            ensemble_model._estimator_type = "regressor"
        ensemble_model.classes_ = ["a"]

        update_partial_dependence_info(X, ensemble_model, values, model_name, columns)

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

    elif library == "Auto-sklearn":
        update_partial_dependence_info(X, ensemble_model, values, model_name, columns)
        for weight, model in ensemble_model.get_models_with_weights():
            for i in range(X.shape[1]):
                try:
                    if task == "classification":
                        values[X.columns[i]].append(partial_dependence_custom(model, X, [i])['average'][0])
                    else:
                        values[X.columns[i]].append(partial_dependence(model, X, [i])['average'][0])
                    name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
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

    plots = {}
    for variable in columns:
        plot_x_value = sorted(X[variable].unique())
        plots[variable] = partial_dependence_line_plot(values[variable], plot_x_value, model_name[variable], variable)

    return plots


def partial_dependence_line_plot(y_values, x_values, model_names, name):
    """
    Generate a line plot illustrating the partial dependence of a variable.

    Parameters
    ----------
    y_values : list
        List of y-axis values (partial dependence values).

    x_values : list
        List of x-axis values (feature values).

    model_names : list
        List of model names corresponding to each line in the plot.

    name : str
        The name of the variable for which partial dependence is plotted.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly figure representing the partial dependence line plot.
    """
    fig = empty_fig()
    fig.update_layout(matrix_layout,
        title=f"{name} variable partial dependence plot"
    )

    for line_value, model_name in zip(y_values, model_names):
        if model_name == 'Ensemble':
            fig.add_trace(go.Scatter(x=x_values, y=line_value, mode='lines', name=model_name, line=dict(color='#ffaef4', width=5)))
        else:
            fig.add_trace(go.Scatter(x=x_values, y=line_value, mode='lines', name=model_name))
    return fig

def update_partial_dependence_info(X, ensemble_model, values, model_name, columns):
    for i in range(X.shape[1]):
        try:
            values[X.columns[i]] = [partial_dependence(ensemble_model, X, [i])['average'][0]]
            model_name[X.columns[i]] = ['Ensemble']
            columns.append(X.columns[i])
        except TypeError:
            pass

# Code below is modified code from scikit-learn library to operate Auto-sklearn partial dependence
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


def calculate_pfi(model, X, y, task):
    """
    Calculate permutation feature importances for a given model.

    Parameters:
    model: Model object
        The machine learning model for which feature importances are to be calculated.

    X: DataFrame
        The input features.

    y: Series or array-like
        The target variable.

    task: str or bool
        The task type ('regression', 'classification', 'multiclass', or False).

    Returns:
    importances_mean: array-like
        The mean feature importances.
    """
    if task == False:
        r = permutation_importance(model, X, y)
    elif task == "regression":
        r = permutation_importance(model, X, y, scoring='r2')
    elif task == "classification" or task == "multiclass":
        r = permutation_importance(model, X, y, scoring='accuracy')
    return r.importances_mean


def prepare_feature_importance(ensemble_model, X, y, library="FLAML", task="regression"):
    """
    Calculate permutation feature importance for an ensemble model.

    Parameters:
    ensemble_model: Model object
        The ensemble model.

    X: DataFrame
        The input features.

    y: Series or array-like
        The target variable.

    library: str, optional (default="FLAML")
        The library used for the ensemble model: "FLAML", "AutoGluon", or "Auto-sklearn".

    task: str, optional (default="regression")
        The task type: "regression", "classification", or "multiclass".

    Returns:
    DataFrame
        A DataFrame containing feature importances for different models in the ensemble.
    """
    pfi = {'variable': X.columns, 'Ensemble': calculate_pfi(ensemble_model, X, y, task)}

    if library == "FLAML":
        ensemble_models = ensemble_model.model.estimators_
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)
        for model in ensemble_models:
            if task == "regression":
                pfi[type(model).__name__] = calculate_pfi(model, X_transform, y, task)
            else:
                pfi[type(model).__name__] = calculate_pfi(model, X_transform,
                                                          ensemble_model._label_transformer.transform(y), task)

    elif library == "AutoGluon":
        ensemble_models = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_models:
            ensemble_model.set_model_best(model_name)
            pfi[model_name] = calculate_pfi(ensemble_model, X, y, task)
        ensemble_model.set_model_best(final_model)

    elif library == "Auto-sklearn":
        if task == "classification" or "multiclass":
            class_name = y.unique()
            class_index = {name: idx for idx, name in enumerate(class_name)}
            y_class_index = [class_index[y_elem] for y_elem in y]
        unique_names = []
        for weight, model in ensemble_model.get_models_with_weights():
            model_name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
            if model_name in unique_names:
                i = 1
                model_name_new = model_name + "_" + str(i)
                while model_name_new in unique_names:
                    i += 1
                    model_name_new = model_name + "_" + str(i)
                unique_names.append(model_name_new)
                model_name = model_name_new
            else:
                unique_names.append(model_name)
            if task == "classification" or task == "multiclass":
                pfi[model_name] = calculate_pfi(model, X, y_class_index, task)
            else:
                pfi[model_name] = calculate_pfi(model, X, y, task)
    df = pd.DataFrame(pfi)

    return df


def feature_importance_plot(df):
    """
    This function creates a plot displaying the feature importance for different models.

    Parameters:
    df: DataFrame
      DataFrame containing model feature importances.

    Returns:
    fig: plotly.graph_objs._figure.Figure
      Plotly figure displaying the model feature importance plot.
    """
    df_melted = df.melt(id_vars='variable', var_name='Model', value_name='Value')
    fig = empty_fig()
    for model, group in df_melted.groupby('Model'):
        fig.add_trace(go.Bar(
            x=group['variable'],
            y=group['Value'],
            name=model,
            hovertemplate="<b>%{customdata}</b><br>Value: %{y:.3f}<extra></extra>",
            customdata=group['Model'],
            visible=True if model == 'Ensemble' else 'legendonly'
        ))
    fig.update_traces(marker_color='#ffaef4', selector=dict(name='Ensemble'))
    return fig