import base64
import io
import pandas as pd
import zipfile
from autogluon.tabular import TabularPredictor
from dash import html

def parse_data(contents):
    """
    Parse data from a base64 encoded string.

    Parameters
    ----------
    contents : str
        A string containing the content type and content separated by a comma.
        The content should be base64 encoded data representing a CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame parsed from the decoded CSV content.

    Notes
    -----
    This function expects the contents parameter to be in the format: "content_type, content_string",
    where content_type specifies the type of content and content_string is a base64 encoded CSV data.

    Raises
    ------
    Exception
        If there is an error while processing the file, an error message will be printed, and
        an HTML Div with an error message will be returned.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep = ',|;|\t', engine='python')
        return df

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

def parse_model(contents, filename):
    """
    Parse a machine learning model from a base64 encoded string.

    Parameters
    ----------
    contents : str
        A string containing the content type and content separated by a comma.
        The content should be base64 encoded data representing a machine learning model file.

    filename : str
        The filename associated with the contents.

    Returns
    -------
    tuple
        A tuple containing the parsed machine learning model and the library used to create the model.

    Notes
    -----
    This function expects the contents parameter to be in the format: "content_type, content_string",
    where content_type specifies the type of content and content_string is a base64 encoded machine learning model.
    The filename parameter is used to determine the appropriate action based on the file extension.

    Raises
    ------
    Exception
        If there is an error while processing the file, an error message will be printed, and
        an HTML Div with an error message will be returned.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if ".pkl" in filename:
            model = pd.read_pickle(io.BytesIO(decoded))
            if "<class 'flaml" in str(model.__class__).split("."):
                library = "Flaml"
            else:
                library = "AutoSklearn"
            return model, library
        elif ".zip" in filename:
            with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
                zip_ref.extractall('./uploaded_model')
            model = TabularPredictor.load('./uploaded_model', require_py_version_match=False)
            library = "AutoGluon"
            return model, library

    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

def get_task_from_model(ensemble_model, y, library):
    """Function that recognise machine learning task performed by ensemble model.

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    y : target variable vector

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    task {'regression', 'classification', 'multiclass'}
        string that specifies the model task

    Examples
    --------
    get_task_from_model(ensemble_model, y, library)
    """
    if library == 'AutoGluon':
        task = ensemble_model.info()['problem_type']
        if task == 'binary':
            return 'classification'
        return task
    elif library == 'Flaml':
        if y.squeeze().nunique() > 10:
            return 'regression'
        elif y.squeeze().nunique() == 2:
            return 'classification'
        else:
            return 'multiclass'
    elif library == 'AutoSklearn':
        task = ensemble_model.get_models_with_weights()[0][1].__dict__['dataset_properties']['target_type']
        if task == 'classification':
            if y.squeeze().nunique() > 2:
                return 'multiclass'
        return task

def get_predictions_from_model(ensemble_model, X, y, library, task):
    """Function that calculates predictions of component models from given ensemble
    model

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X, y : dataframe

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    task : {'regression', 'binary classification', 'multiclass classification'}
        string that specifies the machine learning task

    Returns
    -------
    predictions: dictionary
        of form {'model_name' : 'prediction_vector'}

    Examples
    --------
    get_predictions_from_model(ensemble_model, X, y, library, task)
    """
    predictions = {}
    if library == "Flaml":
        ensemble_models = ensemble_model.model.estimators_
        predictions['Ensemble'] = ensemble_model.predict(X)
        X_transform = ensemble_model._state.task.preprocess(X, ensemble_model._transformer)
        if task == 'regression':
            for model in ensemble_models:
                predictions[type(model).__name__] = model.predict(X_transform)
        else:
            for model in ensemble_models:
                y_pred_class = model.predict(X_transform)
                y_pred_class_name = ensemble_model._label_transformer.inverse_transform(y_pred_class)
                predictions[type(model).__name__] = y_pred_class_name
    elif library == "AutoGluon":
        ensemble_models = ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']
        predictions['Ensemble'] = ensemble_model.predict(X)
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_models:
            ensemble_model.set_model_best(model_name)
            predictions[model_name] = ensemble_model.predict(X)
        ensemble_model.set_model_best(final_model)

    elif library == "AutoSklearn":
        predictions['Ensemble'] = ensemble_model.predict(X)
        if task == 'regression':
            for weight, model in ensemble_model.get_models_with_weights():
                prediction = model.predict(X)
                name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
                if name in predictions.keys():
                    i = 1
                    name_new = name + "_" + str(i)
                    while name_new in predictions.keys():
                        i += 1
                        name_new = name + "_" + str(i)
                    predictions[name_new] = prediction
                else:
                    predictions[name] = prediction
        else:
            class_names = list(y.unique())
            class_names.sort()
            for weight, model in ensemble_model.get_models_with_weights():
                prediction = model.predict(X)
                prediction_class = [class_names[idx] for idx in prediction]
                name = str(type(model._final_estimator.choice)).split('.')[-1][:-2]
                if name in predictions.keys():
                    i = 1
                    name_new = name + "_" + str(i)
                    while name_new in predictions.keys():
                        i += 1
                        name_new = name + "_" + str(i)
                    predictions[name_new] = prediction_class
                else:
                    predictions[name] = prediction_class

    return predictions

def get_probability_pred_from_model(ensemble_model, X, library):
    """Function that calculates probability of belonging to a class from
     component models from given ensemble model

    Parameters
    ----------
    ensemble_model : Flaml, AutoGluon or AutoSklearn ensemble model.

    X : dataframe without target variable

    library : {'Flaml', 'AutoGluon', 'AutoSklearn'}
            string that specifies the model library

    Returns
    -------
    predictions: list containing vectors of probability of belonging to a class

    Examples
    --------
    get_probabilty_pred_from_model(ensemble_model, X, library)
    """
    proba_predictions = []
    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            prediction = model.predict_proba(X)
            proba_predictions.append(prediction.tolist())
    elif library == "AutoGluon":
        final_model = ensemble_model.get_model_best()
        for model_name in ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['stacker_info'][
            'base_model_names']:
            ensemble_model.set_model_best(model_name)
            proba_predictions.append(ensemble_model.predict_proba(X).values.tolist())
        ensemble_model.set_model_best(final_model)
    return proba_predictions

def get_ensemble_weights(ensemble_model, library):
    """
    Retrieve the weights of individual models in an ensemble.

    Parameters
    ----------
    ensemble_model : object
      The ensemble model object containing multiple models.

    library : str
      The name of the library/framework used to create the ensemble model.
      Supported values are "AutoSklearn" or "AutoGluon".

    Returns
    -------
    list: A list containing the weights of individual models in the ensemble.
    """
    weights = []

    if library == "AutoSklearn":
        for weight, model in ensemble_model.get_models_with_weights():
            weights.append(weight)
    elif library == "AutoGluon":
        weights = list(ensemble_model.info()['model_info'][ensemble_model.get_model_best()]['children_info']['S1F1'][
                           'model_weights'].values())

    return weights