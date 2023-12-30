import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error as mse


def mean_squared_difference(pred1, pred2):
    """Calculates mean squared difference of two prediction vectors, which is
    measure of similarity between two regression models prediction.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction

    Returns
    -------
    Numeric value
        MSD value of given prediction pair
    """
    return mse(pred1, pred2)


def root_mean_squared_difference(pred1, pred2):
    """Calculates root of mean squared difference of two prediction vectors,
    which is measure of similarity between two regression models predictions.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction

    Returns
    -------
    Numeric value
        RMSD value of given predictions pair
    """
    return mse(pred1, pred2, squared=False)


def root_mean_sqaured_error_error_penalty(pred1, pred2, y):
    """Calculates root of mean squared difference of two prediction vectors
    with additional penalty from prediction error considering true values

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        RMSDEP value of given predictions pair and true values vector
    """
    return np.sqrt(mse(pred1, pred2) + mse(pred1, y) / 2 + mse(pred2, y) / 2)

def conjunctive_rmse(pred1, pred2, y):
    """Calculates Root Mean Squared Difference between real values and prediction
    being average of two model outputs.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Conjunctive RMSE value of given predictions pair and true values vector
    """
    pred = []
    for i in range(len(pred1)):
        pred.append((pred1[i]+pred2[i])/2)
    return mse(pred, y, squared=False)


def strong_disagreement_ratio(pred1, pred2, y):
    """Calculates percentage of observations that were predicted strongly
    different by two models.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        SDR value of given predictions pair and true values vector
    """
    pred1, pred2 = np.array(pred1), np.array(pred2)
    difference = np.abs(pred1 - pred2)
    standard_deviation = np.std(y)
    return np.sum(difference > standard_deviation) / len(pred1)

def agreement_ratio(pred1, pred2, y):
    """Calculates percentage of observations that were predicted very
     similarly by two models.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        AR value of given predictions pair and true values vector
    """
    pred1, pred2 = np.array(pred1), np.array(pred2)
    difference = np.abs(pred1 - pred2)
    standard_deviation = np.std(y)
    threshold = standard_deviation / 50
    return np.sum(difference < threshold) / len(pred1)

def uniformity(pred1, pred2):
    """Calculates uniformity of two prediction vectors, which is a measure
    of similarity between two classification models predictions.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Uniformity value of given predictions pair
    """
    return accuracy_score(pred1, pred2)


def disagreement_ratio(pred1, pred2):
    """Calculates percentage of observations where two classification models
    predicted different value

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
       numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
       numeric vector representing results of model prediction

    Returns
    -------
    Numeric value
       DR value of given prediction vectors
    """
    count = sum(1 for p1, p2 in zip(pred1, pred2) if p1 != p2)
    return count/len(pred1)

def disagreement_postive_ratio(pred1, pred2, y, positive=None):
    """Calculates percentage of observations where two classification models
    predicted different value in relation to originally positive values.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable
    positive: value, optional
        value that represents a positive observation in variable

    Returns
    -------
    Numeric value
        DPR value of given predictions pair and true values vector
    """
    if positive is None:
        positive = y.squeeze().unique()[1]
    T = 0
    T_disagreement = 0
    for i in range(len(y)):
        if y[i] == positive:
            T += 1
            if pred1[i] != pred2[i]:
                T_disagreement += 1
    return T_disagreement/T

def conjunctive_accuracy(pred1, pred2, y):
    """Calculates conjunctive accuracy of two prediction vectors
    and true values of variable.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Conjunctive accuracy value of given predictions pair
        and true values vector
    """
    same = sum(1 for p1, p2, yy in zip(pred1, pred2, y) if p1 == p2 == yy)
    return same/len(y)

def conjunctive_precission(pred1, pred2, y, positive=None):
    """Calculates conjunctive precision of two prediction vectors
    and true values of variable.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable
    positive: value, optional
        value that represents a positive observation in variable

    Returns
    -------
    Numeric value
        Conjunctive precision value of given predictions pair
        and true values vector
    """
    if positive is None:
        positive = y.squeeze().unique()[1]
    TTP = 0
    FFN = 0
    for i in range(len(y)):
        if pred1[i] == positive and pred2[i] == positive:
            if y[i] == positive:
                TTP += 1
            else:
                FFN += 1
    if TTP+FFN == 0:
        return 0
    return TTP/(TTP+FFN)

def conjunctive_recall(pred1, pred2, y, positive=None):
    """ Calculates conjunctive recall of two prediction vectors
    and true values of variable.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable
    positive: value, optional
        value that represents a positive observation in variable

    Returns
    -------
    Numeric value
        Conjunctive recall value of given predictions pair
        and true values vector
    """
    if positive is None:
        positive = y.squeeze().unique()[1]
    TTP = 0
    P = 0
    for i in range(len(y)):
        if y[i] == positive:
            P += 1
            if pred1[i] == positive and pred2[i] == positive:
                TTP += 1
    return TTP/P

def correctness_counter(pred1, pred2, y):
    """ Calculates count of observationes predicted on different level
    of correctness: predicted correctly by both, predicted correctly
    by only one model and predicted wrong by both.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    3 numeric values representing percentage of correctly, semi-correctly
     and wrong predicted observations
    """
    double_correct = 0
    disagreement = 0
    double_wrong = 0
    n_observations = len(y)

    for i in range(n_observations):
        if pred1[i] == pred2[i]:
            if pred1[i] == y[i]:
                double_correct += 1
            else:
                double_wrong += 1
        else:
            if pred1[i] == y[i] or pred2[i] == y[i]:
                disagreement += 1
            else:
                double_wrong += 1
    return double_correct/n_observations, disagreement/n_observations, double_wrong/n_observations


def average_collective_score(pred1, pred2, y, positive=None):
    """ Calculates collective score of two models predictions by summing
    number of observations with weights corresponding to its prediciton
    correctness. When both models predicted correctly, observation is given
    weight 1, when only one model predicted right, the weight is equal to 0.5, otherwise
    observation is counted with weight 0. Sum is divided by number of observations.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Average collective score value of given predictions pair
        and true values vector

    """
    if positive is None:
        positive = y.squeeze().unique()[1]
    TTP_TTN = 0
    FFP_FFN = 0
    n_observations = len(y)
    weights = []
    for i in range(len(y)):
        if y[i] == positive:
            if pred1[i] == positive and pred2[i] == positive:
                TTP_TTN += 1
                weights = weights + [1]
            elif pred1[i] != positive and pred2[i] != positive:
                FFP_FFN += 1
                weights = weights + [0]
            else:
                weights = weights + [0.5]

        else:
            if pred1[i] == positive and pred2[i] == positive:
                FFP_FFN += 1
                weights = weights + [0]
            elif pred1[i] != positive and pred2[i] != positive:
                TTP_TTN += 1
                weights = weights + [1]
            else:
                weights = weights + [0.5]
    return (TTP_TTN + 0.5*(n_observations - TTP_TTN - FFP_FFN))/n_observations, weights

def macro_conjunctive_precission(pred1, pred2, y):
    """Calculates macro conjunctive precission for multiclass classification task,
    which is averaged conjunctive recall score for every class in variable

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Macro conjunctive precission value of given predictions pair
        and true values vector
    """
    classes = y.squeeze().unique()
    precisions = []
    for clas in classes:
        precisions.append(conjunctive_precission(pred1, pred2, y, positive=clas))
    return np.array(precisions).mean()

def macro_conjunctive_recall(pred1, pred2, y):
    """Calculates macro conjunctive recall for multiclass classification task,
    which is averaged conjunctive recall score for every class in variable y

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Macro conjunctive recall value of given predictions pair
        and true values vector
    """
    classes = y.squeeze().unique()
    recalls = []
    for clas in classes:
        recalls.append(conjunctive_recall(pred1, pred2, y, positive=clas))
    return np.array(recalls).mean()

def weighted_conjunctive_precission(pred1, pred2, y):
    """Calculates weighted conjunctive precission for multiclass classification task,
    which is conjunctive recall score for every class in variable y with averaged
    with weights corresponding to cardinality of each class.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Macro conjunctive precission value of given predictions pair
        and true values vector
    """
    count = y.value_counts()
    classes = y.squeeze().unique()
    result = 0
    for clas in classes:
        result += conjunctive_precission(pred1, pred2, y, positive=clas)*count[clas]
    return result/len(y)

def weighted_conjunctive_recall(pred1, pred2, y):
    """Calculates weighted conjunctive recall for multiclass classification task,
    which is conjunctive recall score for every class in variable y averaged
    with weights corresponding to cardinality of each class.

    Parameters
    ----------
    pred1 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    pred2 : list, numpy.array, pandas.series
        numeric vector representing results of model prediction
    y: list, numpy.array, pandas.series
        true values of predicted variable

    Returns
    -------
    Numeric value
        Macro conjunctive recall value of given predictions pair
        and true values vector
    """
    count = y.value_counts()
    classes = y.squeeze().unique()
    result = 0
    for clas in classes:
        result += conjunctive_recall(pred1, pred2, y, positive=clas)*count[clas]
    return result/len(y)