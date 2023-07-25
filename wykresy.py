from flaml import AutoML
from sklearn import datasets
from flaml import AutoML
import pandas as pd

import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.inspection import PartialDependenceDisplay


def mse_plot(ensembled_model, X, y, library="Flaml"):
    if library == "Flaml":
        ensemble_models = ensembled_model.model.estimators_
        mse = [mean_squared_error(y, ensembled_model.predict(X))]
        models_name = ['ensemble']

        X_transform = ensembled_model._state.task.preprocess(X, ensembled_model._transformer)
        for model in ensemble_models:
            mse.append(mean_squared_error(y, model.predict(X_transform)))
            models_name.append(type(model).__name__)

    fig = go.Figure()
    for i in range(len(mse)):
        fig.add_trace(go.Bar(x=[i], y=[mse[i]], name=models_name[i]))

    fig.update_traces(marker=dict(color='blue'))
    fig.update_layout(
        title="MSE",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(mse))),
            ticktext=models_name
        ),
        showlegend=False
    )

    return fig