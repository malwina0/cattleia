from dash import html, dcc


ann_metrics_prediction_compare = html.Div([
        "The matrix compares model predictions with actual values for each observation.",
        html.Ul([
            html.Li([html.Strong("Classification "), html.A("task: color indicates prediction correctness.")]),
            html.Li([html.Strong("Regression "),  html.A("task: color indicates percentage difference between predicted and actual value.")])
        ])
    ], className="annotation_str", id="ann_metrics_prediction_compare")

ann_weights_metrics = html.Div([
    html.H4(['Individual model metrics'], className='annotation-title'),
    html.P(["A table displays weight values alongside task-specific metrics for each individual model."]),
    html.Ul([
        html.Li([
            html.Strong("Modifying"), html.A(" values in the "),
            html.Strong("'Weights' column"), html.A(" directly alters a model's weight.")]
        ),
        html.Li([
            html.A("Any adjustments made will proportionally modify other weights to ensure their "),
            html.Strong("sum remains at 1.")]
        )
    ])
], className="annotation_str", id="ann_weights_metrics")

ann_weights_sliders = html.Div([
    html.H4(['Sliders'], className='annotation-title'),
    html.P(["Sliders are provided foo adjusting the weight assigned to each model within the ensemble."]),
    html.Ul([
        html.Li([html.A("Initially, these values are pre-selected by the AutoML package.")]),
        html.Li([html.A("To restore these default values, click the 'Reset weights' button.")])
    ])
], className="annotation_str", id="ann_weights_sliders")

ann_weights_ensemble = html.Div([
    html.H4(['Ensemble Model Metrics'], className='annotation-title'),
    html.A(["In the table, metrics for both the "]),
    html.Strong(["ensemble model's original weights"]),
    html.A([" (set by AutoML) and "]),
    html.Strong(["custom weights"]),
    html.A([" (set manually) are presented side by side."])
], className="annotation_str", id="ann_weights_ensemble")

ann_xai_feature_importance = html.Div([
    "The feature importance plot illustrates the significance of different features in a model's predictions.",
    html.Ul([
        html.Li([html.Strong("Default display "), html.A("The ensemble's feature importance chart is shown automatically.")]),
        html.Li([html.Strong("Comparing multiple models: "),  html.A("To compare additional models, click their names in the chart legend.")])
    ])
], className="annotation_str", id="ann_xai_feature_importance")

ann_xai_partial_dep = html.Div([
    "Partial Dependence isolates the impact of a single feature on the model's output while keeping all other features constant.",
    html.Ul([
        html.Li(["It illustrates how the model's output shifts as the chosen feature varies."]),
        html.Li(["With a large number of observations, to expedite graph generation, calculations are performed using a subset of the data."])
    ])
    ], className="annotation_str", id="ann_xai_partial_dep")

ann_comp_overview = html.Div([
    html.H2(['Compatimetrics'], className='annotation-title'),
    html.A("Compatimetrics are novel indicators of models compatibility and similarity."),
    html.Ul([
        html.Li([html.A("They show the "), html.Strong("distance between prediction vectors"), html.A(" from different base models.")]),
        html.Li(["They are developed on the basis of simple heuristics and corresponding evaluation metrics."])
    ])
], className="annotation_str", id="ann_comp_overview")

ann_comp_uniformity = html.Div([
    html.A("Uniformity counts the "), html.Strong("percentage of observations that both models predicted the same. "),
    html.A("It indicates how similar models are but does not reflect their quality.")
], className="annotation_str", id="ann_comp_uniformity")

ann_comp_incompatibility = html.Div([
    html.A("Incompatibility is an "), html.Strong("opposite of uniformity"),
    html.A(", as it counts the percentage of observations that were predicted differently by models.")
], className="annotation_str", id="ann_comp_incompatibility")

ann_comp_acs = html.Div([
    html.A("Average Collective Score helps "), html.Strong("compare model pairs by considering their correctness"),
    html.A("."), html.Br(),
    html.A("It assigns weights to observations based on prediction correctness:"),
    html.Ul([
        html.Li([html.Strong("doubly correct "), html.A("predictions receive a weight of "), html.Strong("1"), html.A(",")]),
        html.Li([html.Strong("disagreements "), html.A("are weighted at "), html.Strong("0.5"), html.A(",")]),
        html.Li([html.A("and "), html.Strong("doubly incorrect "), html.A("predictions are weighted at "), html.Strong("0"), html.A(".")])
    ]),
    dcc.Markdown(r'$$\text{ACS} = \frac{\text{TT} + 0.5 \cdot (\text{TF + FT})}{n}$$', mathjax=True, id="latex-code"),
    html.A("It gives a result between 0 and 1, where a larger value implies greater agreement and correctness.")
], className="annotation_str", id="ann_comp_acs")

ann_comp_conj_acc = html.Div([
    html.A("Conjunctive Accuracy is an "), html.Strong("equivalent of standard accuracy"),
    html.A(", with the following characteristics:"),
    html.Ul([
        html.Li([html.A("it deems double prediction as correct only when both "), html.Strong("models return the correct value"), html.A(",")]),
        html.Li([html.A("calculates the "), html.Strong("percentage "), html.A("of observations where this dual correctness criterion is met.")])
    ]),
    # html.Br(), html.Br(), html.Br(), html.Br(), html.Br()
], className="annotation_str", id="ann_comp_conj_acc")

ann_comp_dis_ratio = html.Div([
    html.H3(['Disagreement ratio'], className='annotation-title'),
    html.A("Disagreement ratio is measuring how many observations were predicted "), html.Strong("differently "),
    html.A("by two models, "), html.Strong("regarding to the record's class."), html.Br(),
    html.A("It can show which class was more difficult to predict when joining models."), html.Br(), html.Br(), html.Br(), html.Br()
], className="annotation_str", id="ann_comp_dis_ratio")

ann_comp_conj_metrics = html.Div([
    html.H3(['Conjunctive metrics'], className='annotation-title'),
    html.A("Conjunctive metrics are similar to standard evaluation metrics, but they consider "),
    html.Strong("two prediction vectors "),html.A("simultaneously."),
    html.Ul([
        html.Li([html.A("A prediction is marked as correct only if both models predict it accurately.")]),
        html.Li([html.Strong("Worth noting: "), html.A("conjunctive recall is generally lower and conjunctive precision"
        " is generally higher, which is related to their definition.")])
    ])
], className="annotation_str", id="ann_comp_conj_metrics")


ann_comp_conj_precision = html.Div([
    html.H3(['Conjunctive precision'], className='annotation-title'),
    html.A("In case of multiclass classification we additionally distinguish weighted and macro versions of precision."),
    html.Br(), html.Strong("Worth noting:"), html.A(" conjunctive precision is generally higher, which is related to "
    "their definition."),
], className="annotation_str", id="ann_comp_conj_precision")

ann_comp_conj_recall = html.Div([
    html.H3(['Conjunctive recall'], className='annotation-title'),
    html.A("In case of multiclass classification we additionally distinguish weighted and macro versions of recall."),
    html.Br(), html.Strong("Worth noting:"), html.A(" conjunctive recall is generally lower, which is related to "
    "their definition."),
], className="annotation_str", id="ann_comp_conj_recall")

ann_comp_pred_corr = html.Div([
    html.H3(['Prediction correctness'], className='annotation-title'),
    html.A("The plot illustrates the breakdown of prediction correctness levels."),
    html.Ul([
        html.Li([html.Strong("Doubly correct "), html.A("predictions happen when both models predict the observation correctly.")]),
        html.Li([html.Strong("Disagreement "), html.A("occurs when one of the models is missing in its prediction.")]),
        html.Li([html.Strong("Doubly incorrect "), html.A("signifies when both models label the observation with the wrong class.")])
    ])
], className="annotation_str", id="ann_comp_pred_corr")

ann_comp_collective = html.Div([
    html.H3(['Cummulative Collective Score'], className='annotation-title'),
    html.A("On the plot below one can observe the process of increasing average collective score through the whole data set."),
    html.Br(), html.A(" This plot can be helpful when searching for areas of data set where prediction was less effective."),
], className="annotation_str", id="ann_comp_collective")

ann_comp_msd = html.Div([
    html.A("MSD calculates the difference between two prediction vectors as a "),
    html.Strong("mean of quadratic difference between all data samples"), html.A("."),
], className="annotation_str", id="ann_comp_msd")

ann_comp_rmsd = html.Div([
    html.A("RMSD is a squared root of MSD.")
], className="annotation_str", id="ann_comp_rmsd")


ann_comp_ar = html.Div([
    html.A("AR represents percentage of observations that were predicted very closely by two different models. "),
    html.Ul([
        html.Li([html.A("The "), html.Strong("bigger "), html.A("the value of AR, the "), html.Strong("more similar "),
                 html.A("outputs of two models are.")]),
        html.Li([html.Strong("Formula: "),
                 dcc.Markdown(r'$$\mathrm{AR} = \frac{\sum^{n}_{i=1}A_{i}}{n}, \,\ \mathrm{where} \,\ A_i = '
                 r'\begin{cases} 0, d_i > \frac{SD(y)}{50} \\ 1, d_i \leq \frac{SD(y)}{50} \end{cases}, \,\ d_i = '
                 r'|\hat{y}_i - \hat{y}_j|.$$', mathjax=True, id="latex-code")]),
        html.Li([html.A("AR "), html.Strong("does not indicate "), html.A("the accuracy of models.")])
    ])
], className="annotation_str", id="ann_comp_ar")

ann_comp_sdr = html.Div([
    html.A("SDR represents percentage of observations that were predicted very closely by two different models. "),
    html.Ul([
        html.Li([html.A("The "), html.Strong("bigger "), html.A("the value of SDR, the "), html.Strong("less similar "),
                 html.A("outputs of two models are.")]),
        html.Li([html.Strong("Formula: "),
                 dcc.Markdown(r'$$\mathrm{SDR} = \frac{\sum^{n}_{i=1}A_{i}}{n}, \,\ \mathrm{where} \,\ A_i = '
                              r'\begin{cases} 0, d_i > SD(y) \\ 1, d_i \leq SD(y) \end{cases}, \,\ d_i = '
                              r'|\hat{y}_i - \hat{y}_j|.$$', mathjax=True, id="latex-code")]),
        html.Li([html.A("SDR "), html.Strong("does not indicate "), html.A("the accuracy of models.")])
    ])
], className="annotation_str", id="ann_comp_sdr")

ann_comp_conj_rmse = html.Div([
    html.H3(['Conjunctive RMSE between chosen model and other models'], className='annotation-title'),
    html.A("Conjunctive RMSE is calculated based on "), html.Strong("mean of two prediction vectors"), html.A("."),
    html.Br(), html.A(" On this plot score of RMSE of prediction of chosen model is compared to predictions joined with "
                      "other models in ensemble.")
], className="annotation_str", id="ann_comp_conj_rmse")

ann_comp_diff_dist = html.Div([
    html.H3(['Prediction Difference Distribution'], className='annotation-title'),
    html.A("Plot shows actual difference of predictions between chosen model and other models in ensemble through "
           "the whole data set.")
], className="annotation_str", id="ann_comp_diff_dist")

ann_comp_diff_boxplot = html.Div([
    html.H3(['Distribution of absolute difference'], className='annotation-title'),
    html.A("Plot shows distribution of absolute prediction differences of chosen model and other models in ensemble."),
    html.Br(), html.A("Yellow dashed lines outline thresholds of "), html.Strong("agreement (lower line) "),
    html.A("and "), html.Strong("strong disagreement (higher line) "), html.A("which help decide which models are closer"
    " prediction-wise.")
], className="annotation_str", id="ann_comp_diff_boxplot")

