from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import sys
from components import compatimetrics_plots
import pandas as pd
sys.path.append("..")

# callback to update compatimetric plots
@callback(
    Output('compatimetrics_plots', 'children'),
    State('predictions', 'data'),
    Input('model_select', 'value'),
    State('task', 'data'),
    State('csv_data', 'data'),
    State('y_label_column', 'data'),
)
def update_compatimetrics_plot(predictions, model_to_compare, task, df, column):
    children = []
    df = pd.DataFrame.from_dict(df)
    df = df.dropna()
    y = df.iloc[:, df.columns == column["name"]]
    y = y.squeeze()
    if model_to_compare:
        if task == 'classification':
            children = [
                html.H3("""
                        Matrices below show how similar two classifiers are by calculating percentage of observations
                        that two models predicted the same in case of uniformity, and differently in case of incompatibility
                       """,
                        className="annotation_str", id="ann_comp_6"),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                       className="plot")],
                            width=6),
                    html.H3("""
                    Matrix below on the right shows value of Average Collective Score which is a metric that 
                    sums number of doubly correct predictions and number of disagreements with coefficient 0.5 and
                    then dividing it by number of observations. It measures joined performance with consideration
                    of double correct prediction and disagreements.
                   """,
                            className="annotation_str", id="ann_comp_7"),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                ]),
                dbc.Row([dbc.Col([
                    html.H3("""Disagreement ratio presented on plot below on the left is measuring how many 
                        observations were predicted differently by two models regarding to the class of the record. 
                        It can show which class was more difficult to predict when joining models.""",
                        className="annotation_str", id="ann_comp_9"),], width=6),
                    dbc.Col([html.H3("""Conjunctive metrics are analogous to standard evaluation metrics,
                        but instead of comparing target variable with one prediction vector, we use two prediction vectors 
                        at the same time. Simply we mark prediction as correct, if two models predicted it correctly. 
                        Thus, conjunctive accuracy, presented on matrix above, precision and recall, showed together below, 
                        are good indicators of joined model performance as they measure the same ratios as original
                        metrics. Worth mentioning - conjunctive recall is generally lower and conjunctive precision 
                        is generally higher, which is related to their definition.""",
                        className="annotation_str", id="ann_comp_8"),
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.disagreement_ratio_plot(predictions, y, model_to_compare),
                        className="plot")],
                        width=6),
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_metrics_plot(predictions, y, model_to_compare),
                        className="plot")],
                        width=6),
                ]),
                html.H3("""
                       Plot below is showing ratio of predictions on different level of correctness. Doubly correct
                       prediction occurs when two models predicted observation right, disagreement when one of models
                       is missing, and doubly incorrect when two models labeled wrong class.
                       """,
                        className="annotation_str", id="ann_comp_10"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                        className='plot')
                     ]),
                html.H3("""
                        On the plot below one can observe the progess of incresing average collective score 
                        through the whole data set. This plot can be helpful when searching for areas of data set
                        where prediction was less effective. 
                       """,
                        className="annotation_str", id="ann_comp_11"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                        className='plot')
                     ]),
            ]
        elif task == 'regression':
            children = [dbc.Row([
                html.H3("""
                                Matrices below show the distance between two prediction vectros obtained from base models. 
                                MSE calculates mean of squared distance between vectors. RMSE is a root of MSE. The bigger the values,
                                the less similar are two models.
                                """,
                        className="annotation_str", id="ann_comp_1"),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_matrix(predictions), className="plot")],
                        width=6),
                dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_matrix(predictions), className="plot")],
                        width=6),
                ]),
                html.H3("""
                    Matrices below show ratio of agreement and strong disagreement between two models. Agreement ratio 
                    calculates the percentage of observations that two models predicted closer than fiftieth part of 
                    standard deviation of target variable. On the other hand, disagreement ratio calculates the percantage
                    of observations witch have prediction difference bigger than standard deviation of target variable
                    """,
                        className="annotation_str", id="ann_comp_2"),

                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.ar_matrix(predictions, y), className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.sdr_matrix(predictions, y), className="plot"), ],
                            width=6),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.msd_comparison(predictions, model_to_compare),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.rmsd_comparison(predictions, model_to_compare),
                                       className="plot")],
                            width=6),
                ]),
                html.H3("""
                        Conjunctive RMSE is calculated based on mean of two prediction vectors. On this plot 
                        score of RMSE of prediction of chosen model is compared to predictions joined with other models
                        in ensemble. 
                        """,
                        className="annotation_str", id="ann_comp_3"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_rmse_plot(predictions, y, model_to_compare),
                        className='plot')
                    ]),
                html.H3("""
                        Plot below shows actual difference of predictions between chosen model 
                        and other models in ensemble through the whole data set.
                        """,
                        className="annotation_str", id="ann_comp_4"),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_distribution(predictions, model_to_compare),
                              className="plot")
                ]),
                html.H3("""
                            Plot below shows distribution of absolute prediction differences of chosen model and 
                            other models in ensemble. Pink dashed lines outline thresholds of agreement (lower line)
                            and strong disagreement (higher line), which help decide which models are closer
                            prediction-wise. 
                            """,
                        className="annotation_str", id="ann_comp_5"),
                dbc.Row([
                    dcc.Graph(figure=compatimetrics_plots.difference_boxplot(predictions, y, model_to_compare),
                              className="plot")
                ])
            ]
        else:
            children = [
                html.H3("""
                        Matrices below show how similar two classifiers are by calculating percentage of observations
                        that two models predicted the same in case of uniformity, and differently in case of incompatibility
                       """,
                        className="annotation_str", id="ann_comp_12"),
                dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                       className="plot")],
                            width=6),
                ]), dbc.Row([
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                    dbc.Col([dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                       className="plot")],
                            width=6),
                ]),
                html.H3("""
                        Conjunctive metrics are analogous to standard evaluation metrics, but instead of comparing target
                        variable with one prediction vector, we use two prediction vectors at the same time. Simply we
                        mark prediction as correct, if two models predicted it correctly. In case of multiclass 
                        classification we additionally distinguish weighted and macro versions of recall and precision.
                        Thus, conjunctive accuracy, presented on matrix above on the right, precision and recall, showed 
                        below, are good indicators of joined model performance as they measure the same ratios as original
                        metrics. Worth mentioning - conjunctive recall is generally lower and conjunctive precision 
                        is generally higher, which is related to their definition. 
                                   """,
                        className="annotation_str", id="ann_comp_14"),
                dbc.Row([
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_precision_multiclass_plot(predictions, y,
                                                                                          model_to_compare),
                        className="plot")],
                        width=6),
                    dbc.Col([dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_recall_multiclass_plot(predictions, y,
                                                                                       model_to_compare),
                        className="plot")],
                        width=6),
                ]),
                html.H3("""
                     Plot below is showing ratio of predictions on different level of correctness. Doubly correct
                     prediction occurs when two models predicted observation right, disagreement when one of models
                     is missing, and doubly incorrect when two models labeled wrong class.
                     """,
                        className="annotation_str", id="ann_comp_15"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                        className='plot')
                    ]),
                html.H3("""
                      On the plot below one can observe the progress of increasing average collective score 
                      through the whole data set. This plot can be helpful when searching for areas of data set
                      where prediction was less effective. 
                     """,
                        className="annotation_str", id="ann_comp_16"),
                dbc.Row(
                    [dcc.Graph(
                        figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                        className='plot')
                    ])
            ]
    return children