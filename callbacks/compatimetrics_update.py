from dash import html, dcc, Output, Input, callback, State
import dash_bootstrap_components as dbc
import sys
import pandas as pd

from components import compatimetrics_plots
from components.annotations import ann_comp_uniformity, ann_comp_incompatibility, ann_comp_acs, ann_comp_conj_acc, \
    ann_comp_dis_ratio, ann_comp_conj_metrics, ann_comp_conj_precision, ann_comp_conj_recall, ann_comp_pred_corr, \
    ann_comp_collective, ann_comp_msd, ann_comp_rmsd, ann_comp_ar, ann_comp_sdr, ann_comp_conj_rmse, ann_comp_diff_dist, \
    ann_comp_diff_boxplot

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
        if task == 'regression':
            children = [
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Mean Squared Difference'], className='annotation-title'),
                            ann_comp_msd,
                            dcc.Graph(figure=compatimetrics_plots.msd_matrix(predictions), className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Root Mean Squared Difference'], className='annotation-title'),
                            ann_comp_rmsd,
                            dcc.Graph(figure=compatimetrics_plots.rmsd_matrix(predictions), className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Agreement Ratio'], className='annotation-title'),
                            ann_comp_ar,
                            dcc.Graph(figure=compatimetrics_plots.ar_matrix(predictions, y), className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Strong Disagreement Ratio'], className='annotation-title'),
                            ann_comp_sdr,
                            dcc.Graph(figure=compatimetrics_plots.sdr_matrix(predictions, y), className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col')
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Mean Squared Difference Comparison'], className='annotation-title'),
                            dcc.Graph(figure=compatimetrics_plots.msd_comparison(predictions, model_to_compare), className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Root Mean Squared Difference Comparison'], className='annotation-title'),
                            dcc.Graph(figure=compatimetrics_plots.rmsd_comparison(predictions, model_to_compare),
                                      className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                ]),
                dbc.Row([
                    ann_comp_conj_rmse,
                    dcc.Graph(
                        figure=compatimetrics_plots.conjunctive_rmse_plot(predictions, y, model_to_compare),
                        className='plot')
                ], className="custom-caption"),
                dbc.Row([
                     ann_comp_diff_dist,
                    dcc.Graph(figure=compatimetrics_plots.difference_distribution(predictions, model_to_compare),
                              className="plot")
                ], className="custom-caption"),
                dbc.Row([
                    ann_comp_diff_boxplot,
                    dcc.Graph(figure=compatimetrics_plots.difference_boxplot(predictions, y, model_to_compare),
                              className="plot")
                ], className="custom-caption")
            ]
        else:
            children = [
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Uniformity'], className='annotation-title'),
                            ann_comp_uniformity,
                            dcc.Graph(figure=compatimetrics_plots.uniformity_matrix(predictions),
                                      className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Incompatibility'], className='annotation-title'),
                            ann_comp_incompatibility,
                            dcc.Graph(figure=compatimetrics_plots.incompatibility_matrix(predictions),
                                      className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col')
                ], style={'display': 'flex'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Average Collective Score'], className='annotation-title'),
                            ann_comp_acs,
                            dcc.Graph(figure=compatimetrics_plots.acs_matrix(predictions, y),
                                      className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col'),
                    dbc.Col([
                        dbc.Row([
                            html.H3(['Conjunctive Accuracy'], className='annotation-title'),
                            ann_comp_conj_acc,
                            dcc.Graph(figure=compatimetrics_plots.conjuntive_accuracy_matrix(predictions, y),
                                      className="plot")
                        ], className="custom-caption")
                    ], width=6, className='plot-col')
                ], style={'display': 'flex'})
            ]
            if task == 'classification':
                children.extend([
                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                ann_comp_dis_ratio,
                                dcc.Graph(
                                    figure=compatimetrics_plots.disagreement_ratio_plot(predictions, y, model_to_compare),
                                    className="plot")
                            ], className="custom-caption")
                        ], width=6, className='plot-col'),
                        dbc.Col([
                            dbc.Row([
                                ann_comp_conj_metrics,
                                dcc.Graph(
                                    figure=compatimetrics_plots.conjunctive_metrics_plot(predictions, y,
                                                                                         model_to_compare),
                                    className="plot")
                            ], className="custom-caption")
                        ], width=6, className='plot-col'),
                    ], style={'display': 'flex'})
                ])
            else:
                children.extend([
                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                ann_comp_conj_precision,
                                dcc.Graph(
                                    figure=compatimetrics_plots.conjunctive_precision_multiclass_plot(predictions, y, model_to_compare),
                                    className="plot")
                            ], className="custom-caption")
                        ], width=6, className='plot-col'),
                        dbc.Col([
                            dbc.Row([
                                ann_comp_conj_recall,
                                dcc.Graph(
                                    figure=compatimetrics_plots.conjunctive_recall_multiclass_plot(predictions, y,
                                                                                                   model_to_compare),
                                    className="plot")
                            ], className="custom-caption")
                        ], width=6, className='plot-col'),
                    ])
                ])

            children.extend([
                dbc.Row([
                    ann_comp_pred_corr,
                    dcc.Graph(
                        figure=compatimetrics_plots.prediction_correctness_plot(predictions, y, model_to_compare),
                        className='plot')
                ], className="custom-caption"),
                dbc.Row([
                    ann_comp_collective,
                    dcc.Graph(
                        figure=compatimetrics_plots.collective_cummulative_score_plot(predictions, y, model_to_compare),
                        className='plot')
                ], className="custom-caption"),
            ])
    return children