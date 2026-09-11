
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import polars as pl
import re

import Calibration.Data.node_info as node_info
import Calibration.Data.quantities as requestedQuantities
from Calibration.Plotting.plot_general import plotOverTime_stack
from Calibration.Plotting.plot_general import plotOverTime_stack_df
from Calibration.Plotting.plot_general import plotOverTime_line_df
from Calibration.Plotting.plot_general import plotHeatmap

def plot_requestedQuantities(model,
                             nodeName,
                             all_fuels=False,
                             patterns=[],
                             **kwargs):
    """
    `filters` are matched to names of fuels, for plotting only a subset of the fuels.
    """

    if all_fuels or ('gimme_all_fuels_please' in kwargs.keys() and kwargs['gimme_all_fuels_please'] == True):
        diffFrame_pre = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName, join_type="outer")
    else:
        diffFrame_pre = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName)

    if len(patterns) > 0:
        diffFrame = diffFrame_pre.filter(
            pl.any_horizontal([pl.col('fuel').str.contains(p) for p in patterns])
        )
    else:
        diffFrame = diffFrame_pre

    diffFrame = diffFrame.with_columns(pl.col("year").cast(pl.Float64))
    diffFrame = diffFrame.sort("year")

    ret_base = plotOverTime_stack_df(diffFrame, colorCol='fuel', valCol='cims_value', showlegend=False)
    ret_calib = plotOverTime_stack_df(diffFrame, colorCol='fuel', valCol='cal_value', showlegend=True)

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=("Model RequestedQuantity", "Calibration RequestedQuantity"))
    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        fig.add_trace(trace, row=1, col=2)

    # Link the two y-axes so they share the same autoranged scale (correctly
    # accounts for the stacked/cumulative totals in each area plot).
    fig.update_yaxes(matches='y')

    fig.show()


def plot_requestedQuantities_cims(model,
                                  nodeName,
                                  patterns=[]):
    """
    This one only plots the cims values, not the calibration counterfactual
    """

    df = pl.DataFrame(requestedQuantities.get_quantityRequested(model, nodeName, getDict=True))
    df = df.with_columns(pl.col("value").cast(pl.Float64))
    df = df.group_by(["year","fuel"]).agg(pl.col("value").sum()).sort("year")
    fig = plotOverTime_stack_df(df, colorCol='fuel', valCol='value', showlegend=True)
    fig.update_layout(title=f"Node: {nodeName}")
    fig.show()


def plot_requestedQuantities_line(model,
                                  nodeName,
                                  all_fuels=False,
                                  patterns=[],
                                  **kwargs):
    """
    """

    if all_fuels or ('gimme_all_fuels_please' in kwargs.keys() and kwargs['gimme_all_fuels_please'] == True):
        diffFrame_pre = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName, join_type="outer")
    else:
        diffFrame_pre = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName)

    if len(patterns) > 0:
        diffFrame = diffFrame_pre.filter(
            pl.any_horizontal([pl.col('fuel').str.contains(p) for p in patterns])
        )
    else:
        diffFrame = diffFrame_pre
   
    diffFrame = diffFrame.with_columns(pl.col("year").cast(pl.Float64))
    diffFrame = diffFrame.sort("year")

    ret_base = plotOverTime_line_df(diffFrame, colorCol='fuel', valCol='cims_value', showlegend=False)
    ret_calib = plotOverTime_line_df(diffFrame, colorCol='fuel', valCol='cal_value', showlegend=True)

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=("Model RequestedQuantities", "Calibration RequestedQuantities"))
    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        fig.add_trace(trace, row=1, col=2)

    # Link the two y-axes so they share the same autoranged scale.
    fig.update_yaxes(matches='y')

    fig.show()

def plot_requestedQuantities_line_cims(model,
                                  nodeName,
                                  patterns=[]):
    """
    This one only plots the cims values, not the calibration counterfactual
    """

    df = pl.DataFrame(requestedQuantities.get_quantityRequested(model, nodeName, getDict=True))
    df = df.with_columns(pl.col("value").cast(pl.Float64))
    df = df.group_by(["year","fuel"]).agg(pl.col("value").sum()).sort("year")
    fig = plotOverTime_line_df(df, colorCol='fuel', valCol='value', showlegend=True)
    fig.update_layout(title=f"Node: {nodeName}")
    fig.show()

def plot_requestedQuantities_heatmap(model,
                           nodeName,
                           fixedColor=None):
    """
    """
    df = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName)

    # ::TODO:: Why does this not work here, causing "selCols" on the heatmap plotting side
    # to claim things are duplicated?
    #df = df.with_columns(pl.col("year").cast(pl.Float64))
    #df = df.sort("year")

    fig = plotHeatmap(df,
                      valName="diff",
                      dim1Name="fuel",
                      dim2Name="year",
                      nodeName=nodeName,
                      fixedColor=fixedColor
    )
    fig.show()


def plot_requestedQuantities_diffLine(model,
                                      nodeName):
    """
    """
    df = requestedQuantities.get_quantityRequested_diff_frame(
            model,
            nodeName
    )
   
    df = df.with_columns(pl.col("year").cast(pl.Float64))
    df = df.sort("year")

    fig = plotOverTime_line_df(df, 
                valCol="diff",
                colorCol="fuel",
                yearCol="year", 
                fixedY=0.0

    )
    fig.show()
