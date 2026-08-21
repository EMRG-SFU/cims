
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import polars as pl
import re

import Calibration.Data.node_info as node_info
import Calibration.Data.emissions as emissions
from Calibration.Plotting.plot_general import plotOverTime_stack
from Calibration.Plotting.plot_general import plotOverTime_stack_df
from Calibration.Plotting.plot_general import plotOverTime_line_df
from Calibration.Plotting.plot_general import plotHeatmap

def plot_emissions(model,
                   nodeName,
                   patterns=[]):
    """
    """
    diffFrame_pre = emissions.get_emissions_diff_frame(model, nodeName)

    if len(patterns) > 0:
        diffFrame = diffFrame_pre.filter(
            pl.any_horizontal([pl.col('gas').str.contains(p) for p in patterns])
        )
    else:
        diffFrame = diffFrame_pre
    #all_years = diffFrame['year'].unique().sort()
    #all_gases = diffFrame['gas'].unique().sort()

    ret_base = plotOverTime_stack_df(diffFrame, colorCol='gas', valCol='cims_value', showlegend=False)
    ret_calib = plotOverTime_stack_df(diffFrame, colorCol='gas', valCol='cal_value', showlegend=True)

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=("Model Emissions", "Calibration Emissions"))
    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        fig.add_trace(trace, row=1, col=2)

    fig.show()

def plot_emissions_cims(model, nodeName, patterns=[]):
    df = pl.DataFrame(emissions.get_emissions(model, nodeName, getDict=True))
    df = df.with_columns(pl.col("value").cast(pl.Float64))
    df = df.group_by(["year","gas"]).agg(pl.col(["value"]).sum()).sort("year")
    fig = plotOverTime_stack_df(df, colorCol="gas", valCol="value", showlegend=True)
    fig.update_layout(title=f"Node: {nodeName}")
    fig.show()

def plot_emissions_line(model,
                        nodeName,
                        patterns=[]):
    """
    """
    diffFrame_pre = emissions.get_emissions_diff_frame(model, nodeName)

    if len(patterns) > 0:
        diffFrame = diffFrame_pre.filter(
            pl.any_horizontal([pl.col('gas').str.contains(p) for p in patterns])
        )
    else:
        diffFrame = diffFrame_pre

    ret_base = plotOverTime_line_df(diffFrame, colorCol='gas', valCol='cims_value', showlegend=False)
    ret_calib = plotOverTime_line_df(diffFrame, colorCol='gas', valCol='cal_value', showlegend=True)

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=("Model Emissions", "Calibration Emissions"))
    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        fig.add_trace(trace, row=1, col=2)

    # Find the maximum value across both subplots
    all_values = []
    for trace in ret_base.data:
        all_values.extend([v for v in trace.y if v is not None])
    for trace in ret_calib.data:
        all_values.extend([v for v in trace.y if v is not None])

    max_val = max(all_values) if all_values else 1.0
    margin = 0.1
    y_max = max_val * (1 + margin)

    fig.update_yaxes(range=[0, y_max], row=1, col=1)
    fig.update_yaxes(range=[0, y_max], row=1, col=2)

    fig.show()


def plot_emissions_line_cims(model, nodeName, patterns=[]):
    df = pl.DataFrame(emissions.get_emissions(model, nodeName, getDict=True))
    df = df.with_columns(pl.col("value").cast(pl.Float64))
    df = df.group_by(["year","gas"]).agg(pl.col(["value"]).sum()).sort("year")
    fig = plotOverTime_line_df(df, colorCol="gas", valCol="value", showlegend=True)
    fig.update_layout(title=f"Node: {nodeName}")
    fig.show()

def plot_emissions_heatmap(model,
                           nodeName,
                           fixedColor=None):
    """
    """
    df = emissions.get_emissions_diff_frame(model, nodeName)

    fig = plotHeatmap(df,
                      valName="diff",
                      dim1Name="gas",
                      dim2Name="year",
                      nodeName=nodeName,
                      fixedColor=fixedColor
    )
    fig.show()


def plot_emissions_diffLine(model,
                           nodeName):
    """
    """
    df = emissions.get_emissions_diff_frame(model, nodeName)

    fig = plotOverTime_line_df(df,
                      valCol="diff",
                      colorCol="gas",
                      yearCol="year",
                      fixedY=0.0
    )
    fig.show()







