import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import re

import Calibration.Data.node_info as node_info
from Calibration.Data.market_share import get_marketShare_diff_frame
from Calibration.Plotting.plot_general import plotOverTime_stack
from Calibration.Plotting.plot_general import plotOverTime_line_df
from Calibration.Plotting.plot_general import plotHeatmap

# Import (for re-exporting) the stuff in _plot_ms_for_node_line
from Calibration.Plotting._plot_ms_for_node_line import plot_ms_line



def plot_ms(model, 
            nodeName,
            msKey = "market_share_total",
            calMsKey = "calibration_market_share_total",
            techFilters = []):
    """
    `techFilters` here contains strings that serve as regex matches for the technologies. If you want to only plot
        a subset of the technologies, here is where you specify them.
    """


    def maybeFloat(x):
        if x is None:
            return(None)
        elif isinstance(x, str) and x=='NA':
            return(None)
        else:
            return(float(x))

    def maybeFloatDiv100(x):
        if x is None:
            return(None)
        elif isinstance(x, str) and x=='NA':
            return(None)
        else:
            return(float(x)/100.0)

    allTechNames = node_info.list_techs(model.graph, nodeName)

    # Use the `techFilters` list to extract only specific technologies
    if len(techFilters) > 0:
        allTechNames = [a for a in allTechNames 
            if any([
                bool( re.search(b, a, flags = re.IGNORECASE) ) for b in techFilters
                ])
         ]


    allYears = node_info.list_years(model.graph, nodeName)

    res_base = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, msKey)] for tn in allTechNames}
    res_calib = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, calMsKey)] for tn in allTechNames}
    
    ret_base = plotOverTime_stack(res_base, allYears, showlegend=False)[0]
    ret_calib = plotOverTime_stack(res_calib, allYears, showlegend=True)[0]

    fig = make_subplots(rows=1, 
                        cols=2, 
                        subplot_titles=("CIMS calc MS","Counterfactual MS"))

    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        fig.add_trace(trace, row=1, col=2)

    fig.show()




def plot_ms_heatmap(model,
                    nodeName,
                    msKey="market_share_total",
                    calMsKey="calibration_market_share_total",
                    fixedColor=None):
    """
    """
    df = get_marketShare_diff_frame(
            model,
            nodeName,
            key_cims=msKey,
            key_cal=calMsKey
    )
   
    fig = plotHeatmap(df, 
                valName="diff",
                dim1Name="tech",
                dim2Name="year",
                nodeName=nodeName,
                fixedColor=fixedColor

    )
    fig.show()


def plot_ms_diffLine(model,
                    nodeName,
                    msKey="market_share_total",
                    calMsKey="calibration_market_share_total"):
    """
    """
    df = get_marketShare_diff_frame(
            model,
            nodeName,
            key_cims=msKey,
            key_cal=calMsKey
    )
   
    fig = plotOverTime_line_df(df, 
                valCol="diff",
                colorCol="tech",
                yearCol="year",
                fixedY=0.0

    )
    fig.show()




