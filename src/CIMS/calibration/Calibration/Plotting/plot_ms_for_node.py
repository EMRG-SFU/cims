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
    Show a side-by-side stacked-area comparison of CIMS vs counterfactual market share.

    Reads two parameters per technology at `nodeName` — the share CIMS computed
    and the calibration target it is being fitted against — and draws each as a
    stacked area panel in a single one-row, two-column Plotly figure. The two
    y-axes are linked so the panels share one autoranged scale and can be
    compared by eye.

    Parameters
    ----------
    model : object
        The loaded CIMS model. Only its `.graph` attribute is used.
    nodeName : str
        Name of the node to plot, used as a key into the model graph.
    msKey : str
        Tech parameter holding the CIMS-calculated share (left panel).
    calMsKey : str
        Tech parameter holding the counterfactual calibration target (right panel).
    techFilters : list of str
        Regex patterns used to plot only a subset of the technologies. A
        technology is kept if it matches *any* pattern, case-insensitively.
        An empty list (the default) keeps everything.

    Returns
    -------
    None
        Calls `fig.show()` for its side effect and returns nothing, so the
        figure cannot be further customised or saved by the caller.

    Notes
    -----
    Values of `None` or the string 'NA' become `None` and render as gaps.
    Unlike the sibling VizServer version, no percentage rescaling is applied
    here: both series are read with `maybeFloat`, which assumes `calMsKey` is
    already stored as a 0-1 fraction rather than a percentage.
    """


    def maybeFloat(x):
        """Coerce to float, mapping `None` and the string 'NA' to `None`."""
        if x is None:
            return(None)
        elif isinstance(x, str) and x=='NA':
            return(None)
        else:
            return(float(x))

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

    # Link the two y-axes so they share the same autoranged scale.
    fig.update_yaxes(matches='y')

    fig.show()




def plot_ms_heatmap(model,
                    nodeName,
                    msKey="market_share_total",
                    calMsKey="calibration_market_share_total",
                    fixedColor=None):
    """
    Show a technology-by-year heatmap of the gap between CIMS and calibration.

    The plotted value is `diff` = CIMS share minus counterfactual share, so
    positive cells are technologies CIMS over-predicts relative to the
    calibration target and negative cells are ones it under-predicts. The
    colour scale is diverging and centred on zero, making the sign of the
    mismatch readable at a glance.

    Parameters
    ----------
    model : object
        The loaded CIMS model, passed through to the data layer.
    nodeName : str
        Name of the node to plot. Also used in the figure title.
    msKey : str
        Tech parameter holding the CIMS-calculated share.
    calMsKey : str
        Tech parameter holding the counterfactual calibration target.
    fixedColor : float, optional
        Clamp the colour scale to a symmetric range of `-fixedColor` to
        `+fixedColor` instead of autoscaling. Use this to hold the scale steady
        when comparing heatmaps across several nodes, since an autoscaled range
        makes a small mismatch look as dramatic as a large one.

    Returns
    -------
    None
        Calls `fig.show()` for its side effect and returns nothing.

    Notes
    -----
    A technology with no calibration value is treated as a target of 0.0 rather
    than being dropped, so it will show up as a non-zero diff. A missing CIMS
    value instead raises `RuntimeError` from the data layer.
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
    Show the CIMS-minus-calibration gap over time, as one line per technology.

    The line-plot counterpart to `plot_ms_heatmap`, built from the same `diff`
    column: CIMS share minus counterfactual share. A dotted horizontal
    reference line is drawn at zero, so a well-calibrated technology is one
    whose line stays flat against it.

    Parameters
    ----------
    model : object
        The loaded CIMS model, passed through to the data layer.
    nodeName : str
        Name of the node to plot.
    msKey : str
        Tech parameter holding the CIMS-calculated share.
    calMsKey : str
        Tech parameter holding the counterfactual calibration target.

    Returns
    -------
    None
        Calls `fig.show()` for its side effect and returns nothing.

    Notes
    -----
    Shares the missing-value behaviour of `plot_ms_heatmap`: an absent
    calibration value is treated as 0.0, an absent CIMS value raises.
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




