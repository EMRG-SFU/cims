import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import re

import Calibration.Data.node_info as node_info

def plotOverTime_line(res_obj, allYears=None, showlegend=True):
    lenCheck1 = [len(a) for a in res_obj.values()]
    lenCheck2 = [a == lenCheck1[0] for a in lenCheck1]
    assert all(lenCheck2), "sub lengths not the same"
    numVals = lenCheck1[0]
    
    if allYears is not None:
        # If allYears given, make sure it's the length of the implicit number of years in res_obj
        assert len(allYears)==lenCheck1[0], "provided `allYears` not same length as param year val arrays"

        # Build a pandas dataframe for plotly line functions
        dates = [datetime(int(yy), 1, 1) for yy in allYears]
        df_list = []
        for n, vals in res_obj.items():
            df = pd.DataFrame({'dates': dates, 'vals': vals, 'name': n})
            df_list.append(df)

        allDf = pd.concat(df_list)
        # Changed from px.area to px.line
        fig = px.line(allDf, x='dates', y='vals', color='name', markers=True)
        fig.update_traces(showlegend=showlegend)
        return((fig, allDf))
        
    else:
        df_list = []
        for n, vals in res_obj.items():
            # Using range(numVals) as x-axis when dates aren't provided
            df = pd.DataFrame({'dates': range(numVals), 'vals': vals, 'name': n})
            df_list.append(df)

        allDf = pd.concat(df_list)
        # Changed from px.area to px.line
        fig = px.line(allDf, x='dates', y='vals', color='name', markers=True)
        # Note: Original code had update_trace (singular), likely a typo for update_traces.
        # Keeping consistent with standard plotly usage here.
        fig.update_traces(showlegend=showlegend)
        return((fig, allDf))


def plot_ms_line(model, 
            nodeName,
            msKey = "market_share_total",
            calMsKey = "calibration_market_share_total",
            techFilters=[]):
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

    if len(techFilters) > 0:
        allTechNames = [a for a in allTechNames 
            if any([
                bool( re.search(b, a, flags = re.IGNORECASE) ) for b in techFilters
                ])
         ]


    allYears = node_info.list_years(model.graph, nodeName)
    
    # Fetch base and calibration data
    res_base = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, msKey)] for tn in allTechNames}
    res_calib = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, calMsKey)] for tn in allTechNames}
    
    # Generate line plots instead of area plots
    ret_base = plotOverTime_line(res_base, allYears, showlegend=False)[0]
    ret_calib = plotOverTime_line(res_calib, allYears, showlegend=True)[0]

    fig = make_subplots(rows=1, 
                        cols=2, 
                        subplot_titles=("CIMS calc MS", "Counterfactual MS"))

    fig.update_layout(title=f"Node: {nodeName}")

    for ct, trace in enumerate(ret_base.data):
        fig.add_trace(trace, row=1, col=1)

    for ct, trace in enumerate(ret_calib.data):
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



