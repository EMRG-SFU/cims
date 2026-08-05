import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

import Calibration.Data.node_info as node_info

def plotOverTime_stack(res_obj, allYears=None, showlegend=True):
    lenCheck1 = [len(a) for a in res_obj.values()]
    lenCheck2 = [a == lenCheck1[0] for a in lenCheck1]
    assert all(lenCheck2), "sub lengths not the same"
    numVals = lenCheck1[0]
    
    if allYears is not None:
        # If allYears given, make sure it's the length of the implicit number of years in res_obj
        assert len(allYears)==lenCheck1[0], "provided `allYears` not same length as param year val arrays"

        # Instead of repeated calls to `plot` we need to build a pandas dataframe that can be passed
        # to the plotly area functions.
        
        dates = [datetime(int(yy), 1, 1) for yy in allYears]
        values = range(numVals)
        df_list = []
        for n,vals in res_obj.items():
            #ax.plot(dates, vals, label=n, linewidth=4)
            df = pd.DataFrame({'dates': dates, 'vals':vals, 'name':n})
            df_list.append(df)

        allDf = pd.concat(df_list)
        fig = px.area(allDf, x='dates', y='vals', color='name', markers=True)
        fig.update_traces(showlegend=showlegend)
        return((fig, allDf))
        
        #ax.xaxis.set_major_locator(mdates.YearLocator(5))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        #fig.autofmt_xdate()

    else:
        df_list = []
        for n,vals in res_obj.items():
            #ax.plot(range(numVals), vals, label=n, linewidth=4)
            df = pd.DataFrame({'dates':range(numVals), 'vals':vals, 'name':n})
            df_list.append(df)

        allDf = pd.concat(df_list)
        #return(allDf)

        fig = px.area(allDf, x='dates', y='vals', color='name', markers=True)
        fig.update_trace(showlegend=showlegend)
        return((fig, allDf))


def plot_ms(model, 
            nodeName,
            msKey = "market_share_total",
            calMsKey = "calibration_market_share_total"):

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
    allYears = node_info.list_years(model.graph, nodeName)
    res_base = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, msKey)] for tn in allTechNames}
    res_calib = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, calMsKey)] for tn in allTechNames}
    
    #return({'base':res_base, 'calib':res_calib})

    ret_base = plotOverTime_stack(res_base, allYears, showlegend=False)[0]
    ret_calib = plotOverTime_stack(res_calib, allYears, showlegend=True)[0]

    fig = make_subplots(rows=1, 
                        cols=2, 
                        subplot_titles=("CIMS calc MS","Counterfactual MS"))

    fig.update_layout(title=f"Node: {nodeName}")

    for ct,trace in enumerate(ret_base.data):
        #print(f"ret_base, {ct}")
        fig.add_trace(trace, row=1, col=1)

    for ct,trace in enumerate(ret_calib.data):
        #print(f"ret_calib, {ct}")
        fig.add_trace(trace, row=1, col=2)

    fig.show()
