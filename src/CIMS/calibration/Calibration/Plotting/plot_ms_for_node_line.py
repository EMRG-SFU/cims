#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

import Calibration.Data.node_info as node_info


def plotOverTime_line(fig, ax, res_obj, allYears=None ):  # See `res_base` and `res_calib` above.
    lenCheck1 = [len(a) for a in res_obj.values()]
    lenCheck2 = [a == lenCheck1[0] for a in lenCheck1]
    assert all(lenCheck2), "sub lengths not the same"
    numVals = lenCheck1[0]
    
    if allYears is not None:
        # If allYears given, make sure it's the length of the implicit number of years in res_obj
        assert len(allYears)==lenCheck1[0], "provided `allYears` not same length as param year val arrays"
        dates = [datetime(int(yy), 1, 1) for yy in allYears]
        values = range(numVals)
        for n,vals in res_obj.items():
            ax.plot(dates, vals, label=n, linewidth=4)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()

    else:
    
        for n,vals in res_obj.items():
            ax.plot(range(numVals), vals, label=n, linewidth=4)


def plot_ms_line(model, 
                 nodeName,
                 msKey = "market_share_total",
                 msCalKey = "calibration_market_share_total"):

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
    res_calib = {tn:[maybeFloat(x) for x in node_info.getTechParamOverTime(model.graph, nodeName, tn, msCalKey)] for tn in allTechNames}

    # Figure size nonsense
    width_px = 800
    height_px = 600
    dpi = 72
    width_in_inches = width_px / dpi
    height_in_inches = height_px / dpi

    fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    fig.set_size_inches(14,6,forward=True)
    fig.suptitle(f"{nodeName}")
    ax1 = plt.subplot(1,2,1, title="CIMS calc MS")
    plotOverTime_line(fig, ax1, res_base, allYears)
    plt.legend(fontsize=7)
    ax2 = plt.subplot(1,2,2, title="Counterfactual MS")
    plotOverTime_line(fig, ax2, res_calib, allYears)
    plt.legend(fontsize=7)

    # Supposedly this will link the y-axes on both plots so they'll be
    # the same scale, which we need for cims/counterfactual calibration comparison.
    # Yeah not quite... this locks them to be the same as ONE of the axes... not
    # necessarily the one with the highest value, so you can get points off the
    # top of the plot if they're shared the wrong way around.
    # It's best to deal with this as below, and not even involve `sharey` at all.
    #ax1.sharey(ax2)

    # Figure out the global y-max and set the axes to accomodate this. (like facet
    # plotting in R -- it's a little more fiddly in python).
    global_ymax = max(ax.get_ylim()[1] for ax in [ax1, ax2])
    for i,ax in enumerate([ax1, ax2], 1):
        ax.set_ylim(0.0, global_ymax)
        if i == 1:
            ax.set_ylabel(f"Fraction of Market Share")
        ax.set_xlabel('Year')
        ax.grid(True)
    
    plt.tight_layout()
    #plt.show()
    return fig

