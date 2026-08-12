import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import polars as pl
import re

def plotOverTime_stack(res_obj, allYears=None, showlegend=True):
    """
    `res_obj` things are the {tech: [param over date]} objects. The tech can I guess be any
              suitable grouping categorical variable, and repurpose for emissions and quantities.
    """
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



def plotOverTime_stack_df( df, 
                           colorCol,
                           yearCol='year',
                           valCol='value',
                           showlegend=True):
    """
    `df` has specified color, year, value columns.
    """

    fig = px.area(df, x=yearCol, y=valCol, color=colorCol, markers=True)
    fig.update_traces(showlegend=showlegend)
    return fig

def plotOverTime_line_df( df,
                          colorCol,
                          yearCol='year',
                          valCol='value',
                          showlegend=True,
                          fixedY=None):
    """
    `df` has specified color, year, value columns.
    """

    fig = px.line(df, x=yearCol, y=valCol, color=colorCol, markers=True)
    fig.update_traces(showlegend=showlegend)
    if fixedY is None:
        pass
    else:
        fig.add_hline(
                y=fixedY,
                line_dash="dot",
                line_color="black",
                line_width=3
        )
    return fig


def plotHeatmap( df_in, valName, dim1Name="tech", dim2Name="year", nodeName=None, fixedColor=None ):
    """
    """
    df = df_in.pivot(on=dim2Name, index=dim1Name, values=valName)

    selCols = df_in[dim2Name].unique().sort().to_list()

    heatmap_df = df.select(selCols).to_numpy()

    # Yes this duplication is annoying. I quickly need different sets of input args.
    if fixedColor is not None:
        fig = px.imshow(
            heatmap_df,
            labels=dict(x=dim1Name, y=dim2Name, color="Diff"),
            x=df_in[dim2Name].unique().sort(),
            y=df_in[dim1Name].unique().sort(),
            aspect="auto",
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint = 0.0,
            zmin=(-1.0) * fixedColor,
            zmax=fixedColor,
            origin='lower'
        )
    else:
        fig = px.imshow(
            heatmap_df,
            labels=dict(x=dim1Name, y=dim2Name, color="Diff"),
            x=df_in[dim2Name].unique().sort(),
            y=df_in[dim1Name].unique().sort(),
            aspect="auto",
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint = 0.0,
            origin='lower'
        )

    base_title = f"market_share_total Cal Diff: {nodeName}"

    fig.update_layout(
        title={
            'text': base_title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=dim2Name,
        yaxis_title=dim1Name,
        coloraxis_colorbar=dict(
            title="(model - cal)"
        ),
        width=1000,
        height=600
    )

    # Update axes to show labels clearly
    fig.update_yaxes(tickmode='linear', dtick=1)
    
    ## Add hover template for better readability
    #fig.update_traces(
    #    hovertemplate=f"<b>%{y}</b><br>{dim1Name}: %{x}<br>Diff: %{z:.3f}<extra></extra>"
    #)
    
    return fig

    
                
