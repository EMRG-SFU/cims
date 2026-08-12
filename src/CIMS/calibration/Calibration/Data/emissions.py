import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
from operator import itemgetter
import types

import Calibration.Data.node_info as node_info

def numFormat(x):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    return f"{x:.2f}"  



def get_emissions_calibration(model, nodeName, key="calibration_emissions_by_type", getDict=False):
    """
    Retrieve calibration emissions and create dataframe with identifying info in columns, and emissions
    values across years
    """
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allEmDict = {yy:nodeDict[yy][key] for yy in yearHeaders}

    calYearEmissions_pre = [{'year':yk, 'gas': gg, 'value': numFormat(yearDict["year_value"])} for yk,emDict in allEmDict.items() for gg,yearDict in emDict.items()]
    if getDict:
        return calYearEmissions_pre

    calYearEmissions = pl.DataFrame(calYearEmissions_pre)
    calYearEmissions_pivot = calYearEmissions.pivot(on="year", values="value")
    return calYearEmissions_pivot

def get_emissions(model, nodeName, key="emissions_total_cumul_net", getDict=False):
    """
    Retrieve emissions object and create dataframe with identifying info in columns, and emissions
    values across the years.
    """
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    # Here we're just interested in the `key` rowName (and this extra structure is the reason that Emissions (and Quantities)
    # always showed up as an error in the 
    allEmDict = {yy:nodeDict[yy][key]["year_value"].emissions for yy in yearHeaders}

    nodeYearEmissions_pre = [{'year':yk, 'fuel':k, 'gas':kk, 'type':kkk, 'value':numFormat(vvv['year_value'])} 
     for yk,emDict in allEmDict.items() 
     for k,v in emDict.items()
     for kk,vv in v.items()
     for kkk,vvv in vv.items()]
    
    if getDict:
        return nodeYearEmissions_pre

    nodeYearEmissions = pl.DataFrame(nodeYearEmissions_pre)
    #print(nodeYearEmissions)
    nodeYearEmissions_pivot = nodeYearEmissions.pivot(on="year", values="value")
    return nodeYearEmissions_pivot


def get_emissions_both_dict(model,
                            nodeName,
                            key_cims = "emissions_total_cumul_net",
                            key_cal = "calibration_emissions_by_type",
                            missingValFunc = None):
    """

    """
     
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    cimsFrame = pl.DataFrame(get_emissions(model, nodeName, getDict=True))
    cimsFrame = cimsFrame.with_columns(pl.col(["value"]).cast(pl.Float64))
    cimsFrame = cimsFrame.group_by(["year","gas"]).agg(pl.col(["value"]).sum()).sort("year")

    calFrame = pl.DataFrame(get_emissions_calibration(model, nodeName, getDict=True))
    calFrame = calFrame.with_columns(pl.col(["value"]).cast(pl.Float64))

    return {'cims': cimsFrame, 
            'cal': calFrame}


def get_emissions_diff_frame(model,
                             nodeName,
                             key_cims = "emissions_total_cumul_net",
                             key_cal = "calibration_emissions_by_type",
                             missingValFunc = None):

    cims, cal = itemgetter('cims','cal')(get_emissions_both_dict(model,
                                                                 nodeName,
                                                                 key_cims,
                                                                 key_cal,
                                                                 missingValFunc))
    cims = cims.rename({'value':'cims_value'})
    cal = cal.rename({'value':'cal_value'})
    both_frame = cims.join(cal, on=['year','gas'], how="inner")
    out_frame = both_frame.with_columns(
            (pl.col('cims_value') - pl.col('cal_value')).alias("diff")
    )
    out_frame = out_frame.with_columns(
            (pl.col('diff') / pl.col('cims_value')).alias("pctDiff_cims"),
            (pl.col('diff') / pl.col('cal_value')).alias("pctDiff_cal")
    )

    return out_frame


