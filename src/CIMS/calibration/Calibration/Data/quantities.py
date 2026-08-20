
import pandas as pd
import polars as pl
import re
import copy

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


def get_quantityRequested_calibration(model, nodeName, key="calibration_quantity_requested", getDict=False):

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allQDict = {yy:nodeDict[yy][key] for yy in yearHeaders}

    calYearQuants_pre = [{'year':yk, 'fuel': ff, 'value':numFormat(yearDict["year_value"])} for yk,qDict in allQDict.items() for ff,yearDict in qDict.items()]
    if getDict:
        return calYearQuants_pre

    calYearQuants = pl.DataFrame(calYearQuants_pre)
    calYearQuants_pivot = calYearQuants.pivot(on="year", values="value")
    return calYearQuants_pivot


def get_quantityRequested(model, nodeName, key = "quantity_requested", getDict=False):

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allQDict = {yy:nodeDict[yy][key]["year_value"].requested_quantities for yy in yearHeaders}

    nodeYearQuantities_pre = [{'year':yk, 'fuel':k, 'service':kk, 'value': numFormat(vv)} for yk,qDict in allQDict.items() for k,v in qDict.items() for kk,vv in v.items()]
    if getDict:
        return nodeYearQuantities_pre

    nodeYearQuantities = pl.DataFrame(nodeYearQuantities_pre)
    nodeYearQuantities_pivot = nodeYearQuantities.pivot(on="year", values="value")

    return nodeYearQuantities_pivot


def get_quantityRequested_both_dict(model,
                                    nodeName,
                                    key_cims = "quantity_requested",
                                    key_cal = "calibration_quantity_requested",
                                    missingValFunc = None):
    """

    """

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    cimsFrame = pl.DataFrame(get_quantityRequested(model, nodeName, getDict=True))
    cimsFrame = cimsFrame.with_columns(pl.col("value").cast(pl.Float64))

    cimsFrame = cimsFrame.group_by(["year","fuel"]).agg(pl.col("value").sum()).sort("year")

    calFrame = pl.DataFrame(get_quantityRequested_calibration(model, nodeName, getDict=True))
    calFrame = calFrame.with_columns(pl.col("value").cast(pl.Float64))

    return {'cims': cimsFrame, 
            'cal': calFrame}

def get_quantityRequested_diff_frame(model,
                                     nodeName,
                                     key_cims = "quantity_requested",
                                     key_cal = "calibration_quantity_requested",
                                     join_type = "inner",
                                     missingValFunc = None):
    """

    """
    cims, cal = itemgetter('cims','cal')(get_quantityRequested_both_dict(model,
                                                                         nodeName,
                                                                         key_cims,
                                                                         key_cal,
                                                                         missingValFunc))
    cims = cims.rename({'value': 'cims_value'})
    cal = cal.rename({'value': 'cal_value'})
    if join_type == "inner":
        both_frame = cims.join(cal, on=['year','fuel'], how='inner')
    elif join_type == "outer":
        both_frame = cims.join(cal, on=['year','fuel'], how='outer')
    else:
        raise NotImplemented(f"join type {join_type} not accepted here.")

    out_frame = both_frame.with_columns(
        (pl.col('cims_value') - pl.col('cal_value')).alias('diff')
    )
    out_frame = out_frame.with_columns(
        (pl.col('diff') / pl.col('cims_value')).alias("pctDiff_cims"),
        (pl.col('diff') / pl.col('cal_value')).alias("pctDiff_cal")
    )

    return out_frame


