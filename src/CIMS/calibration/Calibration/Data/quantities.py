
import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
import types

import Calibration.Data.node_info as node_info

def numFormat(x):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    return f"{x:.2f}"  


def get_quantityRequested_calibration(model, nodeName, key="calibration_quantity_requested"):

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allQDict = {yy:nodeDict[yy][key] for yy in yearHeaders}

    calYearQuants_pre = [{'year':yk, 'fuel': ff, 'value':numFormat(yearDict["year_value"])} for yk,qDict in allQDict.items() for ff,yearDict in qDict.items()]
    calYearQuants = pl.DataFrame(calYearQuants_pre)
    calYearQuants_pivot = calYearQuants.pivot(on="year", values="value")
    return calYearQuants_pivot


def get_quantityRequested(model, nodeName, key = "quantity_requested"):

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allQDict = {yy:nodeDict[yy][key]["year_value"].requested_quantities for yy in yearHeaders}

    nodeYearQuantities_pre = [{'year':yk, 'fuel':k, 'service':kk, 'value': numFormat(vv)} for yk,qDict in allQDict.items() for k,v in qDict.items() for kk,vv in v.items()]
    nodeYearQuantities = pl.DataFrame(nodeYearQuantities_pre)
    nodeYearQuantities_pivot = nodeYearQuantities.pivot(on="year", values="value")

    return nodeYearQuantities_pivot

