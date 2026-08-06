

import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
import types
import warnings
import marimo as mo
import os
import sys
import math
from contextlib import redirect_stdout, redirect_stderr

import Calibration.Data.node_info as node_info
from Calibration.CIMS_Functions.set_param_calibration import set_param_calibration

def numFormat(x, doNumFormat=True):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    if not doNumFormat:
        return x
    if x is None:
        return None
    else:
        return f"{x:.2f}"  

def get_marketShareTotal_calibration(model, nodeName, key="calibration_market_share_total", doNumFormat=True):

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)

    def handleCalDataMissing(ff, tn):
        try:
            return ff()
        except KeyError as ke:
            warnings.warn(f"{key} not found for tech: {tn}")
            return None 

    allMSDict = {yy:{tn: handleCalDataMissing(lambda : nodeDict[yy]['technologies'][tn][key]["year_value"], tn) for tn in allTechs} for yy in allYears}

    yearTechCalMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv, doNumFormat)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechCalMS = pl.DataFrame(yearTechCalMS_pre)
    yearTechCalMS_pivot = yearTechCalMS.pivot(on="year", values="value")
    return yearTechCalMS_pivot


def set_marketShareTotal_calibration_withDataFrame(model, nodeName, dataFrame, key="calibration_market_share_total"):
    """

    """
    # Ensure that all the values sum to one for each year 
    # No... `math.isclose` is too stringent and annoying. Going to manually make it
    # more lenient, like failing if abs(diff) > 0.0001 or something.
    #
    # Also, if all the entries in the column are entirely None, don't raise an issue. We'll assume that
    # this is a case where we just don't have counterfactual data for that year, and we'll just deal somehow
    # (right now I've only seen this in the last year of the series).
    def checkSumOne(col):
        """`col` is a polars series, and has a `.name` attr which should be column header, which
        in this case is the year"""
        if all([a is None for a in col]):
            pass
        elif abs(sum([float(a) for a in col if a is not None]) - 1.0) > 0.0001:
            raise RuntimeError(f"calibration_market_share for {nodeName} in year {col.name} does not sum to one.\
                    It sums to {sum([float(a) for a in col if a is not None])}")
        else:
            pass

    for ind,col in enumerate(dataFrame.iter_columns()):
        # Skip the 1st col, which is the tech names.
        if ind > 0:
            checkSumOne(col)

    dfu = dataFrame.unpivot(index="tech", variable_name="year", value_name="value")
    
    for r in dfu.iter_rows(named=True):
        set_param_calibration(model, r['value'], key, nodeName, year=r['year'], tech=r['tech'], save=False)

    print(f"Values saved to calibration_market_share_total of {nodeName}")
    return True

def tweak_marketShareTotal_calibration(model, nodeName, key="calibration_market_share_total", doNumFormat=False):
    """

    """
    msFrame = get_marketShareTotal_calibration(model, nodeName, key, doNumFormat)
    return(
        mo.ui.data_editor(msFrame).form(on_change = lambda df: set_marketShareTotal_calibration_withDataFrame(model, nodeName, df, key))
    )


def get_marketShareTotal(model, nodeName, key="market_share_total", doNumFormat=True):

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)

    allMSDict = {yy:{tn: nodeDict[yy]['technologies'][tn][key]["year_value"] for tn in allTechs} for yy in allYears}

    yearTechMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv, doNumFormat)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechMS = pl.DataFrame(yearTechMS_pre)
    yearTechMS_pivot = yearTechMS.pivot(on="year", values="value")
    return yearTechMS_pivot


