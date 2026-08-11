

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

def handleCalDataMissing(ff, tn, key=None):
    try:
        return ff()
    except KeyError as ke:
        if key is None:
            warnings.warn(f"Given key not found for tech: {tn}")
        else:
            warnings.warn(f"{key} not found for tech: {tn}")
        return None 


def get_marketShareTotal_calibration(model, nodeName, key="calibration_market_share_total", doNumFormat = True, transpose = False):
    """
    Mostly for purpose of displaying dataframe in Marimo cell. For data to process further, like for optimization objective functions,
    use `get_marketShare_both` function below
    """

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)


    allMSDict = {yy:{tn: handleCalDataMissing(lambda : nodeDict[yy]['technologies'][tn][key]["year_value"], tn, key) for tn in allTechs} for yy in allYears}

    yearTechCalMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv, doNumFormat)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechCalMS = pl.DataFrame(yearTechCalMS_pre)
    if transpose:
        yearTechCalMS_pivot = yearTechCalMS.pivot(on="tech", values="value")
    else:
        yearTechCalMS_pivot = yearTechCalMS.pivot(on="year", values="value")
    return yearTechCalMS_pivot


def set_marketShareTotal_calibration_withDataFrame(model, nodeName, dataFrame, key="calibration_market_share_total", transpose = False):
    """

    """
    # Ensure that all the values sum to one for each year 
    # No... `math.isclose` is too stringent and annoying. Going to manually make it
    # more lenient, like failing if abs(diff) > 0.0001 or something.
    #
    # Also, if all the entries in the column are entirely None, don't raise an issue. We'll assume that
    # this is a case where we just don't have counterfactual data for that year, and we'll just deal somehow
    # (right now I've only seen this in the last year of the series).
    def checkSumOne(ser):
        """`ser` is a polars series, and has a `.name` attr which should be the series header (whether row or col or whatever, which
        in this case is the year"""
        if all([a is None for a in ser]):
            pass
        elif abs(sum([float(a) for a in ser if a is not None]) - 1.0) > 0.0001:
            raise RuntimeError(f"calibration_market_share for {nodeName} in year {ser.name} does not sum to one.\
                    It sums to {sum([float(a) for a in ser if a is not None])}")
        else:
            pass

    if transpose:
        for ind,row in enumerate(dataFrame.iter_rows()):
            # Skip the 1st row, which is the tech names.
            if ind > 0:
                checkSumOne(row)
    else:
        for ind,col in enumerate(dataFrame.iter_columns()):
            # Skip the 1st col, which is the tech names.
            if ind > 0:
                checkSumOne(col)

    if transpose:
        dfu = dataFrame.unpivet(index="year", variable_name="tech", value_name="value")
    else:
        dfu = dataFrame.unpivot(index="tech", variable_name="year", value_name="value")
    
    for r in dfu.iter_rows(named=True):
        set_param_calibration(model, r['value'], key, nodeName, year=r['year'], tech=r['tech'], save=False)

    print(f"Values saved to calibration_market_share_total of {nodeName}")
    return True

def tweak_marketShareTotal_calibration(model, nodeName, key="calibration_market_share_total", doNumFormat = False, transpose = False):
    """

    """
    msFrame = get_marketShareTotal_calibration(model, nodeName, key, doNumFormat, transpose = transpose)
    return(
        mo.ui.data_editor(msFrame).form(on_change = lambda df: set_marketShareTotal_calibration_withDataFrame(model, nodeName, df, key, transpose = transpose))
    )


def get_marketShareTotal(model, nodeName, key="market_share_total", doNumFormat=True):
    """
    Mostly for purpose of displaying dataframe in Marimo cell. For data to process further, like for optimization objective functions,
    use `get_marketShare_both` function below
    """

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)

    allMSDict = {yy:{tn: nodeDict[yy]['technologies'][tn][key]["year_value"] for tn in allTechs} for yy in allYears}

    yearTechMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv, doNumFormat)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechMS = pl.DataFrame(yearTechMS_pre)
    yearTechMS_pivot = yearTechMS.pivot(on="year", values="value")
    return yearTechMS_pivot



def calMissingToZero(cimsVal, calVal, year=None, tech=None, key_cims=None, key_cal=None):
    """ Here we raise a RuntimeError if cimsVal is None or missing, but we assume a missing calibration value
    just means that the market share is 0.0, in that situation. We return explicit zero values in the corresponding
    locations in the data.
    """
    if cimsVal is None:
        raise RuntimeError(f"cimsVal is None. Year: {year}, Tech: {tech}, key_cims: {key_cims}.")

    if calVal is None:
        return {'cims': cimsVal,
                'cal': 0.0}
    else:
        return {'cims': cimsVal,
                'cal' : calVal}

def calMissingRemove(cimsVal, calVal, year=None, tech=None, key_cims=None, key_cal=None):
    """ Here we raise a RuntimeError if cimsVal is None or missing, but we assume a missing calibration value
    just means that the market share is 0.0, in that situation. We return explicit zero values in the corresponding
    locations in the data.

    """
    if cimsVal is None:
        raise RuntimeError(f"cimsVal is None. Year: {year}, Tech: {tech}, key_cims: {key_cims}.")

    if calVal is None:
        return {'cims': cimsVal,
                'cal': 0.0}
    else:
        return {'cims': cimsVal,
                'cal' : calVal}


def get_marketShare_both_dict(model, 
                         nodeName, 
                         key_cims = "market_share_total",
                         key_cal = "calibration_market_share_total",
                         missingValFunc = calMissingToZero):
    """

    """

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)

    allMSDict = {yy:{tn: calMissingToZero(nodeDict[yy]['technologies'][tn][key_cims]["year_value"],
                                          handleCalDataMissing(lambda : nodeDict[yy]['technologies'][tn][key_cal]["year_value"], tn, key_cal)) 
                     for tn in allTechs} 
                 for yy in allYears}
    return allMSDict


def get_marketShare_diff_frame(model, 
                         nodeName, 
                         key_cims = "market_share_total",
                         key_cal = "calibration_market_share_total",
                         doNumFormat = False,
                         absolute = False):
    """

    """
    
    msDict = get_marketShare_both_dict(model, nodeName, key_cims, key_cal)

    def makeDiff(nCims, nCal, doNumFormat, absolute=False):
        if absolute:
            return numFormat(abs(nCims - nCal), doNumFormat)
        else:
            return numFormat(nCims - nCal, doNumFormat)

    out_pre = [{'year': yk, 
                'tech': tn, 
                'value': makeDiff(vv['cims'], 
                                  vv['cal'], doNumFormat
                                  )} 
               for yk,techDict in msDict.items() 
               for tn, vv in techDict.items()]

    out = pl.DataFrame(out_pre)
    out_pivot = out.pivot(on="year", values="value")
    return out_pivot

