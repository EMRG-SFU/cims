

import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
import types
import warnings

import Calibration.Data.node_info as node_info

def numFormat(x):
    """
    Turn numbers in these tables into formatted strings having a particular
    number (let's start with 2) decimal places.
    """
    if x is None:
        return None
    else:
        return f"{x:.2f}"  

def get_marketShareTotal_calibration(model, nodeName, key="calibration_market_share_total"):

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

    yearTechCalMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechCalMS = pl.DataFrame(yearTechCalMS_pre)
    yearTechCalMS_pivot = yearTechCalMS.pivot(on="year", values="value")
    return yearTechCalMS_pivot


def set_marketShareTotal_calibration_withDataFrame(model, nodeName, dataFrame, key="calibration_market_share_total"):
    """

    """
    pass 


def get_marketShareTotal(model, nodeName, key="market_share_total"):

    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)

    allMSDict = {yy:{tn: nodeDict[yy]['technologies'][tn][key]["year_value"] for tn in allTechs} for yy in allYears}

    yearTechMS_pre = [{'year': yk, 'tech': tn, 'value': numFormat(vv)} for yk,techDict in allMSDict.items() for tn,vv in techDict.items()]
    yearTechMS = pl.DataFrame(yearTechMS_pre)
    yearTechMS_pivot = yearTechMS.pivot(on="year", values="value")
    return yearTechMS_pivot


