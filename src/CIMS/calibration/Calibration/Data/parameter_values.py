import pandas as pd
import polars as pl
import re


from collections.abc import Iterable
from functools import reduce
import types

import Calibration.Data.node_info as node_info

#import Calibration.utility_functions as UF
#import Calibration.plotting as plotting
#from Calibration.CIMS_Functions.set_param_calibration import set_param_calibration


## Get Emissions, Quantities, and any Standard Tech Parameter, over the model years.

# EMISSIONS

def get_emissions(model, nodeName, emissions_key="emissions_total_cumul_net"):
    """
    This function retrieves and deals with the emissions objecst found at 
    """
    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    # Here we're just interested in the `emissions_key` rowName (and this extra structure is the reason that Emissions (and Quantities)
    # always showed up as an error in the 
    allEmDict = {yy:nodeDict[yy][emissions_key]["year_value"].emissions for yy in yearHeaders}

    nodeYearEmissions_pre = [{'year':yk, 'fuel':k, 'gas':kk, 'type':kkk, 'value':numFormat(vvv['year_value'])} 
     for yk,emDict in allEmDict.items() 
     for k,v in emDict.items()
     for kk,vv in v.items()
     for kkk,vvv in vv.items()]

    nodeYearEmissions = pl.DataFrame(nodeYearEmissions_pre)
    #print(nodeYearEmissions)
    nodeYearEmissions_pivot = nodeYearEmissions.pivot(on="year", values="value")
    return nodeYearEmissions_pivot


# QUANTITIES

def get_quantities(model, nodeName, rqKey = "quantity_requested"):

    nodeDict = model.graph.nodes().get(nodeName)
    yearHeaders = node_info.list_years(model.graph, nodeName)

    allQDict = {yy:nodeDict[yy][rqKey]["year_value"].requested_quantities for yy in yearHeaders}

    nodeYearQuantities_pre = [{'year':yk, 'fuel':k, 'service':kk, 'value': numFormat(vv)} for yk,qDict in allQDict.items() for k,v in qDict.items() for kk,vv in v.items()]
    nodeYearQuantities = pl.DataFrame(nodeYearQuantities_pre)
    nodeYearQuantities_pivot = nodeYearQuantities.pivot(on="year", values="value")

    return nodeYearQuantities_pivot


## GENERAL NODE PARAMS (i.e. all the other things in node yearDicts besides "technologies"). These are treated here as things that
##     can by default and without finessing be shown in a table cell, as rendered by Marimo from a polars dataframe

def get_nodeParams(model, nodeName, inclParams=None):
    """
    Gets a dataframe of param values, years in the columns, parameters in the rows. The value of `model.get_param` is 
    shown in the table cell
    """
    nodeDict = model.graph.nodes().get(nodeName)
    allYears = node_info.list_years(model.graph, nodeName)
    rowNames = set(reduce(lambda x,y : x+y, [[a for a in nodeDict[yy].keys()] for yy in allYears]))

    if inclParams is not None:
        # Get sorted rowNames list of params that are somewhere in the inclParams list. This list can be
        # in whatever order, and may contain whatever strings they want. Something will only happen if the
        # string happens to be a param name at that node.
        rowNames = [a for a in sorted(rowNames) if a in inclParams]
    else:
        rowNames = sorted(rowNames)
        
    def getMaybe(ff):
        try:
            return(ff())
        except Exception as e:
            return(f"Error: {str(e)}.")

    retDict = {'paramName' : rowNames}
    retDict.update(
            {f"y_{yv}":[model.get_param(rowName, nodeName, year=yv) for rowName in rowNames] for yv in allYears}
            )
    return pl.DataFrame(retDict)


# GENERAL TECH PARAMS (i.e. market_share_total, market_share_new, calibration_market_share_total, etc)

def get_techParam(model, nodeName, pName):
    """
    Gets a dataframe of a single parameter value for all techs at a node, across time. Years in the columns
    and the different tachnologies are the rows.
    """

    allYears = node_info.list_years(model.graph, nodeName)
    allTechNames = node_info.list_techs(model.graph, nodeName)
    retDict = {"techName": allTechNames}
    retDict.update(
            {f"y_{yv}":[model.get_param(pName, nodeName, year=yv, tech=tv) for tv in allTechNames] for yv in allYears}
            )
    return pl.DataFrame(retDict)

