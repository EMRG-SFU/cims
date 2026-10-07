
import re
from collections.abc import Iterable

# Shared helpers live in the parent calibration package; imported here so existing
# `utility_functions.<name>` callers keep working.
from CIMS.calibration.dict_utils import (
    collect_dict_keys,
    collect_dict_keys_fullPath,
    collect_dict_keys_fullPath_stopTech,
    omit_keys,
)
from CIMS.calibration.list_utils import intersect_sublists, union_of_sublists
from CIMS.calibration import graph_utils as _shared
from CIMS.calibration.graph_utils import (
    get_all_node_years as getAllNodeYears,
    get_all_tech_names_strict as getAllTechNames,
    get_all_tech_names_union as getAllTechNames_union,
    get_named_node as getNamedNode,
    get_node_param_over_time as getParamOverTime,
    get_tech_param_over_time as getTechParamOverTime,
    list_all_node_params as listAllNodeParams,
    list_tech_params_strict as listTechParams,
    list_year_node_params_strict as listYearNodeParams,
    list_year_node_params_at as listYearNodeParams_at,
    list_year_node_params_intersect as listYearNodeParams_intersect,
    list_year_node_params_union as listYearNodeParams_union,
    search_for_param as searchForParam,
    search_for_param_any_years as searchForParam_anyYears,
)

# The only years `listTechParams_union` and `listTechParams_intersect` look at.
_YEARS_2000_2020 = [str(a) for a in range(2000, 2021, 5)]


def listTechParams_union(gr, nodeName, techName):
    """
    The union of `techName`'s parameter names over 2000-2020 in steps of 5.
    """
    return(_shared.list_tech_params_union(gr, nodeName, techName, years=_YEARS_2000_2020))


def listTechParams_intersect(gr, nodeName, techName):
    """
    The intersection of `techName`'s parameter names over 2000-2020 in steps of 5.
    """
    return(_shared.list_tech_params_intersect(gr, nodeName, techName, years=_YEARS_2000_2020))


# No
def getAllTechNames_intersect(gr, nodeName):
    """
    This one returns the INTERSECTION of each year's set of tech names. Techs returned from this are present in all
    the years.
    """
    n = gr.nodes()[nodeName]
    #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yVals = getAllNodeYears(gr, nodeName)
    try:
        return(intersect_sublists([list(n[yv]['technologies']) for yv in yVals]))
    except KeyError as e:
        if (len(e.args) == 1) and (e.args[0] == 'technologies'):
            return([])
        else:
            print(f"Args are: {e.args}")
            raise
    except Exception as e:
        print(f"Nodename here is {nodeName}")
        raise


###########################################
###########################################
###########################################

# No
def listTechParams_allYears(gr, nodeName, techName, filterRE=None):
    n = gr.nodes()[nodeName]
    #yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yearVals = getAllNodeYears(gr, nodeName)
    #pVals = [n[yv]['technologies'][techName][paramName]['year_value'] for yv in yearVals]

    if filterRE is not None:
        def filtList(ll):
            return([a for a in ll if re.search(filterRE, a, flags=re.IGNORECASE)])
        tp = [filtList(list(n[yv]['technologies'][techName])) for yv in yearVals]
        return(tp)
    else:
        tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
        return(tp)

def get_techs_with_ms_cal(gr, 
                          nodeName,
                          calVarName="calibration | market share",
                          msVarName="market_share_total"):
    """
    This produces a dict with two keys -- the 'include' entry is a list of 
    tech names that have both calibration data and the regular total market share data.
    The 'exclude' list is tech names that are missing one or the other or both.
    """
    def parsesAsFloat(x):
        try:
            _ = float(x['year_value'])
            return(True)
        except ValueError:
            return(False)

    n = gr.nodes()[nodeName]
    yearVals = getAllNodeYears(gr, nodeName)
    techVals = getAllTechNames(gr, nodeName)

    retDict = {}
    for tt in techVals:
        localList = []
        for yy in yearVals:
            localParams = list(n[yy]['technologies'][tt])
            if (calVarName in localParams) and (msVarName in localParams):
                cond1 = parsesAsFloat(n[yy]['technologies'][tt][calVarName])
                cond2 = parsesAsFloat(n[yy]['technologies'][tt][msVarName])
                if cond1 and cond2:
                    localList.append(True)
                else:
                    localList.append(False)
            else:
                localList.append(False)
        retDict[tt] = all(localList)
    return({'include':[k for k,v in retDict.items() if v],
            'exclude':[k for k,v in retDict.items() if not v]})




def findTechsWithParam(gr, yearVals, paramRE, returnDict=True):
    """
    Search through the entire graph, returning node addresses and tech names where the tech has a parameter
    that matches `paramRE` at any year (given in `yearVals`).
    `gr`: networkx graph structure
    `yearVals`: list or other iterable with years as they are used here (i.e. strings like "2005")
    `paramRE`: seach string we're looking for in the parameter name.
    `returnDict`: If this is false, just return a string that contains the info (node name, tech name, year value, matched param name). If this
               is True, then return this in a dict, which is easier for following code to deal with.
    """
    outList = []
    for nn in list(gr.nodes()):
        localNode = gr.nodes()[nn]
        for yv in yearVals:
            if 'technologies' in list(localNode[yv]):
                allTechs = localNode[yv]['technologies']
                for tech in allTechs:
                    pList = list(localNode[yv]['technologies'][tech])
                    #if any([paramRE in a for a in pList]):
                    matchList = [a for a in pList if re.search(paramRE, a, flags=re.IGNORECASE)]
                    if len(matchList) > 0:
                        # There's calibration data here.
                        if not returnDict:
                            outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                        else:
                            outList.append({'node':nn, 'tech':tech, 'year':yv, 'match':matchList})
                    else:
                        pass
            else:
                # Check if there is a 'calibration' containing parameter nested in the years
                pass
                
    return( outList )

def findTechsWithParam_anyYears(gr, paramRE, returnDict=True):
    """
    Search through the entire graph, returning node addresses and tech names where the tech has a parameter
    that matches `paramRE` at any year (given in `yearVals`).
    `gr`: networkx graph structure
    `yearVals`: list or other iterable with years as they are used here (i.e. strings like "2005")
    `paramRE`: seach string we're looking for in the parameter name.
    `returnDict`: If this is false, just return a string that contains the info (node name, tech name, year value, matched param name). If this
               is True, then return this in a dict, which is easier for following code to deal with.
    """
    outList = []
    for nn in list(gr.nodes()):
        localNode = gr.nodes()[nn]
        yearVals = getAllNodeYears(gr, nn, asStr=True)
        for yv in yearVals:
            if 'technologies' in list(localNode[yv]):
                allTechs = localNode[yv]['technologies']
                for tech in allTechs:
                    pList = list(localNode[yv]['technologies'][tech])
                    #if any([paramRE in a for a in pList]):
                    matchList = [a for a in pList if re.search(paramRE, a, flags=re.IGNORECASE)]
                    if len(matchList) > 0:
                        # There's calibration data here.
                        if not returnDict:
                            outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                        else:
                            outList.append({'node':nn, 'tech':tech, 'year':yv, 'match':matchList})
                    else:
                        pass
            else:
                # Check if there is a 'calibration' containing parameter nested in the years
                pass
                
    return( outList )



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
