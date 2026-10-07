# This one's from Lumo

# Small trivial change to test the git diff-ing.

from collections.abc import Iterable

# Shared helpers live in the parent calibration package; imported here so existing
# `utility_functions.<name>` callers keep working.
from CIMS.calibration.dict_utils import (
    collect_dict_keys,
    collect_dict_keys_fullPath,
    collect_dict_keys_fullPath_stopTech,
)
from CIMS.calibration.list_utils import intersect_sublists, union_of_sublists
from CIMS.calibration import graph_utils as _shared
from CIMS.calibration.graph_utils import (
    get_all_node_years as getAllNodeYears,
    get_named_node as getNamedNode,
    list_all_node_params as listAllNodeParams,
    list_year_node_params_at as listYearNodeParams_at,
)

# The years VizServer looks at. The shared helpers in graph_utils use every year
# on a node unless told otherwise; the functions below pass this list instead.
VIZ_YEARS = [str(a) for a in range(2000, 2021, 5)]


def getParamOverTime(gr, nodeName, techName, paramName):
    """
    The value of `paramName` for technology `techName` at each of VIZ_YEARS present on the node.
    """
    return(_shared.get_tech_param_over_time(gr, nodeName, techName, paramName, years=VIZ_YEARS))
    

def getAllTechNames(gr, nodeName):
    """
    The union of each year's tech names, over the VIZ_YEARS present on the node.
    """
    return(_shared.get_all_tech_names_union(gr, nodeName, years=VIZ_YEARS))

def listYearNodeParams_union(gr, nodeName):
    """
    Return the union of all parameter name lists nested under the VIZ_YEARS present on the node.
    """
    return(_shared.list_year_node_params_union(gr, nodeName, years=VIZ_YEARS))

def listYearNodeParams_intersect(gr, nodeName):
    """
    Return the intersection of all parameter name lists nested under the VIZ_YEARS present on the node.
    """
    return(_shared.list_year_node_params_intersect(gr, nodeName, years=VIZ_YEARS))

def listTechParams_union(gr, nodeName, techName):
    """
    The union of `techName`'s parameter names over the VIZ_YEARS present on the node.
    """
    return(_shared.list_tech_params_union(gr, nodeName, techName, years=VIZ_YEARS))

def listTechParams_intersect(gr, nodeName, techName):
    """
    The intersection of `techName`'s parameter names over the VIZ_YEARS present on the node.
    """
    return(_shared.list_tech_params_intersect(gr, nodeName, techName, years=VIZ_YEARS))


###########################################
###########################################
###########################################


def listTechParams_allYears(gr, nodeName, techName):
    n = gr.nodes()[nodeName]
    yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    #pVals = [n[yv]['technologies'][techName][paramName]['year_value'] for yv in yearVals]
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    return(tp)

def findTechsWithParam(gr, yearVals, paramRE, getDict=True):
    """
    Search through the entire graph, returning node addresses and tech names where the tech has a parameter
    named `paramName` at any year (given in `yearVals`).
    `gr`: networkx graph structure
    `yearVals`: list or other iterable with years as they are used here (i.e. strings like "2005")
    `paramRE`: seach string we're looking for in the parameter name.
    `getDict`: If this is false, just return a string that contains the info (node name, tech name, year value, matched param name). If this
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
                    matchList = [a for a in pList if paramRE in a]
                    if len(matchList) > 0:
                        # There's calibration data here.
                        if not getDict:
                            outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                        else:
                            outList.append({'node':nn, 'tech':tech, 'year':yv, 'match':matchList})
                    else:
                        pass
            else:
                # Check if there is a 'calibration' containing parameter nested in the years
                pass
                
    return( outList )

def searchForParam(gr, yearVals, paramRE):
    """
    Similar to above, but with broader mandate; here we just look for any occurrence of `paramRE`, whether that be in a tech, a regular node as
    a 'non-year' parameter, or within the year dicts but not in the nested tech dicts.
    """
    outList = []
    for nn in list(gr.nodes()):
        localNode = gr.nodes()[nn]
        matchedParams = [a for a in list(localNode) if paramRE in a]
        if len(matchedParams) > 0:
            outList.append(f"NodeMatch: {nn} -- {matchedParams}")
    
        for yv in yearVals:
            matchedParams = [a for a in list(localNode[yv]) if paramRE in a]
            if len(matchedParams) > 0:
                outList.append(f"NodeYearMatch: {nn} -- {yv} -- {matchedParams}")
            if 'technologies' in list(localNode[yv]):
                allTechs = localNode[yv]['technologies']
                for tech in allTechs:
                    pList = list(localNode[yv]['technologies'][tech])
                    #if any([paramRE in a for a in pList]):
                    matchList = [a for a in pList if paramRE in a]
                    if len(matchList) > 0:
                        # There's calibration data here.
                        outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                    else:
                        pass
                
    return( outList )


##################################
##################################
##################################
##################################

#  These are stolen from updates the Calibration package, and its Data.node_info module. I should abstract that out so
#  new things there don't need to be repeated here... but right now I need some things repeated.
#
#  (Now abstracted out: these names point at the shared functions in graph_utils.)

## Yearly Tech Params
list_yearly_techParams_union = _shared.list_yearly_tech_params_union
list_yearly_techParams_intersect = _shared.list_yearly_tech_params_intersect

## All Years for Node
list_years = getAllNodeYears

## All Techs for Node
list_techs = _shared.get_all_tech_names_strict
