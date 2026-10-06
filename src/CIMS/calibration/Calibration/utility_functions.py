
import re
from collections.abc import Iterable

# Shared helpers live in Calibration/Utility; imported here so existing
# `utility_functions.<name>` callers keep working.
from CIMS.calibration.Calibration.Utility.dict_utils import (
    collect_dict_keys,
    collect_dict_keys_fullPath,
    collect_dict_keys_fullPath_stopTech,
    omit_keys,
)
from CIMS.calibration.Calibration.Utility.list_utils import intersect_sublists, union_of_sublists


def getNamedNode(gr, n):
    """
    This is needed because of the roundabout way you get an actual node object out of a networkX graph.
    `gr`: the graph to look in
    `n`: the name of the node to find
    """
    return( gr.nodes()[n] )

def getTechParamOverTime(gr, nodeName, techName, paramName):
    """
     
    """
    def maybeGet(ff):
        try:
            return(ff())
        except:
            return(None)

    n = gr.nodes()[nodeName]
    #yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yearVals = getAllNodeYears(gr, nodeName)
    pVals = [maybeGet(lambda : n[yv]['technologies'][techName][paramName]['year_value']) for yv in yearVals]
    return(pVals)

def getParamOverTime(gr, nodeName, paramName):
    """

    """
    def maybeGet(ff):
        try:
            return ff()
        except:
            return None

    n = gr.nodes()[nodeName]
    yearVals = getAllNodeYears(gr, nodeName)
    pVals = [maybeGet(lambda: n[yv][paramName]['year_value']) for yv in yearVals]
    return pVals

# No
def getAllTechNames_union(gr, nodeName):
    """
    This one returnsr the UNION of each year's set of tech names. Any techs that appear in at least one year
    will be returned by this.
    """
    n = gr.nodes()[nodeName]
    #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yVals = getAllNodeYears(gr, nodeName)
    try:
        return(union_of_sublists([list(n[yv]['technologies']) for yv in yVals]))
    except KeyError as e:
        if (len(e.args) == 1) and (e.args[0] == 'technologies'):
            return([])
        else:
            print(f"Args are: {e.args}")
            raise
    except Exception as e:
        print(f"Nodename here is {nodeName}")
        raise


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


#No
def getAllTechNames(gr, nodeName):
    """
    The set of technologies SHOULD be the same from year to year. This happens when the union is equal to the
    intersection, so we test that here and we raise an error if that's not the case.

    Here `nodeName` can also be an iterable, in which case we return the union over all the tech sublists
    for each node.
    """
    def innerFunc(nodeName):
        n = gr.nodes()[nodeName]
        #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
        yVals = getAllNodeYears(gr, nodeName)
        try:
            subListInter = intersect_sublists([list(n[yv]['technologies']) for yv in yVals])
            subListUnion = union_of_sublists([list(n[yv]['technologies']) for yv in yVals])
            if frozenset(subListInter) == frozenset(subListUnion):
                return(subListInter)
            else:
                raise RuntimeError("Tech names are inconsistent across years here.")
        except KeyError as e:
            if (len(e.args) == 1) and (e.args[0] == 'technologies'):
                return([])
            else:
                print(f"Args are: {e.args}")
                raise
        except Exception as e:
            print(f"Nodename here is {nodeName}")
            raise

    if isinstance(nodeName, Iterable) and not isinstance(nodeName, str):
        subLists = []
        for nn in nodeName:
            subLists.append(innerFunc(nn))
        return(union_of_sublists(subLists))
    else:
        return(innerFunc(nodeName))

def getAllNodeYears(gr, nodeName, asStr=True):
    """
    We take a slightly different approach to year-finding here; we list all the dict keys at `nodeName` in graph `gr`, and we say
    a year is any key that successfully parses as an int.
    """
    n = gr.nodes()[nodeName]
    n_keys = list(n)

    def parsesAsInt(x):
        try:
            v = int(x)
            return(True)
        except ValueError as ve:
            return(False)
    if not asStr:
        ys = [int(a) for a in n_keys if parsesAsInt(a)]
    else:
        ys = [str(a) for a in n_keys if parsesAsInt(a)]
    return(ys)


def listAllNodeParams(gr, nodeName):
    n = gr.nodes()[nodeName]
    allKeys = list(n)
    return(allKeys)

def listYearNodeParams_at(gr, nodeName, year):
    """
    Return the parameter name list nested under the given `year`.
    """
    n = gr.nodes()[nodeName]
    l = list(n[year])
    return(l)

def listYearNodeParams_union(gr, nodeName):
    """
    Return the union of all parameter name lists nested under all years. This will return all
    parameter names seen at any point in any year.
    """
    n = gr.nodes()[nodeName]
    #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yVals = getAllNodeYears(gr, nodeName)
    l = [list(n[yv]) for yv in yVals]
    return(union_of_sublists(l))

def listYearNodeParams_intersect(gr, nodeName):
    """
    Return the intersection of all parameter name lists nested under all years. This will return the
    set of parameter names which occur consistently in all years.
    """
    n = gr.nodes()[nodeName]
    #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    yVals = getAllNodeYears(gr, nodeName)
    l = [list(n[yv]) for yv in yVals]
    return(intersect_sublists(l))

def listYearNodeParams(gr, nodeName):
    """
    Returns the intersection/union of all parameter name list nested under all year. This one forces
    the union to be equal to the intersection, and it throws a runtime error if it is not. This enforces the condition
    that the set of params must be consistent across the years.
    """
    n = gr.nodes()[nodeName]
    yVals = getAllNodeYears(gr, nodeName)

    lol = [list(n[yv]) for yv in yVals]
    lol_inter = intersect_sublists(lol)
    lol_union = union_of_sublists(lol)
    if frozenset(lol_inter) == frozenset(lol_union):
        return lol_inter
    else:
        diffs = sorted(list(set(lol_union).difference(set(lol_inter))))
        raise RuntimeError(f"Parameter sets inconsistent across years at node: {nodeName}, diffs: {diffs}")


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


def listTechParams_union(gr, nodeName, techName):
    n = gr.nodes()[nodeName]
    yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    return(union_of_sublists(tp))
def listTechParams_intersect(gr, nodeName, techName):
    n = gr.nodes()[nodeName]
    yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    return(intersect_sublists(tp))

def listTechParams(gr, nodeName, techName):
    """
    This one gets all the parameters for the `techName` technology in every year, and it throws an error if
    these are inconsistent across the years
    """
    n = gr.nodes()[nodeName]
    yearVals = getAllNodeYears(gr, nodeName)
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    tp_union = union_of_sublists(tp)
    tp_inter = intersect_sublists(tp)
    if frozenset(tp_inter) == frozenset(tp_union):
        return tp_inter
    else:
        paramDiffs = list(set(tp_union).difference(set(tp_inter)))
        raise RuntimeError(f"Technology parameter sets inconsistent across years at node: {nodeName}, tech: {techName}, differing: {paramDiffs}")


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

def searchForParam(gr, yearVals, paramRE, returnDict=True):
    """
    Similar to above, but with broader mandate; here we just look for any occurrence of `paramRE`, whether that be in a tech, a regular node as
    a 'non-year' parameter, or within the year dicts but not in the nested tech dicts.
    """
    outList = []
    for nn in list(gr.nodes()):
        localNode = gr.nodes()[nn]
        matchedParams = [a for a in list(localNode) if re.search(paramRE, a, flags=re.IGNORECASE)]
        if len(matchedParams) > 0:
            if returnDict:
                outList.append({'type': 'node', 'node': nn, 'match': matchedParams})
            else:
                outList.append(f"NodeMatch: {nn} -- {matchedParams}")
    
        for yv in yearVals:
            matchedParams = [a for a in list(localNode[yv]) if re.search(paramRE, a, flags=re.IGNORECASE)]
            if len(matchedParams) > 0:
                if returnDict:
                    outList.append({'type': 'nodeYear', 'node': nn, 'year': yv, 'match': matchedParams})
                else:
                    outList.append(f"NodeYearMatch: {nn} -- {yv} -- {matchedParams}")

            if 'technologies' in list(localNode[yv]):
                allTechs = localNode[yv]['technologies']
                for tech in allTechs:
                    pList = list(localNode[yv]['technologies'][tech])
                    #if any([paramRE in a for a in pList]):
                    matchList = [a for a in pList if re.search(paramRE, a, flags=re.IGNORECASE)]
                    if len(matchList) > 0:
                        # There's calibration data here.
                        if returnDict:
                            outList.append({'type':'tech', 'node': nn, 'year':yv, 'tech':tech, 'match':matchList})
                        else:
                            outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                    else:
                        pass
                
    return( outList )

def searchForParam_anyYears(gr, paramRE, returnDict=True, *args, **kwargs):
    """
    We attempt to match the `paramRE` search string at a node, within a nodes "year" dictionaries, and within the
    year dictionary's technology dictionary.
    """
    outList = []
    for nn in list(gr.nodes()):
        localNode = gr.nodes()[nn]
        matchedParams = [a for a in list(localNode) if re.search(paramRE, a, flags=re.IGNORECASE)]
        if len(matchedParams) > 0:
            if returnDict:
                outList.append({'type': 'node', 'node': nn, 'match':matchedParams})
            else:
                outList.append(f"NodeMatch: {nn} -- {matchedParams}")

        localYears = getAllNodeYears(gr, nn, asStr=True)
        for yv in localYears:
            matchedParams = [a for a in list(localNode[yv]) if re.search(paramRE, a, flags=re.IGNORECASE)]
            if len(matchedParams) > 0:
                if returnDict:
                    outList.append({'type': 'nodeYear', 'node': nn, 'year': yv, 'match': matchedParams})
                else:
                    outList.append(f"NodeYearMatch: {nn} -- {yv} -- {matchedParams}")

            if 'technologies' in list(localNode[yv]):
                allTechs = localNode[yv]['technologies']
                for tech in allTechs:
                    pList = list(localNode[yv]['technologies'][tech])
                    matchList = [a for a in pList if re.search(paramRE, a, flags=re.IGNORECASE)]
                    if len(matchList) > 0:
                        if returnDict:
                            outList.append({'type':'tech', 'node': nn, 'year':yv, 'tech': tech, 'match':matchList})
                        else:
                            outList.append(f"{nn} -- {tech} -- {yv} -- {matchList}")
                    else:
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
