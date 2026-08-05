
from collections.abc import Mapping, Sequence, Iterable
import re

from Calibration.Utility.list_utils import intersect_sublists, union_of_sublists

##################################
##################################
#  Parameter Information


## Non-Yearly Node Params

def list_nonYearly_nodeParams(gr, nodeName):
    n = gr.nodes()[nodeName]
    allKeys = list(n)
    return(allKeys)


## All Years for Node


def list_years(gr, nodeName, asStr=True):
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

## All Techs for Node
def list_techs(gr, nodeName):
    """
    The set of technologies SHOULD be the same from year to year. This happens when the union is equal to the
    intersection, so we test that here and we raise an error if that's not the case.

    Here `nodeName` can also be an iterable, in which case we return the union over all the tech sublists
    for each node.
    """
    def innerFunc(nodeName):
        n = gr.nodes()[nodeName]
        #yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
        yVals = list_years(gr, nodeName)
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
## Yearly Node Params

# At a specific year
def list_yearly_nodeParams_atYear(gr, nodeName, year):
    """
    Return the parameter name list nested under the given `year`.
    """
    n = gr.nodes()[nodeName]
    l = list(n[year])
    return(l)

# The union across all years
def list_yearly_nodeParams_union(gr, nodeName):
    """
    Return the union of all parameter name lists nested under all years. This will return all
    parameter names seen at any point in any year.
    """
    n = gr.nodes()[nodeName]
    yVals = list_years(gr, nodeName)
    l = [list(n[yv]) for yv in yVals]
    return(union_of_sublists(l))

# The intersection across all years (most useful)
def list_yearly_nodeParams_intersect(gr, nodeName):
    """
    Return the intersection of all parameter name lists nested under all years. This will return the
    set of parameter names which occur consistently in all years.
    """
    n = gr.nodes()[nodeName]
    yVals = list_years(gr, nodeName)
    l = [list(n[yv]) for yv in yVals]
    return(intersect_sublists(l))

# The set of nodeparams across all years, IFF this set is the same at each year.
# Otherwise we throw a runtime error.
def list_yearly_nodeParams(gr, nodeName):
    """
    Returns the intersection/union of all parameter name list nested under all year. This one forces
    the union to be equal to the intersection, and it throws a runtime error if it is not. This enforces the condition
    that the set of params must be consistent across the years.
    """
    n = gr.nodes()[nodeName]
    yVals = list_years(gr, nodeName)

    lol = [list(n[yv]) for yv in yVals]
    lol_inter = intersect_sublists(lol)
    lol_union = union_of_sublists(lol)
    if frozenset(lol_inter) == frozenset(lol_union):
        return lol_inter
    else:
        diffs = sorted(list(set(lol_union).difference(set(lol_inter))))
        raise RuntimeError(f"Parameter sets inconsistent across years at node: {nodeName}, diffs: {diffs}")

## Yearly Tech Params

# The union across all years of a given tech's parameters, at a given node
# If `techName` is None, this will *additionally* intergrate across all the found technologies; if the
# `techSetOp` param is 'union' we return the parameters that aren't necessarily found in every technology,
# or every year, randomly for both. If `techSetOp` is 'intersect', then the only parameter
# names this will return is those that are found in ALL technologies, but not necessarily for every year in each (and missingness pattern may vary).
def list_yearly_techParams_union(gr, nodeName, techName=None, techSetOp='union'):
    n = gr.nodes()[nodeName]
    yVals = list_years(gr, nodeName)
    if techName is not None:
        tp = [list(n[yv]['technologies'][techName]) for yv in yVals]
        return(union_of_sublists(tp))
    else:
        allTechs = list_techs(gr, nodeName)
        if techSetOp == 'union':
            return(union_of_sublists([union_of_sublists([list(n[yv]['technologies'][tn]) for yv in yVals]) for tn in allTechs]))
        elif techSetOp == 'intersect':
            return(intersect_sublists([union_of_sublists([list(n[yv]['technologies'][tn]) for yv in yVals]) for tn in allTechs]))
        else:
            raise RuntimeError("techSetOp param must be 'union' or 'intersect'")




# The intersection across all years of a given tech's parameters, at a given node (most useful)
# If `techName` is None, this will *additionally* intergrate across all the found technologies; if the
# `techSetOp` param is 'union' we return the parameters that aren't necessarily found in every technology,
# but when they are they are found in all the years. If `techSetOp` is 'intersect', then the only parameter
# names this will return is those that are found in ALL technologies, for ALL years found in each tech.
def list_yearly_techParams_intersect(gr, nodeName, techName=None, techSetOp='union'):
    n = gr.nodes()[nodeName]
    yVals = list_years(gr, nodeName)
    if techName is not None:
        tp = [list(n[yv]['technologies'][techName]) for yv in yVals]
        return(intersect_sublists(tp))
    else:
        allTechs = list_techs(gr, nodeName)
        if techSetOp == 'union':
            return(union_of_sublists([intersect_sublists([list(n[yv]['technologies'][tn]) for yv in yVals]) for tn in allTechs]))
        elif techSetOp == 'intersect':
            return(intersect_sublists([intersect_sublists([list(n[yv]['technologies'][tn]) for yv in yVals]) for tn in allTechs]))
        else:
            raise RuntimeError("techSetOp param must be 'union' or 'intersect'")

# The set of a given tech's parameters, at a given node, IFF this set is the same at each year.
# Otherwise we throw a runtime error.
def list_yearly_techParams(gr, nodeName, techName):
    """
    This one gets all the parameters for the `techName` technology in every year, and it throws an error if
    these are inconsistent across the years
    """
    n = gr.nodes()[nodeName]
    yearVals = list_years(gr, nodeName)
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    tp_union = union_of_sublists(tp)
    tp_inter = intersect_sublists(tp)
    if frozenset(tp_inter) == frozenset(tp_union):
        return tp_inter
    else:
        paramDiffs = list(set(tp_union).difference(set(tp_inter)))
        raise RuntimeError(f"Technology parameter sets inconsistent across years at node: {nodeName}, tech: {techName}, differing: {paramDiffs}")


##################################
##################################
#  Parameter search


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

        localYears = list_years(gr, nn, asStr=True)
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


##################################
##################################
#  Parameter Access

def getTechParamOverTime(gr, nodeName, techName, paramName):
    """
     
    """
    def maybeGet(ff):
        try:
            return(ff())
        except:
            return(None)

    n = gr.nodes()[nodeName]
    yearVals = list_years(gr, nodeName)
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
    yearVals = list_years(gr, nodeName)
    pVals = [maybeGet(lambda: n[yv][paramName]['year_value']) for yv in yearVals]
    return pVals


