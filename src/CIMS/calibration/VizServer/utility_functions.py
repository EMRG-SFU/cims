# This one's from Lumo

# Small trivial change to test the git diff-ing.

from collections.abc import Iterable

# Shared helpers live in Calibration/Utility; imported here so existing
# `utility_functions.<name>` callers keep working.
from CIMS.calibration.Calibration.Utility.dict_utils import (
    collect_dict_keys,
    collect_dict_keys_fullPath,
    collect_dict_keys_fullPath_stopTech,
)
from CIMS.calibration.Calibration.Utility.list_utils import intersect_sublists, union_of_sublists


def getNamedNode(gr, n):
    """
    This is needed because of the roundabout way you get an actual node object out of a networkX graph.
    `gr`: the graph to look in
    `n`: the name of the node to find
    """
    return( gr.nodes()[n] )

def getParamOverTime(gr, nodeName, techName, paramName):
    """
    This one contains a rigid, hardcoded definition of the years to look at.
    """
    def maybeGet(ff):
        try:
            return(ff())
        except:
            return(None)

    n = gr.nodes()[nodeName]
    yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    pVals = [maybeGet(lambda : n[yv]['technologies'][techName][paramName]['year_value']) for yv in yearVals]
    return(pVals)
    

def getAllTechNames(gr, nodeName):
    n = gr.nodes()[nodeName]
    yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
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
    yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    l = [list(n[yv]) for yv in yVals]
    return(union_of_sublists(l))

def listYearNodeParams_intersect(gr, nodeName):
    """
    Return the intersection of all parameter name lists nested under all years. This will return the
    set of parameter names which occur consistently in all years.
    """
    n = gr.nodes()[nodeName]
    yVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    l = [list(n[yv]) for yv in yVals]
    return(intersect_sublists(l))


###########################################
###########################################
###########################################


def listTechParams_allYears(gr, nodeName, techName):
    n = gr.nodes()[nodeName]
    yearVals = [yr for yr in [str(a) for a in range(2000, 2021, 5)] if yr in list(n)]
    #pVals = [n[yv]['technologies'][techName][paramName]['year_value'] for yv in yearVals]
    tp = [list(n[yv]['technologies'][techName]) for yv in yearVals]
    return(tp)

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

