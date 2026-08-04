# This one's from Lumo

# Small trivial change to test the git diff-ing.

from collections.abc import Mapping, Sequence, Iterable

def collect_dict_keys(structure):
    """
    Recursively walk through a nested structure of dicts and lists (or other
    sequences) and return a flat list containing *all* dictionary keys found.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.

    Returns
    -------
    list
        A list of keys (as they appear in the original objects). Duplicates are
        preserved because the same key may occur in different branches.
    """
    keys = []

    # If we’re looking at a mapping (dict‑like), record its keys and dive into values
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            keys.append(k)          # store the key itself
            keys.extend(collect_dict_keys(v))   # recurse into the value

    # If it’s a sequence (list, tuple, etc.) but *not* a string/bytes, iterate over items
    elif isinstance(structure, Sequence) and not isinstance(structure, (str, bytes, bytearray)):
        for item in structure:
            keys.extend(collect_dict_keys(item))

    # Anything else (int, float, None, custom objects…) is ignored – it can’t contain dict keys
    return keys



def collect_dict_keys_fullPath(structure, _prefix=None):
    """
    Recursively walk through a nested structure of dicts and sequences,
    returning a flat list of dictionary keys prefixed with the hierarchy
    that led to them.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.
    _prefix : tuple[str|int], optional
        Internal helper used during recursion to accumulate the path components.
        Users should not pass this argument.

    Returns
    -------
    list[str]
        A list where each entry is a hierarchical key such as
        ``key1__key2__blah`` or ``4__7__blah``. Duplicate entries are kept
        because the same key may appear in different branches.
    """
    # Normalise the prefix to a tuple for easy concatenation
    if _prefix is None:
        _prefix = ()

    collected = []

    # ------------------------------------------------------------------
    # Mapping (dict‑like) – prepend the current key to the path and dive
    # ------------------------------------------------------------------
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            # Build the new path that includes this key
            new_prefix = _prefix + (k,)
            # Store the fully‑qualified key
            collected.append("__".join(str(p) for p in new_prefix))
            # Recurse into the value, passing along the updated prefix
            collected.extend(collect_dict_keys_fullPath(v, new_prefix))

    # ---------------------------------------------------------------
    # Sequence (list, tuple, …) – use the element index as the next part
    # ---------------------------------------------------------------
    elif (
        isinstance(structure, Sequence)
        and not isinstance(structure, (str, bytes, bytearray))
    ):
        for idx, item in enumerate(structure):
            # Update the prefix with the list index
            new_prefix = _prefix + (idx,)
            # Recurse into the element; note that we do NOT add anything
            # to `collected` here because indices alone aren’t dict keys.
            collected.extend(collect_dict_keys_fullPath(item, new_prefix))

    # ------------------------------------------------------------------
    # Anything else (int, float, None, custom objects…) – nothing to do
    # ------------------------------------------------------------------
    return collected


def collect_dict_keys_fullPath_stopTech(structure, _prefix=None, stopNow=False):
    """

    This version does not interate into a node's technology dict, if it has one. It prints the keys IN the tech
    dict, but doesn't recursively call itself into those. This is just to see if this is a less overwhelming view into the
    CIMS network of services.
    
    Recursively walk through a nested structure of dicts and sequences,
    returning a flat list of dictionary keys prefixed with the hierarchy
    that led to them.

    Parameters
    ----------
    structure : Any
        The input data – typically a dict, list, tuple, or a combination thereof.
    _prefix : tuple[str|int], optional
        Internal helper used during recursion to accumulate the path components.
        Users should not pass this argument.

    Returns
    -------
    list[str]
        A list where each entry is a hierarchical key such as
        ``key1__key2__blah`` or ``4__7__blah``. Duplicate entries are kept
        because the same key may appear in different branches.
    """
    # Normalise the prefix to a tuple for easy concatenation
    if _prefix is None:
        _prefix = ()

    collected = []

    # ------------------------------------------------------------------
    # Mapping (dict‑like) – prepend the current key to the path and dive
    # ------------------------------------------------------------------
    if isinstance(structure, Mapping):
        for k, v in structure.items():
            # Build the new path that includes this key
            new_prefix = _prefix + (k,)
            # Store the fully‑qualified key
            collected.append("__".join(str(p) for p in new_prefix))
            # Recurse into the value, passing along the updated prefix
            
            #print(f"{k} and {type(v)}")
            
            if stopNow == True:
                print("We are in the stopNow thing")
                pass
            elif isinstance(v, Mapping) and ('year_value' in list(v)):
                print("Next level down has 'year_value'. Stopping.")
                pass
            elif k == 'technologies' or k == 'price multiplier':
                print("We are in technologies thing")
                collected.extend(collect_dict_keys_fullPath_stopTech(v, new_prefix, True))
            else:
                print("We are in default thing")
                collected.extend(collect_dict_keys_fullPath_stopTech(v, new_prefix, False))

    # ---------------------------------------------------------------
    # Sequence (list, tuple, …) – use the element index as the next part
    # ---------------------------------------------------------------
    elif (
        isinstance(structure, Sequence)
        and not isinstance(structure, (str, bytes, bytearray))
    ):
        for idx, item in enumerate(structure):
            # Update the prefix with the list index
            new_prefix = _prefix + (idx,)
            # Recurse into the element; note that we do NOT add anything
            # to `collected` here because indices alone aren’t dict keys.
            collected.extend(collect_dict_keys_fullPath_stopTech(item, new_prefix, False))

    # ------------------------------------------------------------------
    # Anything else (int, float, None, custom objects…) – nothing to do
    # ------------------------------------------------------------------
    return collected






def getNamedNode(gr, n):
    """
    This is needed because of the roundabout way you get an actual node object out of a networkX graph.
    `gr`: the graph to look in
    `n`: the name of the node to find
    """
    return( gr.nodes()[n] )

def intersect_sublists(list_of_lists):
    """
    Return a list with the intersection of all strings found in the nested lists.

    Parameters
    ----------
    list_of_lists : list[list[str]]
        Example: [["apple", "banana", "cherry"],
                  ["banana", "cherry", "date"],
                  ["cherry", "banana"]]

    Returns
    -------
    list[str]
        Strings that are present in *all* sub‑lists.
        Order follows the first sub‑list (you can sort later if you prefer).
    """
    if not list_of_lists:                 # empty input → empty result
        return []

    # Start with the set of the first sub‑list
    common = set(list_of_lists[0])

    # Intersect with each subsequent sub‑list
    for sublist in list_of_lists[1:]:

        # Lumo came up with this one. The explanation says that for sets, & means intersection, and the compound
        # operator with = just updates the set in place.
        common &= set(sublist)            # same as common = common.intersection(set(sublist))

        # Early exit: if nothing is common any more we can stop
        if not common:
            return []

    # Preserve the order from the first sub‑list (optional)
    return [item for item in list_of_lists[0] if item in common]


## ----------------------------------------------------------------------
## Example usage
#data = [
#    ["apple", "banana", "cherry"],
#    ["banana", "cherry", "date"],
#    ["cherry", "banana"]
#]
#
#print(intersect_sublists(data))
## Output: ['banana', 'cherry']

def union_of_sublists(list_of_lists):
    """
    Return a list with the union of all strings found in the nested lists.

    Parameters
    ----------
    list_of_lists : list[list[str]]
        Example: [["apple", "banana"], ["banana", "cherry"], ["date"]]

    Returns
    -------
    list[str]
        A list of the distinct strings, order is preserved by first appearance.
    """
    seen = set()          # tracks strings we’ve already added
    result = []           # final union list

    for sublist in list_of_lists:
        for item in sublist:
            if item not in seen:
                seen.add(item)
                result.append(item)

    return result


## Example usage
#data = [
#    ["apple", "banana"],
#    ["banana", "cherry"],
#    ["date", "apple"]
#]
#
#print(union_of_sublists(data))
# Output: ['apple', 'banana', 'cherry', 'date']



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

