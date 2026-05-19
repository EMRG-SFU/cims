"""
Optimization Routines
"""


import networkx as nx
import pickle
import pandas as pd
import numpy as np
import scipy
import scipy.optimize as SO

import Calibration.utility_functions as uf
import CIMS.lcc_calculation as LCC


###############################################################################
#
# Single Year Objective and Optimization
#
###############################################################################


def makeObjective_oneYear_FicMs(model, 
                          targetNode, 
                          year, 
                          allTechNames,
                          techNamesToChange = None,
                          *args,
                          **kwargs):
    """
    DEPRECATED.
    Objective function for L-BFGS-B optimization of market share total output, varing 
    the `fic` values, for a single year. We still carry out an unneccesary (re)-invocation of the LCC 
    calculation after the stock allocation/retirement function.
    """
    if techNamesToChange is None:
        techNamesToChange = allTechNames

    def objective(x_arr, retAll=False):
        assert len(x_arr) == len(techNamesToChange)
        for x, techCurr in zip(x_arr, techNamesToChange):
            model.set_param_calibration(x, 'fic', targetNode, year=year, tech=techCurr, save=False)

        LCC.lcc_calculation(model.graph, node=targetNode, year=year, model=model)
        model.stock_allocation_and_retirement(model.graph, node=targetNode, year=year)
        LCC.lcc_calculation(model.graph, node=targetNode, year=year, model=model)

        # In some older data, the calibration market share is a percentage out of 100.0. In newer ones it's 
        # a fraction between 0.0 and 1.0. If a 'calDataTransform' function has been passed in kwargs, then
        # apply it to the parameter value. Otherwise just return the parameter value.
        if 'calDataTransform' in kwargs.keys():
            node_cms = [kwargs['calDataTransform'](model.get_param('calibration | market share', 
                                        targetNode, 
                                        year=year,
                                        tech=tv)) for tv in allTechNames]
        else:
            node_cms = [model.get_param('calibration | market share', 
                                        targetNode, 
                                        year=year,
                                        tech=tv) for tv in allTechNames]

        node_ms  = [float(model.get_param('market_share_total', 
                                          targetNode,
                                          year=year,
                                          tech=tv)) for tv in allTechNames]

        diffs = [abs(a-b) for a,b in zip(node_cms, node_ms)]

        if retAll:
            return( {'totalDiff': sum(diffs),
                    'cms': node_cms,
                    'ms': node_ms,
                    'diffs': diffs})
        else:
            return( sum(diffs))

    return(objective)


def optimize_one_year(model,
                     node,
                     year,
                     init_x = None):
    """
    DEPRECATED
    L-BFGS-B optimization of a single year of market share total output, varing 
    that year's `fic` values.
    """
    allTechNames = uf.getAllTechNames(model.graph, node)

    if init_x is None:
        node_fics_init = [model.get_param('fic', node, year=year, tech=tv) for tv in allTechNames]
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = [0.0 for tv in allTechNames]
    else:
        node_fics_init = init_x

    assert len(allTechNames) == len(node_fics_init), f"Length mismatch. len(allTechNames) is {len(allTechNames)}, and len(node_fics_init) is {len(node_fics_init)}."

    objective = makeObjective_oneYear_FicMs(model, node, year, allTechNames)
    bounds = [(0.0, None) for _ in range(len(allTechNames))]

    def print_callback(x):
        print(f"Current x: {x}.")
        print(f"Current value: {objective(x)}.")

    startVal = objective(node_fics_init)
    min_res = SO.minimize(objective,
                          node_fics_init,
                          method='L-BFGS-B',
                          bounds=bounds,
                          callback=print_callback,
                          options={'maxiter':5000})
    endVal = objective(min_res.x)
    return({'optObj': min_res,
            'start': startVal,
            'end': endVal})





###############################################################################
#
# Multi Year Objective and Optimization
#
###############################################################################


def makeObjective_FicMs(model,
                  targetNode,
                  yearList,
                  allTechNames,
                  techNamesToChange = None,
                  *args,
                  **kwargs):
    """
    This objective function also does L-BFGS-B optimization of total market share by adjusting `fic`, but it constructs a single
    optimization problem, defined as the sum of all the year problems; then it fiddles with ALL year `fic` values at the same time. This
    achieves good results, but it's very slow. There is considerable information in the year-to-year flow which this can't take advantage of; in
    the same way the `fic` values can only affect a limited span, so including ALL of them at each timestep pollutes the signal and I think makes
    the gradients very shallow which is what makes learning proceed so slowly.
    """

    if techNamesToChange is None:
        techNamesToChange = allTechNames

    def objective(x_arr, retAll=False):
        """
        `x_arr` is unrolled/unnested values. Innermost iteration is over the years, and outermost over the techs.
        """

        currentInd = 0
        for techCurr in techNamesToChange:
            for yearCurr in yearList:
                model.set_param_calibration(x_arr[currentInd], 'fic', targetNode, year=yearCurr, tech=techCurr, save=False)
                currentInd += 1

        for y in yearList:
            LCC.lcc_calculation(model.graph, node=targetNode, year=y, model=model)
            model.stock_allocation_and_retirement(model.graph, node=targetNode, year=y)
            LCC.lcc_calculation(model.graph, node=targetNode, year=y, model=model)

        node_cms = [[uf.maybeFloat(model.get_param('calibration | market share', targetNode, year=yv, tech=tv)/100.0) for yv in yearList] for tv in allTechNames]
        node_ms = [[uf.maybeFloat(model.get_param('market_share_total', targetNode, year=yv, tech=tv)) for yv in yearList] for tv in allTechNames]

        diffList = []
        for node_cms_inner, node_ms_inner in zip(node_cms, node_ms):
            for a, b in zip(node_cms_inner, node_ms_inner):
                try:
                    diffList.append(abs(a-b))
                except Exception as e:
                    print(f"Caught exception in objective: {e}")
                    print(f"Target node: {targetNode}")
                    raise

        totalDiff = sum(diffList)

        if retAll:
            return({'totalDiff': totalDiff,
                    'cms': node_cms,
                    'ms': node_ms,
                    'diffs': diffs})
        else:
            return(totalDiff)

    return(objective)


def concatInner(listOfLists):
    return([item for subList in listOfLists for item in subList])

def optimize(model,
             node,
             init_x = None):
    """
    DEPRECATED
    L-BFGS-B optimization of total market share using `fic`. This optimization solves
    for all years simultaneously, and is WAAAAY too slow.


    Compared to `optimize_oneYear` method above, the `init_x` here needs to span
    the cross product of the years and the technologies. Here `init_x` is a list
    of lists, with the inner list iterating over year values, and the outer list
    iterating over the technologies. Years must be in increasing order;
    technologies must be in same order that uf.getAllTechNames provides them.
    """

    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)

    if init_x is None:
        node_fics_init = [[model.get_param('fic', node, year=yv, tech=tv) for yv in allYears] for tv in allTechNames]
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = [[0.0 for yv in allYears] for tv in allTechNames]
    else:
        node_fics_init = init_x

    assert len(allTechNames) == len(node_fics_init), f"Outer list needs to have length equal to allTechNames."
    assert len(allYears) == len(node_fics_init[0]), f"First inner list needs length equal to allYears."
    assert all([len(node_fics_init[0]) == len(a) for a in node_fics_init]), f"All inner lists must have equal length."

    objective = makeObjective_FicMs(model, node, allYears, allTechNames)
    bounds = concatInner( [[(0.0, None) for _ in range(len(allYears))] for _ in range(len(allTechNames))] )

    def print_callback(x):
        print(f"Current x: {x}.")
        print(f"Current value: {objective(x)}.")

    startVal = objective(concatInner(node_fics_init))
    min_res = SO.minimize(objective,
                          concatInner(node_fics_init),
                          method='L-BFGS-B',
                          bounds=bounds,
                          callback=print_callback,
                          options={'maxiter':5000})
    endVal = objective(min_res.x)
    return({'optObj': min_res,
            'start': startVal,
            'end': endVal})



