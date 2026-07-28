"""
A single module to aggregate everything that's been learned so far. Objective functions, scipy-based optimization
routines, etc.
"""

import networkx as nx
import pickle
import pandas as pd
import numpy as np
import scipy
import scipy.optimize as SO

import utility_functions as uf
import CIMS.lcc_calculation as LCC



def makeObjective_oneYear(modelIn, targetNodeIn, yearIn, allTechNamesIn, allTechNamesToChangeIn=None, *args, **kwargs):
    """
    
    """
    model = modelIn
    targetNode = targetNodeIn
    year = yearIn
    allTechNames = allTechNamesIn
    if allTechNamesToChangeIn is None:
        allTechNamesToChange = allTechNamesIn
    else:
        allTechNamesToChange = allTechNamesToChangeIn

    def objective(x_arr, retAll=False):
        assert len(x_arr) == len(allTechNamesToChange)
        for x,techCurr in zip(x_arr, allTechNamesToChange):
            model.set_param_calibration(x, 'fic', targetNode, year=year, tech=techCurr, save=False)

        LCC.lcc_calculation(model.graph, node=targetNode, year=year, model=model)
        model.stock_allocation_and_retirement(model.graph, node=targetNode, year=year)
        LCC.lcc_calculation(model.graph, node=targetNode, year=year, model=model)
        node_cms = [model.get_param('calibration | market share', targetNode, year=year, tech=tv)/100.0 for tv in allTechNames]
        node_ms = [float(model.get_param('market_share_total', targetNode, year=year, tech=tv)) for tv in allTechNames]
        diffs = [abs(a-b) for a,b in zip(node_cms, node_ms)]
        totalDiff = sum(diffs)
        if retAll:
            return({'totalDiff':totalDiff, 'cms':node_cms, 'ms':node_ms, 'diffs':diffs})
        else:
            return(totalDiff)

    return(objective)

def optimize_one_year(model, node, year, init_x=None):
    """
    """
    allTechNames = uf.getAllTechNames(model.graph, node)
    if init_x is None:
        node_fics_init = [model.get_param('fic', node, year=year, tech=tv) for tv in allTechNames]
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = [0.0 for tv in allTechNames]
    else:
        node_fics_init = init_x
    assert len(allTechNames) == len(node_fics_init), f"Length mismatch, allTechNames is {len(allTechNames)}, and node_fics_init is {len(node_fics_init)}"

    objective = makeObjective_oneYear(model, node, year, allTechNames)
    bounds = [(0.0, None) for _ in range(len(allTechNames))]
    def print_callback(xk):
        print(f"Current x: {xk}")
        print(f"Current value: {objective(xk)}")

    startObjVal = objective(node_fics_init)
    min_res = SO.minimize(objective,
                          node_fics_init,
                          method='L-BFGS-B',
                          bounds=bounds,
                          callback=print_callback,
                          options={'maxiter': 5000})
    endObjVal = objective(min_res.x)
    return({'optObj':min_res, 'start':startObjVal, 'end':endObjVal})
    



def makeObjective(modelIn, targetNodeIn, yearListIn, allTechNamesIn, allTechNamesToChangeIn=None, *args, **kwargs):
    """
    A different approach that optimizes across ALL the years, instead of one single year. Unsure whether
    this will work well, compared to optimizing from past to future, one year at a time.
    """
    model = modelIn
    targetNode = targetNodeIn
    allTechNames = allTechNamesIn
    yearList = yearListIn
    if allTechNamesToChangeIn is None:
        allTechNamesToChange = allTechNamesIn
    else:
        allTechNamesToChange = allTechNamesToChangeIn

    def objective(x_arr, retAll=False):
        """
        `x_arr`: This is an "unrolled" list of fic values over years and technologies. The inner iteration of
                 the unrolling is over the years, and the outer iteration is over the technologies.
        """

        currentInd = 0
        for techCurr in allTechNamesToChange:
            for yearCurr in yearList:
                model.set_param_calibration(x_arr[currentInd], 'fic', targetNode, year=yearCurr, tech=techCurr, save=False)
                currentInd += 1

        # ::TODO:: Check... but I think this works as long as the years are done from earliest to latest.
        #          This assumes that the years in the yearListIn are organized in this way. 
        for y in yearList:
            LCC.lcc_calculation(model.graph, node=targetNode, year=y, model=model)
            model.stock_allocation_and_retirement(model.graph, node=targetNode, year=y)
            LCC.lcc_calculation(model.graph, node=targetNode, year=y, model=model)

        #node_cms = [model.get_param('calibration | market share', targetNode, year=year, tech=tv)/100.0 for tv in allTechNames]
        node_cms = [[model.get_param('calibration | market share', targetNode, year=yv, tech=tv)/100.0 for yv in yearList] for tv in allTechNames]
        #node_ms = [float(model.get_param('market_share_total', targetNode, year=year, tech=tv)) for tv in allTechNames]
        node_ms = [[float(model.get_param('market_share_total', targetNode, year=yv, tech=tv)) for yv in yearList] for tv in allTechNames]

        #diffs = [abs(a-b) for a,b in zip(node_cms, node_ms)]
        diffList = []
        for node_cms_inner,node_ms_inner in zip(node_cms, node_ms):
            for a,b in zip(node_cms_inner, node_ms_inner):
                diffAcc.append(abs(a-b))
        totalDiff = sum(diffList)

        if retAll:
            return({'totalDiff':totalDiff, 'cms':node_cms, 'ms':node_ms, 'diffs':diffs})
        else:
            return(totalDiff)

    return(objective)

def concatInner(listOfLists):
    return([item for subList in listOfLists for item in subList])

def optimize(model, node, init_x=None):
    """
    The difference here is that the `x_arr` needs to span the cross product of the technologies and the years.
    """
    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)

    if init_x is None:
        node_fics_init = concatInner( [[model.get_param('fic', node, year=yv, tech=tv) for yv in allYears] for tv in allTechNames] )
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = concatInner( [[0.0 for yv in allYears] for tv in allTechNames] )
    else:
        node_fics_init = init_x

    assert len(allTechNames) == len(node_fics_init), f"Outer list needs to have length equal to allTechNames"
    assert len(allYears) == len(node_fics_init[0]), f"First inner list needs to have length equal to allYears"
    assert all([ len(node_fics_init[0]) == len(a) for a in node_fics_init]), f"All inner lists must have equal length"

    objective = makeObjective(model, node, allYears, allTechNames)
    bounds = concatInner( [[(0.0, None) for _ in range(len(allYears))] for _ in range(len(allTechNames))] ) 

    def print_callback(x):
        print(f"Current x: {x}")
        print(f"Current value: {objective(x)}")

    startObjVal = objective(node_fics_init)
    min_res = SO.minimize(objective,
                          node_fics_init,
                          method='L-BFGS-B',
                          bounds=bounds,
                          callback=print_callback,
                          options={'maxiter':5000})
    endObjVal = objective(min_res.x)
    return({'optObj':min_res, 'start':startObjVal, 'end':endObjVal})




        
