"""
This does market share (and more generally any single node parameter) optimization by solving 
separate optimizations for each year sequentially.

This works as well as solving for all the years simultaneously, and is also a LOT faster
because each optimization problem is much smaller, with few FIC inputs, and the estimated
gradients are much better behaved, so the algorithm can tune itself better and it runs faster.
"""



from contextlib import redirect_stdout, redirect_stderr
import networkx as nx
import pickle
import sys
import pandas as pd
import numpy as np
import scipy
import scipy.optimize as SO

from skopt.plots import plot_gaussian_process
from skopt import gp_minimize

# Tracing stuff, to try get a view of the function calls in the objective.
import pyinstrument
import functiontrace

import Calibration.utility_functions as uf
import CIMS.lcc_calculation as LCC

dbgStruct = {}

def make_objective(model, node, year, allTechNames, allTechNamesToChange=None, *args, **kwargs):
    """
    The objective function returned by this optimizes a specific year (unlike the objectives in `optimization.py`,
    and calls the "full" version of the LCC calculation.
    """
    
    y = [model.get_param('calibration | market share', node, year=year, tech=tv)/100.0 for tv in allTechNames]
    if allTechNamesToChange is None:
        allTechNamesToChange = allTechNames

    def objective(x_arr, retAll=False):
        
        dbgStruct['info'] = {'x_arr':x_arr, 'names':allTechNamesToChange}
        assert len(x_arr) == len(allTechNamesToChange)

        for x, techCurr in zip(x_arr, allTechNamesToChange):
            model.set_param_calibration(x, 'fic', node, year=year, tech=techCurr, save=False)

        LCC.lcc_calculation(model.graph, node=node, year=year, model=model)
        model.stock_allocation_and_retirement(model.graph, node=node, year=year)

        y_hat = [float(model.get_param('market_share_total', node, year=year, tech=tv)) for tv in allTechNames]

        diffs = [abs(a-b) for a, b in zip(y, y_hat)]
        totalDiff = sum(diffs)

        if retAll:
            return({'totalDiff': totalDiff,
                    'y': y,
                    'y_hat': y_hat,
                    'diffs': diffs})
        else:
            return(totalDiff)

    return(objective)

def make_objective_faster(model, node, year, allTechNames, allTechNamesToChange=None, *args, **kwargs):
    """
    The objective function returned by this calls a "calibrationHack" version of the `lcc_calculation`, where
    only quantities that are strictly required for generating new market shares are computed. This can cut the
    total optimization runtime almost in half.

    It also only optimizes a specific year, unlike the objective functions in `optimization.py`.
    """
    
    y = [model.get_param('calibration | market share', node, year=year, tech=tv)/100.0 for tv in allTechNames]
    if allTechNamesToChange is None:
        allTechNamesToChange = allTechNames

    def objective(x_arr, retAll=False):
        
        dbgStruct['info'] = {'x_arr':x_arr, 'names':allTechNamesToChange}
        assert len(x_arr) == len(allTechNamesToChange)

        for x, techCurr in zip(x_arr, allTechNamesToChange):
            model.set_param_calibration(x, 'fic', node, year=year, tech=techCurr, save=False)

        LCC.lcc_calculation_calibrationHack(model.graph, node=node, year=year, model=model)
        model.stock_allocation_and_retirement(model.graph, node=node, year=year)

        y_hat = [float(model.get_param('market_share_total', node, year=year, tech=tv)) for tv in allTechNames]

        diffs = [abs(a-b) for a, b in zip(y, y_hat)]
        totalDiff = sum(diffs)

        if retAll:
            return({'totalDiff': totalDiff,
                    'y': y,
                    'y_hat': y_hat,
                    'diffs': diffs})
        else:
            return(totalDiff)

    return(objective)


def optimize(model, node, init_x=None, logFile="log_optimizeYearByYear.txt"):
    """
    Run L-BFGS-B optimization using the standard objective function, on each year consecutively. Each year modifies the
    graph structure passed in (`model`), and because the years only refer to those previous, we can plausibly decompose this way, 
    saying that a history of optimized market shares is a collection where each year was optimized to be the best that it can be
    individually.
    """


    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)
    
    if init_x is None:
        node_fics_init = {str(yr):[model.get_param('fic', node, year=yv, tech=tv) for tv in allTechNames] for yr in allYears}
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = {str(yr):[0.0 for tv in allTechNames] for yr in allYears}
    else:
        # And we just assume that this thing is going to be the right shape, in the right order.
        node_fics_init = init_x

    def print_callback(xk):
        #print(f"Type of xk: {type(xk)}")
        #print(f"dir(xk): {dir(xk)}")
        print(f"Current x: {[format(a, ".3f") for a in xk]}")
        print(f"Current value: {objective(xk)}")

    def print_callback_2(intermediate_result):
        print(f"Objective Value: {format(intermediate_result.fun, '.4f')}")
        print(f"Objective Compd: {format(objective(intermediate_result.x), '.4f')}")

        print(f"Current x: {[format(a, '.3f') for a in intermediate_result.x]}")

    optResults = {}

    with open(logFile, 'w') as f:
        for yr in allYears:
            print(f"Optimizing year {yr}")
            with redirect_stdout(f):
                with redirect_stderr(f):
                    objective = make_objective(model, node, yr, allTechNames)
                    bounds = [(0.0, None) for _ in range(len(allTechNames))]
                    startVal = objective(node_fics_init[yr])
                    # Repeat this so it's written into the log file.
                    print(f"Optimizing year {yr}")
                    min_res = SO.minimize(objective,
                                          node_fics_init[yr],
                                          #method='BFGS',
                                          method='L-BFGS-B',
                                          #method='Nelder-Mead',
                                          #method='CG',
                                          bounds=bounds,
                                          callback=print_callback_2,
                                          options={'maxiter':5000})
            endVal = objective(min_res.x)
            optResults[yr] = {'optObj':min_res, 'start': startVal, 'end': endVal}

    return(optResults)


def optimize_faster(model, node, init_x=None, logFile="log_optimizeYearByYear_calibrationHack.txt"):
    """
    As above, but this one uses the pared-down objective that uses a minimal LCC calculation.
    """

    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)
    
    if init_x is None:
        node_fics_init = {str(yr):[model.get_param('fic', node, year=yv, tech=tv) for tv in allTechNames] for yr in allYears}
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = {str(yr):[0.0 for tv in allTechNames] for yr in allYears}
    else:
        # And we just assume that this thing is going to be the right shape, in the right order.
        node_fics_init = init_x

    def print_callback(xk):
        print(f"Current x: {[format(a, ".3f") for a in xk]}")
        print(f"Current value: {objective(xk)}")

    def print_callback_2(intermediate_result):
        print(f"Objective Value: {format(intermediate_result.fun, '.4f')}")
        print(f"Objective Compd: {format(objective(intermediate_result.x), '.4f')}")
        print(f"Current x: {[format(a, '.3f') for a in intermediate_result.x]}")

    optResults = {}

    with open(logFile, 'w') as f:
        for yr in allYears:
            print(f"Optimizing year {yr}")
            with redirect_stdout(f):
                with redirect_stderr(f):
                    objective = make_objective_faster(model, node, yr, allTechNames)
                    bounds = [(0.0, None) for _ in range(len(allTechNames))]
                    startVal = objective(node_fics_init[yr])
                    # Repeat this so it's written into the log file.
                    print(f"Optimizing year {yr}")
                    min_res = SO.minimize(objective,
                                          node_fics_init[yr],
                                          #method='BFGS',
                                          method='L-BFGS-B',
                                          #method='Nelder-Mead',
                                          #method='CG',
                                          bounds=bounds,
                                          callback=print_callback_2,
                                          options={'maxiter':5000})
            endVal = objective(min_res.x)
            optResults[yr] = {'optObj':min_res, 'start': startVal, 'end': endVal}

    return(optResults)


def optimize_faster_bayes(model, node, acq_func="gp_hedge", logFile="log_optimizeBayes.txt"):
    """
    Experimental Bayesian Gaussian process modelling of the optimization landscape.
    `gp_minimize` function is really slow and needs a lot of tweaking.
    Actually not that useful to do at all years like this, as in depth focus on a single year
    is really what this is for. Because of the runtime this function is *really* slow.
    """
    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)
    optResults = {}

    with open(logFile, 'w', buffering=1) as f:
        for yr in allYears:
            print(f"Optimizing year {yr}")
            #with redirect_stdout(f):
                #with redirect_stderr(f):
            try:
                objective = make_objective_faster(model, node, yr, allTechNames)
                print("now are here")
                min_res = gp_minimize(objective,
                                      [(0.0, 9999999.0) for _ in range(len(allTechNames))],
                                      acq_func=acq_func,
                                      verbose=True,
                                      random_state=1234)
                print("and now here...")
                optResults[yr] = min_res
            except:
                print("chucking from the except here...")
                f.flush()
                raise
    return(optResults)

def optimize_basinHop(model, node, init_x=None, logFile="log_optimizeYearByYear_basinHop.txt"):
    """
    Experimental Globel L-BFGS-B optimization incorporating basin hopping to better explore the 
    optimization landscape.
    Theoretically I don't think there's any way this should be able to do *worse* that the non-basinhopping
    L-BFGS, but it does seem to be.
    """


    allTechNames = uf.getAllTechNames(model.graph, node)
    allYears = uf.getAllNodeYears(model.graph, node, asStr=True)
    
    if init_x is None:
        node_fics_init = {str(yr):[model.get_param('fic', node, year=yv, tech=tv) for tv in allTechNames] for yr in allYears}
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = {str(yr):[0.0 for tv in allTechNames] for yr in allYears}
    else:
        # And we just assume that this thing is going to be the right shape, in the right order.
        node_fics_init = init_x


    def bfgs_callback(intermediate_result):
        print("**** Running bfgs callback ****")
        print(f"Objective Value: {format(intermediate_result.fun, '.4f')}")
        print(f"Current x: {[format(a, '.3f') for a in intermediate_result.x]}")

    def basin_callback(x, f, accept):
        print("**** Running basin callback ****")
        print(f"Basin_Callback: x: {x}, f: {f}, accept: {accept}")

    optResults = {}

    with open(logFile, 'w') as f:
        for yr in allYears:
            print(f"Optimizing year {yr}")
            with redirect_stdout(f):
                with redirect_stderr(f):
                    objective = make_objective(model, node, yr, allTechNames)
                    bounds = [(0.0, None) for _ in range(len(allTechNames))]
                    startVal = objective(node_fics_init[yr])
                    # Repeat this so it's written into the log file.
                    print(f"Optimizing year {yr}")
                    min_res = SO.basinhopping(func=objective,
                                          x0=node_fics_init[yr],
                                          minimizer_kwargs={'method':'L-BFGS-B', 'callback':bfgs_callback, 'bounds':bounds},
                                          callback=basin_callback)
            endVal = objective(min_res.x)
            optResults[yr] = {'optObj':min_res, 'start': startVal, 'end': endVal}

    return(optResults)



