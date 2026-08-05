
import pandas as pd
import polars as pl
import networkx as nx
import sys
import os, os.path
import re
from collections.abc import Iterable
from contextlib import redirect_stdout, redirect_stderr
import types
import scipy
import scipy.optimize as SO

import Calibration.Data.node_info as node_info


def optimize_years_sequential(
        objective_maker,
        model,
        nodeName,
        init_x='zero',
        logFile="log_optimize_years_sequential.txt"):

    allTechNames = node_info.list_techs(model.graph, nodeName)
    allYears = node_info.list_years(model.graph, nodeName)

    if init_x is None:
        node_fics_init = {str(yr):[model.get_param('fic', nodeName, year=yr, tech=tv) for tv in allTechNames] for yr in allYears}
    elif isinstance(init_x, str) and init_x=='zero':
        node_fics_init = {str(yr):[0.0 for tv in allTechNames] for yr in allYears}
    else:
        # And we just assume that this thing is going to be the right shape, in the right order.
        node_fics_init = init_x

    def print_callback(intermediate_result):
        print(f"Objective Value: {format(intermediate_result.fun, '.4f')}")
        print(f"Current Optimum: {[format(a, '.3f') for a in intermediate_result.x]}")

    optResults = {}

    with open(logFile, 'w') as f:

        # Here is the year-after-year iterative loop. A complete optimization is done at each year, which alters
        # the parameters in the current shared graph object for that year. Future timepoints thus have access to
        # the already optimized parameters of previous years.
        for yr in allYears:
           
            # Print here so the year appears in the marimo cell output
            print(f"Optimizing year {yr}.")

            # Redirect the standard out/error into the logfile, to make sure that even `print` statements
            # within the optimization libraries make it
            with redirect_stdout(f):
                with redirect_stderr(f):

                    print(f"Optimizing year {yr}.")

                    objective = objective_maker(model, nodeName, yr, allTechNames)
                    # Give the optimized FIC values a floor of 0.0, so they can only be positive.
                    # ::TODO:: This assumption needs to be revisited.
                    bounds = [(0.0, None) for _ in range(len(allTechNames))]
                    startVal = objective(node_fics_init[yr])

                    min_res = SO.minimize(
                        objective,
                        node_fics_init[yr],
                        method = "L-BFGS-B",
                        bounds = bounds,
                        callback = print_callback,
                        options = {'maxiter': 5000}
                    )

                    endVal = objective(min_res.x)
                    optResults[yr] = {
                        'optObj': min_res,
                        'start': startVal,
                        'end': endVal
                    }
    return optResults

def eval_objective_function(
        objective_maker,
        model, 
        nodeName):
    """
    Calculate and return the value of the objective function produced by `objective_maker`, for the service
    node at `nodeName`, given the values of the fic parameter that are currently stored in the `calModel`'s 
    graph. This function uses the same year-by-year logic as the optimization function above. It returns
    the objective function as a sum across all years (so, a single value), and also as a dict, with the 
    objective function broken out by year.
    """

    allTechNames = node_info.list_techs(model.graph, nodeName)
    allYears = node_info.list_years(model.graph, nodeName, asStr=True)

    retDict = {}
    retSum = 0.0

    node_fics = {
        str(yv):[
            model.get_param(
                'fic', 
                nodeName, 
                year=yv, 
                tech=tv) 
            for tv in allTechNames
        ] for yv in allYears
    }

    for yr in allYears:
        objective = objective_maker(model, nodeName, yr, allTechNames)
        obj_val = objective(node_fics[yr])
        retDict[yr] = obj_val
        retSum += obj_val

    return {
        'sum': retSum,
        'byYear': retDict
    }

