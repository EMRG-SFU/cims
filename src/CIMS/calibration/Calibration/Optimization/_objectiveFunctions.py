
from contextlib import redirect_stdout, redirect_stderr
import networkx as nx
import pandas as pd
import polars as pl
import numpy as np
import scipy
import scipy.optimize as SO
import pickle
import sys

# Functions for computing objective function
from Calibration.CIMS_Functions.lcc_calculation_calibration import lcc_calculation_faster
from Calibration.CIMS_Functions.set_param_calibration import set_param_calibration 

def make_objective_localNode(
        model,
        nodeName,
        year,
        allTechNames,
        objective_counterFactual = 'calibration_market_share_total',
        objective_estimate = 'market_share_total',
        free_param = 'fic',
        # This was a transform for back in the day when the counterfactual ranged 0-100, instead of 0.0-1.0.
        counterFactual_transform = lambda x: x,
        # This LCC method is a completely separate function that takes the model as input
        lcc_calc = lcc_calculation_faster,
        # This refers to a member function within the model class.
        marketShare_calc = 'stock_allocation_and_retirement',
        techNamesSubset = None,
        *args,
        **kwargs):
    """

    """

    y = [
        counterFactual_transform(
            model.get_param(
                objective_counterFactual,
                nodeName,
                year = year,
                tech = tv
            )
        )
        for tv in allTechNames
    ]

    if techNamesSubset is None:
        techNamesSubset = allTechNames

    def objective(x_arr, retAll = False):

        assert len(x_arr) == len(techNamesSubset)

        for x, techCurr in zip(x_arr, techNamesSubset):
            set_param_calibration(
                model,
                x,
                free_param,
                nodeName,
                year = year,
                tech = techCurr,
                save = False
            )

        # Do the LCC calculation part of this year's CIMS routine
        lcc_calc(model.graph, node = nodeName, year = year, model = model)

        # Do the stock allocation/retirement part of this year's CIMS routine.
        # Get the actual market share method from the class by str name, and call it.
        ms_method = getattr(model, marketShare_calc, None)
        if ms_method is None or not callable(ms_method):
            raise ValueError(f"No such method: {marketShare_calc}")

        ms_method(model.graph, node = nodeName, year = year)

        # Get the resulting array of estimates of the objective parameter
        y_est = [float(
            model.get_param(
                objective_estimate,
                nodeName,
                year = year,
                tech = tv
            ))
            for tv in allTechNames
        ]

        # Calculate the difference between the objective parameter estimates
        # and the supplied counterfactual numbers.
        diffs = [abs(a-b) for a,b in zip(y, y_est)]
        totalDiff = sum(diffs)

        if retAll:
            return {
                'totalDiff': totalDiff,
                'y': y,
                'y_est': y_est,
                'diffs': diffs
            }
        else:
            return totalDiff

    return objective

