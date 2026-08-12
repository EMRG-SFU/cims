
import networkx as nx
import pandas as pd
import polars as pl
import numpy as np
import scipy
import scipy.optimize as SO

import Calibration.Data.node_info as node_info
from Calibration.CIMS_Functions.lcc_calculation_calibration import lcc_calculation_faster

def update_market_shares(model, 
                         nodeName,
                         lcc_calc = lcc_calculation_faster,
                         marketShare_calc = "stock_allocation_and_retirement"):
    """
    This function calculates market shares based on a set of FICs. If the FICS at `nodeName`
    have been modified in any way, this is the function that will recalculate the new and 
    corresponding market share values.

    This is done year-to-year, starting at the initial year, exactly as in the optimization
    objecting functions.
    """
    allYears = node_info.list_years(model.graph, nodeName)
    
    # Get the actual market share method from the class by str name, and call it.
    ms_method = getattr(model, marketShare_calc, None)
    if ms_method is None or not callable(ms_method):
        raise ValueError(f"No such method: {marketShare_calc}")

    for yy in allYears:

        lcc_calc(model.graph, node=nodeName, year=yy, model=model)
        ms_method(model.graph, node=nodeName, year = yy)

