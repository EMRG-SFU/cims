import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
import types
import marimo as mo
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

import Calibration.Data.node_info as node_info
from Calibration.CIMS_Functions.set_param_calibration import set_param_calibration

def get_FICs(model, nodeName, key="fic"):

    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)
    retDict = {"tech": allTechs}
    retDict.update(
            {yv:[model.get_param(key, nodeName, year=yv, tech=tv) for tv in allTechs] for yv in allYears}
    )
    return pl.DataFrame(retDict)

def set_FICs_withDataframe(model, nodeName, dataFrame, key="fic"):
    """
    Here we expect `dataFrame` to have a technology column named "tech", and the remaining columns should
    have year headers (as strings). The table should be full of numerical values. We get the service node
    within the graph of `model` identified by `nodeName`, and we load up the value in this table to the `key`
    parameter for each tech and year. (where `key` is fic by default).

    This function is designed to be used as the callback save/submit method in the `tweak_FICs` function
    below.
    """

    dfu = dataFrame.unpivot(index="tech", variable_name="year", value_name="value")

    # Redirect the rather copious output that `set_param_calibration` produces to dev/null, just in this
    # case. It really clutters up the calibration Marimo notebooks.
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for r in dfu.iter_rows(named=True):
                set_param_calibration(model, r['value'], key, nodeName, year=r['year'], tech=r['tech'], save=False)
    
    print(f"Values saved to FICS of {nodeName}.")
    return True


def tweak_FICs(model, nodeName, key='fic'):
    
    ficFrame = get_FICs(model, nodeName, key)
    return(
        mo.ui.data_editor(ficFrame).form(on_change = lambda df: set_FICs_withDataframe(model, nodeName, df))
    )

