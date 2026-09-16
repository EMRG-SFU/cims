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
from Calibration.CIMS_Functions.update_market_shares import update_market_shares
from Calibration.SubGraphs.graphFunctions import getDescendants

def get_FICs(model, nodeName, key="fic"):

    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)
    retDict = {"tech": allTechs}
    retDict.update(
            {yv:[model.get_param(key, nodeName, year=yv, tech=tv) for tv in allTechs] for yv in allYears}
    )
    return pl.DataFrame(retDict)

def get_FICs_transpose(model, nodeName, key="fic"):

    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)
    retDict = {"year": allYears}
    retDict.update(
            {tv:[model.get_param(key, nodeName, year=yv, tech=tv) for yv in allYears] for tv in allTechs}
    )
    return pl.DataFrame(retDict)

###################
###################
###################

def set_FICs_withDataframe(model, nodeName, dataFrame, key="fic", transpose=False):
    """
    Here we expect `dataFrame` to have a technology column named "tech", and the remaining columns should
    have year headers (as strings). The table should be full of numerical values. We get the service node
    within the graph of `model` identified by `nodeName`, and we load up the value in this table to the `key`
    parameter for each tech and year. (where `key` is fic by default).

    This function is designed to be used as the callback save/submit method in the `tweak_FICs` function
    below.
    """

    if transpose:
        dfu = dataFrame.unpivot(index="year", variable_name="tech", value_name="value")
    else:
        dfu = dataFrame.unpivot(index="tech", variable_name="year", value_name="value")

    # Redirect the rather copious output that `set_param_calibration` produces to dev/null, just in this
    # case. It really clutters up the calibration Marimo notebooks.
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            for r in dfu.iter_rows(named=True):
                set_param_calibration(model, r['value'], key, nodeName, year=r['year'], tech=r['tech'], save=False)
    
    print(f"Values saved to FICS of {nodeName}.")
    update_market_shares(model, nodeName)
    print(f"Market shares recalculated at {nodeName}.")

    return True


def tweak_FICs(model, nodeName, key='fic', transpose=False):
    
    if transpose:
        ficFrame = get_FICs_transpose(model, nodeName, key)
    else:
        ficFrame = get_FICs(model, nodeName, key)

    return(
        mo.ui.data_editor(ficFrame).form(on_change = lambda df: set_FICs_withDataframe(model, nodeName, df, transpose = transpose))
    )


def toCSV_FICs(model, nodeName, filePath, key="fic", recursive=True):
    """
    Write the FIC values at a node (and optionally its whole subtree) out to CSV.

    The FIC counterpart of `toCSV_marketShareTotal_calibration` in
    `market_share.py`, producing the same wide layout: one row per
    node/technology pair, one column per year, plus a leading `nodeName` column
    identifying which node each row came from. That `nodeName` column is what
    lets `fromCSV_FICs` put the values back where they belong.

    Parameters
    ----------
    model : CIMS.Model
        The model to read FIC values from.
    nodeName : str
        The node to export. With `recursive=True` this is the root of the
        subtree to export rather than the only node exported.
    filePath : str
        Path of the CSV file to write.
    key : str
        The tech parameter holding the FIC value.
    recursive : bool
        If True, also export every structural descendant of `nodeName`.

    Returns
    -------
    None
        Writes the file and prints a confirmation.

    Notes
    -----
    Nodes with no technologies are skipped, since they have no FICs to report.
    Years are written in ascending numeric order, and nodes covering different
    year ranges are merged into a common set of columns, leaving blanks where a
    node has no value for a given year.
    """
    if recursive:
        allNodes = [nodeName] + list(getDescendants(model, nodeName))
    else:
        allNodes = [nodeName]

    # Skip nodes with no technologies, since there are no tech-level FICs to report
    # for them and `get_FICs` would otherwise contribute an empty frame.
    allNodes = [n for n in allNodes if node_info.list_techs(model.graph, n)]

    def addNodeName(df, nName):
        return df.with_columns(pl.lit(nName).alias('nodeName'))

    allFICFrame = pl.concat(
        [addNodeName(get_FICs(model, n, key), n) for n in allNodes],
        how="diagonal_relaxed"
    )

    yearCols = sorted((c for c in allFICFrame.columns if c not in ('nodeName', 'tech')), key=int)
    allFICFrame = allFICFrame.select(['nodeName', 'tech'] + yearCols)

    allFICFrame.write_csv(filePath)
    print(f"Node/tech {key} information written to: {filePath}.")


def fromCSV_FICs(model, filePath, key="fic"):
    """
    Load FIC values from a CSV written by `toCSV_FICs` back into the model.

    Rows are grouped by the `nodeName` column and each node's block is handed to
    `set_FICs_withDataframe`, so the values land at whichever nodes the file
    names. The node list comes entirely from the file, meaning a CSV can be
    edited down to a subset of nodes and only those will be touched.

    Parameters
    ----------
    model : CIMS.Model
        The model to load FIC values into. Modified in place.
    filePath : str
        Path of the CSV file to read, in the layout `toCSV_FICs` writes.
    key : str
        The tech parameter to write the FIC values to.

    Returns
    -------
    bool
        True once every node in the file has been processed.

    Notes
    -----
    `set_FICs_withDataframe` recalculates market shares at each node as it goes,
    so loading a large subtree triggers one `update_market_shares` call per
    node and can take a while.
    """
    allFICFrame = pl.read_csv(filePath)

    for nName in allFICFrame['nodeName'].unique():
        nodeFrame = allFICFrame.filter(pl.col('nodeName') == nName).drop('nodeName')
        set_FICs_withDataframe(model, nName, nodeFrame, key)

    return True
