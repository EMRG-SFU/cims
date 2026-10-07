import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import os, os.path
    import sys
    import pickle
    import gzip
    from pathlib import Path
    import pandas as pd
    import polars as pl
    import networkx as nx
    import importlib
    import copy
    import re

    # For using Flask in a cell without blocking
    import threading


    # Custom cell output functions
    def mao(x):
        mo.output.append(mo.as_html(x))
    def hh(*args):
        return(mo.hstack(args))
    def vv(*args):
        return(mo.vstack(args))

    import VizServer

    from Calibration import bind_data


    from Calibration.Optimization.optimize_ms import optimize_total_market_share_fic
    from Calibration.Optimization.optimize_ms import optimize_new_market_share_fic
    from Calibration.Optimization.optimize_ms import optimize_total_market_share_fic_lifetime

    from Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal

    import Calibration.CIMS_Functions as CIMS_Functions

    import Calibration.Data.node_info as node_info
    import Calibration.Data.parameter_values as parameter_values
    import Calibration.Data.emissions as emissions
    import Calibration.Data.quantities as requestedQuantities
    import Calibration.Data.market_share as market_share
    import Calibration.Data.FICs as FICs

    from Calibration.SubGraphs.get_subGraph_model import get_subGraph_model
    from Calibration.SubGraphs.get_subGraph_model import write_subGraph_pickle

    import Calibration.Plotting.plot_ms_for_node as plotMS
    import Calibration.Plotting.plot_emissions_for_node as plotEmissions
    import Calibration.Plotting.plot_requestedQuantities_for_node as plotRequestedQuantities

    from CIMS.utils.parameter import list as PARAM

    # Passenger Vehicle Motors is calibrated to new-vehicle shares; every other
    # passenger node to total shares. (modelled share key, calibration target key)
    SHARE_KEYS = {
        "total": ("market_share_total", "calibration_market_share_total"),
        "new_share": ("market_share_new", "calibration_market_share_new"),
    }

    def share_keys_for(model, node):
        """(modelled share key, calibration target key) for the target `node` has."""
        for _ms_key, _cal_key in SHARE_KEYS.values():
            if node in node_info.find_nodes_with_parameter(model, _cal_key):
                return _ms_key, _cal_key
        raise ValueError(f"{node} has no market share calibration target")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Load Model Pickle File
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Path to pkl file
    """)
    return


@app.cell
def _():
    # model_pickle_path = "/path/to/your/model/here.pkl"
    # or "C:\path\to\your\model.pkl"
    model_pickle_path = "C:/calibration/cims/results/Reference/buildings_transportation_optimized.pkl"
    # model_pickle_path = "C:/_dev/data_processing_calibration/cims/results/residential/model.pkl"
    return (model_pickle_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Unpickling

    For a 3-region, all-sector, 26 year pickled model, this should take less than 2 mins.

    If the pickle is already loaded into `model`, and code updating forces it to *re*-load, this can take a while (probably due to memory constraints). If this happens, it's often quicker to just restart the notebook kernel and take it again from the top.
    """)
    return


@app.cell
def _(model_pickle_path):
    with gzip.open(model_pickle_path, 'rb') as _f:
        model = pickle.load(_f)
    return (model,)


@app.cell
def _():
    nodeName = "CIMS.CAN.ON.Transportation Passenger.Passenger Vehicle Motors"
    return (nodeName,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Find Nodes With Counterfactual
    """)
    return


@app.cell
def _(model):
    {_mode: node_info.find_nodes_with_parameter(model, _cal_key)
     for _mode, (_, _cal_key) in SHARE_KEYS.items()}
    return


@app.cell
def _(model):
    node_info.find_nodes_with_parameter(model, "calibration_emissions_by_type")
    return


@app.cell
def _(model):
    node_info.find_nodes_with_parameter(model, "calibration_quantity_requested")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plots
    """)
    return


@app.cell
def _(model):
    plotRequestedQuantities.plot_requestedQuantities_line(model,"CIMS.CAN.AB.Commercial.Buildings.Shell.Transportation and Warehousing (Cold)")
    return


@app.cell
def _(model):
    plotMS.plot_ms_line_cims(model, "CIMS.CAN.QC.Commercial.Buildings.Shell.Transportation and Warehousing (Cold)")
    return


@app.cell
def _(model):
    _node = "CIMS.CAN.NB.Transportation Passenger.Mode.Urban"
    _ms_key, _cal_key = share_keys_for(model, _node)
    plotMS.plot_ms_line(model, _node, msKey = _ms_key, calMsKey = _cal_key)
    return


@app.cell
def _():
    # Total-share nodes only (new share has no lifetime lever):
    # optimize_total_market_share_fic_lifetime(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    FICs.tweak_FICs(model, nodeName, transpose = True)
    return


@app.cell
def _():
    # aggregation_traversal(model)
    return


@app.cell
def _():
    #market_share.tweak_marketShareTotal_calibration(model, nodeName=nodeName, key=share_keys_for(model, nodeName)[1], doNumFormat=False)
    return


@app.cell
def _():
    # optimize_new_market_share_fic(model, nodeName, ridge = 0)
    return


@app.cell
def _(model):
    plotEmissions.plot_emissions_line(model, "CIMS.CAN.AB.Transportation Passenger")
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions_diffLine(model, nodeName)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # FICs
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Find Nodes With FIC Values Defined in Technologies
    """)
    return


@app.cell
def _(model):
    node_info.find_nodes_with_parameter(model, "fic")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node FICs For Technologies
    """)
    return


@app.cell
def _(model, nodeName):
    FICs.get_FICs(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    FICs.tweak_FICs(model, nodeName)
    return


if __name__ == "__main__":
    app.run()
