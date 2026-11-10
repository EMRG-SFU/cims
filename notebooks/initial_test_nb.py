import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

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



    # Custom cell output functions
    def mao(x):
        mo.output.append(mo.as_html(x))
    def hh(*args):
        return(mo.hstack(args))
    def vv(*args):
        return(mo.vstack(args))



    #from Calibration.cal_model import Cal_Model, find_calibration_nodes

    from CIMS.calibration.Calibration.Plotting import plot_ms_for_node

    from CIMS.calibration.Calibration.Optimization.optimize_ms import optimize_total_market_share_fic
    #from Calibration.Optimization_objectiveFunctions import make_objective_localNode

    from CIMS.calibration.Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal

    import CIMS.calibration.Calibration.Data.node_info as node_info
    import CIMS.calibration.Calibration.Data.parameter_values as parameter_values
    import CIMS.calibration.Calibration.Data.emissions as emissions
    import CIMS.calibration.Calibration.Data.quantities as requestedQuantities
    import CIMS.calibration.Calibration.Data.market_share as market_share
    import CIMS.calibration.Calibration.Data.FICs as FICs

    from CIMS.calibration.Calibration.Plotting.plot_ms_for_node import plot_ms, plot_ms_line


    from CIMS.calibration.Calibration.paramLoc import ParamLoc
    from CIMS.calibration.Calibration.paramLoc import RegexSearch as Search
    from CIMS.calibration.Calibration.paramLoc import RegexMatch as Match
    from CIMS.calibration.Calibration.paramLoc import All

    import CIMS.calibration.Calibration.utility_functions as UF

    import CIMS.calibration.Calibration.CIMS_Functions as CIMS_Functions


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
    # Set this to your own model pickle, e.g.
    #   macOS/Linux: "/path/to/your/model.pkl"
    #   Windows:     "C:/path/to/your/model.pkl"

    model_pickle_path = "C:/cims/results/Reference/model.pkl"
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



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Calibration / Model Debug Stuff

    The cells below can just be run as-is. They default to using the model loaded from the pickle at the top of this notebook file, and the node set in `selected_node` below.

    These parameters can be set to anything else, if you have an alternative graph object to examine, or if you want to look at some other node in the graph.
    """)
    return


@app.cell
def _():
    # Node used by the debug cells below. This used to be set by shift-clicking
    # a node in the embedded graph viewer (VizServer), which has been removed
    # for now. Edit this to inspect a different node.
    selected_node = "CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)"
    return (selected_node,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Non-Yearly Node Parameters
    """)
    return


@app.cell
def _(model, selected_node):
    node_info.list_nonYearly_nodeParams(model.graph, nodeName = selected_node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Node Technologies
    """)
    return


@app.cell
def _(model, selected_node):
    node_info.list_techs(model.graph, nodeName = selected_node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Node Years
    """)
    return


@app.cell
def _(model, selected_node):
    node_info.list_years(model.graph, nodeName = selected_node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Yearly Node Parameters
    """)
    return


@app.cell
def _(model, selected_node):
    node_info.list_yearly_nodeParams_intersect(model.graph, nodeName = selected_node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Yearly Tech Parameters
    """)
    return


@app.cell
def _(model, selected_node):
    node_info.list_yearly_techParams_intersect(model.graph, nodeName = selected_node, techName = "Electricity_GSHP")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## View Node Parameter Values

    <span style="color:red;">TBD</span>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## View Technology Parameter Values

    <span style="color:red;">TBD</span>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Emissions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Find Nodes With Emissions Calibration Data
    """)
    return


@app.cell
def _(model):
    sorted(list(set([a['node'] for a in node_info.searchForParam_anyYears(model.graph, "calibration_emissions_by_type")])))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Emissions

    <span style="color:red;">TBD</span>
    """)
    return


@app.cell
def _(model):
    emissions.get_emissions(model, nodeName="CIMS.CAN.AB.Residential")
    return


@app.cell
def _(model):
    emissions.get_emissions_calibration(model, nodeName="CIMS.CAN.AB.Residential")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Requested Quantities
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Find Nodes With Requested Quantities Calibration Data
    """)
    return


@app.cell
def _(model):
    sorted(list(set([a['node'] for a in node_info.searchForParam_anyYears(model.graph, "calibration_quantity_requested")])))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Requested Quantities
    """)
    return


@app.cell
def _(model):
    requestedQuantities.get_quantityRequested(model, nodeName="CIMS.CAN.AB.Residential")
    return


@app.cell
def _(model):
    requestedQuantities.get_quantityRequested(model, nodeName="CIMS.CAN.AB.Residential")
    return


@app.cell
def _(model):
    requestedQuantities.get_quantityRequested_calibration(model, nodeName="CIMS.CAN.AB.Residential")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Market Share
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Find Nodes With Total Market Share Calibration Data
    """)
    return


@app.cell
def _(model):
    mao(node_info.searchForParam_anyYears(model.graph, "calibration_market_share_total"))
    mao(sorted(list(set([a['node'] for a in node_info.searchForParam_anyYears(model.graph, "calibration_market_share_total")]))))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Market Shares

    <span style="color:red;">TBD</span>
    """)
    return


@app.cell
def _(model):
    market_share.get_marketShareTotal(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell
def _(model):
    market_share.get_marketShareTotal_calibration(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell
def _(model):
    plot_ms(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell
def _(model):
    plot_ms_line(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
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
    mao(node_info.searchForParam_anyYears(model.graph, "fic"))
    mao(sorted(list(set([a['node'] for a in node_info.searchForParam_anyYears(model.graph, "fic")]))))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node FICs For Technologies
    """)
    return


@app.cell
def _(model):
    FICs.get_FICs(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Optimization
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Optimize FICs To Fit Market Share
    """)
    return


@app.cell
def _(model):
    res = optimize_total_market_share_fic(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell
def _(model):
    aggregation_traversal(model)
    return


if __name__ == "__main__":
    app.run()
