import marimo

__generated_with = "0.24.0"
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


    from Calibration.Optimization.optimize_ms import optimize_ms_via_fics
    from Calibration.Optimization.optimize_ms_v2 import optimize_ms_via_fics_v2
    from Calibration.Optimization.optimize_ms_v2 import optimize_ms_via_fics_and_lifetimes

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


@app.cell
def _():
    nodeName="CIMS.CAN.AB.Commercial.Buildings.Hot Water"
    # nodeName="CIMS.CAN.AB.Commercial.HVAC (Cold)"
    # nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.LowMed Density.Vintage.1981-2000 Bldg Code.Heating (Cold)"
    # nodeName="CIMS.CAN.AB.Residential.Water Heating.LowMed Density"
    return (nodeName,)


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
    model_pickle_path = "C:/_dev/data_processing_calibration/cims/results/commercial/model.pkl"
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Start Graph Viz Server
    """)
    return


@app.cell
def _(model):
    # Start up Graph Server
    vizVars = {}
    threading.Thread(
        target=lambda: VizServer.server.run_server_modelObject(
            model, 
            cims_funcs=CIMS_Functions,
            vizVars = vizVars,
            PORT=5353
        ), 
        name="flask-server", 
        daemon=True
    ).start()
    return (vizVars,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bind Viz and Notebook
    """)
    return


@app.cell
def _(vizVars):
    bind_data(vizVars = vizVars)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Embed Graph Viz

    This pops up a very basic window into the CIMS graph visualizer.

    * In this window the `Fold All` and `Unfold All` buttons do what they advertise, and `Reset Zoom` rescales the zoom to 1.0, and moves the root node to the very top-left of the graph viewport.
    * The node address bar underneath that is loaded with all the node names in the loaded graph pickle. Type in here to get useful autocompletions, and hit enter to reload a new graph, with the node address as-entered as the root.
    * Clicking on a node will fold/unfold all of its children.
    * All other panning/zooming behaviour should be fairly intuitive
    * Shift-Clicking on a node will **SELECT** it, for use in notebook cell plotting functions.

    <span style="color:red; font-weight:bold; size:large">Notebook -> Graph Viz Communication Disabled Temporarily (sorry!)</span> This means that there's no indication of which node you've selected, no visual indicators of where calibration data is loaded, and no summary graphs of calibration fit. These are coming back soon.

    For now, the `vizVars` object sitting in the cell below the graph is storing the name of the currently selected node.
    """)
    return


@app.cell
def _():
    # Embed the Graph Viewer
    mo.Html(f''' 
    <iframe 
        src="http://localhost:5353/getVizPage/" 
        style="width:100%; height:700px; border:none;"
    ></iframe>
    ''')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Currently SELECTED node

    This will contain the node node that you have shift-clicked on in the graph above.
    """)
    return


@app.cell
def _(vizVars):
    vizVars
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test Calibration Functions
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
    node_info.find_nodes_with_parameter(model, "calibration_emissions_by_type")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Emissions
    """)
    return


@app.cell
def _(model, nodeName):
    edf = emissions.get_emissions(model, nodeName)
    edf
    return


@app.cell
def _(model, nodeName):
    emissions.get_emissions_calibration(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    emissions.get_emissions_calibration(model, nodeName)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Node Emissions
    """)
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions_line(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions_line(model, nodeName, patterns=['CH4','N2O'])
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions_heatmap(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotEmissions.plot_emissions_diffLine(model, nodeName)
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
    node_info.find_nodes_with_parameter(model, "calibration_quantity_requested")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Requested Quantities
    """)
    return


@app.cell
def _(model, nodeName):
    cimsRq = requestedQuantities.get_quantityRequested(model, nodeName)
    cimsRq
    return


@app.cell
def _(model, nodeName):
    calRq = requestedQuantities.get_quantityRequested_calibration(model, nodeName)
    calRq
    return


@app.cell
def _(model, nodeName):
    bla = requestedQuantities.get_quantityRequested_diff_frame(model, nodeName, join_type="outer")
    bla
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Node Requested Quantities
    """)
    return


@app.cell
def _(model, nodeName):
    plotRequestedQuantities.plot_requestedQuantities(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotRequestedQuantities.plot_requestedQuantities_line(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotRequestedQuantities.plot_requestedQuantities_heatmap(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotRequestedQuantities.plot_requestedQuantities_diffLine(model, nodeName)
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
    node_info.find_nodes_with_parameter(model, "calibration_market_share_total")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get Node Market Shares
    """)
    return


@app.cell
def _(model, nodeName):
    market_share.get_marketShareTotal(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    market_share.get_marketShareTotal_calibration(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    market_share.get_marketShare_diff_frame(model, nodeName, doNumFormat=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tweak Node Calibration Market Share Data

    Manually tweak the `calibration_market_share_total`. Update will fail unless the market shares sum to 1.0 within a given year.
    """)
    return


@app.cell
def _(model, nodeName):
    market_share.tweak_marketShareTotal_calibration(model, nodeName, doNumFormat=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Node Market Shares

    Model Data and Calibration Counterfactual
    """)
    return


@app.cell
def _(model, nodeName):
    plotMS.plot_ms(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotMS.plot_ms_line(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotMS.plot_ms_heatmap(model, nodeName)
    return


@app.cell
def _(model, nodeName):
    plotMS.plot_ms_diffLine(model, nodeName)
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
    FICs.tweak_FICs(model, nodeName, transpose = True)
    return


@app.cell
def _(model, nodeName):
    FICs.tweak_FICs(model, nodeName)
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
    ### Optimize FICs To Fit Market Share (v2)

    See `optimize_ms_v2.py` for documentation
    """)
    return


@app.cell
def _(model, nodeName):
    ### Use this function to adjust lifetimes and FICs

    fit_techs = optimize_ms_via_fics_and_lifetimes(model, nodeName, plot=True)
    return


@app.cell
def _():
    ### Use this function to adjust FICs only

    # fit_techs = optimize_ms_via_fics_v2(model, nodeName)
    return


if __name__ == "__main__":
    app.run()
