import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import os, os.path
    import sys
    import pickle
    from pathlib import Path
    import pandas as pd
    import polars as pl
    import networkx as nx
    import importlib
    import copy

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

    #from Calibration.cal_model import Cal_Model, find_calibration_nodes

    from Calibration.Plotting import plot_ms_for_node, plot_ms_for_node_line

    from Calibration.Optimization.optimize_ms import optimize_ms_via_fics
    #from Calibration.Optimization_objectiveFunctions import make_objective_localNode

    from Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal

    import Calibration.Data.node_info as node_info
    import Calibration.Data.parameter_values as parameter_values
    import Calibration.Data.emissions as emissions
    import Calibration.Data.quantities as requestedQuantities
    import Calibration.Data.market_share as market_share
    import Calibration.Data.FICs as FICs

    from Calibration.Plotting.plot_ms_for_node import plot_ms
    from Calibration.Plotting.plot_ms_for_node_line import plot_ms_line


    from Calibration.paramLoc import ParamLoc
    from Calibration.paramLoc import RegexSearch as Search
    from Calibration.paramLoc import RegexMatch as Match
    from Calibration.paramLoc import All

    import Calibration.utility_functions as UF

    import Calibration.CIMS_Functions as CIMS_Functions


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

    model_pickle_path = "/Users/matt/Projects/CIMS/Calibration/CIMS_Calibration_Folder/TestData/model_3regions_withCalibration_abres.pickle"
    #model_pickle_path = "/Users/matt/Projects/CIMS/Calibration/CIMS_Calibration_Folder/TestData/modelPost_3regions_withCalibrationQuantFix.pickle"
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
    with open(model_pickle_path, 'rb') as _f:
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
    # Calibration / Model Debug Stuff

    The cells below can just be run as-is, if a node has been selected from the graph. They default to using the model loaded from the pickle at the top of this notebook file, and the currently selected node in the graph viz.

    These parameters can be set to anything else, if you have an alternative graph object to examine, or if you want to hardcode the retrieval to some other node in the graph.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Non-Yearly Node Parameters
    """)
    return


@app.cell
def _(model, vizVars):
    node_info.list_nonYearly_nodeParams(model.graph, nodeName = vizVars["selectedNode"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Node Technologies
    """)
    return


@app.cell
def _(model, vizVars):
    node_info.list_techs(model.graph, nodeName = vizVars["selectedNode"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Node Years
    """)
    return


@app.cell
def _(model, vizVars):
    node_info.list_years(model.graph, nodeName = vizVars["selectedNode"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Yearly Node Parameters
    """)
    return


@app.cell
def _(model, vizVars):
    node_info.list_yearly_nodeParams_intersect(model.graph, nodeName = vizVars["selectedNode"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## List Yearly Tech Parameters
    """)
    return


@app.cell
def _(model, vizVars):
    node_info.list_yearly_techParams_intersect(model.graph, nodeName = vizVars["selectedNode"], techName = "Electricity_GSHP")
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
    res = optimize_ms_via_fics(model, nodeName="CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.1981-2000 Bldg Code.Heating (Cold)")
    return


@app.cell
def _(model):
    aggregation_traversal(model)
    return


if __name__ == "__main__":
    app.run()
