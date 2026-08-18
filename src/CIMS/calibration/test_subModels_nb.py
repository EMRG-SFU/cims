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
    import re

    # For using Flask in a cell without blocking
    import threading

    import VizServer


    # Custom cell output functions
    def mao(x):
        mo.output.append(mo.as_html(x))
    def hh(*args):
        return(mo.hstack(args))
    def vv(*args):
        return(mo.vstack(args))

    import Calibration.CIMS_Functions as CIMS_Functions
    from Calibration.SubGraphs.get_subGraph_model import get_subGraph_model
    #from Calibration.SubGraphs.get_subGraph_model import get_custom_model
    from Calibration.SubGraphs.get_subGraph_model import write_subGraph_pickle

    from Calibration.SubGraphs.single_sector_all_region import get_all_region_names
    from Calibration.SubGraphs.single_sector_all_region import get_all_sector_names
    from Calibration.SubGraphs.single_sector_all_region import get_single_sector_all_region
    from Calibration.SubGraphs.single_sector_all_region import write_single_sector_all_region_pickle


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
    model_pickle_path = "/Users/matt/Projects/CIMS/Calibration/TestData/modelPost_3regions_withCalibrationQuantFix.pickle"
    return (model_pickle_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Unpickling

    If the pickle is already loaded into `model`, and code updating forces it to *re*-load, this can take a while (probably due to memory constraints). If this happens, it's often quicker to just restart the notebook kernel and take it again from the top.
    """)
    return


@app.cell
def _(model_pickle_path):
    with open(model_pickle_path, 'rb') as _f:
        model = pickle.load(_f)
    return (model,)


@app.cell
def _(model):
    get_all_region_names(model)
    return


@app.cell
def _(model):
    get_all_sector_names(model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Subgraph Model Creation

    With functions from `SubGraphs` you can split apart large model objects. The CIMS graph is structurally a tree (with `CIMS` as the root), giving addresses such as `CIMS.CAN.BC.Commercial. ...`. These functions re-root the tree at the node given by `nodeName`, and discard any parts of the graph that are not connected by outgoing graph edges. If this is done at the sector level (i.e. "CIMS.CAN.BC.Residential"), then the graph will only contain information on that sector, in that region, but it will contain ALL such information necessary for running cims.

    The `DCC` classes are recomputed, using the appropriate model member functions, as otherwise they will include information on regions the graph no longer contains, causing errors in any calculations involving dcc. The penalty for this is that the value is no longer a truly accurate DCC, being computed on a regional subset of information.

    The `get_subgraph_model` function returns the submodel object.
    """)
    return


@app.cell
def _(model):
    bc_residential_model = get_subGraph_model(model, 'CIMS.CAN.BC.Residential')
    return


@app.cell
def _(model):
    bc_residential_clothesDrying_model = get_subGraph_model(model, 'CIMS.CAN.BC.Residential.Clothes Drying')
    return


@app.cell
def _(model):
    all_residential_model = get_single_sector_all_region(model, "Residential")
    return


@app.cell
def _():
    regions = ['CIMS.CAN.AB', 'CIMS.CAN.BC']
    sectors = ['Ethanol', 'Electricity', 'Light Industrial']

    # Not implemented yet, but will be soon.
    #customModel = get_custom_model(model, regions = regions, sectors = sectors)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `write_subgraph_pickle` function creates the submodel as above, and writes it to a (gzipped) pickle file at the given filepath.
    """)
    return


@app.cell
def _(model):
    write_subGraph_pickle(model, 'CIMS.CAN.BC.Residential', output_filepath="test_bc_residential_submodel.pkl")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same thing with `write_single_sector_all_region_pickle`. It creates the requested sub-model, then saves it
    to the specified pickle filepath as a gzipped pickle.
    If the `output_filepath` is `None` it saves it as `allRegions_{sectorName}.pkl`.
    """)
    return


@app.cell
def _(model):
    write_single_sector_all_region_pickle(model, 'Commercial', output_filepath=None)
    return


if __name__ == "__main__":
    app.run()
