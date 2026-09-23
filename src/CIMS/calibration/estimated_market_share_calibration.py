import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import pickle
    import gzip

    from Calibration.Optimization.optimize_ms_v2 import optimize_ms_via_fics_v2
    from Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal
    from Calibration.Utility.write_fics import write_fics

    import Calibration.Data.node_info as node_info
    import Calibration.Data.market_share as market_share

    import Calibration.Plotting.plot_ms_for_node as plotMS
    import Calibration.Plotting.plot_emissions_for_node as plotEmissions
    import Calibration.Plotting.plot_requestedQuantities_for_node as plotRequestedQuantities


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Estimated Market Share Calibration

    Calibrates nodes against `estimated_market_share_total` -- an external
    (non-CEUD) technology-mix estimate -- instead of
    `calibration_market_share_total`. Use this for nodes CEUD doesn't survey
    by technology (e.g. Lighting's Incandescent/CFL/LED split).

    `nodeName` below drives the single-node plots/tweak/optimize cells;
    `nodeNames` drives the all-nodes batch loops (Optimize All + Write FICs
    All), same split `batch_optimization.py` and `calibration_residential.py`
    use.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Model Pickle File
    """)
    return


@app.cell
def _():
    model_pickle_path = "C:/calibration/cims/results/Reference/transportation_freight_optimized.pkl"
    return (model_pickle_path,)


@app.cell
def _(model_pickle_path):
    with gzip.open(model_pickle_path, 'rb') as _f:
        model = pickle.load(_f)
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nodes
    """)
    return


@app.cell
def _():
    nodeNames = [
        "CIMS.CAN.ON.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.AB.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.BC.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.SK.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.MB.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.QC.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.NB.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.NS.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.PE.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.NL.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.NT.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.NU.Fuel Blends.Gasoline_Transportation",
        "CIMS.CAN.YT.Fuel Blends.Gasoline_Transportation",
    ]
    return (nodeNames,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plots — All Nodes

    One figure per node in `nodeNames`, printed with a node-name header above
    each -- same pattern as `batch_optimization.py`'s "Plot Market Shares —
    All Nodes" cell, just for all three plot types.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Market Share — All Nodes
    """)
    return


@app.cell
def _(model, nodeNames):
    for _node in nodeNames:
        print(f"--- {_node} ---")
        plotMS.plot_ms_line(model, _node, calMsKey="estimated_market_share_total")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Demand (Requested Quantities) — All Nodes
    """)
    return


@app.cell
def _(model, nodeNames):
    for _node in nodeNames:
        print(f"--- {_node} ---")
        plotRequestedQuantities.plot_requestedQuantities_line_cims(model, _node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Emissions — All Nodes

    Lighting nodes genuinely have no emissions data of their own (it
    attributes upstream at the fuel/supply node), so most/all of these will
    print "no emissions data" rather than a figure -- expected, not an error.
    """)
    return


@app.cell
def _(model, nodeNames):
    for _node in nodeNames:
        print(f"--- {_node} ---")
        try:
            plotEmissions.plot_emissions_line(model, _node)
        except Exception as _e:
            print(f"  no emissions data: {type(_e).__name__}: {_e}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Tweak Estimated Market Shares
    """)
    return


@app.cell
def _(model, nodeNames):
    for _node in nodeNames:
        mo.output.append(mo.md(f"### {_node}"))
        mo.output.append(
            market_share.tweak_marketShareTotal_calibration(
                model, _node, key = "estimated_market_share_total", doNumFormat=False, transpose=True
            )
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Optimize All Nodes
    """)
    return


@app.cell
def _(model, nodeNames):
    fit_results = {}
    for _i, _n in enumerate(nodeNames, start=1):
        fit_results[_n] = optimize_ms_via_fics_v2(
            model, _n, objective_counterFactual="estimated_market_share_total", verbose=False
        )
        print(f"[{_i}/{len(nodeNames)}] fit {_n}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Re-aggregate

    Whole-model recompute (no node argument) -- run once, after fitting, before
    exporting FICs.
    """)
    return


@app.cell
def _(model):
    aggregation_traversal(model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Write FICs — All Nodes
    """)
    return


@app.cell
def _():
    calibration_output_dir = "C:/calibration/cims/data/model_inputs/calibration_outputs"
    return (calibration_output_dir,)


@app.cell
def _(calibration_output_dir, model, nodeNames):
    for _n in nodeNames:
        write_fics(model, _n, calibration_output_dir, source="calibration_estimated_fic_export")
    print(f"wrote fics for {len(nodeNames)} node(s) to {calibration_output_dir}")
    return


if __name__ == "__main__":
    app.run()
