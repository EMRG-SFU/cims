import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import pickle
    import gzip
    import time

    import polars as pl

    from Calibration.Optimization.optimize_ms import (
        optimize_total_market_share_fic,
        optimize_total_market_share_fic_lifetime,
        optimize_on_slice,
    )
    from Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal
    from Calibration.Utility.calibration_outputs import (
        group_by_sector,
        last_recorded_fits,
        log_file_for,
        market_share_l1,
        record_fit,
        write_calibration_outputs,
    )

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

    `nodeNames` drives every all-nodes loop (plots, tweak, optimize, write),
    the same way `batch_optimization.py` does. `target_key` sets the target
    series for the plots, the tweak tables, the fit and the fit check together,
    so they always describe the same quantity.

    Outputs go to `<calibration_output_root>/<sector>/fitted_fics` (and
    `fitted_lifetimes` when lifetimes are fitted), one sector folder per node,
    which is where Reference.py's `calibration_output_sectors` reads them from.
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
    # The target series every section below compares against and fits to.
    target_key = "estimated_market_share_total"
    return (target_key,)


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


@app.cell
def _(model, nodeNames, target_key):
    # Nodes without the target series cannot be fitted; the fit and write
    # cells skip them.
    _with_target = set(node_info.find_nodes_with_parameter(model, target_key))
    fit_nodes = [n for n in nodeNames if n in _with_target]
    _missing = [n for n in nodeNames if n not in _with_target]
    print(f"{len(fit_nodes)}/{len(nodeNames)} node(s) have {target_key}")
    if _missing:
        print(f"no {target_key} (will be skipped):", _missing)
    return (fit_nodes,)


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
def _(model, nodeNames, target_key):
    for _node in nodeNames:
        print(f"--- {_node} ---")
        plotMS.plot_ms_line(model, _node, calMsKey=target_key)
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
    ## Check Fit — All Nodes

    L1 error against `target_key` of the loaded model (all years but the base
    year), next to the error the last recorded fit ended at
    (`<sector>/logs/fit_summary.csv`). Nothing is fitted or recomputed. See
    the same section in `batch_optimization.py`.
    """)
    return


@app.cell
def _(calibration_output_root, fit_nodes, model, target_key):
    _recorded = last_recorded_fits(calibration_output_root, fit_nodes)
    _rows = []
    for _node in fit_nodes:
        _last = _recorded.get(_node, {})
        _now = market_share_l1(model, _node, target_key)
        _fit = _last.get("L1_after")
        _rows.append({
            "node": _node,
            "L1_now": round(_now, 4),
            "L1_last_fit": None if _fit is None else round(_fit, 4),
            "drift": None if _fit is None else round(_now - _fit, 4),
            "last_fit_stage": _last.get("Stage"),
            "last_fit_time": _last.get("Time"),
        })
    pl.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Tweak Estimated Market Shares
    """)
    return


@app.cell
def _(model, nodeNames, target_key):
    for _node in nodeNames:
        mo.output.append(mo.md(f"### {_node}"))
        mo.output.append(
            market_share.tweak_marketShareTotal_calibration(
                model, _node, key = target_key, doNumFormat=False, transpose=True
            )
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Optimize All Nodes

    - **fics only** runs `optimize_total_market_share_fic` (the fit this
      notebook has always run) and writes fics only, leaving lifetimes alone.
    - **fics + lifetimes** runs `optimize_total_market_share_fic_lifetime`
      against the same target and writes both.

    Each node is fitted on a slice of `model` and copied back, so the plots
    above show the result once re-run. Solver logs go to `<sector>/logs/`.
    """)
    return


@app.cell
def _(nodeNames):
    # Root of the per-sector output folders. Each node is written to
    # <calibration_output_root>/<sector>/fitted_fics (and fitted_lifetimes), and
    # read back in by the next Reference.py run via calibration_output_sectors —
    # every sector listed below must be in that list.
    calibration_output_root = "C:/calibration/cims/data/model_inputs/calibration_outputs"
    for _sector, _nodes in group_by_sector(nodeNames).items():
        print(f"{_sector}: {len(_nodes)} node(s) -> {calibration_output_root}/{_sector}")
    return (calibration_output_root,)


@app.cell
def _():
    # e.g. dict(ridge=1e-5) -- see the optimize_ms.py module docstring
    fit_kwargs = dict()
    fit_mode = mo.ui.radio(
        options=["fics only", "fics + lifetimes"], value="fics only", label="Fit")
    fit_run = mo.ui.run_button(label="Run fit")
    mo.hstack([fit_mode, fit_run], justify="start")
    return fit_kwargs, fit_mode, fit_run


@app.cell
def _(
    calibration_output_root,
    fit_kwargs,
    fit_mode,
    fit_nodes,
    fit_run,
    model,
    target_key,
):
    mo.stop(not fit_run.value, mo.md("_press **Run fit** to fit_"))
    fit_lifetimes = fit_mode.value == "fics + lifetimes"
    _fit = optimize_total_market_share_fic_lifetime if fit_lifetimes else optimize_total_market_share_fic
    _stage = "estimated_fic_lifetime" if fit_lifetimes else "estimated_fic"

    fit_results = {}
    for _i, _n in enumerate(fit_nodes, start=1):
        _kwargs = dict(fit_kwargs)
        _kwargs.setdefault("logFile", log_file_for(calibration_output_root, _n, _stage))
        _start = time.time()
        try:
            _before = market_share_l1(model, _n, target_key)
            _result = optimize_on_slice(
                model, _n, fit=_fit, objective_counterFactual=target_key,
                verbose=False, **_kwargs)
            if fit_lifetimes:
                _after = _result['final']
                _note = f"({len(_result['changed'])} lifetime(s) changed)"
            else:
                _after = sum(r['end'] for r in _result.values())
                _note = ""
            fit_results[_n] = (_before, _after)
            print(f"[{_i}/{len(fit_nodes)}] {_n}  {time.time() - _start:6.1f}s  "
                  f"L1 {_before:.4f} -> {_after:.4f}  {_note}")
        except Exception as _exc:
            print(f"[{_i}/{len(fit_nodes)}] {_n}  {time.time() - _start:6.1f}s  FAILED: {_exc}")
    return fit_lifetimes, fit_results


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Re-aggregate

    Whole-model recompute (no node argument) -- run once, after fitting, before
    exporting FICs.
    """)
    return


@app.cell
def _(fit_results, model):
    mo.stop(not fit_results, mo.md("_nothing fitted yet_"))
    aggregation_traversal(model)
    reaggregated = True
    return (reaggregated,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Write FICs (and Lifetimes) — All Nodes
    """)
    return


@app.cell
def _(
    calibration_output_root,
    fit_lifetimes,
    fit_results,
    model,
    reaggregated,
    target_key,
):
    mo.stop(not reaggregated)
    for _n, (_before, _after) in fit_results.items():
        _out_dir = write_calibration_outputs(
            model, _n, calibration_output_root, lifetimes=fit_lifetimes,
            fic_source="calibration_estimated_fic_export",
            lifetime_source="calibration_estimated_lifetime_export")
        record_fit(calibration_output_root, _n,
                   "estimated_fic_lifetime" if fit_lifetimes else "estimated_fic",
                   target_key, _before, _after)
    _what = "fics + lifetimes" if fit_lifetimes else "fics"
    print(f"wrote {_what} for {len(fit_results)} node(s) under {calibration_output_root}")
    return


if __name__ == "__main__":
    app.run()
