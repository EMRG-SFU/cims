import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import os
    import re
    import pickle
    import gzip
    import time

    import Calibration.Data.node_info as node_info
    import Calibration.Plotting.plot_ms_for_node as plotMS
    from Calibration.Optimization.optimize_ms_v2 import (
        optimize_ms_via_fics_v2,
        optimize_ms_via_fics_and_lifetimes,
    )
    from Calibration.Utility.write_fics import write_fics
    from Calibration.Utility.write_lifetimes import write_lifetimes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Batch Calibration Pipeline

    All-nodes versions of the single-node calibration functions, run over
    every node with `calibration_market_share_total` data. Reference.py is run
    separately (outside this notebook) — this notebook picks up the model
    pickle it produces at each point in the workflow.

    After ANY Reference.py run (including the very first, uncalibrated one),
    load its pkl below and run **Plot All Nodes**. Then:

    1. `optimize_ms_via_fics_and_lifetimes`, all nodes -> **Stage 1** (fits and exports lifetimes + fics)
    2. Reference.py (re-run with fitted fics/lifetimes) -> **Load Model** + **Plot All Nodes** + **Stage 2/4/5 export**
    3. `optimize_ms_via_fics_v2`, all nodes -> **Stage 3**
    4. Turn on `dcc` in Reference.py, re-run -> **Load Model** + **Plot All Nodes** + **Stage 2/4/5 export**
    5. same as 4, after `dcc` is on

    Between points in the workflow: update `model_pickle_path` below, re-run
    **Load Model**, then run only the section for the stage you're on.
    **Plot All Nodes** applies every time.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Model
    """)
    return


@app.cell
def _():
    # Point this at the model.pkl produced by the Reference.py run for this stage
    model_pickle_path = "C:/calibration/cims/results/Reference/transportation_freight.pkl"
    return (model_pickle_path,)


@app.cell
def _(model_pickle_path):
    with gzip.open(model_pickle_path, 'rb') as _f:
        model = pickle.load(_f)
    return (model,)


@app.cell
def _(model):
    calibrated_nodes = node_info.find_nodes_with_parameter(model, "calibration_market_share_total")
    print(f"{len(calibrated_nodes)} node(s) with market share calibration data")
    calibrated_nodes
    return (calibrated_nodes,)


@app.cell
def _():
    # Where fitted fics/lifetimes get written, and read back in by the next
    # Reference.py run via its calibration_output_data inputs
    calibration_output_dir = "C:/cims/data/model_inputs/calibration_outputs"
    return (calibration_output_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Market Shares — All Nodes

    Run this after loading any stage's pkl. One `plot_ms_line` figure per node,
    printed with a node-name header above it.
    """)
    return


@app.cell
def _(calibrated_nodes, model):
    for _node in calibrated_nodes:
        print(f"--- {_node} ---")
        plotMS.plot_ms_line(model, _node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stage 1 — Fit FICs + Lifetimes (All Nodes)

    Runs `optimize_ms_via_fics_and_lifetimes` at every calibrated node, then
    exports both lifetimes and fics. Tune `fit_kwargs_stage1` (e.g. `ridge`,
    `smooth`) — see `optimize_ms_v2.py` module docstring. Each node gets its
    own log file under `<calibration_output_dir>/logs/`.
    """)
    return


@app.cell
def _():
    fit_kwargs_stage1 = dict()  # e.g. dict(ridge=1e-5)
    return (fit_kwargs_stage1,)


@app.cell
def _(calibrated_nodes, calibration_output_dir, fit_kwargs_stage1, model):
    _log_dir = os.path.join(calibration_output_dir, "logs")
    os.makedirs(_log_dir, exist_ok=True)

    results_stage1 = {}
    errors_stage1 = {}
    for _i, _node in enumerate(calibrated_nodes, start=1):
        _kwargs = dict(fit_kwargs_stage1)
        _kwargs.setdefault(
            "logFile",
            os.path.join(_log_dir, re.sub(r"[^A-Za-z0-9]+", "_", _node) + "_stage1.log"),
        )
        _start = time.time()
        try:
            _result = optimize_ms_via_fics_and_lifetimes(
                model, _node, plot=False, verbose=False, **_kwargs)
            results_stage1[_node] = _result
            _elapsed = time.time() - _start
            _n_changed = len(_result['changed'])
            print(f"[{_i}/{len(calibrated_nodes)}] {_node}  "
                  f"{_elapsed:6.1f}s  "
                  f"L1 {_result['final_baseline']:.4f} -> {_result['final']:.4f}  "
                  f"({_n_changed} lifetime(s) changed)")
        except Exception as _exc:
            errors_stage1[_node] = _exc
            _elapsed = time.time() - _start
            print(f"[{_i}/{len(calibrated_nodes)}] {_node}  {_elapsed:6.1f}s  FAILED: {_exc}")

    print(f"\nfitted {len(results_stage1)}/{len(calibrated_nodes)} node(s)")
    if errors_stage1:
        print("failed:", errors_stage1)
    return (results_stage1,)


@app.cell
def _(calibration_output_dir, model, results_stage1):
    for _node in results_stage1:
        write_lifetimes(model, _node, calibration_output_dir, include_subtree=False)
        write_fics(model, _node, calibration_output_dir, include_subtree=False)
    print(f"wrote lifetimes + fics for {len(results_stage1)} node(s) to {calibration_output_dir}")
    return


if __name__ == "__main__":
    app.run()
