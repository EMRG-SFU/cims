import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import pickle
    import gzip
    import time

    import polars as pl

    import Calibration.Data.node_info as node_info
    import Calibration.Plotting.plot_ms_for_node as plotMS
    from Calibration.Optimization.optimize_ms import (
        optimize_total_market_share_fic,
        optimize_total_market_share_fic_lifetime,
        run_stage1_nodes_parallel,
        optimize_on_slice,
    )
    from Calibration.Utility.calibration_outputs import (
        group_by_sector,
        last_recorded_fits,
        log_file_for,
        market_share_l1,
        record_fit,
        sector_output_dir,
        write_calibration_outputs,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Batch Calibration Pipeline

    All-nodes versions of the single-node calibration functions, run over
    every node in `nodeNames`. Reference.py is run separately (outside this
    notebook) — this notebook picks up the model pickle it produces at each
    point in the workflow.

    After ANY Reference.py run (including the very first, uncalibrated one),
    load its pkl below and run **Plot All Nodes** and **Check Fit**. Then:

    1. Reference.py, uncalibrated -> **Load Model** + **Plot All Nodes**
    2. **Fit FICs + Lifetimes** (`optimize_total_market_share_fic_lifetime`) -> writes fics + lifetimes
    3. Reference.py (re-run with fitted fics/lifetimes) -> **Load Model** + **Plot All Nodes** + **Check Fit**
    4. **Refit FICs** (`optimize_total_market_share_fic`, lifetimes kept) -> writes fics
    5. Turn on `dcc` in Reference.py, re-run -> **Load Model** + **Plot All Nodes** + **Check Fit**
    6. **Refit FICs** again, with `dcc` on -> writes fics
    7. Reference.py re-run to verify -> **Load Model** + **Plot All Nodes** + **Check Fit**; repeat 6–7 for nodes that drifted

    Step 4 is optional when only the final calibrated state matters: **Check
    Fit** after step 3 shows which nodes moved once every node's fitted values
    were in the same run.

    Between points in the workflow: update `model_pickle_path`, re-run
    **Load Model**, then run only the section for the step you're on. The fit
    sections only run when their button is pressed.

    Outputs go to `<calibration_output_root>/<sector>/fitted_fics` and
    `.../fitted_lifetimes`, one sector folder per node, which is where
    Reference.py's `calibration_output_sectors` reads them from.
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
    model_pickle_path = "results/Reference/fuel_blends.pkl"
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
def _(calibrated_nodes):
    # The nodes every section below runs over: every node with calibration
    # data. Replace with an explicit list to work on a subset.
    nodeNames = list(calibrated_nodes)
    return (nodeNames,)


@app.cell
def _(nodeNames):
    # Root of the per-sector output folders. Each node is written to
    # <calibration_output_root>/<sector>/fitted_fics (and fitted_lifetimes), and
    # read back in by the next Reference.py run via calibration_output_sectors —
    # every sector listed below must be in that list.
    calibration_output_root = "data/model_inputs/calibration_outputs"
    for _sector, _nodes in group_by_sector(nodeNames).items():
        print(f"{_sector}: {len(_nodes)} node(s) -> {calibration_output_root}/{_sector}")
    return (calibration_output_root,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot Market Shares — All Nodes

    Run this after loading any stage's pkl. One `plot_ms_line` figure per node,
    printed with a node-name header above it.
    """)
    return


@app.cell
def _(model, nodeNames):
    for _node in nodeNames:
        print(f"--- {_node} ---")
        plotMS.plot_ms_line(model, _node)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Check Fit — All Nodes

    L1 market-share error of the loaded model at each node (all years but the
    base year, the same number the fits report), next to the error the last
    recorded fit ended at (`<sector>/logs/fit_summary.csv`). Nothing is fitted
    or recomputed: this measures the Reference.py run that produced the pickle.

    A node whose `L1_now` is close to `L1_last_fit` kept its fit once every
    node's fitted values were in the same run, and does not need refitting. A
    large `drift` means something the node depends on (a child node's price, a
    fuel price, demand, DCC) moved after it was fitted.
    """)
    return


@app.cell
def _(calibration_output_root, model, nodeNames):
    _recorded = last_recorded_fits(calibration_output_root, nodeNames)
    _rows = []
    for _node in nodeNames:
        _last = _recorded.get(_node, {})
        _now = market_share_l1(model, _node, "calibration_market_share_total")
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
    ## Step 2 — Fit FICs + Lifetimes (All Nodes)

    Runs `optimize_total_market_share_fic_lifetime` at every node, then
    exports both lifetimes and fics to each node's sector folder. Tune
    `fit_kwargs_stage1` (e.g. `ridge`, `smooth`) — see the `optimize_ms.py`
    module docstring. Each node gets its own log file under
    `<sector>/logs/`.

    Pick a mode with the toggle below; only the selected path runs.

    - **parallel** runs the nodes `max_workers` at a time, one subprocess
      each, and writes each node's fics/lifetimes as it finishes. `model` in
      this notebook is NOT updated — re-load the next Reference.py pickle
      instead. Sectors are run one after another.
    - **serial** fits one node at a time on a slice of `model` and copies the
      result back, so `model` ends up exactly as an in-place fit would leave
      it, then writes the outputs. Use it for a handful of nodes you want to
      plot here.
    """)
    return


@app.cell
def _():
    fit_kwargs_stage1 = dict()  # e.g. dict(ridge=1e-5)
    return (fit_kwargs_stage1,)


@app.cell
def _():
    stage1_mode = mo.ui.radio(
        options=["parallel", "serial"], value="serial", label="Step 2 mode")
    stage1_run = mo.ui.run_button(label="Run step 2")
    mo.hstack([stage1_mode, stage1_run], justify="start")
    return stage1_mode, stage1_run


@app.cell
def _(
    calibration_output_root,
    fit_kwargs_stage1,
    model,
    model_pickle_path,
    nodeNames,
    stage1_mode,
    stage1_run,
):
    mo.stop(not stage1_run.value, mo.md("_press **Run step 2** to fit_"))
    mo.stop(stage1_mode.value != "parallel", mo.md("_parallel step 2 skipped (mode is serial)_"))
    # One subprocess per node, max_workers at a time, one sector folder at a
    # time. Reuses the loaded pickle on disk so the model is not re-pickled.
    # Status per node in results_parallel.
    results_parallel = {}
    for _sector, _nodes in group_by_sector(nodeNames).items():
        _out_dir = sector_output_dir(calibration_output_root, _nodes[0])
        print(f"=== {_sector} ({len(_nodes)} node(s)) -> {_out_dir}")
        _res = run_stage1_nodes_parallel(
            model, _nodes, _out_dir, fit_kwargs_stage1,
            max_workers=None,          # default: capped by physical cores and available memory (see the docstring)
            timeout_seconds=3600,
            model_path=model_pickle_path,
        )
        results_parallel.update(_res)
        for _node, (_status, _payload) in _res.items():
            if _status == 'ok':
                record_fit(calibration_output_root, _node, "fic_lifetime",
                           "calibration_market_share_total",
                           _payload['final_baseline'], _payload['final'])
    return


@app.cell
def _(
    calibration_output_root,
    fit_kwargs_stage1,
    model,
    nodeNames,
    stage1_mode,
    stage1_run,
):
    # The write cell below depends on results_stage1, so marimo skips it too
    # whenever this cell is stopped.
    mo.stop(not stage1_run.value, mo.md("_press **Run step 2** to fit_"))
    mo.stop(stage1_mode.value != "serial", mo.md("_serial step 2 skipped (mode is parallel)_"))

    results_stage1 = {}
    errors_stage1 = {}
    for _i, _node in enumerate(nodeNames, start=1):
        _kwargs = dict(fit_kwargs_stage1)
        _kwargs.setdefault("logFile", log_file_for(calibration_output_root, _node, "stage1"))
        _start = time.time()
        try:
            # Fits a slice of the model (the node and its request targets),
            # then copies the fitted nodes back into `model` — identical
            # result, a fraction of the memory and faster ladder rungs.
            _result = optimize_on_slice(
                model, _node, plot=False, verbose=False, **_kwargs)
            results_stage1[_node] = _result
            _elapsed = time.time() - _start
            _n_changed = len(_result['changed'])
            print(f"[{_i}/{len(nodeNames)}] {_node}  "
                  f"{_elapsed:6.1f}s  "
                  f"L1 {_result['final_baseline']:.4f} -> {_result['final']:.4f}  "
                  f"({_n_changed} lifetime(s) changed)")
        except Exception as _exc:
            errors_stage1[_node] = _exc
            _elapsed = time.time() - _start
            print(f"[{_i}/{len(nodeNames)}] {_node}  {_elapsed:6.1f}s  FAILED: {_exc}")

    print(f"\nfitted {len(results_stage1)}/{len(nodeNames)} node(s)")
    if errors_stage1:
        print("failed:", errors_stage1)
    return (results_stage1,)


@app.cell
def _(calibration_output_root, model, results_stage1):
    for _node, _result in results_stage1.items():
        _out_dir = write_calibration_outputs(model, _node, calibration_output_root, lifetimes=True)
        record_fit(calibration_output_root, _node, "fic_lifetime",
                   "calibration_market_share_total",
                   _result['final_baseline'], _result['final'])
        print(f"wrote lifetimes + fics for {_node} -> {_out_dir}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Steps 4 and 6 — Refit FICs (All Nodes)

    `optimize_total_market_share_fic` at every node, keeping the lifetimes the
    loaded model already has (the ones step 2 chose, if Reference.py loaded
    them). Run it on the pickle from step 3 (before `dcc`) and again on the
    pickle from step 5 (after `dcc` is turned on). Use **Check Fit** first to
    see which nodes need it; you can cut `nodeNames` down to those.

    Fits one node at a time on a slice of `model` and copies the result back,
    so the result can be plotted here. Only fics are written: this fit does
    not choose lifetimes, and re-writing the loaded ones would overwrite the
    fitted lifetimes with the originals if the Reference.py run did not load
    them.
    """)
    return


@app.cell
def _():
    fit_kwargs_refit = dict()  # e.g. dict(ridge=1e-5); match step 2 for comparable FICs
    refit_label = mo.ui.dropdown(
        options=["refit_pre_dcc", "refit_dcc"], value="refit_pre_dcc",
        label="Which refit (log label)")
    refit_run = mo.ui.run_button(label="Run refit")
    mo.hstack([refit_label, refit_run], justify="start")
    return fit_kwargs_refit, refit_label, refit_run


@app.cell
def _(
    calibration_output_root,
    fit_kwargs_refit,
    model,
    nodeNames,
    refit_label,
    refit_run,
):
    mo.stop(not refit_run.value, mo.md("_press **Run refit** to fit_"))

    results_refit = {}
    errors_refit = {}
    for _i, _node in enumerate(nodeNames, start=1):
        _kwargs = dict(fit_kwargs_refit)
        _kwargs.setdefault("logFile", log_file_for(calibration_output_root, _node, refit_label.value))
        _start = time.time()
        try:
            # Error of the loaded run, before refitting (the fit's own per-year
            # 'start' is the error with every FIC at zero, not this).
            _before = market_share_l1(model, _node, "calibration_market_share_total")
            _result = optimize_on_slice(
                model, _node, fit=optimize_total_market_share_fic, verbose=False, **_kwargs)
            _after = sum(r['end'] for r in _result.values())
            results_refit[_node] = (_before, _after)
            print(f"[{_i}/{len(nodeNames)}] {_node}  {time.time() - _start:6.1f}s  "
                  f"L1 {_before:.4f} -> {_after:.4f}")
        except Exception as _exc:
            errors_refit[_node] = _exc
            print(f"[{_i}/{len(nodeNames)}] {_node}  {time.time() - _start:6.1f}s  FAILED: {_exc}")

    print(f"\nrefitted {len(results_refit)}/{len(nodeNames)} node(s)")
    if errors_refit:
        print("failed:", errors_refit)
    return (results_refit,)


@app.cell
def _(calibration_output_root, model, refit_label, results_refit):
    for _node, (_before, _after) in results_refit.items():
        _out_dir = write_calibration_outputs(model, _node, calibration_output_root, lifetimes=False)
        record_fit(calibration_output_root, _node, refit_label.value,
                   "calibration_market_share_total", _before, _after)
        print(f"wrote fics for {_node} -> {_out_dir}")
    return


if __name__ == "__main__":
    app.run()
