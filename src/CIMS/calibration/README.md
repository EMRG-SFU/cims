# CIMS Calibration

Tools for calibrating CIMS against historical data. The main job is fitting
**fixed intangible costs (FICs)**, and where needed **technology lifetimes**, at
tech-compete nodes so that modelled market shares match a calibration
counterfactual (`calibration_market_share_total`, or an external estimate such
as `estimated_market_share_total`).

Fitted values are written to CSVs in the standard model-input layout. The next
`scenarios/Reference.py` run reads them back in, so calibration is an iterative
loop between Reference.py and the notebooks in this folder.

For how the optimizer itself works (the math, solver choices, and tuning knobs),
see [Calibration/Optimization/README.md](Calibration/Optimization/README.md).

---

## Contents

| Path | What it is |
|---|---|
| [batch_optimization.py](batch_optimization.py) | **Main pipeline notebook (marimo).** Runs the FIC + lifetime fit (serial or parallel), the FIC-only refits, and a fit check over a list of nodes, and writes the outputs to each node's sector folder. |
| [estimated_market_share_calibration.py](estimated_market_share_calibration.py) | Notebook for nodes that CEUD doesn't survey by technology (e.g. Lighting's Incandescent/CFL/LED split). Fits against `estimated_market_share_total` (set by `target_key`), FICs only or FICs + lifetimes. |
| [transportation_passenger.py](transportation_passenger.py) | Exploratory notebook for passenger transportation: find nodes with counterfactuals, plot them, inspect and edit FICs. |
| [_stage1_worker.py](_stage1_worker.py) | Subprocess entry point for parallel and timeout-guarded Stage 1 fits. You don't run it directly. |
| `initial_test_nb.py`, `test_calibration_func_nb*.py`, `test_subModels_nb.py` | Older development and test notebooks. |
| [Calibration/](Calibration/) | The importable library. Notebooks use `import Calibration.…`. |
| [VizServer/](VizServer/) | Flask + D3 web viewer for browsing the model tree, viewing tables, editing FICs and re-running a node. |

### The `Calibration` package

| Module | Purpose |
|---|---|
| `Optimization/optimize_ms.py` | The FIC / lifetime fitters, plus the serial-on-slice and parallel runners. See the [Optimization README](Calibration/Optimization/README.md). |
| `Optimization/_objectiveFunctions.py` | `make_objective_localNode` sets FICs for one node and year, runs the LCC and stock allocation, and returns the share error. `optimize_ms.py` uses it. |
| `Optimization/_optimize_years_sequential.py` | The original unseeded year-by-year L-BFGS-B loop. `optimize_ms.py` has replaced it. |
| `CIMS_Functions/` | Calibration versions of CIMS internals: `lcc_calculation_faster`, `set_param_calibration` (creates the param if it's missing), `update_market_shares` (recompute shares from the current FICs), `aggregation_traversal` (re-aggregate quantities and emissions across the whole model). |
| `Data/` | Getters and setters that return polars tables. `node_info` lists years, techs and params, and has `find_nodes_with_parameter`. `market_share`, `FICs`, `quantities` and `emissions` return model-vs-calibration tables and `tweak_*` editors. |
| `Plotting/` | Plotly figures per node: `plot_ms_for_node.plot_ms_line` (the one used most), plus requested quantities and emissions as stack, line, heatmap or diff plots. |
| `SubGraphs/` | Model slicing. `node_slice.build_node_slice` is the one used by the optimizer (see below). `get_subGraph_model` and `single_sector_all_region` are older helpers. |
| `Utility/` | `write_fics` and `write_lifetimes` export to CSV. `calibration_outputs` routes each node to its sector folder (`write_calibration_outputs`), measures a loaded run's error (`market_share_l1`) and keeps `fit_summary.csv` (`record_fit`, `last_recorded_fits`). There are also dict and list helpers. |
| `paramLoc.py`, `cal_model.py`, `config.py`, `utility_functions.py` | Earlier scaffolding (parameter locators, a `Cal_Model` wrapper, module config). Not used by the current pipeline. |

---

## Running the notebooks

The notebooks are [marimo](https://marimo.io) apps (`pip install -e ".[notebooks]"`).
Launch them **from the repo root**, because paths such as `results/Reference/...`
and `data/model_inputs/...` are relative to it:

```bash
marimo edit src/CIMS/calibration/batch_optimization.py
```

marimo puts the notebook's folder on `sys.path`, so `import Calibration` works.
Every notebook starts by loading a gzipped model pickle. Reference.py writes
one to `results/<scenario>/model.pkl`.

---

## The calibration workflow

```
 Reference.py ──► model.pkl ──► batch_optimization.py ──► fitted_fics/ + fitted_lifetimes/
      ▲                                                            │
      └───────────── read back via calibration_outputs ◄───────────┘
```

The steps below are written out in the header of `batch_optimization.py`. After
**every** Reference.py run, load its pickle and run **Plot All Nodes** and
**Check Fit**.

1. **Uncalibrated run.** Run Reference.py and load `model.pkl`.
2. **Fit FICs and lifetimes.** Run `optimize_total_market_share_fic_lifetime`
   at every node. This writes `fitted_fics` and `fitted_lifetimes`.
3. **Re-run Reference.py** with the fitted values. Load, plot and check the fit.
4. **FIC-only refit.** Run `optimize_total_market_share_fic`, keeping the
   lifetimes from step 2. This writes `fitted_fics` only.
5. **Turn on `dcc`** (declining capital cost) in Reference.py and re-run. Load,
   plot and check the fit.
6. **Final FIC-only refit with `dcc` on.**
7. **Re-run Reference.py to verify.** Load, plot and check the fit. Repeat
   steps 6–7 for any nodes that drifted.

Between steps: update `model_pickle_path`, re-run **Load Model**, then run only
the section for the step you're on. The fit sections only run when you press
their button.

### Why refit at all

Each fit recomputes only its own node. Everything the node reads from the rest
of the model stays frozen at the values in the loaded pickle: the prices of the
services it requests (a child node's price is its share-weighted lifecycle
cost), fuel prices, demand, and stock in other members of a DCC class. A full
Reference.py run with every node's fitted values in at once moves those
inputs, so linked nodes (a parent and a calibrated child, or transport and the
calibrated Fuel Blends nodes) drift. Isolated nodes refit to the same FICs.
Turning on `dcc` changes capital costs everywhere, which is why step 6 is
needed.

A refit doesn't depend on the FICs already in the model. Each year starts from
FICs of zero and an analytic seed. With the same surroundings and settings, a
refit reproduces the earlier fit.

### Check Fit

**Check Fit** shows each node's L1 error in the loaded run (`L1_now`, measured
with `market_share_l1`, without fitting) next to the error its last recorded fit
ended at (`L1_last_fit`, from `<sector>/logs/fit_summary.csv`). Nodes with a
small `drift` kept their fit and can be dropped from `nodeNames` before
refitting.

### Choosing nodes

`node_info.find_nodes_with_parameter(model, "calibration_market_share_total")`
lists every node that has calibration data. The notebooks drive their loops from
an explicit `nodeNames` list, so you can work on one sector or region at a time.

### Step 2: serial vs parallel

Use the toggle in the notebook to choose a mode.

- **serial** uses `optimize_on_slice`. It fits each node on a small slice of the
  model and copies the result back into the notebook's `model`. That means you
  can plot the result straight away. Best for a handful of nodes.
- **parallel** uses `run_stage1_nodes_parallel`. It runs one subprocess per node
  with a timeout, and each worker writes its own outputs. The notebook's `model`
  is **not** updated, so re-run Reference.py and re-load to see the results. The
  number of workers is capped by physical cores and available memory, and it
  refuses to start (`MemoryError`) when not even one worker fits. Sectors run
  one after another.

The step 4/6 refits always run serially, because the parallel runner has no
total-share FIC-only mode.

Fit options go in `fit_kwargs_stage1` and `fit_kwargs_refit`, e.g.
`dict(ridge=1e-5)`. Each node gets a solver log under `<sector>/logs/`.

---

## Outputs and how Reference.py picks them up

The notebooks write through `write_calibration_outputs`, which puts each node
in its own sector folder under `calibration_output_root`
(`data/model_inputs/calibration_outputs`). Each folder gets one CSV per region:

```
<calibration_output_root>/<sector>/fitted_fics/fitted_fics_<region>.csv
<calibration_output_root>/<sector>/fitted_lifetimes/fitted_lifetimes_<region>.csv
<calibration_output_root>/<sector>/logs/<node>_<step>.log
<calibration_output_root>/<sector>/logs/fit_summary.csv
```

The sector folder comes from the node name: `Fuel Blends` → `fuels`, and every
other sector is lower-cased with spaces turned into underscores
(`Transportation Passenger` → `transportation_passenger`). The notebooks print
which folder each group of nodes will go to.

The columns follow the standard model-input layout (`Branch, Type, Region,
Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source,
Unit, Year, Value`).

- FICs are written with one row per tech per year.
- Lifetimes are written with one row per tech and a blank `Year`.
- Re-exporting a node **replaces** that node's rows and leaves other nodes' rows
  alone.
- `Source` records where a value came from: `calibration_fic_export`,
  `calibration_lifetime_export`, `calibration_estimated_fic_export` or
  `calibration_estimated_lifetime_export`.
- Lifetimes are only written after a fit that chose them. A FIC-only fit
  re-writing the loaded lifetimes would overwrite the fitted ones with the
  originals if that Reference.py run hadn't loaded them.
- The `Sector` column is left blank for `Fuel Blends` nodes on purpose. Those
  nodes are shared across sectors, and a sector name would get them dropped by
  the `sector_list` filter.

Reference.py loads these from
`data/model_inputs/calibration_outputs/<sector>/fitted_fics` and
`.../<sector>/fitted_lifetimes` for every sector in
`calibration_output_sectors` (currently `transportation_passenger`,
`transportation_freight`, `residential`, `commercial` and `fuels`). To
calibrate a node in any other sector, add that sector's folder name to
`calibration_output_sectors`, or Reference.py won't read its outputs.

---

## VizServer

The VizServer is a small Flask app for browsing a pickled model in the browser.
It shows the node tree (D3), service, tech, FIC, emissions and
requested-quantity tables, and market-share plots. It also lets you edit a
node's FICs and re-run that node.

```bash
cd src/CIMS/calibration
python -m VizServer path/to/model.pkl [port]
```

It can also be started from a thread inside a notebook (see
`transportation_passenger.py`).
