# Market-Share Optimization

[optimize_ms.py](optimize_ms.py) fits **FICs**, and optionally **lifetimes**, at
a tech-compete node so that the modelled market shares reproduce a calibration
target series. The module docstring at the top of `optimize_ms.py` is the full
reference, with measurements behind each design choice. This README is a
summary and a practical guide.

For how this fits into the wider workflow (Reference.py → fit → export →
re-run), see the [calibration README](../../README.md).

---

## Entry points

| Function | Fits | Against | Use when |
|---|---|---|---|
| `optimize_total_market_share_fic(model, node, **kw)` | FIC per tech per year | `calibration_market_share_total` vs `market_share_total` | FIC-only fit. Stage 3, after lifetimes are fixed. |
| `optimize_total_market_share_fic_lifetime(model, node, **kw)` | FICs + shorter lifetimes | same | Stage 1. A total-share residual that FICs alone can't close. |
| `optimize_new_market_share_fic(model, node, **kw)` | FIC per tech per year | `calibration_market_share_new` vs `market_share_new` | The target is new-stock share. It's an easier fit, and there's no lifetime lever. |
| `optimize_on_slice(model, node, fit=None, **kw)` | wraps any of the above | — | Serial batch runs. It fits on a small slice of the model and copies the result back. |
| `run_stage1_nodes_parallel(model, nodes, out_dir, fit_kwargs, ...)` | Stage 1, many nodes | — | Parallel batch runs, one subprocess per node. It writes the CSVs itself. |
| `run_stage1_node_with_timeout(...)` | Stage 1, one node | — | A single fit you can kill if it runs too long. |

All the fitters **mutate `model` in place**. Re-load the pickle between runs
you want to compare.

```python
from Calibration.Optimization.optimize_ms import (
    optimize_total_market_share_fic, optimize_total_market_share_fic_lifetime)

# FICs only
res = optimize_total_market_share_fic(model, node, ridge=1e-5)
total_l1 = sum(r['end'] for r in res.values())     # res is keyed by year

# FICs + lifetimes, with a plot at every step
res = optimize_total_market_share_fic_lifetime(model, node, plot=True)
res['lifetimes']   # tech -> fitted lifetime
res['changed']     # only the techs that were shortened
res['rosters']     # every tech considered at each pass, and why it was or wasn't picked
res['final_baseline'], res['final']   # L1 before and after, at the final settings
```

Per-year solver output goes to `logFile`. `verbose` controls the console
summary.

---

## How the fit works

Each year is solved in turn. The base year is skipped because its shares are
exogenous.

1. **Pick the free variables.** These are the techs inside their
   available/unavailable window whose objective actually responds to a FIC
   probe. Techs with a counterfactual share of exactly 0 are handled separately
   (see `zero_targets` below).
2. **Seed analytically.** A tech's new share is proportional to
   `softplus(lcc)^-v`. That relationship can be inverted relative to the
   biggest-share tech:
   `lcc_i = lcc_ref * (target_i / target_ref)^(-1/v)`.
   The gap between the required and actual LCC, divided by the measured
   `d(lcc)/d(fic)`, gives a starting FIC. **This step matters most.** Once a
   share saturates at 0 or 1 the gradient vanishes, and a solver started there
   doesn't move.
3. **Solve in scaled units.** The solver works on `z = fic / fic_scale`
   (default 100), so the variables and penalties are of order one.
4. **Minimise the squared share residuals.** The default solver is
   `least_squares` (trust-region reflective), which takes Gauss-Newton steps and
   was 2.4–3.4× faster than L-BFGS-B in tests. The **L1 error** is still what
   gets reported, so read it as "total share error".

### Why the answer isn't unique

Where one tech takes essentially all new share, a wide band of FIC vectors give
identical shares. Without a tie-breaker, the fitted FICs can land anywhere in
that band and jump by 100+ between years. There are two tie-breakers:

- **`ridge`** prefers the smallest FICs that fit, which pins down the *level*.
  **This isn't portable between nodes.** It caps FIC magnitude, so at a node
  that needs FICs in the thousands (e.g. one tech pushed to 0.3%), a ridge that
  was harmless elsewhere can triple the error. Run once at `ridge=0` to see
  what magnitudes the node needs.
- **`smooth`** penalises year-to-year movement, which pins down the *path* and
  doesn't care about the level. Use it when the goal is a steady series.
  Setting `smooth_passes=2` or more anchors each year to both of its neighbours.

`pin_reference` (hold one tech at FIC 0) is kept for experiments only. It
measured badly.

### What a FIC can't fix

The objective is **total** share, which includes surviving vintage stock. A FIC
only moves **new** share. When a tech stays above a falling target even with a
large FIC, you're usually looking at old stock that hasn't retired yet. That's
why the lifetime search exists.

Two cases that no fit can fix, both of them problems in the model description:

- The counterfactual gives share to a tech the model has already retired
  through its availability window.
- The counterfactual asks a tech to grow while the techs it would replace are
  unavailable.

---

## Lifetime search (`optimize_total_market_share_fic_lifetime`)

Lifetime is one number per tech, applied to every year, and it is **only ever
reduced**.

1. Fit FICs at the current lifetimes (the baseline).
2. Build a **roster** of every tech. A tech becomes a candidate if:
   - its counterfactual is flat or falling for at least `min_decline_run`
     years, and
   - the model overshoots it by `min_overshoot` on average, or by
     `peak_overshoot` in any single year.

   Techs that hold stock but are unavailable for the whole period go first,
   because retirement is the only lever that reaches them.
3. Take the **single worst offender** and walk its `lifetime_ladder`
   (0.9×, 0.8×, … 0.1× of the original, floored at `lifetime_min`). Refit at
   each rung. Keep each step only while it cuts the error by at least
   `min_gain` × baseline, and stop at the first step that doesn't pay.
4. Re-rank against the new fit and repeat, up to `max_techs` ladders. A tech
   that has already walked its ladder is never re-opened.

The search fits run at loose tolerances (`fast_search`). The final fit uses
exactly the settings you passed. Shortening a lifetime also raises the
annualised capital cost wherever `capital_recovery` isn't set, so a fitted
lifetime isn't purely a statement about retirement.

---

## Key parameters

Shared by all the FIC fitters (`**fit_kwargs`):

| Parameter | Default | Notes |
|---|---|---|
| `solver` | `'least_squares'` | `'lbfgsb'` is the old behaviour. |
| `ridge` | `0.0` | Preference for small FICs. Scale it to the node (see above). |
| `smooth`, `smooth_passes` | `0.0`, `1` | Penalty on year-to-year movement. |
| `zero_targets` | `'freeze'` | For techs with a target of 0. `'freeze'` suppresses them once to `suppress_floor` and holds them fixed. `'free'` optimizes them like other techs, which chases infinity and is slow. `'drop'` leaves their FIC at 0. |
| `skip_retrofits` | `'auto'` | Skips `calc_retrofits` (~40% of an evaluation) only when `retrofit_existing_max` is 0 everywhere, so the result is unchanged. |
| `scale_by_output` | `False` | Turn this on at nodes with large `output` (e.g. Mode.Urban, ~20,000). Without it, the FICs needed are around 1e7 and the solver stalls at the clipped seed. |
| `ftol`, `xtol`, `gtol` | `1e-10`, `1e-10`, `1e-7` | Keep these tight when the FICs are the deliverable. |
| `objective_counterFactual` / `objective_estimate` | total-share pair | Change both together, or the fit compares mismatched quantities. |
| `logFile`, `verbose` | | Logging. |

Lifetime search only: `lifetime_ladder`, `lifetime_min` (3 yrs), `min_gain`
(0.02), `max_techs` (3), `min_decline_run` (5), `min_overshoot`,
`peak_overshoot`, `fast_search`, `search_kwargs`, `plot`, `plot_kwargs`.

---

## Speed and memory

- **Cost per evaluation** is about 6–10 ms at a 13-tech node. `get_param` is
  about two thirds of that. Wall time is roughly the number of evaluations times
  that cost.
- **Slicing** ([SubGraphs/node_slice.py](../SubGraphs/node_slice.py)): a fit at
  node N only reads N and the nodes it requests services from. Declining capital
  cost from the rest of the model is frozen into `_dcc_external`. Fitting the
  slice gave bit-identical results at a fraction of the memory (~2.8 MB vs
  37 MB). Don't rebuild `dcc_classes` from a slice's graph, because that drops
  the class members outside the slice.
- **Parallel workers** each peak at about 20× the uncompressed pickle size when
  they're not slicing. The worker count is the smallest of: `max_workers`,
  physical cores − 1, what fits in 70% of free memory, and the number of nodes.
  If not even one worker fits, it raises `MemoryError`. Workers serialise their
  CSV writes through `<out_dir>/.write_lock`.

---

## Legacy code

`_optimize_years_sequential.optimize_years_sequential` is the original approach:
an L1 objective with an unseeded L-BFGS-B start at 0 and no penalties. Only
`initial_test_nb.py` still uses it. Use `optimize_ms.py` for new work.

`_objectiveFunctions.make_objective_localNode` is **not** legacy.
`optimize_ms.py` still uses it to set FICs, run the year's LCC and stock
allocation, and read the modelled shares back.
