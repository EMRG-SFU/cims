# CIMS speed profile, Reference scenario: November 2026 update

**Date:** 2026-11-13 · **Branch:** `staging` @ `0b5a7b1` · **Scenario:** same as the September profile below, with `coal_mining` and the corrected sector list (`'Coal Mining', 'Petroleum Refining'`)

This update replaces the ranking in the September profile. That profile came from code before #609, and #609 removed the `deepcopy` and `numpy.isclose` hot spots. The September sections below are kept for history. Section 2.A has a correction (see "What September got wrong").

## Inputs used (profiling copy only)

Current `staging` can't run Reference with the inputs in `data/model_inputs`. Since #511, `validate_files` fails on two errors:

- 87 techs have `dic_initial` but no `dic_class`.
- 7 market share limits name a class that no tech belongs to (BC `Clean`, `ZEV`, `Gasoline_Transportation`, `Diesel_Transportation`, and `Diesel_Industrial`).

The profile ran on a cloud copy of the inputs with two patches. Your files were not changed.

1. **EMH TF2 class-name fix** (from `emh_mmcf_2026/scenarios/tf2_inputs.py`, `fix_class_name`). It moves `market_share_class` names from Context to Value in `ref_clean_electricity`, `ref_ldv_zev_federal`, `ref_ldv_zev_prov`, `ref_renewable_fuel_content_fed`, and `ref_renewable_fuel_content_prov`. This fixes the 7 limit errors.
2. **Singleton DIC classes.** Each tech with `dic_initial` and no class gets its own `dic_class` (`<branch>|<tech>`). This reproduces the pre-#511 fallback, where DIC follows the tech's own new market share. EMH has no fix for this one because its code predates #511.

The real fix for both belongs in the CIMS inputs.

## Headline numbers (same container, same patched inputs)

| Code | `model.run`, no profiler |
|---|---:|
| `84991a0` (staging just before the #609 merge) | 210.0 s |
| `0b5a7b1` (current staging) | 148.6 s |

**29% faster.** That matches the 30% that #608 measured for the `deepcopy` and `isclose` patch.

Function calls per run fell from 1.59 B to 0.68 B. Under `cProfile` the run took 378 s (2.2× inflation, down from 2.7×).

| Phase (profiled) | Time | Share |
|---|---:|---:|
| `Model.__init__` | 5.3 s | 1.4% |
| `validate_files` | 11.6 s | 3.1% |
| `construct_graph` | 28.8 s | 7.6% |
| **`model.run`** | **332.2 s** | **87.9%** |

`construct_graph` rose from 17 s to 29 s since September. Setup is still small, but this is worth a look if it keeps growing.

## Where the time goes now

The profile is much flatter than in September. No single cheap fix will deliver another 30%.

Self time (time spent in a function's own code), as a % of the 370 s profiled total:

| Function | Calls | Self time | Share |
|---|---:|---:|---:|
| `query.get_param` | 50.0 M | 87.6 s | 23.7% |
| `Model.get_param` (wrapper) | 50.0 M | 22.9 s | 6.2% |
| `isinstance` (137 M of these from `get_param`) | 191.7 M | 15.9 s | 4.3% |
| `construction.create_value_dict` (almost all from `EmissionsCost`) | 29.5 M | 12.6 s | 3.4% |
| `setters.set_param_internal` | 4.7 M | 11.0 s | 3.0% |
| `round` (all from `vintage_weighting.py:46`) | 3.6 M | 10.3 s | 2.8% |
| `EmissionsCost.__add__` (emissions.py:109) | 1.1 M | 10.1 s | 2.7% |
| NetworkX `NodeView.__getitem__` | 28.0 M | 7.8 s | 2.1% |
| `copy.deepcopy` (mostly `inherit_parameter`) | 8.0 M | 6.2 s | 1.7% |

Self time summed by file: all of `emissions.py` comes to **60 s (16%)**.

Cumulative time by area: `lcc_calculation` 54%, stock allocation 21%, vintage weighting 14%, revenue recycling 11%, retrofits 11%, and cumulative emissions aggregation 6%.

## Inside `get_param` (line_profiler, full run)

What happens to the 50.0 M calls:

| Outcome | Calls | Share |
|---|---:|---:|
| Value found directly at node/year/tech | 29.8 M | 60% |
| Falls through to a default | 6.2 M | 12% |
| Falls back to last year's value (one recursion each) | 6.9 M | 14% |
| Calculated (`do_calc`) | 4.4 M | 9% |
| Returns `None` at base year | 1.5 M | 3% |
| Inherited | 1.2 M | 2% |

Calculator bodies account for 62% of `get_param`'s line-profiled time and the year-walk recursion for 21%. Everything else is spread thinly across about 40 lines, each costing 100–250 ns per hit. In other words, the cost comes from how often `get_param` is called, not from any one expensive line.

### What September got wrong

Section 2.A blamed the "use last year's value" recursion for turning 12.9 M primitive calls into 53.7 M. That reading is wrong. In `cProfile`, "primitive" excludes every call made while `get_param` is already on the stack, and most of those come from `do_calc` chains (`get_param` → calculator → `get_param`), not from the year walk. The year walk is 6.9 M of the 50 M calls. Memoizing it could save at most about 14% of `get_param` calls, and it would need care, because values change between iterations.

## Where to look next, ranked by expected gain per effort

1. **Bypass the `Model.get_param` wrapper** (about 6% profiled, numerically free). It forwards 10 keyword arguments and does nothing else. Callers could call `param_query.get_param(model, ...)` directly, or `Model.get_param` could be bound to it. This changes the hottest call path, so confirm it with the value-by-value comparison.
2. **Add a fast path for direct hits in `get_param`.** 60% of calls find the value directly but still run the `isinstance`/`dict_expected`/`unit` bookkeeping, about 20 lines in all. A short early return for the common shape (a value dict with `year_value` and no context or target) should cut a large part of the 87.6 s self time.
3. **Cache vintage weights.** `_get_vintage_weights` runs 1.1 M times, once for every parameter that `calculate_vintage_weighted_parameter` weights, even though the weights depend only on (node, tech, year) and the current stock. Computing them once per (node, tech, year) per iteration would remove most of the 8.2 M `get_param` calls and 3.6 M `round` calls from vintage weighting, which is 14% cumulative. Check the numbers, because the weights are rounded to 5 digits.
4. **`EmissionsCost` arithmetic** (16% self time across `emissions.py`). `__add__`/`__mul__` build new nested dicts through `create_value_dict` 29.5 M times. In-place accumulation (`__iadd__`) at hot call sites such as `calc_financial_emissions_cost` and `calc_competition_emissions_cost` would avoid most of those allocations.
5. **Use `graph._node` in the setters and emissions code.** #608 did this for `get_param`. The other 28 M `NodeView.__getitem__` calls come from `set_param_internal`, `set_tech_param`, `set_node_param`, and `emissions.py:595/821`.
6. **Defaults.** 6.2 M calls work through the calculation and inheritance checks before reaching a default. Checking `node_tech_defaults` sooner for parameters that are never calculated or inherited would shorten that path.

## Files

- `reference_2026-11-13.prof`: the cProfile output (open with `snakeviz`)
- `get_param_lines.txt`: the line_profiler output for `query.get_param` and `Model.get_param`
- `apply_profiling_input_patches.py`: the two input patches described above

---

# CIMS speed profile — Reference scenario

**Date:** 2026-09-10 · **Branch:** `staging` @ `30f3fae` · **Scenario:** `scenarios/Reference.py`
(BC + CAN + CIMS + RoW, 11 model years 2000–2050, 20 sectors, `max_iterations=10`, `equilibrium_threshold=0.05`)

Profiled with `cProfile` on a Linux container (Python 3.11). Absolute times will differ on your
ThinkPad — the *proportions* are what transfer.

---

## 1. Where the time goes

| Phase | Wall time | Share |
|---|---:|---:|
| `Model.__init__` (read inputs) | 2.9 s | 0.8% |
| `validate_files` | 2.1 s | 0.6% |
| `construct_graph` | 17.0 s | 4.9% |
| `validate_graph` | 0.00 s | 0.0% |
| **`model.run`** | **327.2 s** | **93.6%** |
| **Total** | **349.7 s** | 100% |

Setup is not the problem. Everything below is about `model.run`.

Under the profiler the same run took 950 s — a 2.7× inflation, because the run makes
**1.59 billion function calls**. That call count is itself the headline finding: the model is
bounded by Python interpreter overhead, not by arithmetic.

### Cumulative time by function (profiled run, % of 950 s)

| Function | Calls | Cumulative | Share |
|---|---:|---:|---:|
| `lcc_calculation.lcc_calculation` | 57 k | 564.8 s | 59.6% |
| `utils/parameter/query.get_param` | 53.7 M (12.9 M primitive) | 568.5 s | 59.9% |
| `copy.deepcopy` | 133.4 M | 297.9 s | 31.4% |
| `numpy.isclose` | 1.81 M | 66.4 s | 7.0% |
| `declining_costs._calc_all_stock` | 19.7 k | 58.0 s | 6.1% |

(`lcc_calculation` and `get_param` overlap — the former is the traversal that calls the latter.)

---

## 2. The four hot spots

### A. `get_param` — 53.7 M calls, and 4× more than it needs to be

`utils/parameter/query.py:37`. 12.9 M primitive calls expand to 53.7 M through the
**"use last year's value"** fallback at the end of the function: when a parameter isn't found for
`year`, it recurses to `year - step`, and so on back to `base_year`. By 2050 that's up to
10 stacked frames for a single lookup, each redoing the full node/year/tech dict walk,
the `calculation_directory` check, and the inheritance check.

Two costs sit on top of every one of those calls:

- `model.py:1170 get_param` is a pure pass-through wrapper to `query.get_param`. It contributes
  **34 s of tottime doing nothing but forwarding 12 arguments.**
- `model.graph.nodes[node]` goes through NetworkX's `NodeView.__getitem__`
  (78.6 M calls, 39 s). Hoisting `_node = model.graph._node` once — or caching the view on the
  model — removes a Python-level `__getitem__` from the hottest path in the program.

The structural fix is to memoize the year-walk: once `(param, node, tech, context)` has resolved
to a value with `param_source == 'previous_year'`, that resolution is stable for the rest of the
year. A cache keyed on the lookup tuple and cleared at each year boundary should collapse most of
the 41 M recursive calls.

### B. `EmissionsCost.__add__` deepcopies its own payload — 915 k times

`emissions.py:55`:

```python
result = copy.deepcopy(self)
```

`self.emissions_cost` is a 3-level dict of plain scalar value-dicts. `copy.deepcopy` walks it
generically with memo tables, `id()` calls and `_keep_alive` bookkeeping — 133 M `deepcopy`
frames in total, **298 s cumulative (31%)**. A hand-written nested rebuild is
**16× faster** on a representative payload (266 µs → 16 µs, measured).

The same pattern appears seven more times, all on the same two shapes:

- `emissions.py:628, 637, 650, 659` — `deepcopy(gross_emissions)` / `deepcopy(net_emissions)`
- `emissions.py:869, 877, 890` — same, in `calc_financial_emissions_cost`
- `emissions.py:779` — `removal_rates = copy.deepcopy(tax_rates)`, where `tax_rates` was built by
  a dict comprehension of `create_value_dict(0)` three lines earlier. Re-running the comprehension
  is **13× faster** than deepcopying its result.

Note `EmissionsCost.__mul__` (line 90) already does the right thing — it builds the result
dict directly. `__add__` just needs the same treatment.

### C. `numpy.isclose` on scalars — 1.77 M calls, 181× slower than needed

`stock_allocation/market_share_limits.py:337, 340` (`_min_max_ms_compliant`) and
`:174, 177` (`_min_max_ms_class_compliant`). `numpy.isclose` on two Python floats costs
**16.4 µs**; the equivalent scalar expression costs **0.11 µs**.

`numpy.isclose(a, b)` is exactly `abs(a-b) <= 1e-8 + 1e-5*abs(b)`. Substituting that expression
gave **zero mismatches over 200 000 random samples**, so this is a drop-in swap, worth ~65 s of
profiled time.

### D. `_calc_all_stock` recomputes the same number ~500× per call

`declining_costs.py:95`. 19.7 k calls issue **9.93 M** `get_param` calls — about 504 each.
Three layers of redundancy:

1. The inner loop iterates over **every calendar year** from `base_year` to `year`, but
   `reference_year` quantises to the model step — so with `step=5` each stock value is fetched
   **5 times identically**.
2. `unit_convert` and `bs_k` are base-year values, constant for the life of the run, refetched on
   every call.
3. The returned `all_stock` depends only on `(dcc_class, year)` — every technology in a class gets
   the same answer — and every input year is strictly *before* `year`, so the value is stable
   across that year's equilibrium iterations. It is currently recomputed per technology per
   iteration.

---

## 3. Measured effect of the fixes

I applied B, C and D to a scratch copy of the tree (**your repo was not modified**) and re-ran the
same scenario. Each configuration was compared value-by-value against the baseline across
**58 283 stored model values**.

| Configuration | `model.run` | vs baseline | Values differing from baseline | Max relative error |
|---|---:|---:|---:|---:|
| Baseline | 327.2 s | — | — | — |
| Baseline, re-run (control) | 320.7 s | — | 2 | 1.3e-16 |
| **B + C only** (deepcopy + isclose) | **229.0 s** | **−30%** | 16 | **5.2e-16** |
| B + C + D (`_calc_all_stock`) | 205.8 s | **−37%** | 29 | 1.8e-05 |

Read the last two columns carefully:

- **B and C are numerically free.** Their max deviation (5e-16) is the same order as the control
  run against itself — i.e. ordinary float noise, not a behaviour change. Take these two.
- **D is not bit-identical.** Reordering the summation perturbs values by up to ~2e-5 relative on
  four nodes. That is far below the solver's 5% equilibrium threshold, but it is large enough to
  flip an iteration-count decision, so it needs your judgement about whether the scenario outputs
  are still acceptable. The `(dcc_class, year)` cache is the part to scrutinise first — it rests
  on the assumption that no previous year's `stock_new` is revised during the current year's
  iterations.

Note the baseline is itself not perfectly reproducible: two identical runs differed on 2 of
58 283 values. Worth knowing before you chase any small delta.

**Headroom beyond this:** hot spot A (`get_param`) is untouched and is the largest single item at
~60% of cumulative time. Memoizing the previous-year walk and dropping the `model.get_param`
wrapper is more invasive than B–D, but it is where the next factor of two lives.

---

## 4. Two bugs found along the way

Both are in `scenarios/Reference.py`, unrelated to performance, but they change what your
Reference scenario actually runs.

**1. Missing comma silently merges two sectors (line ~140).**

```python
sector_list = [
    'Coal Mining'          # <-- no comma
    # 'Natural Gas',
    # 'Petroleum Crude',
    'Petroleum Refining',
```

Python concatenates adjacent string literals, so this list starts with the single entry
`'Coal MiningPetroleum Refining'`. **Neither Coal Mining nor Petroleum Refining is being
simulated**, and no error is raised because an unmatched sector name is simply filtered out.

**2. `'coal mining'` in `sector_req` (line ~48)** uses a space where the directory is
`data/model_inputs/model/coal_mining`. The load table confirms it: `coal mining  0/4  none`.
Every other sector in that list uses underscores.

**3. (Environmental, not a bug in your setup.)** The model only runs on a case-insensitive
filesystem. The reader looks for `<name>_BC.csv` / `_CAN.csv` / `_RoW.csv`, but many files on disk
are lowercase (`natural_gas_bc.csv`, `market_share_limits_bc.csv`, all of
`policies/reference/*/*_bc.csv`). Windows resolves these; Linux and macOS-with-case-sensitive-APFS
do not, and the run dies in `validate_files` with 22 undefined parent nodes. If CI, a container, or
a colleague on Linux ever needs to run this, normalising the region suffixes is the fix.

---

## 5. Reproducing

`profile_reference.py` is a plain-Python transcription of the `Reference.py` marimo cells with
per-phase timers, wrapped in `cProfile`. `reference.prof` is the raw profile — open it with
`snakeviz reference.prof` or `python -m pstats reference.prof` for an interactive call tree.
