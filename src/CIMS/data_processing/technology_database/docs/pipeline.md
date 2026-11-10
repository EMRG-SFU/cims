# Pipeline

> **Short paths.** `records/` stands for `data/technology_database/`, `output/` for
> `data/processed_data/technology_database/`, and `pipeline/` for the package's
> `pipeline/` folder (`src/technology_database/pipeline/`, or
> `src/CIMS/data_processing/technology_database/pipeline/` in CIMS). The extractions are in
> `data/raw_data/technology_database/` and the documents in `primary-sources/`.

One entry point, run from the repo root:

```
python -m technology_database.build              # everything
python -m technology_database.build --check      # stop after the checks
python -m technology_database.build --quantified-only   # the wide view lists only archetypes with pooled numbers
```

It reads `data/raw_data/technology_database/` and `data/technology_database/`. It writes
`data/processed_data/technology_database/` and `data/processed_data/technology_database/vocabulary.md`, and it appends new examples to
`records/source-example.csv`. A broken rule stops it with a non-zero exit, so it can
gate a commit.

## Stages

| # | Stage | Module | What it does |
| -: | - | - | - |
| 1 | hierarchy | `pipeline/hierarchy.py` | Loads `records/cims-hierarchy.csv`, stamps it (`output/hierarchy-stamp.csv`), and lists the nodes added and removed since the last build against `output/cims-hierarchy.csv`, the copy the last build wrote. |
| 2 | records | `pipeline/records.py` | Loads the archetypes, service links, annotations and vocabularies, and checks each archetype key and path. |
| 3 | extract | `pipeline/extract.py` | Reads every extraction as reported and keeps each source's own technology name. |
| 4 | clean | `pipeline/facts.py` | Evaluates arithmetic-only formulas, spells each unit one way, writes fractions as %, links derived rows to their source rows, flags missing bases, and numbers records within each source. |
| 5 | examples | `pipeline/examples.py` | Attaches each example to the archetype `records/source-example.csv` names. Appends new examples with a blank archetype. |
| 5b | whole-plant | `pipeline/whole_plant.py` | Adds UKERC's host baseline cost to each measure cost where `records/reference/ukerc-host-baseline.csv` allows, as a derived row (`tp-<source>-wNNN`). It sets the other measure costs aside. |
| 5c | assembly | `pipeline/assembly.py` | Sums one source's component costs into a whole-plant cost as listed in `records/reference/whole-plant-assembly.csv`, as a derived row (`tp-<source>-aNNN`). Stops the build on components from another source, another currency-year or an unreconciled unit. |
| 5d | corrections | `pipeline/correction.py` | Takes in figures a source prints wrongly on an assumed correction, as listed in `records/reference/source-correction.csv`: `add` writes a value the extraction set aside as a new row (`tp-<source>-xNNN`) on an existing example; `replace` writes a corrected copy of a loaded row and excludes the original as superseded; `mark` tags a row the extraction already corrected. `rebase` restates a loaded row stated per another product of the same source (per t cement on a per-t-clinker kiln) with the source's own ratio, registered in conversion-factor.csv (class same-source); it carries no correction marker. Every such row carries `correction` = sc-NNN, a note starting ASSUMED CORRECTION, and the factor `assume:source-correction-sc-NNN` (conversion-class `assumption`). |
| 6 | checks | `pipeline/checks.py` | The gate. |
| 7 | normalise | `pipeline/normalise.py`, `pipeline/currency.py`, `pipeline/convert.py`, `pipeline/heating.py`, `pipeline/bridge.py` | Puts every fact on its archetype's basis, or excludes it with the reason. Unit definitions and the product factors in `records/reference/conversion-factor.csv` (per example: `example-conversion.csv`) bridge units the plain table cannot. LHV numbers go to HHV (`heating-value-stream.csv`); numbers per unit of input go per GJ of output by the efficiency bridge (`efficiency-bridge.csv`). Each row records its `conversion-class` and `conversion-factors`. |
| 7a | gross energy use | `pipeline/net_sec.py` | Specific energy consumption is gross input (ruling 2026-10-13). A negative energy use (an export) is excluded and, where it names a carrier, written as a derived output flow `tf-<source>-dNNN`; the values a source states net of a credit are grossed up by the printed credit or excluded, as `records/reference/net-energy.csv` lists. |
| 7a' | feedstock | `pipeline/feedstock.py` | On the conversion archetypes of `records/reference/feedstock.csv`, an energy use that counts the feed whose energy ends in the product has that energy taken off, and the feed written as `feedstock-consumption` (target = carrier) on a derived row `tp-<source>-dNNN` (ruling 2026-10-13). |
| 7a'' | CO2-eq basis | `pipeline/gwp.py` | A CO2-eq figure that is clearly one gas (`records/reference/co2e-species.csv`) gets a derived row in tonnes of that gas from the source's own GWP, and its CO2-eq is restated on the AR5 100-year GWPs (ruling 2026-10-14). |
| 7a''' | supplied heat | `pipeline/supplied_heat.py` | On the archetypes of `records/reference/supplied-heat.csv`, the boiler or fired-heater fuel the sources include becomes a `steam` (or `heat`) request at the generator efficiency on a derived row `tp-<source>-hNNN`; the fuel rows are excluded with the reason and the total loses the generator's losses; `capex-share` and `emissions` = refill take the heat plant's cost and combustion CO2 off (general rule 1, 2026-11-05). |
| 7a'''' | capital share | `pipeline/capital_share.py` | Multiplies the capital cost rows of the archetypes in `records/reference/capital-share.csv` by their multiplier: the capital a unit chain misses against its reference whole plant, added back on the route's own units (rule 3, 2026-11-05). |
| 7b | capture energy | `pipeline/capture_energy.py` | Sets aside the capture plant's energy that sources state on host-with-capture archetypes (`output/ccs-addon-energy.csv`) and, where `records/reference/ccs-host-energy.csv` names the capture archetype, restates it per t CO2 captured on a derived example `<host example>|capture`. |
| 7c | UKERC savings | `pipeline/ukerc_saving.py` | Sets aside UKERC efficiency-bucket SECs and derives each variant's SEC from its base archetype's pooled total and the bucket's median saving. |
| 8 | summarise | `pipeline/summarise.py` | One value set per archetype and parameter: central, low, high, n. |
| 8a | carriers | `pipeline/carriers.py` | One consistent energy split per archetype: the carriers of specific-energy-consumption are the split example's shares of the pooled total; archetypes with carriers but no total get one. |
| 8a' | share to capture | `pipeline/capture_share.py` | On hosts with capture, derives the share of the host's CO2 sent to capture from CO2 captured, the capture plant's rate and the host's CO2 (same example, else the base archetype's total). |
| 8b | fill | `pipeline/fill.py` | Grades every summary `sourced`, then writes graded values (identity, estimated, proxy, guess) and `not-applicable` markers into empty cells by the rules in `records/fill/fill-rule.csv` (run in file order, then the judgements, then the rules again), the hosts in `records/fill/host-archetype.csv`, the equipment classes in `records/fill/equipment-class.csv`, the class constants in `records/fill/fill-assumption.csv` and the judgements in `records/fill/fill-judgement.csv`. Never overwrites a sourced value, except the post phase that makes variants consistent with their hosts (a variant's own sourced value moves to target `as-sourced`); a final phase writes the calculated levelized costs. The summaries before this step are written to `output/unfilled/`; `output/coverage.csv` reports each CIMS cell's grade. |
| 8c | requests and chains | `pipeline/service_request.py`, `pipeline/chain_validation.py` | Writes `output/archetype-service-request.csv` from `records/reference/service-request.csv`, and `output/chain-validation.csv`: each reference whole plant beside the quantity-weighted sum of its unit chain (`records/reference/chain-validation.csv`). Both tables are checked at stage 6 (rules 2 and 3, 2026-11-05). |
| 9 | views | `pipeline/views.py` | Writes `archetype.csv`, `archetype-application.csv`, `cims-hierarchy.csv`, `cims-hierarchy_technology.csv`, and `column-legend.csv`. It builds the wide views in memory for the workbook only; no wide CSVs are written (2026-11-06). Before summarise, `pipeline/roll_up.py` pools targets that restate their base parameter; after the fill step `pipeline/roll_in.py` applies the CHP, temperature and single-stream-efficiency roll-ins; `archetype-secondary-parameter.csv` holds the secondary parameters (`pipeline/placement.py`). |
| 10 | reports | `pipeline/sources.py`, `pipeline/reports.py` | Writes `source-list.csv`, `citation-review.md`, `mapping-review.md` and `data/processed_data/technology_database/vocabulary.md`. |
| 11 | workbook | `pipeline/workbook.py` | Writes `technology-database.xlsx`, last. It is never read back. |

## Records (edited by hand)

| File | One row per | Key |
| - | - | - |
| `records/archetype.csv` | archetype | `archetype` |
| `records/archetype-service.csv` | archetype × hierarchy path or pattern | both |
| `records/source-example.csv` | example | `example-id` = `<source-id>:<source-technology>[@<process>]` |
| `records/family.csv` | display family | `family` |
| `records/hierarchy-annotation.csv` | annotation group of hierarchy paths | `annotation` |
| `records/cims-hierarchy.csv` | hierarchy node (the CIMS snapshot) | `branch` |
| `records/vocabulary/*.csv` | term: equipment, discriminator, modifier, carrier, the other lists, parameter, commodity | the term |
| `records/reference/*.csv` | source priority, source citations, TRL scale map, unit conversions, currency references, conversion factors (with class and source-id), the heating-value streams per archetype and the sources' default basis (`source-heating-basis.csv`), the efficiency-bridge review, rows ruled out (`excluded-row.csv`), figures a source prints wrongly and their assumed corrections (`source-correction.csv`), capture energy on hosts (`ccs-host-energy.csv`) | |
| `records/fill/fill-rule.csv` | fill rule: grade, method, scope, formula, rationale, status | `rule-id` |
| `records/fill/host-archetype.csv` | archetype with incremental costs, or a host plant with capture, and the host plant whose cost it takes | `archetype` |
| `records/fill/fill-judgement.csv` | a guessed value: central within low-high, basis, rationale, author, status (draft, accepted, changed, rejected) | `archetype`, `parameter`, `target` |
| `records/reference/energy-price.csv` | reference 2025 price per carrier (CAD per GJ, HHV) for the calculated levelized cost: kind sourced, derived (from an archetype's own levelized cost), internal (0) or pending | `carrier` |
| `records/fill/equipment-class.csv` | each equipment term's lifetime class and variable-O&M class (material, minor, none) | `equipment` |
| `records/fill/fill-assumption.csv` | a class constant from an outside catalogue (proxy grade): group, parameter, applies-to, value, low, high, source-id, rationale | `assumption-id` |
| `data/raw_data/technology_database/source-details_reference.csv` | reference source behind a conversion factor or the currency conversion (kept apart from the technology sources; `used_for` says which, `file_on_disk` names a held table) | `source_id` |

## Outputs (generated)

| File | What it is |
| - | - |
| `README.md` | What each output file is for, grouped by use (kept in the output folder; not written by the build). |
| `example.csv` | Every example, with its archetype, its source's context and the readiness it reports. |
| `example-parameter.csv`, `example-flow.csv`, `example-constraint.csv` | Every fact as reported, and converted to the archetype's basis (`*-norm`), or `excluded` with the reason. Flows are put on HHV like the parameters (2026-11-06); `example-flow.csv`'s `conversion` column says what was done. |
| `archetype.csv` | Every archetype with its parsed key, the nodes it applies to, its examples and its status. |
| `archetype-parameter.csv`, `archetype-flow.csv` | The summaries; `n-corrected` counts the examples that rest on a source correction. `archetype-parameter.csv` also holds the filled rows, with `grade`, `method`, `rule-id`, `inputs` and `rationale`. |
| `archetype-application.csv` | Every hierarchy node each archetype serves, patterns expanded. |
| `cims-hierarchy.csv`, `cims-hierarchy_technology.csv` | The CIMS hierarchy snapshot the build used, and the same with `technology` as the second column: each node repeated once per archetype serving it, blank where none does. |
| `column-legend.csv` | Each column of the workbook's archetypes and carriers-and-species sheets: its sheet, group, unit and note. The review packages' column legend is built from it. |
| `source-list.csv`, `citation-review.md` | Every source, its citation and where it is used. |
| `set-aside-host-cost.csv` | UKERC measure costs set aside until a host-plant cost is established. |
| `ccs-addon-cost.csv` | Costs sources give on host plants with capture: capture cost, set aside to move to a co2-capture archetype. |
| `ccs-addon-energy.csv` | Energy sources give on host plants with capture: the capture plant's, set aside; the mapped rows are restated per t CO2 on the capture archetypes. |
| `set-aside-efficiency.csv` | UKERC numbers per tonne of a host plant's product: `needs-efficiency` (not yet reviewed for the bridge) or `not-bridged` (reviewed; the reason is in `excluded`), with each example's fuel use per tonne. |
| `fill-backtest.csv` | Each borrowing rule run on cells that have a sourced value: the estimate, its range, and whether the sourced value falls inside. |
| `archetype-alternative-carrier.csv` | The energy carriers left out of the wide view so its carriers add up (`pipeline/fuel_choice.py`, `records/reference/fuel-choice.csv`): other fuels, purchased steam or heat against on-site firing, other estimates, and fuels listed with no quantity; ranked, not in the workbook. |
| `archetype-service-request.csv` | Each service request that is not an energy carrier or a feedstock: a bought intermediate, or a zero or negative request for a step a route skips or supplies itself, per unit of the archetype's product. |
| `chain-validation.csv` | Each reference whole plant beside the sum of the unit chain that replaces it: energy by carrier, capital and fixed operating cost, direct emissions and calculated levelized cost, with the ratio. |
| `ccs-variant.csv` | Each host-with-capture variant and cost parameter: the host plant's value, the capture add-on (tier 1 same source, tier 2 capture node x CO2 captured, and which was used), the CO2 captured per unit of product and the all-in value. |
| `coverage.csv` | One row per archetype: the grade (sourced, identity, estimated, proxy, guess), the `not-applicable` marker or `missing` for each CIMS cell (capex, opex-fixed, opex-variable, energy, lifetime, annual-output, emissions, trl). |
| `unfilled/` | `archetype-parameter.csv` and `archetype-secondary-parameter.csv` as the sources alone give them, before the fill step, so any check can be run with and without the fills. |
| `conversion-factor-use.csv` | Each conversion factor, pooled efficiency and assumption, with its class and the pooled rows, examples and archetypes that rest on it. |
| `reference-source-list.csv` | Each reference source, the factors it backs (or `currency-conversion.csv`) and the pooled rows that rest on it. |
| `mapping-review.md` | How examples reach archetypes, what was not pooled, and what needs a decision. |
| `currency-conversion.csv`, `hierarchy-stamp.csv` | The factors and the snapshot the build used. |
| `technology-database.xlsx` | The workbook: readme (with a list of the CSV tables and what they hold), archetypes (the wide view), energy, emissions and feedstocks (the carriers-and-species view, split), and source-list. The other tables are CSV only (2026-11-04). |

## How to

**Add a source.** Pick it from `source-intake/to-get.csv` where it is listed there,
and remove it from that list once it is held. If it cannot be obtained, move its row to
`source-intake/unavailable.csv` with the reason and any substitute. If the
document turns out to hold nothing usable, put it in `primary-sources/background-no-data-extracted/`
and add a row to `data/raw_data/technology_database/background-register.csv` instead of registering it.
1. Put the document in its publisher folder under `primary-sources/` (new downloads arrive in
   `primary-sources/to-process/`), named `author_year_slug[_source]`.
2. Extract it to a long CSV in the template format (`src/technology_database/extraction/input_template.py`),
   in the same publisher folder under `data/raw_data/technology_database/`. A multi-context source also
   gets a `_nodes.csv`, and every extraction a `_meta.csv` (licence, citation and extraction notes). A
   source that needs code (a workbook, a bulk download, a PDF table) gets a script
   beside its CSVs in the raw data (so it goes to FRDR with them), which reads the document
   from the same folder under `primary-sources/`.
3. Register it with `src/technology_database/extraction/register_source.py` (the manifest,
   `source-details_sources.csv` with its `licence_type`, and `source-details_files.csv`; the citation
   and priority records; the loader line), or by hand.
4. Add its loader to `pipeline/extract.py` if `register_source.py` did not: a file in the template format needs one line in `TEMPLATE_FILES`. A file in the ETSAP-brief shape (process/technology rows with `_nodes` and optional `_flows`) needs one line in `ETSAP_BRIEFS`, or in `BRIEF_SHAPE` with the region of its data for a non-ETSAP source.
5. Build. The new examples are appended to `records/source-example.csv` with a blank archetype.
6. Fill in each new example's `archetype` (or `(none)`), then build again.
7. Give the source a row in `records/reference/source-priority.csv`.
8. Run `check_manifest.py` and `check_names.py`. The new extraction goes to FRDR with the next
   deposit of `data/raw_data/technology_database/` (its README).

**Correct a figure a source prints wrongly.** Leave the extraction as printed. Add a row to `records/reference/source-correction.csv`: `add` (with `example-id` and the corrected parameter, value, unit and bases) for a value the extraction set aside, `replace` (with `record-id` and only the fields that change) for a loaded row, `mark` (with `record-id`) for a row the extraction already corrected. `as-printed`, `correction`, `evidence` and `decided` are required. Build.

**Change a mapping.** Edit the `archetype` cell in `records/source-example.csv`.

**Add an archetype.**
1. Apply the test in [design.md](design.md) §2.
2. Add the row to `records/archetype.csv`, a new discriminator (with its definition) to `vocabulary/discriminator.csv`, and its hierarchy paths to `records/archetype-service.csv`.
3. A new equipment term is a modelling decision. Add it to `vocabulary/equipment.csv` deliberately.

**Update the hierarchy.**
1. Replace `records/cims-hierarchy.csv` and build.
2. `mapping-review.md` lists the nodes added and removed.
3. The build fails on any service link or annotation whose path has gone. Repoint those, then rebuild.

**Refresh currency.** Replace the raw downloads in `data/raw_data/technology_database/currency/`
(the file names `deflator_exchange.py` expects) and build. The build turns them into
`output/currency/currency_deflator.csv` and `currency_exchange.csv` with CIMS's
deflator_exchange builder (`src/CIMS/data_processing/source/deflator_exchange/`).

## Other checks

- `python src/technology_database/extraction/check_manifest.py`: the source manifest against the disk and against the build's outputs.
- `python src/technology_database/extraction/check_names.py`: names in the extraction layer and the YAML tree.
