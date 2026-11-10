# Build outputs

Everything in `data/processed_data/technology_database/` is written by `python -m CIMS.data_processing.technology_database.build` and is overwritten on the next build. Do not edit these files: change the records (`data/technology_database/`) or the source extractions (`data/raw_data/technology_database/`) and rebuild. [`src/technology_database/docs/pipeline.md`](pipeline.md) describes the build stages, and [`src/technology_database/docs/design.md`](design.md) the rules behind the values.

All energy is on a higher heating value (HHV) basis, all costs are in 2025 Canadian dollars, and every value is per unit of the archetype's product (`product-unit` in `archetype.csv`) unless a column says otherwise.

## Start here

| File | Use it for |
| - | - |
| `technology-database.xlsx` | The workbook for reading, and the only wide view: one row per archetype with its central values for every CIMS parameter (cost, energy use, efficiency, emissions, lifetime, size), then energy by carrier, emissions by species, feedstocks, and the source list. It is a view: anything typed into it is lost on the next build. |
| `column-legend.csv` | Each column of the workbook's archetypes and carriers-and-species sheets: its group, unit, and definition. |
| `archetype.csv` | Every archetype: label, family, type, product and unit, carriers, the CIMS nodes it serves, its examples, and its status. |

## Archetype values in long form

| File | Use it for |
| - | - |
| `archetype-parameter.csv` | Every pooled or filled value: one row per archetype, parameter, and target, with central, low, and high, the number of examples (`n`), the sources, and, for filled values, the `grade` (sourced, identity, estimated, proxy, guess), `method`, `rule-id`, `inputs`, and `rationale`. This is the table the workbook and the CIMS values come from. |
| `archetype-secondary-parameter.csv` | The other targets of each parameter (capex by cost basis, as-sourced values, host-plant and capture add-on parts), in the same form as `archetype-parameter.csv`. |
| `archetype-alternative-carrier.csv` | Energy carriers left out of the wide view so that its carriers add up (2026-11-07): other fuels that could serve the same end use, purchased steam or heat against on-site firing (with the boiler-efficiency equivalent), other estimates of a carrier, and fuels a source lists with no quantity. Ranked by the number of examples (`rank`; the wide view's fuel is rank 0). Not in the workbook. Which fuel the wide view carries is set in `data/technology_database/reference/fuel-choice.csv`. |
| `archetype-flow.csv` | Pooled input and output streams per unit of product: materials (air, water, chemicals, ore), co-products, and energy streams where sources list them as flows. Pooled separately from the parameters, from the sources that give flows (mostly the MIDDEN cards), so an energy flow can differ from the energy use in `archetype-parameter.csv`. CIMS energy requests come from the parameters; flows are supporting detail. |
| `archetype-service-request.csv` | Service requests that are not energy carriers: bought intermediates (reduced iron, anodes, oxygen, acid), CO2 to capture, gas to abatement nodes, and zero or negative requests for a step a route skips or supplies itself. |
| `archetype-application.csv` | Every CIMS node each archetype serves, with patterns expanded. |
| `unfilled/` | `archetype-parameter.csv` and `archetype-secondary-parameter.csv` as the sources alone give them, before the fill step, so any check can be run with and without the fills. |

## Examples and source facts

| File | Use it for |
| - | - |
| `example.csv` | Every example: the source technology, its archetype, and the source's context and reported readiness. |
| `example-parameter.csv` | Every parameter fact as the source reports it, and converted to the archetype's basis (`*-norm` columns), or `excluded` with the reason. Use it to trace a pooled value back to its source rows. |
| `example-flow.csv` | Every flow fact as reported and converted (`*-norm`); `conversion` says what was done (unit, product basis, gas density, LHV to HHV). |
| `example-constraint.csv` | Market-share and other constraints the sources give. |
| `source-list.csv`, `citation-review.md` | Every source, its citation, and where it is used. |
| `reference-source-list.csv` | Reference sources behind conversion factors and the currency conversion, with what rests on them. |

## CIMS hierarchy

| File | Use it for |
| - | - |
| `cims_hierarchy.csv` | The CIMS hierarchy snapshot the build used. |
| `cims_hierarchy_technology.csv` | The same, with each node repeated once per archetype serving it (blank where none does). |
| `hierarchy-stamp.csv` | The snapshot's hash and the nodes added or removed since the last build. |

## Capture variants and whole plants

| File | Use it for |
| - | - |
| `ccs-variant.csv` | Each host-with-capture variant and cost parameter: the host plant's value, the capture add-on (tier 1 same source, tier 2 capture node times CO2 captured, and which was used), the CO2 captured, and the all-in value. |
| `ccs-addon-cost.csv`, `ccs-addon-energy.csv` | Costs and energy that sources give on host plants with capture, set aside or restated per t CO2 on the capture archetypes. |
| `chain-validation.csv` | Each reference whole plant beside the sum of the unit chain that replaces it in CIMS (energy by carrier, capital, fixed operating cost, emissions, levelized cost), with the ratio. A ratio far from 1 points at a missing component or a basis difference. |

## Checks and diagnostics

| File | Use it for |
| - | - |
| `mapping-review.md` | How examples reach archetypes, what was not pooled and why, hierarchy nodes no archetype serves, and what needs a decision. |
| `coverage.csv` | One row per archetype: the grade of each CIMS cell (capex, fixed and variable O&M, energy, lifetime, annual output, emissions, readiness), or `not-applicable` or `missing`. |
| `carrier-check.csv` | Carriers that examples name but the archetype does not list. |
| `fill-backtest.csv` | Each estimating rule run on cells that have a sourced value: the estimate, its range, and whether the sourced value falls inside. |
| `conversion-factor-use.csv` | Each conversion factor, pooled efficiency, and assumption, with the rows, examples, and archetypes that rest on it. |
| `currency-conversion.csv` | The deflator and exchange factors the build used. |
| `set-aside-host-cost.csv` | Incremental and retrofit costs set aside until a host-plant cost is known. |
| `set-aside-efficiency.csv` | UKERC per-tonne figures that need, or were refused, an efficiency bridge. |
