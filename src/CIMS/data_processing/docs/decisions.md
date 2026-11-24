# Decisions

The decisions behind the data processing pipeline, newest first. The
rules they lead to are stated in [conventions.md](conventions.md) and, as they are
written, in `design.md` and `pipeline.md`. Superseded decisions stay in the table, so
the chain can be followed.

A commit that changes outputs on purpose adds a row here (conventions §8).

**Status:** *current* is in force on `staging`; *in review* is on a branch with an open
pull request; *planned* is decided but not yet built; *open* still needs a ruling.

**Records:** *plan* is `data-reorganization-plan.md` (the section is given); a commit or
pull request is in this repository unless noted. The technology database keeps its own
log in [`technology_database/docs/decisions.md`](../technology_database/docs/decisions.md).

| Date | Decision | Status | Record |
| - | - | - | - |
| 2026-11-24 | Policies will be built from sources like any other input (extracted, processed, then written by the policy stage), not kept as hand-edited records in `config/`. Until then, they're made by hand in `model_inputs/policies/`, and `flatten_policies` has no sources to read. Narrows D3. | planned | Brad, 2026-11-24 |
| 2026-11-23 | `uv` resolves only package versions that have a Windows wheel (`required-environments` in `pyproject.toml`), because `loro` 1.19.0, a marimo dependency, shipped without one and broke `uv sync` on Windows. | in review | `e79072e`, branch `uv-windows-wheels` |
| 2026-11-23 | Outputs are checked against a recorded snapshot (`tools/output_snapshot.py`). Files with the same rows in any order count as the same, and numbers are compared at 1e-9 relative. The Phase 0 baseline is snapshot `1eecb766d2`. | current on `data-reorganization` | `4a2f254`; plan §6, Phase 0 step 6 |
| 2026-11-20 | CIMS is the technology database's home: its code in the package and its records in `data/technology_database/`. The technology-database repository keeps the extractions' authoring copy and their scripts, the reviews, the review packages, and source intake, and installs CIMS to use the package. The package moved without its history, which stays in that repository (`97a29cf`). | in review | `88d1897`, `de4724b`, branch `techdb-into-cims`; D6 |
| 2026-11-20 | `CIMS/__init__.py` loads the model's public names (`Model`, `load_model`, and the rest) on first use, so importing the data processing package doesn't load the model. | in review | `3e00e96`, branch `techdb-into-cims` |
| 2026-11-19 | CER data is an interim source. Each series moves, over time, to the primary sources CER draws on (StatCan, NRCan, ECCC, and provincial data, for example). | current | D11; plan Appendix D, issue 9 |
| 2026-11-19 | Anything in the data processing pipeline may change: code, data, names, organization, file and stage order, methods, and outputs. Output changes are deliberate, in their own commits, and recorded here. The model core changes only for data paths and the phase 14 vocabulary. Calibration (the FIC fitting in `src/CIMS/calibration/`) is outside this decision: it changes only through the phase 11 renames and its own reviews (2026-11-23). | current | D10; plan §6 |
| 2026-11-19 | The Natural Gas Market is an optional sector, off by default. Off, each region's production serves its own region and Generic Fuels sets the price. On, the market file overrides that path, and the market's supply shares come from each region's share of national production requests in each year, replacing the JCIMS shares. | in review | `0ac7384`, `686d88c`, branch `natural-gas-market`; `docs/design/natural-gas-market.md` |
| 2026-11-19 | Exogenous electricity demand targets `.Electricity.Utility Generation` instead of `.Electricity`, so the demand reaches generation. | in review | `90a774c`, branch `emh-port` |
| 2026-11-19 | CER source processing: `cer_source_files.py` reads CER's published files, removes carbon from end-use prices, extends CER's series back to 2000, and writes to `processed_data/cer/`, which the energy price, price multiplier, and population and GDP stages read. National oil product prices move up to about 2% (2015 to 2020), and price multipliers up to about 59% (2008 to 2024, where provincial carbon prices had been embedded). | in review | `f6156d3`, branch `emh-port`; plan Phase 0 step 3 |
| 2026-11-19 | Policies: `flatten_policies.py` uses emh's wide-file handling and naming, with `data-processing-calibration`'s long-file pass-through and currency step. | in review | `8f90b47`, branch `emh-port`; plan Phase 0 step 4 |
| 2026-11-19 | Currency labels: a blank `fic` or `dic_min` unit takes the untagged currency year, and new FIC fits carry the run's label. One control key, `currency_year_untagged`, sets the untagged year. | in review | `c2c824f`, branch `emh-port`; D9 |
| 2026-11-19 | Steam fuel targets are built in a source stage, so no script that builds calibration targets reads another's output. This removes an order dependence: with one sector's files missing, another sector's CHP shares changed. | current | #628 |
| 2026-11-12 | Data layers: `primary-sources/`, `extracted/`, `config/`, `processed/`, and `model/` under `data/`. | planned (phase 3) | D1; plan §2.1 |
| 2026-11-12 | Everything in `fixed_data` moves to the layer it belongs in (structure to `config/cims-hierarchy.csv`, sourced values to extracted and processed data, assumptions to `config/model-parameter/`), and `fixed_data` is retired. The technology database's hierarchy is the starting point. | planned (phase 13) | D2; plan §2.1 |
| 2026-11-12 | The hand-made model inputs (defaults, policy records, hierarchy, parameter records) are versioned, in `config/`. `model/` holds only build output. | planned (phase 3); policies superseded 2026-11-24 | D3 |
| 2026-11-12 | Names are lowercase kebab-snake throughout, including parameters, regions, fuels, units, branch paths, and headers. `config/display-name.csv` gives display spellings for results. | planned (phases 3 to 14) | D4, D5; plan §3 |
| 2026-11-12 | One pull request per phase. | current | D7 |
| 2026-11-12 | The calibration package is renamed after the pipeline work, in its own pull request. | planned (phase 11) | D8 |
| 2026-11-12 | Currency: `utils/currency.py`, which converts with the model's own converter and handles the defaults, with the key `currency_year_untagged`, plus emh's blank-unit rule for `fic` and `dic_min`. The model converts defaults when it builds. | current (the blank-unit rule in review) | D9; plan Appendix A |
| 2026-11-12 | Processed tables and model files share one schema, in lowercase kebab-snake. Columns the build can derive may be left blank in processed tables, and only anchor years are kept for parameters the model interpolates. | planned | Q1; plan §2.3 |
| 2026-11-12 | Generated files always come from the build. A release tags the model and the FRDR version together. | planned (phase 15) | Q2 |
| 2026-11-12 | Fetching primary sources stays manual. The build never reads them. | current | Q3 |
| 2026-11-12 | Which sources FRDR can host. Meanwhile, every raw data source goes through the technology database's intake, with its licence recorded, and doubtful cases go to a hosting review. | open | Q4; plan phase 9 |
| 2026-11-06 | CSV files are written as UTF-8 with a byte-order mark, so Excel reads accents and symbols. | current for the technology database; proposed for the pipeline | technology database `pipeline/common.py`; conventions §4 |

## Earlier decisions in the sector modules

These were recorded, undated, in the "Decisions/notes" lists of the sector modules'
docstrings. They are in force. Each proxy moves to `config/proxy-region.csv`, with its
rationale (phase 4), and the method and data review (phase 16) looks at each one again.

| Sector | Decision | Record |
| - | - | - |
| Iron and Steel | Slabs and billets, and their technologies, are removed from SK. | `sector/iron_and_steel_model_inputs.py` |
| Iron and Steel | SK is the proxy for the missing BC, AB, and MB data. | same |
| Iron and Steel | Whether to remove the COREX technology. | same (open) |
| Light Industrial | The missing rubber parameters for AT and TR are copied from BC (all regions had the same values). | `sector/light_industrial_model_inputs.py` |
| Light Industrial | The missing fixed O&M row for AB is copied from BC (all regions had the same values). | same |
| Metal Smelting | Heat processes use thermal coal. | `sector/metal_smelting_model_inputs.py` |
| Metal Smelting | The blast furnace, electrochemical furnace, vertical retort, and sinter roast use coke. | same |
| Metal Smelting | QC, ON, and MB use AT's lead rows (BC's are too detailed). | same |
| Metal Smelting | Aluminium smelting is removed from NL. | same |
| Metal Smelting | BC's magnesium rows are copied from MB (ON, MB, and QC are the same). | same |
| Metal Smelting | ON is the proxy for AB (the QC and BC sectors are too big). | same |
| Metal Smelting | QC's fixed data has titanium, which the activity script doesn't produce. | same (open) |
| Mining | "Size reduced iron and non-iron product" is renamed "iron and non-iron product". | `sector/mining_model_inputs.py` |
| Mining | Underground mining keeps the JCIMS data, because the activity data lacks it. | same (open: add it to the activity script?) |
| Mining | Sintering uses coke. | same |
| Mining | MB's iron product comes from BC (a better proxy than QC's detailed rows). | same |
| Mining | SK's open-pit mining comes from BC. | same |
| Mining | SK is the proxy for NB, the only other region with potash. | same; plan Appendix D, issue 1 |
| Mining | QC is the proxy for NL, which has a similarly detailed mining sector. | same |
| Mining | BC is the proxy for NS (a smaller mining sector), NT, YT, and AB. | same |
| Mining | BC is the proxy for NU, with QC's more detailed iron product nodes, which BC lacks. | same |
| Natural Gas | Only SK, AB, and BC have natural gas production. The other regions are small, and CER's Energy Futures supply data, which the stage reads, doesn't report them. | `sector/natural_gas_model_inputs.py` |
| Petroleum Crude | Exploration stays in the fixed data, because the activity script doesn't produce it. | `sector/petroleum_crude_model_inputs.py` |
| Petroleum Crude | Heavy oil is added to NL and BC, copied from AB. | same |
| Petroleum Crude | Onshore production is removed from NL, and offshore from the regions without it. | same |
| Petroleum Crude | MB is the proxy for NT. | same (open: NT likely has more exploration) |
| Pulp and Paper | MB and SK use AB's fixed data. | `sector/pulp_and_paper_model_inputs.py` |
