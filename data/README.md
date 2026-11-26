# CIMS Data Processing

The data processing step transforms data into **model input** files and **calibration** files for the CIMS energy-economy model. Raw data is **never committed** to this repository — only scripts, mappings, and this documentation are tracked. Raw data is stored on the External-CIMS sharepoint. Model input files are used to build the model with information like sectors, activity levels, services, technologies, model parameters, and fuel prices. Calibration files include the historical demand, emissions, and technology shares publicly available. During the calibration step, the user will attempt to match market shares, energy demand, and emissions in the model from 2000-last historical year to the calibration data. This step will help determine intangible costs and trends, and will be a vetted starting point from which the projection begins. 

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Utility Library](#utility-library)
6. [Output Format](#output-format)
7. [Common Patterns](#common-patterns)
8. [Dependencies](#dependencies)
9. [What is and is not Committed](#what-is-and-is-not-committed)

---

## Overview

The data follows five layers, in the order the data flows (conventions §2). Each layer is written by one kind of process, and nothing reads a layer upstream of its own place.

**Step 1 — Assemble the sources** (`data/primary-sources/`, `data/extracted/`): documents as obtained go in `primary-sources/`, and the values the build reads go in `extracted/`, both in publisher folders (`gov-can/statcan/`, `gov-usa/eia/`, …). Neither is committed: get them from the External-CIMS SharePoint (and, for the technology database, FRDR). The last historical year for each source goes in the control file, `data/config/control.py`, with the scenario and currency settings. Scripts use it to start projections in the following year.

**Step 2 — Source processing** (`src/CIMS/data_processing/source/`): each script reads `extracted/` and `config/`, and writes one table per source dataset to `data/processed/` for vetting. The data is usually yearly, 2000–2100, for every Canadian region the source covers. The functions and data frames in these scripts are also called by the next step.

**Step 3 — Sector assembly** (`src/CIMS/data_processing/sector/`): one module per sector and stage, `<sector>_model_inputs.py` and `<sector>_calibration.py`. Each model-inputs module combines the JCIMS structural parameters (`data/extracted/legacy/fixed-data/`) with processed data, and writes the model files to `data/model/sector/<sector>/`. Each calibration module writes calibration targets to `data/model/calibration/target/<sector>/`.

Every path comes from one module, `src/CIMS/data_processing/common/paths.py`. To move an existing data folder from the old layout (`raw_data/`, `processed_data/`, `model_inputs/`, …), run `uv run python tools/migrate_data_layout.py --apply`; its map is `tools/data-layout_renames.csv`.

---

## Directory Structure

```
<repo>/data/
├── primary-sources/            # Documents as obtained, by publisher. Not committed; the build never reads them.
│   └── technology-database/    # The technology database's documents, by publisher (its manifest's file paths)
│
├── extracted/                  # As-published values the build reads, by publisher. Not committed (SharePoint, FRDR).
│   ├── gov-can/                # statcan/, cer/, eccc/nir/, nrcan/ceud/
│   ├── gov-ab/  gov-bc/        # provincial data
│   ├── gov-usa/                # eia/, epa/, doe/
│   ├── ceedc/  ipcc/           # other publishers
│   ├── currency/               # deflator and exchange-rate sources
│   ├── legacy/                 # JCIMS-era inputs: fixed-data/ (structural parameters per sector,
│   │                           #   retired in phase 13), clean-fuel-production/, energy-prices/
│   └── technology-database/    # The technology database's extractions and manifest (the FRDR deposit)
│
├── config/                     # Hand-edited records. Committed.
│   ├── control.py              # Pipeline-wide settings (data years, scenarios, currency)
│   ├── mapping/                # cer-to-cims, energy, region, sector, sector-region, nir-to-cims, nir-crosswalk, cims-node
│   ├── conversion/             # energy-conversion.csv (GJ ↔ TJ, L ↔ tonnes, …)
│   ├── assumption/             # Growth rates and shares that extend the data through the projection period
│   ├── calibration/            # fic-proxy-map.csv and fic-decline-techs.csv
│   ├── model-default/          # default-parameter.csv and default-list.csv, read by the model
│   ├── interim-source/         # Hand-entered CER tables, until the primary sources replace CER
│   ├── technology-database/    # The technology database's records: archetypes, mapping, reference tables
│   ├── excluded-branch.csv
│   └── results-general.txt, results-tech.txt   # Which results the model writes
│
├── processed/                  # Written by the source stages. Not committed.
│   ├── gov-can/  gov-usa/      # One table per source dataset
│   ├── activity/  energy-prices/  emission-factors/  steam/
│   ├── currency/               # currency-deflator.csv, currency-exchange.csv
│   └── technology-database/    # The technology database's build outputs, incl. the workbook
│
└── model/                      # Everything CIMS reads. Written by the build, not committed (except its README).
    ├── base/                   # cims-base_<region>.csv
    ├── sector/<sector>/        # <sector>_<region>.csv, for example iron-and-steel/iron-and-steel_ab.csv
    ├── cost/                   # declining-capital-cost/, declining-intangible-cost/, fixed-intangible-cost/
    ├── exogenous/              # exogenous-price/, exogenous-demand/, market-share-limit/
    ├── policy/<policy>/        # <policy>_<region>.csv. Made by hand until the policy stage exists.
    └── calibration/
        ├── target/<sector>/    # Calibration targets, written by the sector calibration modules
        └── fitted/<sector>/    # Written by calibration: fitted_fics/, fitted_lifetimes/, extended_fics/
```

Logs go to `results/logs/` (gitignored), outside the data folder.

Pipeline **code** lives outside the data tree, inside the installed `CIMS` package:

```
<repo>/src/CIMS/data_processing/
├── process_source_data.py      # Runs all Stage 1 source modules in dependency order
├── run_all_model_inputs.py     # Runs all Stage 2 model_inputs modules
├── run_all_calibration.py      # Runs all Stage 2 calibration modules
├── source/                     # Stage 1 — extracted → processed
│   ├── activity/               # Activity for all sectors (res/com/trans from ceud source)
│   ├── deflator_exchange/      # GDP deflators and exchange rates (currency conversion)
│   ├── eccc/nir/               # National Inventory Report processing
│   ├── emission_factors/       # Fuel-level emission factors (NIR Annex 6, CEEDC)
│   ├── energy_prices/          # Energy prices and their multipliers (various sources)
│   └── nrcan/ceud/             # Comprehensive Energy Use Database extraction
│       ├── residential/
│       ├── commercial/
│       ├── transportation_passenger/
│       └── transportation_freight/
├── technology_database/        # Technology database build (python -m CIMS.data_processing.technology_database.build)
├── sector/                     # Stage 2 — processed + fixed data → model
│   ├── agriculture_model_inputs.py
│   ├── agriculture_calibration.py
│   ├── biodiesel_model_inputs.py
│   ├── biodiesel_calibration.py
│   ├── commercial_model_inputs.py
│   ├── commercial_calibration.py
│   └── ...                     # one <sector>_model_inputs.py, and where the
│                               # sector is calibrated, a <sector>_calibration.py
└── utils/                      # Shared utility functions
    ├── extractors/             # Source-specific readers (Stats Can, CEUD, CER)
    ├── output_builder.py       # CIMS output row formatter
    ├── data_extensions.py      # CAGR projection and trend fitting
    ├── data_fill.py            # Interpolation and gap filling
    ├── flatten_fixed_data.py   # Wide fixed-data CSVs → annual long format
    ├── add_cims_totals.py
    ├── dict_ops.py
    └── controls_conversions.py  # Path anchors + CONTROLS loader


---

## Quick Start

### Prerequisites

1. Install Python dependencies (see [Dependencies](#dependencies)).
2. Obtain the sources from the **External-CIMS SharePoint** and place them under `extracted/` and `primary-sources/` following the structure above. If you have a data folder in the old layout, move it with `uv run python tools/migrate_data_layout.py --apply`.
   - `primary-sources/source-details.xlsx` documents every source.

### Running the pipeline

You can run scripts individually (useful when only one source has been updated) or run everything at once:

```powershell
# Run the full pipeline in one go
python -m CIMS.data_processing.process_source_data       # all Stage 1 source processors
python -m CIMS.data_processing.run_all_model_inputs      # all Stage 2 model input assemblers
python -m CIMS.data_processing.run_all_calibration        # all Stage 2 calibration assemblers
```

Or run stages selectively. Stage 1 source processors (order generally does not matter, except `energy_prices.py` before `energy_price_multipliers.py`):

```powershell
python -m CIMS.data_processing.source.activity.emissions_drivers
python -m CIMS.data_processing.source.activity.electricity
python -m CIMS.data_processing.source.nrcan.ceud.residential.residential
python -m CIMS.data_processing.source.nrcan.ceud.commercial.commercial
python -m CIMS.data_processing.source.nrcan.ceud.transportation_passenger.transportation_passenger
python -m CIMS.data_processing.source.nrcan.ceud.transportation_freight.transportation_freight
python -m CIMS.data_processing.source.energy_prices.energy_prices
python -m CIMS.data_processing.source.energy_prices.energy_price_multipliers
python -m CIMS.data_processing.source.eccc.nir.nir_to_cims
python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange
python -m CIMS.data_processing.source.emission_factors.emission_factors
```

Run Stage 2 sector assemblers:

```powershell
python -m CIMS.data_processing.sector.agriculture_model_inputs
python -m CIMS.data_processing.sector.agriculture_calibration
python -m CIMS.data_processing.sector.commercial_model_inputs
python -m CIMS.data_processing.sector.commercial_calibration
...
```

Model inputs land in `model/sector/<sector>/`, one CSV per province or territory (13 files per sector). Calibration targets land in `model/calibration/target/<sector>/`.

---

## Configuration

All pipeline-wide parameters are set in **`config/control.py`**. This file exports a `CONTROLS` dict that every script imports via `src/CIMS/data_processing/utils/controls_conversions.py`.

Key parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `currency_year_cims` | Dollar-year for CIMS prices | `2025` |
| `currency_year_jcims` | Dollar-year for JCIMS prices | `2005` |
| `currency_year_cer` | Dollar-year for CER prices | `2025` |
| `cer_scenario` | CER scenario to extract | `"Current Measures"` |
| `data_start` | First historical year in outputs | `2000` |
| `projection_end` | Last projected year in outputs | `2100` |
| `last_data_year_ceud` | Last year available in CEUD data | `2022` |
| `last_data_year_nir` | Last year available in NIR data | `2024` |
| `last_data_year_cer` | Last year available in CER data | `2050` |
| `last_data_year_statscan_resd` | Last year of Stats Can RESD table | `2024` |
| `last_data_year_statscan_pop` | Last year of Stats Can population table | `2026` |

### Updating Data

When a new data vintage arrives:

1. Download the file and replace it in its folder under `extracted/` (or `primary-sources/`, for a document the build doesn't read).
2. Open the `control.py` Marimo notebook:
   ```powershell
   cd <repo>/data/config
   marimo edit control.py
   ```
3. Update the relevant `last_data_year_*` value (and currency year or scenario name if needed). The pipeline uses the year *after* the last data year as the start of its projection assumptions, so keeping this current is important.
4. Hit **Save to control.py** and exit the Marimo session.
5. Re-run the affected source script(s) and sector script(s).

---

## Utility Library

All utilities live in `src/CIMS/data_processing/utils/`.

### `output_builder.py`

Core formatter for CIMS-compatible output rows.

- `META_COLS` — ordered list of the 12 metadata column names every output row carries:
  `Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit`
- `make_row(meta, year_values, extend_fn=None)` — builds a single output row dict combining metadata with a `{year: value}` series. Optional `extend_fn` projects values past the last data year.
- `pl_to_series(df, filters)` — extracts a `{year: value}` dict from a Polars long-format DataFrame given a set of column filters.
- `pl_get_scalar(df, filters, col)` — extracts a single scalar value from a Polars DataFrame.

### `data_extensions.py`

Functions for projecting historical time series to 2100.

| Function | Description |
|----------|-------------|
| `compute_cagr(series, start, end)` | Compound annual growth rate between two years |
| `extend_constant(series, from_year, to_year)` | Hold the last value constant |
| `extend_series_linear(series, cagr, periods)` | Apply CAGR over defined periods |
| `extend_cagr_periods(series, periods, dampeners)` | Multi-period CAGR with progressive dampening |
| `extend_series_trend_decline(series, ...)` | Fit linear trend to history then apply declining growth |
| `extend_series_trend_dampener(series, ...)` | Fit trend then progressively dampen growth rate |

Dampening is commonly applied as: full CAGR through 2050, 50% through 2075, 20% through 2100.

### `data_fill.py`

Gap-filling utilities: linear interpolation between known values, forward-fill, backward-fill. Used after suppression handling to produce complete 2000–last-data-year series before projection.

### `flatten_fixed_data.py`

Converts wide fixed-data CSVs (one row per parameter, year columns 2000–2050) into long format with annual rows. Applies linear interpolation across the 5-year breakpoints and holds the 2050 value constant through 2100.

### `add_cims_totals.py`

Rolls up leaf-node rows to parent aggregate rows in the CIMS branch hierarchy. For example, shell technology rows (Retail Cold, Retail Marine, Office Cold, …) are summed to produce a Shell total row.

### `controls_conversions.py`

Loads and exposes the `CONTROLS` dict from `control.py` as module-level constants:

- `BASE_PATH` — data root directory, derived automatically from this file's location (portable across machines)
- `DATA_START` — first historical year (2000)
- `PROJECTION_END` — last projected year (2100)
- `LAST_DATA_YEAR` — dict of last available year per source
- `ARCHIVED_DATA` — dict of previously used data vintage years (for reference)
- Energy conversion factors loaded from `energy_conversions.csv`

### `extractors/stats_can.py`

| Function | Description |
|----------|-------------|
| `read_statscan_csv(path)` | Loads a Stats Can CSV preserving suppression codes as strings |
| `load_resd(path)` | Standardizes RESD table (25-10-0029-01) column names |
| `build_population_shares(path)` | Converts quarterly population data to annual regional shares (sum to 1.0 per year) |

### `extractors/nrcan_ceud.py`

| Function | Description |
|----------|-------------|
| `extract_year_cols(df)` | Finds columns whose headers are four-digit years |
| `find_row_indices(df, label)` | Finds row positions matching a label string |
| `get_row_series(df, label)` | Extracts a `{year: value}` dict from a named row |
| `_to_float(cell)` | Converts cell to float; returns `NaN` for 'X', None, or non-numeric |

---

## Output Format

All sector model input files share a common long-format schema:

| Column | Description | Example |
|--------|-------------|---------|
| `Branch` | Full CIMS node path | `CIMS.CAN.AB.Residential.Dwellings.Building Type.High Density.Vintage.2001-2020 Bldg Code.Heating (Cold)` |
| `Type` | Scope level | `Region`, `Sector`, `Service`, `Technology, etc.` |
| `Region` | Province/territory code | `AB` |
| `Sector` | Sector name | `Residential` |
| `Service` | Service name | `Heating(Cold)` |
| `Technology` | Technology name | `Natural Gas_Furnace_High Efficiency` |
| `Parameter` | Model parameter | `market_share_total`, `service_request`, `multiplier_price, etc.` |
| `Context` | Additional context (often blank) | |
| `Sub_Context` | Sub-context (often blank) | |
| `Target` | Demand/supply node this row applies to (often blank) | |
| `Source` | Data provenance | `CEUD`, `CER`, `ECCC, etc.` |
| `Unit` | Value unit | `%`, `PJ`, `2025$/GJ, etc.` |
| `Year` | Year for each value | `2000 - 2100` |

Files are saved as CSVs.

---

Some scripts document their own outputs in a README beside the script — for
instance the currency deflator and exchange rate tables written to
`data/processed/currency/`, described in
[`src/CIMS/data_processing/source/deflator_exchange/README.md`](../src/CIMS/data_processing/source/deflator_exchange/README.md).

## Common Patterns

### Source Script Structure

Despite pulling from different sources, all scripts in `src/CIMS/data_processing/source/` share the same structure:

**1. Module-level docstring** — Every script opens with a prominent docstring (sometimes a `===` banner) that states what it does, its input files and output location, and its specific suppression-handling strategy. This is the first place to look when diagnosing unexpected output.

**2. Package imports** — The pipeline is a package under `src/CIMS/data_processing/`, so scripts use absolute imports and need no `sys.path` manipulation:

```python
from CIMS.data_processing.utils.controls_conversions import BASE_PATH, DATA_START
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years
import CIMS.data_processing.source.activity.emissions_drivers as _emissions_mod
```

Because `CIMS` is installed (editable) into the project venv, this works from any working directory, whether the script is run as a module (`python -m CIMS.data_processing.source.activity.emissions_drivers`) or by path.

**3. Configuration block** — Input/output paths and key constants (e.g. `DATA_START`, `LAST_HIST_YEAR`, `REGIONS`) are declared at module level, loaded from `controls_conversions.py`. All paths are built with `BASE_PATH / '...'` so they resolve correctly regardless of working directory.

**4. `main()` function + `__main__` guard** — All processing is wrapped in a `main()` function that returns a DataFrame. The `if __name__ == '__main__': main()` guard at the bottom means:
- Scripts can be run directly: `python emissions_drivers.py`
- The `run_all_*.py` runners sit at the top of `src/CIMS/data_processing/` and run each module as a subprocess (`python -m ...`), and a sector assembler can import a source module and call its `main()` directly
- Sector assemblers that need to chain outputs can import a source script's `main()` directly

**5. Console summary on completion** — Every script prints a ✅ block at the end showing row count, regions/variables processed, years covered, and the output path. This gives quick confirmation that the data landed where expected without opening the output file.

---

### Suppression Handling

Government data sources use various codes for confidential or unavailable cells:

| Source | Suppression codes | Handling strategy |
|--------|-------------------|-------------------|
| Stats Can CSVs | `'x'`, `'..'`, `'F'`, `'E'` | Read as string, detect, fill by interpolation or population proxy |
| CEUD Excel | `'X'` | `_to_float()` converts `'X'` (and any non-numeric cell) to `NaN`; `pct_series()` additionally filters negatives to catch cases where Polars numerically coerces `'X'` to `-1.0` in percentage columns |
| NIR inventory | `'x'` | Algebraic back-calculation from Canada total + known provinces |

### Projections

Historical data typically ends between 2022 and 2024. The pipeline extends series to 2100 in various ways using the data_extensions.py util. A common one used across activity drivers is the CAGR function. It uses paramters defined  `data/raw_data/assumptions/activity_cagr_projections.csv` to determine the CAGR calculation period, and determine the projection period splits and dampeners. 

CAGR Example
```
2014 → 2024       : Computed CAGR
2025 → 2050       : 50%  of computed CAGR
2036 → 2100       : 20%  of computed CAGR
```

Sector- and province-specific overrides are applied where the national historical CAGR is unrepresentative (e.g., Alberta Waste: 2%, 1%, 0.5% across the three periods).

### Wide-to-Long Conversion

Fixed structural data is stored as wide CSVs (columns = years 2000, 2005, 2010, 2015, 2020, 2025, 2030, …, 2050). The pipeline:
1. Reads the wide file.
2. Linearly interpolates to fill every year between anchor columns.
3. Holds the 2050 value constant through 2100.
4. Writes rows in long format (one row per parameter per year).

### Row Ordering in CIMS Outputs

Sector assemblers maintain an `_order` integer column during construction and sort by it before writing. This preserves the structural hierarchy CIMS expects (e.g., shell rows must precede HVAC rows, which must precede output rows for a given building type).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `polars` | Primary DataFrame library in all recent modules |
| `pandas` | Legacy support and Excel reading |
| `numpy` | Numeric operations |
| `scipy` | Linear regression for trend fitting (`data_extensions.py`) |
| `openpyxl` | Reading `.xlsx` CEUD and NIR workbooks |
| `marimo` | Optional — interactive UI for `control.py` |

Install with:

```powershell
pip install polars pandas numpy scipy openpyxl marimo
```

**Path convention**: every layer's folder is a constant in `src/CIMS/data_processing/common/paths.py`. The data root is the `data/` folder beside the imported code, unless the `CIMS_DATA_ROOT` environment variable points somewhere else. The runners print the code and data roots first, and warn when the data root isn't inside the current folder (`--strict` makes that an error). Keep the layout under the data root as described above.

---

## What is and is not Committed

### Committed (tracked in git)

- All Python scripts in `src/CIMS/data_processing/`
- Everything in `config/`: the control file, mappings, conversions, assumptions, the FIC
  lists, the model defaults, the interim CER tables, and the technology database's records.
  The database build is
  [`src/CIMS/data_processing/technology_database/`](../src/CIMS/data_processing/technology_database/README.md);
  it reads the extractions from `extracted/technology-database/` (from FRDR) and writes
  `processed/technology-database/`.
- `model/README.md`, and this README and the sub-READMEs

### Not committed (gitignored)

| Directory | Reason |
|-----------|--------|
| `primary-sources/` | Documents as obtained: SharePoint |
| `extracted/` | Values the build reads: SharePoint or FRDR |
| `processed/` | Written by the source stages |
| `model/` | Written by the sector stages (policies are made by hand until the policy stage exists) |
| `__pycache__/` | Python bytecode |
