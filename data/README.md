# CIMS Data Processing

This directory transforms energy data into model input files and calibration files for the CIMS energy-economy model. Raw data is **never committed** to this repository — only pipeline scripts, mappings, and this documentation are tracked. Raw data is stored on the External-CIMS sharpoint. Model input files are used to build the model with information like sectors, activity levels, services, technologies, model parameters, and fuel prices. Calibration files include the historical demand, emissions, and technology shares publicly available. During the calibration step, the user will match the model from 2000-last historical year to the calibration data in order to determine intangible costs, trends, and a vetted starting point from which the projection begins. 

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Utility Library](#utility-library)
7. [Output Format](#output-format)
8. [Common Patterns](#common-patterns)
9. [Dependencies](#dependencies)
10. [What is and is not Committed](#what-is-and-is-not-committed)

---

## Overview

Raw data from various sources is converted into CIMS-formatted CSVs covering Canada's 13 provinces and territories. It operates in two stages:

- **Stage 1 — Source processing** (`pipeline/source/`): Each script ingests data and writes a CSV to `processed_data/` for user vetting. The processed_data csvs are not used elsewhere in the pipeline.
- **Stage 2 — Sector assembly** (`pipeline/sector/`): Each sector's model_inputs script combines fixed structural parameters (`raw_data/fixed_data/`) with one or more processed data frames and writes final CIMS model input files to `model_inputs/model/{sector}/`. Each sector's calibration script combines processed historical energy demand, emissions, and technology market shares and writes final CIMS calibration files to `calibration/{sector}/`.

All configuration (currency years, data years, scenarios) lives in a single file: `mappings_conversions/control.py`.

---

## Directory Structure

```
C:\cims\data\
├── mappings_conversions/       # Master configuration
│   ├── control.py              # Pipeline-wide settings (edit this to change data years, scenarios)
│   ├── energy_map.csv          # Map for energy types across sources
│   ├── region_map.csv          # Map for regions across sources
│   ├── sector_map.csv          # Map for sectors across sources 
│   ├── energy_conversions.csv  # Unit conversions (GJ ↔ TJ, L ↔ tonnes, …)
│   ├── NIR_to_CIMS_map.csv     # Maps NIR rows to CIMS branch nodes
│   └── README.md
│
├── pipeline/                   # All pipeline scripts (committed)
│   ├── source/                 # Stage 1 — raw → processed_data
│   │   ├── activity/           # Activity levels for all sectors (res/com/trans from ceud source)
│   │   ├── eccc/nir/           # National Inventory Report processing
│   │   ├── emission_factors/   # Fuel-level emission factors (NIR Annex 6, CEEDC)
│   │   ├── energy_prices/      # Energy prices and their multipliers (various sources)
│   │   └── nrcan/ceud/         # Comprehensive Energy Use Database extraction
│   │       ├── residential/
│   │       └── commercial/
│   │       └── transport_passenger/
│   │       └── transport_freight/
│   ├── sector/                 # Stage 2 — processed_data + fixed_data → model_inputs
│   │   ├── agriculture/
│   │   ├── biodiesel/
│   │   ├── chemical_products/
│   │   ├── commercial/
│   │   └── residential/
│   │   └── .../
│   └── utils/                  # Shared utility functions (committed)
│       ├── extractors/         # Source-specific readers (Stats Can, CEUD, CER)
│       ├── output_builder.py   # CIMS output row formatter
│       ├── data_extensions.py  # CAGR projection and trend fitting
│       ├── data_fill.py        # Interpolation and gap filling
│       ├── flatten_fixed_data.py
│       ├── add_cims_totals.py
│       ├── dict_ops.py
│       └── controls_conversions.py
│
├── raw_data/                   # Source files — gitignored, never committed
│   ├── assumptions/            # Assumption parameters that extend model input data through the projection period
│   ├── fixed_data/             # Structural parameters per sector from the JCIMS model (wide-format CSVs; should be updated over time)
│   │   ├── Agriculture/
│   │   ├── Biodiesel/
│   │   ├── Chemical Products/
│   │   ├── Commercial/
│   │   └── Residential/
│   ├── eccc/nir/               # NIR data
│   ├── cer/                    # CER data
│   ├── nrcan/ceud/             # CEUD data
│   ├── stats_can/              # Stats Can data
│   ├── ceedc/                  # CEEDC data
│   ├── ipcc/                   # IPCC data
│   ├── ab_gov/                 # Alberta government data
│   └── bc_gov/                 # BC government data
│
├── processed_data/             # Stage 1 outputs — gitignored, auto-generated, for quick visualization of processed data
│   ├── activity/
│   ├── eccc/
│   ├── energy_prices/
│   └── nrcan/ceud/
│
├── model_inputs/               # Stage 2 outputs — gitignored, auto-generated
│   └── model/
│       ├── agriculture/
│       ├── biodiesel/
│       ├── chemical_products/
│       ├── commercial/
│       └── residential/
│
└── calibration/                # Calibration outputs — gitignored, auto-generated
    ├── commercial/
    └── residential/
```

---

## Quick Start

### Prerequisites

1. Install Python dependencies (see [Dependencies](#dependencies)).
2. Obtain raw data from the **External-CIMS SharePoint** and place it under `raw_data/` following the structure above.
   - `raw_data/data_dictionary.xlsx` documents every source: what it contains, where to download it, and how often it is published.

### Running the pipeline

You can run scripts individually (useful when only one source has been updated) or run everything at once:

```powershell
# Run the full pipeline in one go
python pipeline/source/run_all.py   # all Stage 1 source processors
python pipeline/sector/run_all.py   # all Stage 2 sector assemblers
```

Or run stages selectively. Stage 1 source processors (order generally does not matter):

```powershell
# Examples — run each script that corresponds to the data you have
python pipeline/source/activity/emissions_drivers.py
python pipeline/source/nrcan/ceud/residential/residential.py
python pipeline/source/nrcan/ceud/commercial/commercial.py
python pipeline/source/energy_prices/energy_prices.py
python pipeline/source/energy_prices/energy_price_multipliers.py
python pipeline/source/eccc/nir/nir_to_cims.py
python pipeline/source/emission_factors/emission_factors.py
```

Run Stage 2 sector assemblers (order of sector vs source assembler running does not matter):

```powershell
python pipeline/sector/agriculture/model_inputs.py
python pipeline/sector/agriculture/calibration.py
python pipeline/sector/commercial/model_inputs.py
python pipeline/sector/commercial/calibration.py
...
```

Model inputs outputs land in `model_inputs/model/{sector}/` — one CSV per province/territory (13 files per sector). Calibration outputs land in `calibration/{sector}/`


---

## Configuration

All pipeline-wide parameters are set in **`mappings_conversions/control.py`**. This file exports a `CONTROLS` dict that every script imports via `pipeline/utils/controls_conversions.py`.

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

1. Download the file and replace it in its folder under `raw_data/`.
2. Open the `control.py` Marimo notebook:
   ```powershell
   cd C:\cims\data\mappings_conversions
   marimo edit control.py
   ```
3. Update the relevant `last_data_year_*` value (and currency year or scenario name if needed). The pipeline uses the year *after* the last data year as the start of its projection assumptions, so keeping this current is important.
4. Hit **Save to control.py** and exit the Marimo session.
5. Re-run the affected source script(s) and sector script(s).

---

## Utility Library

All utilities live in `pipeline/utils/`.

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

Converts wide fixed-data CSVs (one row per parameter, year columns 2000–2050) into long format with annual rows. Applies linear interpolation across the 2000–2050 range and holds the 2050 value constant through 2100.

### `add_cims_totals.py`

Rolls up leaf-node rows to parent aggregate rows in the CIMS branch hierarchy. For example, shell technology rows (Retail Cold, Retail Marine, Office Cold, …) are summed to produce a Shell total row.

### `controls_conversions.py`

Loads and exposes the `CONTROLS` dict from `control.py` as module-level constants:

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
| `Branch` | Full CIMS node path | `CIMS.CAN.AB.Commercial.Buildings.Shell` |
| `Type` | Scope level | `Region`, `Sector`, `Service`, `Technology` |
| `Region` | Province/territory code | `AB` |
| `Sector` | Sector name | `Commercial` |
| `Service` | Service name | `Shell` |
| `Technology` | Technology name | `Retail (Cold)` |
| `Parameter` | Model parameter | `market_share_total`, `service_request`, `multiplier_price` |
| `Context` | Additional context (often blank) | |
| `Sub_Context` | Sub-context (often blank) | |
| `Target` | Demand/supply node this row applies to | `CIMS.CAN.AB.Commercial.Buildings.Shell.Retail (Cold)` |
| `Source` | Data provenance | `CEUD`, `CER`, `fixed_data` |
| `Unit` | Value unit | `% of m2`, `PJ`, `2025$/GJ` |
| `2000` … `2100` | One column per year, value for that year | `0.35` |

Files are saved as CSV with the first 12 columns as metadata and years 2000–2100 as remaining columns.

---

## Common Patterns

### Source Script Structure

Despite pulling from different government sources, all scripts in `pipeline/source/` share the same structure:

**1. Module-level docstring** — Every script opens with a prominent docstring (sometimes a `===` banner) that states what it does, its input files and output location, and its specific suppression-handling strategy. This is the first place to look when diagnosing unexpected output.

**2. `sys.path` setup** — Immediately after imports, every script locates the project root relative to `__file__` and inserts it into `sys.path`:

```python
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
```

This lets the script be run from any working directory (`python pipeline/source/activity/emissions_drivers.py` from `C:\cims\data\`, or directly from its own folder) without import errors.

**3. Configuration block** — Input/output paths and key constants (e.g. `DATA_START`, `LAST_HIST_YEAR`, `REGIONS`) are declared at module level, loaded from `controls_conversions.py`. All paths are built with `BASE_PATH / '...'` so they resolve correctly regardless of working directory.

**4. `main()` function + `__main__` guard** — All processing is wrapped in a `main()` function that returns a DataFrame. The `if __name__ == '__main__': main()` guard at the bottom means:
- Scripts can be run directly: `python emissions_drivers.py`
- `run_all.py` can import and call each `main()` in-process without spawning subprocesses
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

Historical data typically ends between 2022 and 2024. The pipeline extends series to 2100 in various ways using the data_extensions.py util. A common one used across activity drivers is the CAGR function. It uses paramters defined  C:\cims\data\raw_data\assumptions\activity_cagr_projections.csv to determine the CAGR calculation period, and determine the projection period splits and dampeners. 

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

**Path convention**: All scripts hardcode `C:/cims/data` as the base path. The raw data directory layout under `raw_data/` must match the structure described above.

---

## What is and is not Committed

### Committed (tracked in git)

- All Python scripts in `pipeline/`
- Mapping and conversion files in `mappings_conversions/`
- `.gitkeep` files preserving empty output directory structure
- This README and sub-READMEs

### Not committed (gitignored)

| Directory | Reason |
|-----------|--------|
| `raw_data/` | Large source files — obtain from SharePoint or FRDR |
| `processed_data/` | Auto-generated by Stage 1 scripts |
| `calibration/` | Auto-generated by sector calibration scripts |
| `model_inputs/` | Auto-generated by Stage 2 sector assemblers |
| `__pycache__/` | Python bytecode |
