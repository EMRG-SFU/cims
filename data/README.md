# CIMS Data Pipeline

This directory contains the full ETL pipeline that transforms government statistical releases into model-ready input files for the CIMS energy-economy model. Raw data is **never committed** to this repository — only pipeline scripts, mappings, and this documentation are tracked.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Source Processors](#source-processors)
7. [Sector Assemblers](#sector-assemblers)
8. [Utility Library](#utility-library)
9. [Output Format](#output-format)
10. [Common Patterns](#common-patterns)
11. [Dependencies](#dependencies)
12. [What is and is not Committed](#what-is-and-is-not-committed)

---

## Overview

The pipeline converts raw releases from Statistics Canada, NRCan, ECCC, and CER into CIMS-formatted CSVs covering Canada's 13 provinces and territories over 2000–2100. It operates in two stages:

- **Stage 1 — Source processing** (`pipeline/source/`): Each script ingests one government data source and writes a normalized intermediate CSV to `processed_data/`.
- **Stage 2 — Sector assembly** (`pipeline/sector/`): Each sector script combines fixed structural parameters (`raw_data/fixed_data/`) with one or more processed intermediates and writes final CIMS model input files to `model_inputs/model/{sector}/`.

A parallel **calibration** branch writes validation outputs to `calibration/{sector}/`.

All configuration (currency years, data years, scenarios) lives in a single file: `mappings_conversions/control.py`.

---

## Directory Structure

```
C:\cims\data\
├── mappings_conversions/       # Master configuration and crosswalk tables (committed)
│   ├── control.py              # Pipeline-wide settings (edit this to change data years, scenarios)
│   ├── energy_map.csv          # Standardised energy carrier names
│   ├── region_map.csv          # NIR region names → CIMS province/territory codes
│   ├── sector_map.csv          # Sector classifications
│   ├── energy_conversions.csv  # Unit conversions (GJ ↔ TJ, L ↔ tonnes, …)
│   ├── NIR_to_CIMS_map.csv     # Maps NIR rows to CIMS branch nodes
│   └── README.md
│
├── pipeline/                   # All pipeline scripts (committed)
│   ├── source/                 # Stage 1 — raw → processed_data
│   │   ├── activity/           # Emissions-driver time series (ECCC GHG, Stats Can GDP)
│   │   ├── eccc/nir/           # National Inventory Report → CIMS mapping
│   │   ├── emission_factors/   # Fuel-level emission factors (NIR Annex 6, CEEDC)
│   │   ├── energy_prices/      # CER + AFDC energy price multipliers
│   │   └── nrcan/ceud/         # Comprehensive Energy Use Database extraction
│   │       ├── residential/
│   │       └── commercial/
│   ├── sector/                 # Stage 2 — processed_data + fixed_data → model_inputs
│   │   ├── agriculture/
│   │   ├── biodiesel/
│   │   ├── chemical_products/
│   │   ├── commercial/
│   │   └── residential/
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
│   ├── assumptions/            # Projection assumption CSVs
│   ├── fixed_data/             # Structural parameters per sector (wide-format CSVs)
│   │   ├── Agriculture/
│   │   ├── Biodiesel/
│   │   ├── Chemical Products/
│   │   ├── Commercial/
│   │   └── Residential/
│   ├── eccc/nir/               # GHG_Econ_Can_Prov_Terr.csv
│   ├── cer/                    # Canada's Energy Future tables
│   ├── nrcan/ceud/             # CEUD Excel workbooks (residential + commercial)
│   ├── stats_can/              # Stats Can CSVs (GDP by industry, population, RESD)
│   ├── ceedc/                  # CEEDC coal emission factors
│   ├── ipcc/                   # IPCC 2006 reference emission factors
│   ├── ab_gov/                 # Alberta government data
│   └── bc_gov/                 # BC government data
│
├── processed_data/             # Stage 1 outputs — gitignored, auto-generated
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
2. Obtain raw data from SharePoint or FRDR and place it under `raw_data/` following the structure above.

### Running the pipeline

Run Stage 1 source processors first (order within Stage 1 generally does not matter):

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

Then run Stage 2 sector assemblers (each is independent):

```powershell
python pipeline/sector/agriculture/model_inputs.py
python pipeline/sector/residential/model_inputs.py
python pipeline/sector/commercial/model_inputs.py
python pipeline/sector/biodiesel/model_inputs.py
python pipeline/sector/chemical_products/model_inputs.py
```

Outputs land in `model_inputs/model/{sector}/` — one CSV per province/territory (13 files per sector).

---

## Data Flow

```
RAW DATA (gitignored)
        │
        ├─ ECCC GHG inventory CSV
        │   └─→ activity/emissions_drivers.py
        │       └─→ processed_data/activity/emissions_drivers.csv
        │
        ├─ NRCan CEUD Excel workbooks
        │   ├─→ nrcan/ceud/residential/residential.py
        │   │   └─→ processed_data/nrcan/ceud/residential.csv
        │   └─→ nrcan/ceud/commercial/commercial.py
        │       └─→ processed_data/nrcan/ceud/commercial.csv
        │
        ├─ Stats Can GDP table (36-10-0711-01)
        │   └─→ activity/light_industrial.py
        │       └─→ processed_data/activity/light_industrial.csv
        │
        ├─ CER + AFDC price data
        │   └─→ energy_prices/energy_price_multipliers.py
        │       └─→ processed_data/energy_prices/energy_price_multipliers.csv
        │
        └─ NIR Annex 6 + CEEDC emission factors
            └─→ emission_factors/emission_factors.py
                └─→ (used directly by sector scripts)

STAGE 2 — SECTOR ASSEMBLY
        │
        ├─ raw_data/fixed_data/{Sector}/         (structural parameters)
        ├─ processed_data/ intermediates above
        ├─ mappings_conversions/ crosswalks
        └─ mappings_conversions/control.py       (configuration)
                │
                ├─→ sector/agriculture/model_inputs.py
                │   └─→ model_inputs/model/agriculture/agriculture_{region}.csv  ×13
                ├─→ sector/residential/model_inputs.py
                │   └─→ model_inputs/model/residential/residential_{region}.csv  ×13
                ├─→ sector/commercial/model_inputs.py
                │   └─→ model_inputs/model/commercial/commercial_{region}.csv    ×13
                ├─→ sector/biodiesel/model_inputs.py
                │   └─→ model_inputs/model/biodiesel/biodiesel_{region}.csv      ×13
                └─→ sector/chemical_products/model_inputs.py
                    └─→ model_inputs/model/chemical products/..._{region}.csv    ×13
```

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

To update for a new data vintage, change the relevant `last_data_year_*` value and re-run the affected source scripts followed by dependent sector scripts.

---

## Source Processors

### `pipeline/source/activity/`

Extracts emissions-driver time series used as activity inputs in sector models.

**`emissions_drivers.py`**
- **Input**: `raw_data/eccc/nir/GHG_Econ_Can_Prov_Terr.csv` (ECCC national GHG inventory by province)
- **Sectors covered**: Agriculture, Waste, Construction, Forestry
- **Output**: `processed_data/activity/emissions_drivers.csv`
- Suppressed 'x' cells are filled by interpolation then forward-fill.
- Historical data (2000–last NIR year) is extended to 2100 with multi-period CAGR and dampening. Province-specific CAGR overrides are applied where the national rate is inappropriate (e.g., Alberta Waste).

**`light_industrial.py`**
- **Input**: `raw_data/stats_can/activity/36100711.csv` (Stats Can GDP by industry, chained 2017 dollars)
- **Sectors covered**: 7 light manufacturing industries
- **Output**: `processed_data/activity/light_industrial.csv`

Other activity scripts (`petroleum_refining.py`, `coal_mining.py`, `oil_production.py`, `gas_production.py`, `heavy_industry.py`) follow the same pattern for their respective sectors.

---

### `pipeline/source/eccc/nir/`

**`nir_to_cims.py`**
- **Input**: `raw_data/eccc/nir/GHG_Econ_Can_Prov_Terr.csv`
- **Output**: `processed_data/eccc/NIR_to_CIMS.csv`

Two-step process:
1. **Solve suppressed values** — Where provinces are suppressed ('x'), uses the Canada total and the known provincial values to back-calculate suppressed cells algebraically. Handles co-suppression years where multiple provinces are suppressed simultaneously (e.g., 2005 where NS, NWT, and NU are all suppressed in the same row).
2. **Map to CIMS nodes** — Applies `NIR_to_CIMS_map.csv` to assign each NIR row to a CIMS branch identifier. Rolls up leaf-level rows to parent aggregates using `add_cims_totals.py`.

---

### `pipeline/source/energy_prices/`

**`energy_prices.py`**
- **Inputs**: CER Canada's Energy Future tables, AFDC biofuel prices
- Extracts production cost time series by fuel and region.

**`energy_price_multipliers.py`**
- **Input**: `energy_prices.py` outputs + mapping files
- **Output**: `processed_data/energy_prices/energy_price_multipliers.csv`
- Computes region × fuel price multipliers relative to a base, adjusted for currency-year differences using CPI deflators.

See `pipeline/source/energy_prices/README.md` for source-specific detail.

---

### `pipeline/source/emission_factors/`

**`emission_factors.py`**
- **Primary source**: NIR Annex 6 Emission Factors Tables (Excel)
- **Secondary sources**: CEEDC (coal), IPCC 2006 (LPG)
- Covers 30+ fuels: natural gas, diesel, gasoline, coal grades, renewable fuels, hydrogen, electricity
- Outputs CO₂, CH₄, and N₂O factors in t/GJ
- Results are consumed directly by sector assemblers (not written to an intermediate CSV).

---

### `pipeline/source/nrcan/ceud/`

Extracts data from NRCan's Comprehensive Energy Use Database (CEUD) Excel workbooks.

**`residential/residential.py`**
- **Inputs**: Provincial CEUD Excel files (`res_{prov}_e.xls`, `res_ca_e_32.xls`)
- **Output**: `processed_data/nrcan/ceud/residential.csv`
- Extracts: housing stock, vintage distribution (age bins), space heating technology market shares, space cooling, domestic hot water heating
- **Regional disaggregation**: Canada-level totals are split to provinces using territory population shares (from Stats Can table 17-10-0009-01). Where CEUD suppresses provincial data ('X'), population-share proxies fill the gap.
- See `pipeline/source/nrcan/ceud/residential/README.md` for workbook layout details.

**`commercial/commercial.py`**
- **Inputs**: Provincial CEUD Excel files for commercial buildings
- **Output**: `processed_data/nrcan/ceud/commercial.csv`
- Extracts: floorspace by building type, HVAC technology splits, hot water technology splits
- **Regional grouping**: Data grouped into AB, AT (Atlantic), BC, MB, ON, QC, SK
- **BC climate split**: BC floorspace and HVAC shares split 80% Cold / 20% Marine to reflect two distinct climate zones in the CIMS model.

---

## Sector Assemblers

Each sector script in `pipeline/sector/{sector}/model_inputs.py` follows the same broad pattern:

1. Load fixed structural parameters from `raw_data/fixed_data/{Sector}/{sector}_{region}.csv` and flatten from wide (year columns) to long format.
2. Load relevant processed intermediates from `processed_data/`.
3. Interleave rows from both sources, maintaining the structural ordering CIMS expects.
4. Write one output CSV per region (13 files: AB, BC, MB, NB, NL, NS, NT, NU, ON, PE, QC, SK, YT).

### Agriculture

- Fixed data + emissions driver activity series (from ECCC GHG, extended to 2100)
- Service request rows for total Agriculture and sub-services (Process heat, etc.)
- Energy price multipliers for Diesel, Natural Gas, Electricity, Hydrogen

### Commercial

The most complex assembler. Interleaves in strict order:

1. Fixed structural data (lifetime, capital cost, output capacity, etc.)
2. CEUD floorspace rows as `service_request` inputs
3. Energy price multipliers (inserted after the Commercial header row)
4. Building shell market shares (from CEUD, with BC Cold/Marine split)
5. HVAC and hot water technology `market_share_total` rows (inserted between fixed lifetime and output rows per technology)

### Residential

- Fixed data + CEUD calibration extracts (housing stock, vintage shares, technology market shares)
- One file per province/territory, structured identically to commercial

### Biodiesel

- Fixed data + energy price multipliers
- No emissions drivers or CEUD data

### Chemical Products

- Fixed data + energy price multipliers

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

### Suppression Handling

Government data sources use various codes for confidential or unavailable cells:

| Source | Suppression codes | Handling strategy |
|--------|-------------------|-------------------|
| Stats Can CSVs | `'x'`, `'..'`, `'F'`, `'E'` | Read as string, detect, fill by interpolation or population proxy |
| CEUD Excel | `'X'`, `-1.0` | `_to_float()` returns NaN; filled by interpolation or province share |
| NIR inventory | `'x'` | Algebraic back-calculation from Canada total + known provinces |

### Regional Disaggregation

Where only a Canada-level total is available:
1. **Preferred**: Use the province-level CEUD workbook directly.
2. **Fallback**: Split Canada total by annual population shares (from Stats Can table 17-10-0009-01).
3. **BC special case**: BC totals are split 80% Cold climate / 20% Marine climate for HVAC and shell market shares.

### Multi-Period CAGR Projection

Historical data typically ends between 2022 and 2024. The pipeline extends series to 2100 in three phases:

```
Historical → 2050 : 100% of computed CAGR
2050 → 2075       : 50%  of computed CAGR
2075 → 2100       : 20%  of computed CAGR
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
