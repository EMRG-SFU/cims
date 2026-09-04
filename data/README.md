# CIMS Data

This directory holds CIMS **data** — raw inputs, generated outputs, and the small set
of version-controlled reference tables. The scripts that process it live in the
package, under [`src/CIMS/data_processing/`](../src/CIMS/data_processing/). Raw data
is **never committed** to this repository.

## Quick Start

1. **Get raw data**:
   - Raw data may be sourced from SharePoint, FRDR, or another data repository
   - Place files in `raw_data/` — this directory is gitignored

2. **Run a processing script** from `src/CIMS/data_processing/`:
   - `source/` — scripts that process raw source data into `processed_data/`
   - `sector/` — one directory per sector; scripts output to `calibration/` and `model_inputs/`
   - `utils/` — shared utility functions

   Each script is runnable on its own, e.g.
   `python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange`.
   Scripts resolve `data/` from their own location, so the working directory does
   not matter.

3. **Check outputs**:
   - `processed_data/` — intermediate CSVs for validation (gitignored)
   - `calibration/` — calibration CSVs output by sector scripts (gitignored)
   - `model_inputs/` — model input CSVs output by sector scripts (gitignored)

Some scripts document their own outputs in a README beside the script — for
instance the currency deflator and exchange rate tables in
`processed_data/deflator_exchange/`, described in
[`src/CIMS/data_processing/source/deflator_exchange/README.md`](../src/CIMS/data_processing/source/deflator_exchange/README.md).

## What IS committed

- Mappings and conversion files (`mappings_conversions/`)
- Empty directory structure (`.gitkeep` files)
- This README

Processing scripts are committed too, but they live in `src/CIMS/data_processing/`,
not here.

## What is NOT committed

| Directory         | Reason                         |
|-------------------|--------------------------------|
| `raw_data/`       | Large source files — store locally or access via SharePoint/FRDR |
| `processed_data/` | Generated — do not commit      |
| `calibration/`    | Generated — do not commit      |
| `model_inputs/`   | Generated — do not commit      |
