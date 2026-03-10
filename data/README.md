# CIMS Data

This directory contains the data pipeline for CIMS. Raw data is **never committed** to this repository.

## Quick Start

1. **Get raw data**:
   - SFU team: access via SharePoint
   - Local development: place files in `raw_data/` — this directory is gitignored

2. **Run the pipeline**:
   - `pipeline/source/` — scripts that process raw source data into `processed_data/`
   - `pipeline/sector/` — one directory per sector; scripts output to `calibration/` and `model_inputs/`
   - `pipeline/utils/` — shared utility functions

3. **Check outputs**:
   - `processed_data/` — intermediate CSVs for validation (gitignored)
   - `calibration/` — calibration CSVs output by sector scripts (gitignored)
   - `model_inputs/` — model input CSVs output by sector scripts (gitignored)

## What IS committed

- Pipeline scripts (`pipeline/`)
- Mappings and conversion files (`mappings_conversions/`)
- Empty directory structure (`.gitkeep` files)
- This README and `cims_data_mapping.txt`

## What is NOT committed

| Directory         | Reason                         |
|-------------------|--------------------------------|
| `raw_data/`       | Sensitive / large source files |
| `processed_data/` | Generated — do not commit      |
| `calibration/`    | Generated — do not commit      |
| `model_inputs/`   | Generated — do not commit      |
