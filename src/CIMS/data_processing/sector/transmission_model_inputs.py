"""
Flatten transmission fixed data to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    extracted/legacy/fixed-data/transmission/transmission_CIMS.csv
    Flattened from wide (2000-2050 year columns) to long format via
    utils/flatten_fixed_data.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""


import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_file

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.currency import write_model_csv
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = paths.FIXED_DATA / 'transmission'
OUTPUT_DIR      = paths.MODEL / 'transmission'
OUTPUT_FILE     = OUTPUT_DIR / 'transmission_cims.csv'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Flatten transmission fixed data and write a single CSV."""
    print('=' * 60)
    print('TRANSMISSION MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed data...')
    output = read_fixed_data_file(FIXED_INPUT_DIR / 'transmission_cims.csv').cast(pl.String).select(OUTPUT_COLS)
    print(f'  Rows: {len(output):,}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = collapse_constant_years(output)
    write_model_csv(output, OUTPUT_FILE)
    print(f'  Wrote {len(output):,} rows -> {OUTPUT_FILE.name}')

    return output


if __name__ == '__main__':
    main()
