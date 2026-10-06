"""
Flatten market share limit fixed data to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/market_share_limits/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""


import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_folder

from CIMS.data_processing.utils.controls_conversions import BASE_PATH
from CIMS.data_processing.utils.output_builder import write_per_region_csvs

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/market_share_limits'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/market_share_limits'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Flatten market share limit fixed data and write one CSV per region."""
    print('=' * 60)
    print('MARKET SHARE LIMITS MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed data...')
    fixed = read_fixed_data_folder(FIXED_INPUT_DIR)
    print(f'  Rows: {len(fixed):,}')

    output = fixed.cast(pl.String).select(OUTPUT_COLS)

    regions = write_per_region_csvs(output, OUTPUT_DIR, 'market_share_limits', collapse_years=True)

    print(f'\nMarket share limits model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
