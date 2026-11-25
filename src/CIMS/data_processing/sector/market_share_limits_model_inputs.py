"""
Flatten market share limit fixed data to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    extracted/legacy/fixed-data/market_share_limits/*.csv
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

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.output_builder import write_per_region_csvs
from CIMS.data_processing.utils.techdb_steam import drop_replaced_steam_rows

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = paths.FIXED_DATA / 'market_share_limits'
OUTPUT_DIR      = paths.MODEL_EXOGENOUS / 'market-share-limit'

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

    output = drop_replaced_steam_rows(fixed.cast(pl.String).select(OUTPUT_COLS))

    regions = write_per_region_csvs(output, OUTPUT_DIR, 'market-share-limit', collapse_years=True, finalise_currency=True)

    print(f'\nMarket share limits model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
