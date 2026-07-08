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

import sys
import tempfile
import importlib.util
from pathlib import Path

import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flatten_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/market_share_limits'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/market_share_limits'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _flatten_mod.main(
            input_folder=FIXED_INPUT_DIR,
            output_folder=tmp_path,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR['cer'],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        frames = [
            pl.read_csv(f, infer_schema_length=0)
            for f in sorted(tmp_path.rglob('*.csv'))
        ]
    return pl.concat(frames)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Flatten market share limit fixed data and write one CSV per region."""
    print('=' * 60)
    print('MARKET SHARE LIMITS MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed data...')
    fixed = _read_flattened_fixed_data()
    print(f'  Rows: {len(fixed):,}')

    output = fixed.cast(pl.String).select(OUTPUT_COLS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'market_share_limits_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows -> {out_path.name}')

    print(f'\nMarket share limits model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
