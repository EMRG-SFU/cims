"""
DCC Pipeline — Model Inputs

Flattens all DCC fixed-data CSVs from wide (2000–2050 year columns)
to long format and writes one output CSV per input file.

Sources
-------
Fixed DCC parameters
    raw_data/fixed_data/dcc/DCC_{region}.csv
    One file per Canadian province/territory.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import tempfile

from pathlib import Path

import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────
import CIMS.data_processing.utils.flatten_fixed_data as _flatten_mod

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.controls_conversions import DATA_START, PROJECTION_END, LAST_DATA_YEAR
from CIMS.data_processing.utils.currency import write_model_csv
from CIMS.data_processing.utils.techdb_steam import drop_replaced_steam_rows

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = paths.FIXED_DATA / 'dcc'
OUTPUT_DIR      = paths.MODEL / 'dcc'


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Flatten all DCC fixed CSVs and write outputs to model_inputs/model/dcc."""
    print('=' * 60)
    print('DCC MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening DCC fixed data...')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        output_files = sorted(tmp_path.rglob('*.csv'))
        total_rows = 0
        for f in output_files:
            dest = OUTPUT_DIR / f.name
            df = drop_replaced_steam_rows(pl.read_csv(f, infer_schema_length=0))
            total_rows += len(df)
            write_model_csv(df, dest)
            print(f'  Wrote {len(df):,} rows -> {dest.name}')

    print(f'\nDCC model inputs complete')
    print(f'   Total rows: {total_rows:,}')
    print(f'   Files:      {len(output_files)}')


if __name__ == '__main__':
    main()
