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

import importlib.util
import shutil
import sys
import tempfile

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
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/dcc'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/dcc'


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
            df = pl.read_csv(f, infer_schema_length=0)
            total_rows += len(df)
            shutil.copy(f, dest)
            print(f'  Wrote {len(df):,} rows -> {dest.name}')

    print(f'\nDCC model inputs complete')
    print(f'   Total rows: {total_rows:,}')
    print(f'   Files:      {len(output_files)}')


if __name__ == '__main__':
    main()
