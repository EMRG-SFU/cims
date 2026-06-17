"""
Flatten transmission fixed data to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Transmission/transmission_CIMS.csv
    Flattened from wide (2000-2050 year columns) to long format via
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

from utils.controls_conversions import DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ─────────────────────────────────────────────────────────────
BASE_PATH       = Path('C:/cims/data')
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Transmission'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/transmission'
OUTPUT_FILE     = OUTPUT_DIR / 'transmission_CIMS.csv'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    src = FIXED_INPUT_DIR / 'transmission_CIMS.csv'
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_file = tmp_path / src.name
        _flatten_mod.process_file(
            input_path=src,
            output_path=out_file,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR['cer'],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        return pl.read_csv(out_file, infer_schema_length=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Flatten transmission fixed data and write a single CSV."""
    print('=' * 60)
    print('TRANSMISSION MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed data...')
    output = _read_flattened_fixed_data().cast(pl.String).select(OUTPUT_COLS)
    print(f'  Rows: {len(output):,}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_csv(OUTPUT_FILE)
    print(f'  Wrote {len(output):,} rows -> {OUTPUT_FILE.name}')

    return output


if __name__ == '__main__':
    main()
