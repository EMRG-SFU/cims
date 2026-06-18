"""
FIC Pipeline — Model Inputs

Flattens fixed incremental cost (FIC) data into CIMS-formatted CSVs
(one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/FIC/FIC_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Each region has its own file; FIXED_TEMPLATE maps 1:1.

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

# ── path setup ─────────────────────────────────────────────────────────────────
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

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/FIC'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/FIC'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

FIXED_TEMPLATE: dict[str, str] = {
    'AB': 'AB', 'BC': 'BC', 'MB': 'MB', 'NB': 'NB', 'NL': 'NL',
    'NS': 'NS', 'NT': 'NT', 'NU': 'NU', 'ON': 'ON', 'PE': 'PE',
    'QC': 'QC', 'SK': 'SK', 'YT': 'YT',
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    """Flatten one FIC CSV and return as a long-format DataFrame."""
    fixed_path = FIXED_INPUT_DIR / f'FIC_{region}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'FIC_{region}.csv'
        _flatten_mod.process_file(
            input_path=fixed_path,
            output_path=out_file,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR['cer'],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)
    return df


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Flatten FIC fixed data and write one CSV per region."""
    print('=' * 60)
    print('FIC MODEL INPUTS')
    print('=' * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'FIC_{template}.csv'
        if not fixed_path.exists():
            print(f'  ⚠  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            output = _read_flattened_fixed(template).select(OUTPUT_COLS)

            out_path = OUTPUT_DIR / f'FIC_{region}.csv'
            output.write_csv(str(out_path))
            print(f'  Wrote {len(output):,} rows → {out_path.name}')
            results[region] = output

        except Exception as exc:
            print(f'  ERROR: {exc}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Regions complete: {len(results)}/{len(FIXED_TEMPLATE)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
