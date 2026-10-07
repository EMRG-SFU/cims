"""
FIC Pipeline — Model Inputs

Flattens fixed incremental cost (FIC) data into CIMS-formatted CSVs
(one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/fic/fic_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Each region has its own file; FIXED_TEMPLATE maps 1:1.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""


import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_file

from CIMS.data_processing.utils.controls_conversions import BASE_PATH
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/fic'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/fic'

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

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Flatten FIC fixed data and write one CSV per region."""
    print('=' * 60)
    print('FIC MODEL INPUTS')
    print('=' * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'fic_{template.lower()}.csv'
        if not fixed_path.exists():
            print(f'  ⚠  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            output = read_fixed_data_file(fixed_path).select(OUTPUT_COLS)

            out_path = OUTPUT_DIR / f'fic_{region.lower()}.csv'
            output = collapse_constant_years(output)
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
