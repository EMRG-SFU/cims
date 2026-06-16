"""
Ethanol Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Ethanol/ethanol_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Each region has its own file; FIXED_TEMPLATE maps 1:1.

Energy price multipliers  (multiplier_price rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Inserted after the Ethanol sector is_supply row.

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

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
BASE_PATH       = Path('C:/cims/data')
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Ethanol'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/ethanol'

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

REGION_SPECIFIC_ENERGIES: set[str] = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    """Flatten one fixed Ethanol CSV and return as a row-indexed DataFrame."""
    fixed_path = FIXED_INPUT_DIR / f'ethanol_{region}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'ethanol_{region}.csv'
        _flatten_mod.process_file(
            input_path=fixed_path,
            output_path=out_file,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR["cer"],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)
    return df.with_row_index('_order')


def _build_price_mult_rows(multipliers: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """multiplier_price rows for the Ethanol sector."""
    data = (
        multipliers
        .filter((pl.col('Sector') == 'Ethanol') & (pl.col('Region') == region))
        .sort('Energy', 'Year')
    )
    n = len(data)
    return data.select([
        pl.lit(f'CIMS.CAN.{region}.Ethanol').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Ethanol').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('multiplier_price').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.when(pl.col('Energy').is_in(list(REGION_SPECIFIC_ENERGIES)))
        .then(pl.lit(f'CIMS.CAN.{region}.') + pl.col('Energy'))
        .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('Energy'))
        .alias('Target'),
        pl.col('Source').alias('Source'),
        pl.lit('').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Multiplier').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i * 1e-4 for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _find_max_order(df: pl.DataFrame, service: str, parameter: str,
                    require_tech: bool = False) -> float | None:
    """Return the max _order value for rows matching service + parameter."""
    mask = (pl.col('Service').fill_null('') == service) & (pl.col('Parameter').fill_null('') == parameter)
    if require_tech:
        mask = mask & (pl.col('Technology').fill_null('') != '')
    subset = df.filter(mask)
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


def _assemble_region(
    fixed: pl.DataFrame,
    multipliers: pl.DataFrame,
    region: str,
) -> pl.DataFrame:
    """Build the complete model-inputs DataFrame for one region."""
    # Price multipliers: just after Ethanol sector is_supply row
    is_supply_max = _find_max_order(fixed, '', 'is_supply') or 0.0
    price_rows = _build_price_mult_rows(
        multipliers, region, start_order=is_supply_max + 0.5
    )

    combined = pl.concat(
        [f for f in [fixed.cast({'_order': pl.Float64}), price_rows] if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble Ethanol model inputs and write one CSV per region."""
    print('=' * 60)
    print('ETHANOL MODEL INPUTS')
    print('=' * 60)

    print('\nLoading pipeline data...')
    multipliers = pl.from_pandas(_energy_price_mod.main())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'ethanol_{template}.csv'
        if not fixed_path.exists():
            print(f'  ⚠  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            fixed = _read_flattened_fixed(region)

            print('  Assembling...')
            output = _assemble_region(fixed, multipliers, region)

            out_path = OUTPUT_DIR / f'ethanol_{region}.csv'
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
