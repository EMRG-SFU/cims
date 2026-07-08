"""
Extract Petroleum Refining model input data and save to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/petroleum_refining/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.

Activity demand  (service_request)
    processed_data/activity/petroleum_refining.csv
    Produced by pipeline/source/activity/petroleum_refining.py.
    Variable: 'Petroleum Refining'  (m3 crude input, per province, 2000–2100)

Energy price multipliers  (multiplier_price rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Produced by pipeline/source/energy_prices/energy_price_multipliers.py.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request              — region-level demand (from petroleum_refining activity)
2. service_provide, competition — sector header from fixed data
3. multiplier_price             — from energy_price_multipliers
4. service_request              — sector tail from fixed data (→ Refined Petroleum Products)
5. rest of fixed data (all sub-service rows)
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

_pr_spec = importlib.util.spec_from_file_location(
    'petroleum_refining_activity',
    _PIPELINE_ROOT / 'source' / 'activity' / 'petroleum_refining.py',
)
_petroleum_refining_mod = importlib.util.module_from_spec(_pr_spec)
_pr_spec.loader.exec_module(_petroleum_refining_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/petroleum_refining'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/petroleum_refining'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Petroleum Refining fixed CSVs and return combined DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _flatten_mod.main(
            input_folder=FIXED_INPUT_DIR,
            output_folder=tmp_path,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR["cer"],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        frames = [
            pl.read_csv(f, infer_schema_length=0)
            for f in sorted(tmp_path.rglob('*.csv'))
        ]
    return pl.concat(frames, how='diagonal_relaxed').cast(pl.String)


def _build_activity_rows(activity: pl.DataFrame) -> pl.DataFrame:
    """
    Region-level service_request rows pointing to the Petroleum Refining sector.
    Value = total crude input to refineries in m3 from petroleum_refining.
    """
    df = activity.filter(pl.col('Variable') == 'Petroleum Refining')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Petroleum Refining').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Petroleum Refining')).alias('Target'),
        pl.col('Source'),
        pl.col('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Petroleum Refining sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Petroleum Refining')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Petroleum Refining')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Petroleum Refining').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('multiplier_price').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.when(pl.col('Energy').is_in(list(REGION_SPECIFIC_ENERGIES)))
            .then(pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.') + pl.col('Energy'))
            .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('Energy'))
            .alias('Target'),
            pl.col('Source').alias('Source'),
            pl.lit('').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Multiplier').cast(pl.String).alias('Value'),
        ])
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble petroleum refining model inputs and write one CSV per region."""
    print('=' * 60)
    print('PETROLEUM REFINING MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading petroleum refining activity data...')
    activity = _petroleum_refining_mod.main()
    activity_rows = _build_activity_rows(activity)
    print(f'  Rows: {len(activity_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    fixed_str = fixed.cast(pl.String)
    _sector_branch = pl.col('Branch').str.ends_with('.Petroleum Refining')
    _header_params = pl.col('Parameter').is_in(['service_provide', 'competition'])

    fixed_sector_header = fixed_str.filter(_sector_branch & _header_params)
    fixed_sector_tail   = fixed_str.filter(_sector_branch & ~_header_params)
    fixed_rest          = fixed_str.filter(~_sector_branch)

    output = (
        pl.concat(
            [activity_rows, fixed_sector_header, price_rows,
             fixed_sector_tail, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = activity_rows['Region'].unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'petroleum_refining_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Petroleum Refining model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
