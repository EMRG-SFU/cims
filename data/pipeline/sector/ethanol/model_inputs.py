"""
Ethanol Pipeline — Model Inputs

Combines fixed structural parameters with energy price multipliers into
CIMS-formatted CSVs (one per region). Ethanol has no external activity
inputs — production levels are demand-driven and all structural parameters
are sourced from the fixed data.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Ethanol/ethanol_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Ethanol hierarchy:
      Ethanol (Sector)
        └── Production (Service, Fixed Ratio)
              ├── Corn       (Service, Fixed Ratio)
              │     ├── Agricultural Input  (shared utility service)
              │     └── Steam              (shared utility service)
              └── Cellulosic (Service, Fixed Ratio)
                    └── Steam              (shared utility service)
      Ethanol.Agricultural Input  (shared utility service)
      Ethanol.Steam               (shared utility service)
      Ethanol.Methane Fuel        (shared utility service)
      Ethanol.CCS                 (shared utility service)

    All service_provide rows have null values (sector is demand-driven).
    All service_request rows are static intensity coefficients.
    CCS.Target is region-specific (CIMS.Transmission.CCS TS {region}).

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. competition      — from fixed data (Sector level)
2. is_supply        — generated (TRUE)
3. multiplier_price — from energy_price_multipliers
4. rest of fixed data
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

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Ethanol fixed CSVs and return combined DataFrame."""
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


def _build_is_supply_rows(regions: list[str]) -> pl.DataFrame:
    """Generate a single is_supply=TRUE row per region for the sector header."""
    return pl.DataFrame([
        {
            'Branch':      f'CIMS.CAN.{r}.Ethanol',
            'Type':        'Sector',
            'Region':      r,
            'Sector':      'Ethanol',
            'Service':     '',
            'Technology':  '',
            'Parameter':   'is_supply',
            'Context':     'TRUE',
            'Sub_Context': '',
            'Target':      '',
            'Source':      '',
            'Unit':        '',
            'Year':        '',
            'Value':       '',
        }
        for r in regions
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Ethanol sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Ethanol')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Ethanol')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Ethanol').alias('Sector'),
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
    """Assemble Ethanol model inputs and write one CSV per region."""
    print('=' * 60)
    print('ETHANOL MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')

    # Sector-level branch: ends with '.Ethanol' (no further sub-path)
    _sector_branch = pl.col('Branch').str.ends_with('.Ethanol')

    # Keep only competition from fixed data at sector level;
    # service_provide has null values throughout (sector is demand-driven)
    # and is retained in fixed_rest unchanged.
    fixed_sector_competition = fixed.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # Everything from fixed except sector-level competition, which is
    # repositioned above is_supply to match standard output ordering.
    fixed_rest = fixed.filter(
        ~(_sector_branch & (pl.col('Parameter').fill_null('') == 'competition'))
    )

    regions = fixed['Region'].drop_nulls().unique().sort().to_list()
    is_supply_rows = _build_is_supply_rows(regions)

    output = (
        pl.concat(
            [fixed_sector_competition, is_supply_rows, price_rows, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'ethanol_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Ethanol model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
