"""
Hydrogen Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Hydrogen/hydrogen_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Hydrogen hierarchy:
      Hydrogen (Sector)
        ├── Production (Service, Tech Compete)
        │     ├── Coal gasification_centralised_CCS
        │     ├── Steam methane reforming_centralised
        │     ├── Steam methane reforming_centralised_CCS
        │     ├── Biomass gasification_centralised
        │     ├── Biomass gasification_centralised_CCS
        │     ├── Electrolysis_centralised
        │     ├── Steam methane reforming_distributed
        │     ├── Ethanol_distributed
        │     └── Electrolysis_distributed
        ├── Infrastructure (Service, Tech Compete)
        └── CCS (Service, Tech Compete)

    No external activity data source is used for this sector.
    All parameters including market shares, costs, and service_request
    routing are fully contained in the fixed data.

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Hydrogen is classified as Industrial in sector_map.csv.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. multiplier_price — from energy_price_multipliers
2. rest of fixed data
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
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Hydrogen'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/hydrogen'

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
    """Flatten all Hydrogen fixed CSVs and return combined DataFrame."""
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


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Hydrogen sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Hydrogen')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Hydrogen')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Hydrogen').alias('Sector'),
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
    """Assemble Hydrogen model inputs and write one CSV per region."""
    print('=' * 60)
    print('HYDROGEN MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    output = (
        pl.concat(
            [price_rows, fixed],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'hydrogen_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Hydrogen model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
