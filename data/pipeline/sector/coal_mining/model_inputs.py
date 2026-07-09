"""
Coal Mining Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/coal_mining/coal_mining_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Coal Mining hierarchy:
      Coal Mining (Sector)
        └── Coal (Service, Fixed Ratio)
              ├── Raw Product (Service, Fixed Ratio)
              │     ├── Extraction (Service, Tech Compete)
              │     ├── Transportation (Service, Tech Compete)
              │     └── Mine Ventilation (Service, Tech Compete)
              ├── Size Reduced Product (Service, Fixed Ratio)
              │     ├── Primary Crushing (Service, Tech Compete)
              │     ├── Primary Milling (Service, Tech Compete)
              │     └── Secondary Milling (Service, Tech Compete)
              └── Metallurgical Finishing (Service, Fixed Ratio) — AB and BC only
                    ├── Washing (Service, Tech Compete)
                    ├── Cleaning (Service, Tech Compete)
                    └── Tailings Disposal (Service, Tech Compete)
    Plus HVAC and Lighting at the sector level.

    NB and NS have a simplified Extraction service (underground mining) directly
    under the Coal branch rather than under Raw Product. SK, NB, and NS do not
    have the Metallurgical Finishing sub-tree (no met coal production).

Total production  (service_request at sector level)
    pipeline/source/activity/coal_mining.py  (called directly via main())
    Variable: 'Coal Mining'  (kt, per province, 2000–2100)

Metallurgical finishing share  (service_request at Coal branch level)
    pipeline/source/activity/coal_mining.py  (called directly via main())
    Variable: 'Coal Mining.Coal.Metallurgical Finishing'  (%)
    Only produced for AB and BC (the met coal regions).

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Coal Mining is classified as Industrial in sector_map.csv.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request  — total production (from coal_mining)
2. competition      — from fixed data (Sector level)
3. multiplier_price — from energy_price_multipliers
4. rest of fixed data (Coal branch + all sub-services)
5. service_request  — Coal → Metallurgical Finishing (from coal_mining, AB and BC only)
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

_cm_spec = importlib.util.spec_from_file_location(
    'coal_mining',
    _PIPELINE_ROOT / 'source' / 'activity' / 'coal_mining.py',
)
_coal_mining_mod = importlib.util.module_from_spec(_cm_spec)
_cm_spec.loader.exec_module(_coal_mining_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/coal_mining'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/coal_mining'

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
    """Flatten all Coal Mining fixed CSVs and return combined DataFrame."""
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


def _build_total_rows(coal: pl.DataFrame) -> pl.DataFrame:
    """
    service_request row for the Coal Mining sector.
    Value = total coal production kt from coal_mining.
    """
    df = coal.filter(pl.col('Variable') == 'Coal Mining')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Coal Mining').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Coal Mining')).alias('Target'),
        pl.col('Source'),
        pl.lit('kt').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Coal Mining sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Coal Mining')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Coal Mining')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Coal Mining').alias('Sector'),
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


def _build_met_finishing_rows(coal: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Coal branch level pointing to Metallurgical Finishing.
    Value = met share as %(only produced for AB and BC).
    Branch: CIMS.CAN.{region}.Coal Mining.Coal
    Target: CIMS.CAN.{region}.Coal Mining.Coal.Metallurgical Finishing
    """
    df = coal.filter(pl.col('Variable') == 'Coal Mining.Coal.Metallurgical Finishing')
    if df.is_empty():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Coal Mining.Coal')).alias('Branch'),
        pl.lit('Service').alias('Type'),
        pl.col('Region'),
        pl.lit('Coal Mining').alias('Sector'),
        pl.lit('Coal').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Coal Mining.Coal.Metallurgical Finishing')).alias('Target'),
        pl.col('Source'),
        pl.lit('%').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Coal Mining model inputs and write one CSV per region."""
    print('=' * 60)
    print('COAL MINING MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading coal mining activity data...')
    coal = _coal_mining_mod.main()

    print('Building total production rows...')
    total_rows = _build_total_rows(coal)
    print(f'  Rows: {len(total_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building metallurgical finishing service request rows...')
    met_rows = _build_met_finishing_rows(coal)
    print(f'  Rows: {len(met_rows):,}')

    print('Combining...')

    # Sector-level branch: ends with '.Coal Mining' (no further sub-path)
    _sector_branch = pl.col('Branch').str.ends_with('.Coal Mining')

    # Keep only competition from fixed data at sector level;
    # service_provide rows remain null (fixed data); activity data inserted as service_request.
    fixed_sector_competition = fixed.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # Everything else from fixed: Coal branch, all sub-services, HVAC, Lighting.
    # This includes the Coal branch service_request rows pointing to Raw Product,
    # Size Reduced Product, HVAC, and Lighting (all fixed constants).
    # The Met Finishing service_request is injected separately as met_rows.
    fixed_rest = fixed.filter(
        ~(_sector_branch & pl.col('Parameter').fill_null('').is_in(['competition']))
    )

    regions = total_rows['Region'].unique().sort().to_list()

    output = (
        pl.concat(
            [total_rows, fixed_sector_competition,
             price_rows, fixed_rest, met_rows],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'coal_mining_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Coal Mining model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
