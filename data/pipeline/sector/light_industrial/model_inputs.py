"""
Light Industrial Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/light_industrial/light_industrial_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Light Industrial hierarchy:
      Light Industrial (Sector)
        └── Manufacturing (Service, Fixed Ratio)
              ├── Food Tobacco and Beverage  (Service, Tech Compete)
              ├── Rubber and Plastics        (Service, Tech Compete)
              ├── Leather and Textiles       (Service, Tech Compete)
              ├── Wood Products              (Service, Tech Compete)
              ├── Furniture Printing and Machinery (Service, Tech Compete)
              ├── Transportation Equipment   (Service, Tech Compete)
              └── Electronics and Other      (Service, Tech Compete)
    Plus Water Heating, Space Heating, Lighting, Ventilation AC at sector level.

    The sector-level service_request and the Manufacturing-branch service_request
    rows (splits into each sub-service) have been removed from the fixed data
    and are sourced from the pipeline instead (see below).

    Decisions/notes:
    1. Copied in missing rubber parameters to AT/TR (used BC as proxy but all regions were the same)
    2. Copied in missing FOM row to AB (used BC as proxy but all regions were the same)

Total activity  (service_request at sector level)
    pipeline/source/activity/light_industrial.py  (called directly via main())
    Variable: 'Light Industrial'  ($M 2017 GDP, per province/territory, 2000–2100)

Manufacturing sub-service splits  (service_request rows at Manufacturing level)
    pipeline/source/activity/light_industrial.py  (called directly via main())
    Variables (%, routed into Manufacturing sub-services):
      'Light Industrial.Manufacturing.Food Tobacco and Beverage'
      'Light Industrial.Manufacturing.Rubber and Plastics'
      'Light Industrial.Manufacturing.Leather and Textiles'
      'Light Industrial.Manufacturing.Wood Products'
      'Light Industrial.Manufacturing.Furniture Printing and Machinery'
      'Light Industrial.Manufacturing.Transportation Equipment'
      'Light Industrial.Manufacturing.Electronics and Other'

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Light Industrial is classified as Industrial in sector_map.csv.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request  — total activity (from light_industrial)
2. competition      — from fixed data (Sector level)
3. multiplier_price — from energy_price_multipliers
4. fixed_mfg_header — Manufacturing branch competition
5. service_request  — Manufacturing → each sub-service (from light_industrial)
6. rest of fixed data
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

_li_spec = importlib.util.spec_from_file_location(
    'light_industrial',
    _PIPELINE_ROOT / 'source' / 'activity' / 'light_industrial.py',
)
_light_industrial_mod = importlib.util.module_from_spec(_li_spec)
_li_spec.loader.exec_module(_light_industrial_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/light_industrial'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/light_industrial'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Manufacturing sub-service names — must match both the light_industrial.py
# variable suffixes and the CIMS sub-service names in the fixed data.
MANUFACTURING_NODES = [
    'Food Tobacco and Beverage',
    'Rubber and Plastics',
    'Leather and Textiles',
    'Wood Products',
    'Furniture Printing and Machinery',
    'Transportation Equipment',
    'Electronics and Other',
]

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Light Industrial fixed CSVs and return combined DataFrame."""
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


def _build_total_rows(light_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request row for the Light Industrial sector.
    Value = total Light Industrial $M 2017 GDP from light_industrial.
    """
    df = light_ind.filter(pl.col('Variable') == 'Light Industrial')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Light Industrial').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Light Industrial')).alias('Target'),
        pl.col('Source'),
        pl.lit('M$ GDP').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Light Industrial sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Light Industrial')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Light Industrial')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Light Industrial').alias('Sector'),
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


def _build_mfg_split_rows(light_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Manufacturing service level.
    One block per sub-service; value = share (%) from light_industrial.
    Branch: CIMS.CAN.{region}.Light Industrial.Manufacturing
    Target: CIMS.CAN.{region}.Light Industrial.Manufacturing.{sub-service}
    """
    parts = []
    for node in MANUFACTURING_NODES:
        df = light_ind.filter(
            pl.col('Variable') == f'Light Industrial.Manufacturing.{node}'
        )
        if df.is_empty():
            continue
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Light Industrial.Manufacturing')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Light Industrial').alias('Sector'),
            pl.lit('Manufacturing').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Light Industrial.Manufacturing.{node}')).alias('Target'),
            pl.col('Source'),
            pl.lit('%').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Light Industrial model inputs and write one CSV per region."""
    print('=' * 60)
    print('LIGHT INDUSTRIAL MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading light industrial activity data...')
    light_ind = _light_industrial_mod.main()

    print('Building total activity rows...')
    total_rows = _build_total_rows(light_ind)
    print(f'  Rows: {len(total_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building manufacturing split rows...')
    mfg_split_rows = _build_mfg_split_rows(light_ind)
    print(f'  Rows: {len(mfg_split_rows):,}')

    print('Combining...')

    # Sector-level branch: ends with '.Light Industrial' (no further sub-path)
    _sector_branch = pl.col('Branch').str.ends_with('.Light Industrial')

    # Manufacturing service branch: ends with '.Light Industrial.Manufacturing' exactly
    _mfg_branch = pl.col('Branch').str.ends_with('.Light Industrial.Manufacturing')

    # Keep only competition from fixed data at sector level;
    # service_provide rows remain null (fixed data); activity data inserted as service_request.
    fixed_sector_competition = fixed.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # Keep competition from fixed at Manufacturing level;
    # service_request splits come from the pipeline as mfg_split_rows.
    fixed_mfg_header = fixed.filter(
        _mfg_branch & pl.col('Parameter').fill_null('').is_in(['competition'])
    )

    # Everything else: all Manufacturing sub-services, Water Heating,
    # Space Heating, Lighting, Ventilation AC, etc.
    fixed_rest = fixed.filter(
        ~(_sector_branch & pl.col('Parameter').fill_null('').is_in(['competition']))
        & ~_mfg_branch
    )

    regions = total_rows['Region'].unique().sort().to_list()

    output = (
        pl.concat(
            [total_rows, fixed_sector_competition,
             price_rows, fixed_mfg_header, mfg_split_rows, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'light_industrial_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Light Industrial model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
