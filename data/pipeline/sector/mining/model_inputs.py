"""
Mining Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/mining/mining_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Contains the full Mining hierarchy including the structural 1.0
    service_request at the sector level. The iron/non-iron product
    splits and potash split are NOT in the fixed data; they are built
    here from the activity pipeline and injected as service_request rows.

    Decisions/notes:
    1. Changed "sized reduced iron/non-iron product" to "iron/non-iron product"
    2. Missing underground in our activity data - keeping JCIMS data. Consider adding to activity script?
    3. Assuming coke for sintering processes
    4. MB missing iron product - pulled from BC (better proxy than detailed QC)
    5. SK missing open pit mining - pulled from BC
    6. NB - using SK as a proxy because only other region with potash
    7. NL - using QC as proxy because detailed mining sector like QC
    8. NS - using BC as a proxy because smaller mining sector
    9. NT/YT - using BC as proxy
    10. NU - using BC as proxy but pulling in more detailed QC iron product nodes that are missing from BC file
    11. AB - using BC as proxy

Total mining production  (service_request at sector level)
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variable: 'Mining'  (tonnes, per province, 2000–2100)
    Inserted above the fixed data; the fixed data retains its own
    structural 1.0 service_request at the sector level.

Sub-product splits at Metal Open Pit level
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variable: 'Mining.Size Reduced Iron Product'     (%)
    Variable: 'Mining.Size Reduced Non Iron Product' (%)
    Target format:
        CIMS.CAN.{region}.Mining.Final Product.Metal Open Pit.Size Reduced Iron Product
        CIMS.CAN.{region}.Mining.Final Product.Metal Open Pit.Size Reduced Non Iron Product

Potash split at Final Product level  (SK and NB only)
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variable: 'Mining.Potash'  (%)
    Target format: CIMS.CAN.{region}.Mining.Final Product.Potash

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request  — sector-level total tonnes (from heavy_industry)
2. multiplier_price — from energy_price_multipliers
3. service_request  — Metal Open Pit → Size Reduced Iron Product (from heavy_industry)
4. service_request  — Metal Open Pit → Size Reduced Non Iron Product (from heavy_industry)
5. service_request  — Final Product → Potash (from heavy_industry, SK and NB only)
6. fixed data       — full fixed hierarchy unchanged (service_provide, competition,
                      structural 1.0 service_request, and all sub-service parameters)
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

_hi_spec = importlib.util.spec_from_file_location(
    'heavy_industry',
    _PIPELINE_ROOT / 'source' / 'activity' / 'heavy_industry.py',
)
_heavy_industry_mod = importlib.util.module_from_spec(_hi_spec)
_hi_spec.loader.exec_module(_heavy_industry_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/mining'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/mining'

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
    """Flatten all Mining fixed CSVs and return combined DataFrame."""
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


def _build_total_rows(heavy_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the region level for Mining.
    Value = total Mining tonnes from heavy_industry.
    Inserted above the fixed data; the fixed data retains its own
    structural 1.0 service_request at the sector level.
    """
    df = heavy_ind.filter(pl.col('Variable') == 'Mining')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Mining').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Mining')).alias('Target'),
        pl.col('Source'),
        pl.lit('tonne').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Mining sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Mining')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Mining')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Mining').alias('Sector'),
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


def _build_iron_product_rows(heavy_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Metal Open Pit service level.
    One row per year per region for each of:
      - Size Reduced Iron Product
      - Size Reduced Non Iron Product
    Branch: CIMS.CAN.{region}.Mining.Final Product.Metal Open Pit
    Target: CIMS.CAN.{region}.Mining.Final Product.Metal Open Pit.{sub-service}
    Value:  %
    """
    parts = []
    for var_suffix, sub_service in [
        ('Size Reduced Iron Product',     'Size Reduced Iron Product'),
        ('Size Reduced Non Iron Product', 'Size Reduced Non Iron Product'),
    ]:
        df = heavy_ind.filter(pl.col('Variable') == f'Mining.{var_suffix}')
        if df.is_empty():
            continue
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Mining.Final Product.Metal Open Pit')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Mining').alias('Sector'),
            pl.lit('Metal Open Pit').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Mining.Final Product.Metal Open Pit.{sub_service}')).alias('Target'),
            pl.col('Source'),
            pl.lit('%').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))

    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


def _build_potash_rows(heavy_ind: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Final Product service level for Potash.
    Only generated for regions present in the activity data with a Potash variable
    (currently SK and NB).
    Branch: CIMS.CAN.{region}.Mining.Final Product
    Target: CIMS.CAN.{region}.Mining.Final Product.Potash
    Value:  %
    """
    df = heavy_ind.filter(pl.col('Variable') == 'Mining.Potash')
    if df.is_empty():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})

    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Mining.Final Product')).alias('Branch'),
        pl.lit('Service').alias('Type'),
        pl.col('Region'),
        pl.lit('Mining').alias('Sector'),
        pl.lit('Final Product').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Mining.Final Product.Potash')).alias('Target'),
        pl.col('Source'),
        pl.lit('%').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Mining model inputs and write one CSV per region."""
    print('=' * 60)
    print('MINING MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading heavy industry activity data...')
    heavy_ind = _heavy_industry_mod.main()

    print('Building service request rows (sector total)...')
    total_rows = _build_total_rows(heavy_ind)
    print(f'  Rows: {len(total_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building iron/non-iron product split rows...')
    iron_rows = _build_iron_product_rows(heavy_ind)
    print(f'  Rows: {len(iron_rows):,}')

    print('Building potash split rows...')
    potash_rows = _build_potash_rows(heavy_ind)
    print(f'  Rows: {len(potash_rows):,}')

    print('Combining...')

    regions = total_rows['Region'].unique().sort().to_list()

    output = (
        pl.concat(
            [
                total_rows,
                price_rows,
                iron_rows,
                potash_rows,
                fixed,
            ],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'mining_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Mining model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
