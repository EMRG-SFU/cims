"""
Extract Industrial Minerals model input data and save to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/industrial_minerals/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.
    Encodes the full hierarchy:
      Industrial Minerals (Sector)
        ├── Industrial Minerals.Products  (Service, Fixed Ratio competition)
        │     ├── Industrial Minerals.Products.Cement
        │     └── Industrial Minerals.Products.Lime
        ├── Industrial Minerals.HVAC      (Service, fixed structural)
        ├── Industrial Minerals.Lighting  (Service, fixed structural)
        └── Industrial Minerals.Machine Drive (Service, fixed structural)
    Note: service_request rows for Products sub-services are excluded from
    the fixed data and replaced with dynamic values from heavy_industry.py.

Activity demand  (service_provide levels)
    pipeline/source/activity/heavy_industry.py  (called directly via main())
    Variables used:
      'Industrial Minerals'        → total tonnes (region-level service_request)
      'Industrial Minerals.Cement' → fraction of total (%)
      'Industrial Minerals.Lime'   → fraction of total (%)

Energy price multipliers  (price_mult rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())

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
_PIPELINE_ROOT = Path(__file__).parent.parent.parent   # .../data/pipeline
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
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/industrial_minerals'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/industrial_minerals'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    """
    Flatten all Industrial Minerals fixed CSVs and return as a combined DataFrame.

    Wide year columns (2000, 2005, …, 2050) are expanded to annual rows
    covering 2000–2100, with a Comments column dropped.
    market_share_total rows are reduced to a single year-2000 row.

    Note: service_request rows on the Products branch are excluded from the
    fixed data and replaced with dynamic values from heavy_industry.py.
    """
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
    return pl.concat(frames, how='diagonal_relaxed')


def _build_emission_rows(
    activity: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build service_request rows from the heavy_industry output.

    Returns (total, shares) where:
      total  = Industrial Minerals tonnes row (region-level service_request)
      shares = Industrial Minerals.Cement and .Lime rows
               (service_request on the Products branch)
    """
    _CEMENT = 'Industrial Minerals.Cement'
    _LIME   = 'Industrial Minerals.Lime'

    total = (
        activity
        .filter(pl.col('Variable') == 'Industrial Minerals')
        .select([
            ('CIMS.CAN.' + pl.col('Region')).alias('Branch'),
            pl.lit('Region').alias('Type'),
            pl.col('Region'),
            pl.lit('Industrial Minerals').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            ('CIMS.CAN.' + pl.col('Region') + pl.lit('.Industrial Minerals')).alias('Target'),
            pl.col('Source'),
            pl.lit('tonnes').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ])
    )

    def _product_rows(variable: str, target_suffix: str) -> pl.DataFrame:
        return (
            activity
            .filter(pl.col('Variable') == variable)
            .select([
                ('CIMS.CAN.' + pl.col('Region') + '.Industrial Minerals.Products').alias('Branch'),
                pl.lit('Service').alias('Type'),
                pl.col('Region'),
                pl.lit('Industrial Minerals').alias('Sector'),
                pl.lit('Products').alias('Service'),
                pl.lit('').alias('Technology'),
                pl.lit('service_request').alias('Parameter'),
                pl.lit('').alias('Context'),
                pl.lit('').alias('Sub_Context'),
                ('CIMS.CAN.' + pl.col('Region') + pl.lit('.') + pl.lit(target_suffix)).alias('Target'),
                pl.col('Source'),
                pl.lit('%').alias('Unit'),
                pl.col('Year').cast(pl.String).alias('Year'),
                pl.col('Value').cast(pl.String).alias('Value'),
            ])
        )

    shares = pl.concat([
        _product_rows(_CEMENT, 'Industrial Minerals.Products.Cement'),
        _product_rows(_LIME,   'Industrial Minerals.Products.Lime'),
    ])

    return total, shares


def _build_price_mult_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """
    Build price_mult rows from the energy price multipliers output.

    All Industrial Minerals energies flow through directly; the energy name
    is used as the Target so no manual fuel mapping is required.
    """
    return (
        multipliers
        .filter(pl.col('Sector') == 'Industrial Minerals')
        .select([
            ('CIMS.CAN.' + pl.col('Region') + '.Industrial Minerals').alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Industrial Minerals').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('multiplier_price').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.when(pl.col('Energy').is_in([
                'Electricity', 'Biodiesel', 'Renewable Diesel',
                'Ethanol', 'Renewable Gasoline', 'Hydrogen',
            ]))
            .then(pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.') + pl.col('Energy'))
            .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('Energy'))
            .alias('Target'),
            pl.col('Source').alias('Source'),
            pl.lit('').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Multiplier').cast(pl.String).alias('Value'),
        ])
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble industrial minerals model inputs and write one CSV per region."""
    print('=' * 60)
    print('INDUSTRIAL MINERALS MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed_data()
    print(f'  Rows: {len(fixed):,}')

    print('Building emission rows...')
    activity = (
        _heavy_industry_mod.main()
        .filter(pl.col('Variable').str.starts_with('Industrial Minerals'))
    )
    total_emissions, share_emissions = _build_emission_rows(activity)
    print(f'  Rows: {len(total_emissions) + len(share_emissions):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_mult_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    fixed_str = fixed.cast(pl.String)
    _im_branch           = pl.col('Branch').str.ends_with('.Industrial Minerals')
    _products_branch     = pl.col('Branch').str.ends_with('.Industrial Minerals.Products')
    _hvac_branch         = pl.col('Branch').str.ends_with('.Industrial Minerals.HVAC')
    _lighting_branch     = pl.col('Branch').str.ends_with('.Industrial Minerals.Lighting')
    _machine_drive_branch = pl.col('Branch').str.ends_with('.Industrial Minerals.Machine Drive')
    _header_params       = pl.col('Parameter').is_in(['service_provide', 'competition'])

    _sub_branches = _products_branch | _hvac_branch | _lighting_branch | _machine_drive_branch

    fixed_im_header           = fixed_str.filter(_im_branch & _header_params)
    fixed_im_tail             = fixed_str.filter(_im_branch & ~_header_params)
    fixed_products_header     = fixed_str.filter(_products_branch & _header_params)
    fixed_products_tail       = fixed_str.filter(_products_branch & ~_header_params)
    fixed_hvac_header         = fixed_str.filter(_hvac_branch & _header_params)
    fixed_hvac_tail           = fixed_str.filter(_hvac_branch & ~_header_params)
    fixed_lighting_header     = fixed_str.filter(_lighting_branch & _header_params)
    fixed_lighting_tail       = fixed_str.filter(_lighting_branch & ~_header_params)
    fixed_machine_drive_header = fixed_str.filter(_machine_drive_branch & _header_params)
    fixed_machine_drive_tail   = fixed_str.filter(_machine_drive_branch & ~_header_params)
    fixed_rest                = fixed_str.filter(~_im_branch & ~_sub_branches)

    output = (
        pl.concat(
            [total_emissions, fixed_im_header, price_rows,
             fixed_im_tail, fixed_products_header, share_emissions,
             fixed_products_tail, fixed_hvac_header, fixed_hvac_tail,
             fixed_lighting_header, fixed_lighting_tail,
             fixed_machine_drive_header, fixed_machine_drive_tail,
             fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'industrial_minerals_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Industrial Minerals model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
