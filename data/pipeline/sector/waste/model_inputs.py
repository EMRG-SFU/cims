"""
Extract Waste model input data and save to JCIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/waste/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.
    Encodes the full hierarchy:
      Waste (Sector)
        └── Waste.Sites (Service, Fixed Ratio competition)
              ├── Waste.Sites.Large Sites  (Service, Tech Compete)
              ├── Waste.Sites.Medium Sites (Service, Tech Compete)
              └── Waste.Sites.Small Sites  (Service, Tech Compete)
                    └── Technologies: No Control, Flaring,
                                      Electricity Generation
    The constant Waste → Waste.Sites passthrough (value=1) and the fixed
    fractional splits from Sites to Large/Medium/Small Sites are all
    structural parameters already present in the fixed data CSVs.

Activity demand  (service_provide levels)
    pipeline/source/activity/emissions_drivers.py  (called directly via main())
    Variables used:
      'Waste'   → total tCO2e (region-level service_request)

    Unlike Agriculture, Waste has no fractional sub-service splits sourced
    from emissions_drivers. The Large/Medium/Small Sites fractions are
    fixed constants already present in the fixed data CSVs.

Energy price multipliers
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Waste is classified as Industrial in sector_map.csv and receives
    multipliers accordingly.

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

_ed_spec = importlib.util.spec_from_file_location(
    'emissions_drivers',
    _PIPELINE_ROOT / 'source' / 'activity' / 'emissions_drivers.py',
)
_emissions_mod = importlib.util.module_from_spec(_ed_spec)
_ed_spec.loader.exec_module(_emissions_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/waste'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/waste'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    """
    Flatten all Waste fixed CSVs and return as a combined DataFrame.

    Wide year columns (2000, 2005, …, 2050) are expanded to annual rows
    covering 2000–2100, with a Comments column dropped.
    market_share_total rows are reduced to a single year-2000 row.

    The fixed data encodes the full Waste → Sites → {Large, Medium, Small}
    hierarchy, including the constant sector→Sites passthrough (value=1)
    and the fixed fractional splits from Sites to each sub-service.
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
    return pl.concat(frames)


def _build_emission_rows(emissions: pl.DataFrame) -> pl.DataFrame:
    """
    Build the region-level service_request row from emissions_drivers.

    Waste has a single top-level tCO2e demand. The sector→Sites passthrough
    (value=1) and the Sites→{Large,Medium,Small} fixed fractional splits are
    structural parameters already in the fixed data, so they are not
    produced here.

    Returns a DataFrame with one row per (Region, Year) representing the
    region-level service_request pointing to the Waste sector node.
    """
    return (
        emissions
        .filter(pl.col('Variable') == 'Waste')
        .select([
            ('JCIMS.CAN.' + pl.col('Region')).alias('Branch'),
            pl.lit('Region').alias('Type'),
            pl.col('Region'),
            pl.lit('Waste').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            ('JCIMS.CAN.' + pl.col('Region') + pl.lit('.Waste')).alias('Target'),
            pl.col('Source'),
            pl.lit('tCO2e').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ])
    )


def _build_price_mult_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """
    Build price_mult rows from the energy price multipliers output.

    All Waste energies flow through directly; the energy name is used
    as the Target so no manual fuel mapping is required.
    """
    return (
        multipliers
        .filter(pl.col('Sector') == 'Waste')
        .select([
            ('JCIMS.CAN.' + pl.col('Region') + '.Waste').alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Waste').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('multiplier_price').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.when(pl.col('Energy').is_in([
                'Electricity', 'Biodiesel', 'Renewable Diesel',
                'Ethanol', 'Renewable Gasoline', 'Hydrogen',
            ]))
            .then(pl.lit('JCIMS.CAN.') + pl.col('Region') + pl.lit('.') + pl.col('Energy'))
            .otherwise(pl.lit('JCIMS.Generic Fuels.') + pl.col('Energy'))
            .alias('Target'),
            pl.col('Source').alias('Source'),
            pl.lit('').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Multiplier').cast(pl.String).alias('Value'),
        ])
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble waste model inputs and write one CSV per region."""
    print('=' * 60)
    print('WASTE MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed_data()
    print(f'  Rows: {len(fixed):,}')

    print('Building emission rows...')
    emissions = (
        _emissions_mod.main()
        .filter(pl.col('Variable').str.starts_with('Waste'))
    )
    total_emissions = _build_emission_rows(emissions)
    print(f'  Rows: {len(total_emissions):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_mult_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    fixed_str = fixed.cast(pl.String)
    _waste_branch = pl.col('Branch').str.ends_with('.Waste')
    _sites_branch = pl.col('Branch').str.ends_with('.Waste.Sites')
    _header_params = pl.col('Parameter').is_in(['service_provide', 'competition'])

    fixed_waste_header = fixed_str.filter(_waste_branch & _header_params)
    fixed_waste_tail   = fixed_str.filter(_waste_branch & ~_header_params)
    fixed_sites_header = fixed_str.filter(_sites_branch & _header_params)
    fixed_sites_tail   = fixed_str.filter(_sites_branch & ~_header_params)
    fixed_rest         = fixed_str.filter(~_waste_branch & ~_sites_branch)

    output = (
        pl.concat(
            [total_emissions, fixed_waste_header, price_rows,
             fixed_waste_tail, fixed_sites_header, fixed_sites_tail, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'waste_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Waste model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
