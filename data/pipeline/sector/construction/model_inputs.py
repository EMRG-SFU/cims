"""
Extract Construction model input data and save to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/construction/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.
    Includes: service_provide, competition, technology, market_share_total,
    lifetime, and the constant service_request (value=1) routing the
    Construction sector to its Transport sub-service.

Activity demand  (service_provide levels)
    pipeline/source/activity/emissions_drivers.py  (called directly via main())
    Variables used:
      'Construction'   → total tCO2e (region-level service_request)

Energy price multipliers  (price_mult rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Energies applied: Diesel, Electricity (matching Transport technologies).

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
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/construction'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/construction'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    """
    Flatten all Construction fixed CSVs and return as a combined DataFrame.

    Wide year columns (2000, 2005, …, 2050) are expanded to annual rows
    covering 2000–2100, with a Comments column dropped.
    market_share_total rows are reduced to a single year-2000 row.

    The fixed data encodes the full Construction→Transport→{Diesel,Electric}
    structure, including the constant service_request of 1 from the
    Construction sector node to its Transport sub-service.
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

    Construction has a single top-level tCO2e demand. The sector→Transport
    passthrough (service_request value=1) is a fixed structural parameter
    already present in the fixed data, so it is not produced here.

    Returns a DataFrame with one row per (Region, Year) representing the
    region-level service_request pointing to the Construction sector node.
    """
    return (
        emissions
        .filter(pl.col('Variable') == 'Construction')
        .select([
            ('CIMS.CAN.' + pl.col('Region')).alias('Branch'),
            pl.lit('Region').alias('Type'),
            pl.col('Region'),
            pl.lit('Construction').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            ('CIMS.CAN.' + pl.col('Region') + pl.lit('.Construction')).alias('Target'),
            pl.col('Source'),
            pl.lit('tCO2e').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ])
    )


def _build_price_mult_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """
    Build price_mult rows from the energy price multipliers output.

    All Construction energies flow through directly; the energy name is used
    as the Target so no manual fuel mapping is required.
    """
    return (
        multipliers
        .filter(pl.col('Sector') == 'Construction')
        .select([
            ('CIMS.CAN.' + pl.col('Region') + '.Construction').alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Construction').alias('Sector'),
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
    """Assemble construction model inputs and write one CSV per region."""
    print('=' * 60)
    print('CONSTRUCTION MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed_data()
    print(f'  Rows: {len(fixed):,}')

    print('Building emission rows...')
    emissions = (
        _emissions_mod.main()
        .filter(pl.col('Variable').str.starts_with('Construction'))
    )
    total_emissions = _build_emission_rows(emissions)
    print(f'  Rows: {len(total_emissions):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_mult_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    fixed_str = fixed.cast(pl.String)
    _con_branch       = pl.col('Branch').str.ends_with('.Construction')
    _transport_branch = pl.col('Branch').str.ends_with('.Construction.Transport')
    _header_params    = pl.col('Parameter').is_in(['service_provide', 'competition'])

    fixed_con_header       = fixed_str.filter(_con_branch & _header_params)
    fixed_con_tail         = fixed_str.filter(_con_branch & ~_header_params)
    fixed_transport_header = fixed_str.filter(_transport_branch & _header_params)
    fixed_transport_tail   = fixed_str.filter(_transport_branch & ~_header_params)
    fixed_rest             = fixed_str.filter(~_con_branch & ~_transport_branch)

    output = (
        pl.concat(
            [total_emissions, fixed_con_header, price_rows,
             fixed_con_tail, fixed_transport_header, fixed_transport_tail, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'construction_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Construction model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
