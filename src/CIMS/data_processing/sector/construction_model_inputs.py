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


import pandas as pd
import polars as pl

# ── path setup ────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_folder

import CIMS.data_processing.source.activity.emissions_drivers as _emissions_mod

import CIMS.data_processing.source.energy_prices.energy_price_multipliers as _energy_price_mod

import CIMS.data_processing.source.cer.cer_resd_demand as _cer_resd_mod

from CIMS.data_processing.utils.controls_conversions import BASE_PATH
from CIMS.data_processing.utils.output_builder import write_per_region_csvs
from CIMS.data_processing.utils.feedstock_demand import build_feedstock_rows_all_regions

# ── configuration ─────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/construction'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/construction'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]


# ── helpers ───────────────────────────────────────────────────────────────────

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
                'Electricity', 'Biodiesel',
                'Ethanol', 'Hydrogen',
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


def _build_feedstock_rows(emissions: pl.DataFrame) -> pl.DataFrame:
    """
    Feedstock service rows for the Construction sector (CER vFsDmd-CIMS.csv,
    via cer_resd_demand.load_feedstock_demand()), tied to the same
    'Construction' tCO2e driver _build_emission_rows() already uses.
    """
    feedstock_demand = _cer_resd_mod.load_feedstock_demand()
    feedstock_demand = feedstock_demand[feedstock_demand['Node'] == '.Construction']
    if feedstock_demand.empty:
        return pl.DataFrame()

    driver = emissions.filter(pl.col('Variable') == 'Construction').with_columns(
        pl.col('Year').cast(pl.Int64), pl.col('Value').cast(pl.Float64)
    )
    scale_by_region = {
        region: pd.Series(sub['Value'].to_list(), index=sub['Year'].to_list())
        for region, sub in driver.to_pandas().groupby('Region')
    }

    return build_feedstock_rows_all_regions(
        sector_name='Construction',
        feedstock_demand=feedstock_demand,
        scale_by_region=scale_by_region,
        scale_unit='tCO2e',
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble construction model inputs and write one CSV per region."""
    print('=' * 60)
    print('CONSTRUCTION MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    # The fixed data encodes the full Construction→Transport→{Diesel,Electric}
    # structure, including the constant service_request of 1 from the
    # Construction sector node to its Transport sub-service.
    fixed = read_fixed_data_folder(FIXED_INPUT_DIR)
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

    print('Building feedstock rows...')
    feedstock_rows = _build_feedstock_rows(emissions)
    print(f'  Rows: {len(feedstock_rows):,}')

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
            [total_emissions, fixed_con_header, price_rows, feedstock_rows,
             fixed_con_tail, fixed_transport_header, fixed_transport_tail, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    regions = write_per_region_csvs(output, OUTPUT_DIR, 'construction', collapse_years=True)

    print(f'\n✅ Construction model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
