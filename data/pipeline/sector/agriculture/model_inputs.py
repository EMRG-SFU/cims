"""
Extract Agriculture model input data and save to CIMS-formatted CSV.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Agriculture/*.csv
    Flattened from wide (2000–2050 year columns) to long format via
    utils/flatten_fixed_data.

Activity demand  (service_provide levels)
    processed_data/activity/emissions_drivers.csv
    Produced by pipeline/source/activity/emissions_drivers.py.
    Variables used:
      'Agriculture'                                          → total tCO2e
      'Agriculture.Heat'                                     → fraction of total
      'Agriculture.Process.Soils'                            → fraction of total
      'Agriculture.Process.Enteric Fermentation and Manure Management'
                                                             → fraction of total

Energy price multipliers  (price_mult rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Produced by pipeline/source/energy_prices/energy_price_multipliers.py.
    Energies applied: Diesel, Natural Gas, Electricity, Hydrogen.

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
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Agriculture'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/agriculture'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]



# ── helpers ───────────────────────────────────────────────────────────────────

def _read_flattened_fixed_data() -> pl.DataFrame:
    """
    Flatten all Agriculture fixed CSVs and return as a combined DataFrame.

    Wide year columns (2000, 2005, …, 2050) are expanded to annual rows
    covering 2000–2100, with a Comments column dropped.
    market_share_total rows are reduced to a single year-2000 row.
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


def _build_emission_rows(emissions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build service_provide emission rows from the emissions_drivers output.

    The four agriculture variables in emissions_drivers are:
      'Agriculture'         tCO2e  – total agriculture emissions (absolute)
      'Agriculture.Heat'    %(fraction 0–1)
      'Agriculture.Process.Soils'         %
      'Agriculture.Process.Enteric ...'   %

    Returns (total, shares) where:
      total  = Agriculture tCO2e row
      shares = Agriculture.Process, .Soils, and .Enteric… rows
    """
    _SOILS   = 'Agriculture.Process.Soils'
    _ENTERIC = 'Agriculture.Process.Enteric Fermentation and Manure Management'

    # Preserve Source per (Region, Year) before pivoting drops it
    source_map = (
        emissions
        .filter(pl.col('Variable') == 'Agriculture')
        .select(['Region', 'Year', 'Source'])
    )

    wide = (
        emissions
        .filter(pl.col('Variable').is_in(['Agriculture', 'Agriculture.Heat', _SOILS, _ENTERIC]))
        .select(['Region', 'Year', 'Variable', 'Value'])
        .pivot(index=['Region', 'Year'], on='Variable', values='Value')
        .with_columns([
            (pl.col('Agriculture') * pl.col(_SOILS).fill_null(0)).alias('soils_tco2e'),
            (pl.col('Agriculture') * pl.col(_ENTERIC).fill_null(0)).alias('enteric_tco2e'),
            (pl.col(_SOILS).fill_null(0) + pl.col(_ENTERIC).fill_null(0)).alias('process_pct'),
        ])
        .with_columns(
            (pl.col('soils_tco2e') + pl.col('enteric_tco2e')).alias('process_tco2e'),
        )
        .join(source_map, on=['Region', 'Year'], how='left')
    )

    def _service_rows(
        branch_suffix: str,
        target_suffix: str,
        value_col: str,
    ) -> pl.DataFrame:
        return wide.select([
            ('CIMS.CAN.' + pl.col('Region') + '.' + pl.lit(branch_suffix)).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Agriculture').alias('Sector'),
            pl.lit('Process').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            ('CIMS.CAN.' + pl.col('Region') + '.' + pl.lit(target_suffix)).alias('Target'),
            pl.col('Source'),
            pl.lit('tCO2e').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col(value_col).cast(pl.String).alias('Value'),
        ])

    total = wide.select([
        ('CIMS.CAN.' + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Agriculture').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        ('CIMS.CAN.' + pl.col('Region') + pl.lit('.Agriculture')).alias('Target'),
        pl.col('Source'),
        pl.lit('tCO2e').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Agriculture').cast(pl.String).alias('Value'),
    ])

    shares = pl.concat([
        _service_rows('Agriculture.Process',
                      'Agriculture.Process.Enteric Fermentation and Manure Management',
                      _ENTERIC),
        _service_rows('Agriculture.Process',
                      'Agriculture.Process.Soils',
                      _SOILS),
        _service_rows('Agriculture.Process',
                      'Agriculture.Heat',
                      'Agriculture.Heat'),
    ])

    return total, shares


def _build_price_mult_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """
    Build price_mult rows from the energy price multipliers output.

    All Agriculture energies flow through directly; the energy name is used
    as the Target so no manual fuel mapping is required.
    """
    return (
        multipliers
        .filter(pl.col('Sector') == 'Agriculture')
        .select([
            ('CIMS.CAN.' + pl.col('Region') + '.Agriculture').alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Agriculture').alias('Sector'),
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
    """Assemble agriculture model inputs and write one CSV per region."""
    print('=' * 60)
    print('AGRICULTURE MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed_data()
    print(f'  Rows: {len(fixed):,}')

    print('Building emission rows...')
    emissions = (
        _emissions_mod.main()
        .filter(pl.col('Variable').str.starts_with('Agriculture'))
    )
    total_emissions, share_emissions = _build_emission_rows(emissions)
    print(f'  Rows: {len(total_emissions) + len(share_emissions):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_mult_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Combining...')
    fixed_str = fixed.cast(pl.String)
    _ag_branch   = pl.col('Branch').str.ends_with('.Agriculture')
    _proc_branch = pl.col('Branch').str.ends_with('.Agriculture.Process')
    _header_params = pl.col('Parameter').is_in(['service_provide', 'competition'])

    fixed_ag_header   = fixed_str.filter(_ag_branch & _header_params)
    fixed_ag_tail     = fixed_str.filter(_ag_branch & ~_header_params)
    fixed_proc_header = fixed_str.filter(_proc_branch & _header_params)
    fixed_rest        = fixed_str.filter(~_ag_branch & ~_proc_branch)

    output = (
        pl.concat(
            [total_emissions, fixed_ag_header, price_rows,
             fixed_ag_tail, fixed_proc_header, share_emissions, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'agriculture_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Agriculture model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
