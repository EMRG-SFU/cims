"""
Exogenous Demand — Electricity Model Inputs

Emits two sets of rows per region:

1. Top-level electricity activity  (service_request at Region level)
   Source: processed_data/activity/electricity.csv
   Variable: Electricity.Utility Generation  (MWh)
   Branch CIMS.CAN.{region} → Target CIMS.CAN.{region}.Electricity

2. Zero-value service_request rows  (Electricity sector → Utility Generation)
   Branch: CIMS.CAN.{region}.Electricity
   Type: Sector
   Parameter: service_request
   Target: CIMS.CAN.{region}.Electricity.Utility Generation
   Unit: MWh, Value: 0  (all years DATA_START–PROJECTION_END)

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import sys
from pathlib import Path

import polars as pl

_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END

ACTIVITY_PATH = BASE_PATH / 'processed_data/activity/electricity.csv'
OUTPUT_DIR    = BASE_PATH / 'model_inputs/model/exogenous demand'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

REGIONS = [
    'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT',
]


def _build_activity_rows(activity: pl.DataFrame, region: str) -> pl.DataFrame:
    """Region-level service_request rows from total utility generation MWh."""
    data = (
        activity
        .filter(
            (pl.col('Region') == region) &
            (pl.col('Variable') == 'Electricity.Utility Generation')
        )
        .sort('Year')
    )
    if len(data) == 0:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return data.select([
        pl.lit(f'CIMS.CAN.{region}').alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Electricity').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit(f'CIMS.CAN.{region}.Electricity').alias('Target'),
        pl.col('Source'),
        pl.col('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_zero_rows(region: str) -> pl.DataFrame:
    """Zero-value service_request rows from Electricity sector to Utility Generation."""
    years = list(range(DATA_START, PROJECTION_END + 1))
    n = len(years)
    return pl.DataFrame({
        'Branch':      [f'CIMS.CAN.{region}.Electricity'] * n,
        'Type':        ['Sector'] * n,
        'Region':      [region] * n,
        'Sector':      ['Electricity'] * n,
        'Service':     [''] * n,
        'Technology':  [''] * n,
        'Parameter':   ['service_request'] * n,
        'Context':     [''] * n,
        'Sub_Context': [''] * n,
        'Target':      [f'CIMS.CAN.{region}.Electricity.Utility Generation'] * n,
        'Source':      [''] * n,
        'Unit':        ['MWh'] * n,
        'Year':        [str(y) for y in years],
        'Value':       ['0'] * n,
    })


def main() -> dict[str, pl.DataFrame]:
    """Assemble exogenous demand electricity model inputs and write one CSV per region."""
    print('=' * 60)
    print('EXOGENOUS DEMAND — ELECTRICITY MODEL INPUTS')
    print('=' * 60)

    print('\nLoading electricity activity data...')
    activity = pl.read_csv(ACTIVITY_PATH, infer_schema_length=0)
    act_regions = sorted(activity['Region'].unique().to_list())
    print(f'  Rows: {len(activity):,}  regions: {act_regions}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region in REGIONS:
        if region not in act_regions:
            print(f'\n  Skipping {region} — no activity data')
            continue

        act_rows  = _build_activity_rows(activity, region)
        zero_rows = _build_zero_rows(region)

        output = pl.concat([act_rows, zero_rows], how='diagonal_relaxed').select(OUTPUT_COLS)

        out_path = OUTPUT_DIR / f'exogenous demand_{region}.csv'
        output.write_csv(str(out_path))
        print(f'  Wrote {len(output):,} rows -> {out_path.name}')
        results[region] = output

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Regions complete: {len(results)}/{len(REGIONS)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
