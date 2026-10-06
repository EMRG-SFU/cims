"""
Extract Transportation Freight calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total from crosswalk; calibration_emissions_by_type from nir_to_cims)
    nir_crosswalk_tables_cims.py  → total tCO2e per transportation freight CIMS branch
                                    5-year intervals (2000–2020)
    nir_to_cims.py                → per-gas kt per transportation freight CIMS branch,
                                    annual resolution (2000–latest NIR year);
                                    summed to tCO2e using AR5 GWP100 factors

Energy demand  (calibration_quantity_requested)
    cer_resd_demand.py            → energy demand in PJ by fuel and CIMS node

Technology market shares  (calibration_market_share_total)
    transportation_freight.py     → CEUD-derived market shares (2000–last CEUD year):
                                      Light Medium service: fuel-based tech shares
                                      Heavy service: Trucks vs Rail shares

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

from pathlib import Path

import polars as pl
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────


import CIMS.data_processing.source.eccc.nir.nir_crosswalk_tables_cims as _crosswalk_mod
import CIMS.data_processing.source.eccc.nir.nir_to_cims as _nir_mod
import CIMS.data_processing.source.cer.cer_resd_demand as _cer_mod
import CIMS.data_processing.source.nrcan.ceud.transportation_freight.transportation_freight as _tf_mod
from CIMS.data_processing.utils.controls_conversions import BASE_PATH, load_sector_regions, filter_excluded_branches
from CIMS.data_processing.utils.calibration_helpers import (
    OUTPUT_COLS,
    branch_meta,
    build_cer_energy,
    build_crosswalk_emissions,
    build_nir_emissions,
    empty_df,
)
from CIMS.data_processing.utils.output_builder import write_per_region_csvs

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/transportation_freight'

SECTOR_NAME = 'Transportation Freight'

# Pipeline fuel category → Light Medium technology name, for provinces where
# the source pipeline can't compute the Low/Medium/High Efficiency vintage
# split (no Table 35/36 intensity data) and falls back to an unsplit fuel
# category. Ethanol/Biodiesel are folded into Gasoline/Diesel upstream
# (they're blended fuel volumes, not separate vehicle technologies) and
# never appear here.
LM_CAT_TO_TECH: dict[str, str] = {
    'Diesel':   'Diesel_Low Efficiency',
    'Gasoline': 'Gasoline_Low Efficiency',
    'Propane':  'Propane',
}

# (pipeline variable, service_name, branch suffix after sector, cat_to_tech, year_max)
# year_max=None → all historical years
_TECH_SHARE_SERVICES: list[tuple[str, str, str, dict[str, str] | None, int | None]] = [
    ('Light Medium',       'Light Medium', '.Freight.Land.Light Medium', LM_CAT_TO_TECH, None),
    ('Freight.Land.Heavy', 'Heavy',        '.Freight.Land.Heavy',        None,           None),
]


# ── helpers ───────────────────────────────────────────────────────────────────

# Bottom-up roll-up chain: each parent branch suffix (after "CIMS.CAN.{region}")
# gets a calibration_quantity_requested row equal to the sum of its children's
# values (per Target fuel/Year), plus whatever value is already sitting directly
# on the parent branch itself (e.g. an un-attributable CER residual, such as
# Lubricants feedstock, which has no technology breakdown to drill into).
# Order matters: each level must be computed before the level above it, since
# a parent's own children can include an already-rolled-up lower-level parent
# (e.g. Freight.Land needs Freight.Land.Heavy to already be summed).
_ROLLUP_LEVELS: list[tuple[str, list[str]]] = [
    ('.Transportation Freight.Freight.Land.Heavy', [
        '.Transportation Freight.Freight.Land.Heavy.Trucks',
        '.Transportation Freight.Freight.Land.Heavy.Rail',
    ]),
    ('.Transportation Freight.Freight.Land', [
        '.Transportation Freight.Freight.Land.Light Medium',
        '.Transportation Freight.Freight.Land.Heavy',
    ]),
    ('.Transportation Freight.Freight', [
        '.Transportation Freight.Freight.Land',
        '.Transportation Freight.Freight.Marine',
        '.Transportation Freight.Freight.Air',
    ]),
    ('.Transportation Freight', [
        '.Transportation Freight.Freight',
        '.Transportation Freight.Off Road',
    ]),
]


def _rollup_hierarchy(rows: pl.DataFrame) -> pl.DataFrame:
    """Roll calibration_quantity_requested up from leaf mode nodes to their
    parent Freight/Transportation Freight branches, per _ROLLUP_LEVELS."""
    if rows.is_empty():
        return rows

    df = rows.to_pandas()
    df['Value_f'] = df['Value'].astype(float)

    for parent_suffix, child_suffixes in _ROLLUP_LEVELS:
        child_mask = df['Branch'].str.endswith(tuple(child_suffixes))
        children = df[child_mask]
        if children.empty:
            continue

        summed = children.groupby(['Region', 'Target', 'Year'], as_index=False)['Value_f'].sum()
        parent_branch = 'CIMS.CAN.' + summed['Region'] + parent_suffix
        summed['Branch'] = parent_branch

        existing_mask = df['Branch'] == ('CIMS.CAN.' + df['Region'] + parent_suffix)
        existing = df[existing_mask][['Region', 'Target', 'Year', 'Value_f']]
        if not existing.empty:
            summed = summed.merge(
                existing, on=['Region', 'Target', 'Year'],
                how='outer', suffixes=('', '_existing'),
            )
            summed['Value_f'] = summed['Value_f'].fillna(0) + summed['Value_f_existing'].fillna(0)
            summed = summed.drop(columns=['Value_f_existing'])
            summed['Branch'] = 'CIMS.CAN.' + summed['Region'] + parent_suffix

        df = df[~existing_mask]

        meta_cols = summed['Branch'].apply(branch_meta).apply(pd.Series).drop(columns=['Region'])
        summed = pd.concat([summed.reset_index(drop=True), meta_cols.reset_index(drop=True)], axis=1)
        summed['Technology']  = ''
        summed['Parameter']   = 'calibration_quantity_requested'
        summed['Context']     = ''
        summed['Sub_Context'] = ''
        summed['Source']      = 'CER (rolled up)'
        summed['Unit']        = 'GJ'
        summed['Value']       = summed['Value_f'].astype(str)

        df = pd.concat([df, summed[df.columns]], ignore_index=True)

    df = df.drop(columns=['Value_f'])
    return pl.DataFrame(df, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── technology market share builder ───────────────────────────────────────────

def _build_tf_tech_shares(
    tf: pl.DataFrame,
    variable: str,
    service_name: str,
    branch_suffix: str,
    cat_to_tech: dict[str, str] | None = None,
    year_max: int | None = None,
) -> pl.DataFrame:
    """Extract calibration_market_share_total rows for one transportation freight service."""
    mask = (pl.col('variable') == variable) & (pl.col('parameter') == 'market_share_total')
    if year_max is not None:
        mask = mask & (pl.col('year') <= year_max)
    data = tf.filter(mask)
    if data.is_empty():
        return empty_df()

    rows: list[dict] = []
    for r in data.iter_rows(named=True):
        region   = r['province']
        category = r['category']
        tech     = cat_to_tech.get(category, category) if cat_to_tech else category
        branch   = f'CIMS.CAN.{region}.Transportation Freight{branch_suffix}'
        rows.append({
            'Branch':      branch,
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Freight',
            'Service':     service_name,
            'Technology':  tech,
            'Parameter':   'calibration_market_share_total',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      'CEUD',
            'Unit':        str(r['unit'] or '%'),
            'Year':        str(r['year']),
            'Value':       str(r['value']),
        })
    if not rows:
        return empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble transportation freight calibration data and write one CSV per region."""
    print('=' * 60)
    print('TRANSPORTATION FREIGHT CALIBRATION')
    print('=' * 60)

    print('\nRunning NIR crosswalk (nir_crosswalk_tables_cims)...')
    crosswalk_df = pl.from_pandas(_crosswalk_mod.main())

    print('\nRunning NIR to CIMS (nir_to_cims)...')
    nir_df = _nir_mod.main()

    print('\nRunning CER demand (cer_resd_demand)...')
    cer_df = _cer_mod.main()

    print('\nRunning transportation freight pipeline (CEUD)...')
    tf_results = _tf_mod.main(export_csv=False)
    tf = (
        pl.concat(list(tf_results.values()), how='diagonal_relaxed')
        .filter(pl.col('year') <= _tf_mod.LAST_HIST_YEAR)
    )

    print('\nBuilding CER energy demand rows...')
    cer_rows = build_cer_energy(cer_df, SECTOR_NAME)
    print(f'  Leaf rows: {len(cer_rows):,}')
    cer_rows = _rollup_hierarchy(cer_rows)
    print(f'  Rows after rolling up to Freight/Transportation Freight: {len(cer_rows):,}')

    print('Building crosswalk emission rows...')
    crosswalk_rows = build_crosswalk_emissions(crosswalk_df, SECTOR_NAME)
    print(f'  Rows: {len(crosswalk_rows):,}')

    print('Building NIR annual emission rows (tCO2e via AR5 GWP100)...')
    nir_rows = build_nir_emissions(nir_df, SECTOR_NAME)
    print(f'  Rows: {len(nir_rows):,}')

    print('Building technology market share rows...')
    tech_frames: list[pl.DataFrame] = []
    for variable, service_name, branch_suffix, cat_to_tech, year_max in _TECH_SHARE_SERVICES:
        frame = _build_tf_tech_shares(tf, variable, service_name, branch_suffix, cat_to_tech, year_max)
        tech_frames.append(frame)
        print(f'  {service_name}: {len(frame):,} rows')
    tech_rows = pl.concat(tech_frames, how='diagonal_relaxed') if tech_frames else empty_df()
    print(f'  Tech share total: {len(tech_rows):,} rows')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows, tech_rows], how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    print('Filtering to regions with Transportation Freight (see sector_region_map.csv)...')
    allowed_regions = load_sector_regions().get(SECTOR_NAME)
    if allowed_regions:
        before_count = len(output)
        output = output.filter(pl.col('Region').is_in(list(allowed_regions)))
        dropped_count = before_count - len(output)
        if dropped_count:
            print(f'  Dropped {dropped_count:,} rows for regions without Transportation Freight')

    output = filter_excluded_branches(output)

    written_regions = write_per_region_csvs(
        output, OUTPUT_DIR, 'transportation_freight', skip_if_all_zero=True, report_written=True
    )

    print(f'\n Transportation Freight calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(written_regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
