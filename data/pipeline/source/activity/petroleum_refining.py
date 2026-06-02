"""
=============================================================================
PETROLEUM REFINING ACTIVITY
=============================================================================
Combines two Statistics Canada tables to produce annual crude input to
refineries by province/territory (2000–2024) in cubic metres (m3).

Sources:
  OLD: Table 25-10-0014 (1985-01 to 2016-02)
       GEO = Canada, row = "To refineries, [region]"
       Scalar = thousands  →  VALUE × 1,000 = m3
       Regions: Atlantic provinces (NL+NB combined), Quebec, Ontario,
                Saskatchewan, Alberta, British Columbia

  NEW: Table 25-10-0063 (2016-01 onward)
       GEO = [province], row = "Input to Canadian refineries"
       Units = Cubic metres, scalar = units  →  no conversion needed
       Provinces reported directly: AB, BC (partial), NL, ON, QC (partial), SK
       New Brunswick: never reported directly — always derived

Suppression handling:
  Statistics Canada suppresses province-level cells (STATUS = 'x', VALUE = '')
  where disclosure would identify individual refinery operators.

  Old file (25100014): Data is reported at the Canada level with disposition
    rows per region — no province-level cell suppression occurs. Any null
    values from read_statscan_csv() are filled with 0.0 before aggregation.

  New file (25100063): Province-level cells are suppressed but the Canada
    total is always reported. Rather than filling suppressed cells with 0
    (which creates a spurious NB residual spike), we distribute the
    unallocated residual across suppressed provinces each month using
    reference shares derived from the months they are reported.

  Algorithm (applied month by month):
    1. Reference shares: for each province, compute its average share of
       the Canada total using only months where it IS reported.  NB's
       share is implied as 1 − sum(all other shares).
    2. For each month:
       a. Sum the reported provinces  → reported_sum
       b. residual = Canada − reported_sum
          (this belongs to suppressed provinces + NB combined)
       c. Distribute residual across suppressed provinces (incl. NB)
          proportionally by their reference shares.
       d. Reported provinces keep their exact values; Canada total is
          always preserved exactly (residual error = 0).

Atlantic split (old file only):
  The old file combines NL and NB as "Atlantic provinces". They are split
  using the average NL fraction observed in 2017–2019 from the new file
  (earliest stable period with both provinces individually reported):
      NL fraction = 0.2439,  NB fraction = 0.7561

Output unit: m3/year  (annual sum of monthly volumes)
=============================================================================
"""

import polars as pl
from pathlib import Path

import sys
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_extensions import extend_cagr_periods, compute_cagr, load_cagr_assumptions
from utils.extractors.stats_can import read_statscan_csv
from utils.controls_conversions import load_control_config, BASE_PATH

# Configuration
MAPPINGS_PATH   = BASE_PATH / 'mappings_conversions'
REGION_MAP_FILE = MAPPINGS_PATH / 'region_map.csv'
OUTPUT_DIR      = BASE_PATH / 'processed_data/activity'

DATA_START     = 2000
LAST_DATA_YEAR = load_control_config()["last_data_year"]

ASSUMPTIONS_FILE = BASE_PATH / 'raw_data/assumptions/activity_cagr_projections.csv'
CAGR_START, CAGR_END, CAGR_PERIODS = load_cagr_assumptions('Petroleum Refining', ASSUMPTIONS_FILE)

# Per-region overrides — one explicit annual rate per period.
# Saskatchewan historically has anomalous CAGR due to upgrader capacity swings.
CAGR_OVERRIDES: dict[str, tuple[float, ...]] = {
    "SK": (0.01, 0.005, 0.002),
}

import pandas as _pd   # used only in the suppression-imputation step

OLD_REFINERY_FILE = BASE_PATH / 'raw_data/stats_can/activity/25100014.csv'
NEW_REFINERY_FILE = BASE_PATH / 'raw_data/stats_can/activity/25100063.csv'

_NL_FRAC = 0.2439          # NL share of old-file "Atlantic provinces" total
_NB_FRAC = 1.0 - _NL_FRAC

_OLD_REGION_MAP = {
    'To refineries, Quebec':                'Quebec',
    'To refineries, Ontario':               'Ontario',
    'To refineries, Saskatchewan':          'Saskatchewan',
    'To refineries, Alberta':               'Alberta',
    'To refineries, British Columbia':      'British Columbia',
    'To refineries, Atlantic provinces':    '_Atlantic',
}

# Provinces that appear as GEO rows in the new file
_NEW_PROVINCES = [
    'Alberta', 'British Columbia', 'Newfoundland and Labrador',
    'Ontario', 'Quebec', 'Saskatchewan',
]
# All provinces we want in the output (NB derived, never a direct GEO row)
_ALL_REGIONS = _NEW_PROVINCES + ['New Brunswick']


def main() -> pl.DataFrame:
    """Build petroleum refining DataFrame and write to CSV."""

    # -- R1. Old file: annual totals 2000–2015 ------------------------------------

    old_raw = read_statscan_csv(OLD_REFINERY_FILE)

    old_monthly = (
        old_raw
        .filter(
            (pl.col('GEO') == 'Canada') &
            (pl.col('Supply and disposition').is_in(list(_OLD_REGION_MAP.keys()))) &
            (pl.col('REF_DATE') >= '2000-01') &
            (pl.col('REF_DATE') <= '2015-12')
        )
        .select(['REF_DATE', 'Supply and disposition', 'VALUE'])
        .with_columns(
            (pl.col('VALUE').cast(pl.Float64, strict=False).fill_null(0.0) * 1_000).alias('m3'),
            pl.col('Supply and disposition').replace(_OLD_REGION_MAP).alias('Region'),
            pl.col('REF_DATE').str.slice(0, 4).cast(pl.Int64).alias('Year'),
        )
    )

    old_annual_non_atl = (
        old_monthly
        .filter(pl.col('Region') != '_Atlantic')
        .group_by(['Region', 'Year'])
        .agg(pl.col('m3').sum().alias('Value'))
    )

    old_atl_annual = (
        old_monthly
        .filter(pl.col('Region') == '_Atlantic')
        .group_by('Year')
        .agg(pl.col('m3').sum().alias('atl_m3'))
    )

    old_nl = old_atl_annual.with_columns(
        pl.lit('Newfoundland and Labrador').alias('Region'),
        (pl.col('atl_m3') * _NL_FRAC).alias('Value'),
    ).select(['Region', 'Year', 'Value'])

    old_nb = old_atl_annual.with_columns(
        pl.lit('New Brunswick').alias('Region'),
        (pl.col('atl_m3') * _NB_FRAC).alias('Value'),
    ).select(['Region', 'Year', 'Value'])

    old_combined = (
        pl.concat([old_annual_non_atl, old_nl, old_nb])
        .select(['Region', 'Year', 'Value'])   # enforce column order before concat
        .sort(['Region', 'Year'])
    )


    # -- R2. New file: monthly imputation then annual aggregation -----------------
    #
    #   Suppressed cells are imputed so that Canada totals are exactly preserved.

    new_raw = read_statscan_csv(NEW_REFINERY_FILE)

    new_filt = (
        new_raw
        .filter(
            (pl.col('Supply and disposition') == 'Input to Canadian refineries') &
            (pl.col('Units of measure') == 'Cubic metres') &
            (pl.col('GEO').is_in(['Canada'] + _NEW_PROVINCES)) &
            (pl.col('REF_DATE') >= '2016-01') &
            (pl.col('REF_DATE') <= f'{LAST_DATA_YEAR["stat_can_crude_new"]}-12')
        )
        .select(['REF_DATE', 'GEO', 'VALUE'])
        # Keep null for suppressed cells — we need to distinguish reported vs suppressed
        .with_columns(pl.col('VALUE').cast(pl.Float64, strict=False).alias('m3'))
    )

    new_canada = (
        new_filt.filter(pl.col('GEO') == 'Canada')
        .select(['REF_DATE', pl.col('m3').alias('canada_m3')])
    )
    new_provs = new_filt.filter(pl.col('GEO') != 'Canada')

    # Step R2a: reference shares from clean (reported) months only
    new_with_canada = new_provs.join(new_canada, on='REF_DATE', how='left')
    ref_shares_pl = (
        new_with_canada
        .filter(pl.col('m3').is_not_null())
        .with_columns((pl.col('m3') / pl.col('canada_m3')).alias('share'))
        .group_by('GEO')
        .agg(pl.col('share').mean().alias('ref_share'))
    )
    # NB is never directly reported; its share is the complement of all others
    nb_ref_share = 1.0 - ref_shares_pl['ref_share'].sum()
    ref_shares_pl = pl.concat([
        ref_shares_pl,
        pl.DataFrame({'GEO': ['New Brunswick'], 'ref_share': [nb_ref_share]}),
    ])
    ref = dict(zip(ref_shares_pl['GEO'].to_list(), ref_shares_pl['ref_share'].to_list()))

    # Step R2b: pivot to wide, join Canada total, ensure all province columns exist
    new_wide_pd = (
        new_provs
        .pivot(values='m3', index='REF_DATE', on='GEO', aggregate_function='first')
        .join(new_canada, on='REF_DATE', how='left')
        .sort('REF_DATE')
        .to_pandas()
        .set_index('REF_DATE')
    )
    for p in _NEW_PROVINCES:
        if p not in new_wide_pd.columns:
            new_wide_pd[p] = float('nan')

    # Step R2c: month-by-month imputation
    # Reported provinces keep exact values; suppressed provinces (incl. NB) share
    # the residual (Canada − reported_sum) in proportion to their reference shares.
    imputed_rows = []
    for date, row in new_wide_pd.iterrows():
        canada      = row['canada_m3']
        reported    = {p: row[p] for p in _NEW_PROVINCES if _pd.notna(row[p])}
        suppressed  = [p for p in _ALL_REGIONS if p not in reported]
        residual    = canada - sum(reported.values())
        sup_ref_sum = sum(ref[p] for p in suppressed)
        imputed     = {p: residual * ref[p] / sup_ref_sum for p in suppressed}
        # Merge dicts — reported takes priority over imputed
        all_vals = {**imputed, **reported}
        imputed_rows.append({'REF_DATE': date, **{p: all_vals[p] for p in _ALL_REGIONS}})

    # Step R2d: aggregate to annual
    new_monthly_pd = _pd.DataFrame(imputed_rows)
    new_monthly_pd['Year'] = new_monthly_pd['REF_DATE'].str[:4].astype(int)
    new_annual_pd  = new_monthly_pd.groupby('Year')[_ALL_REGIONS].sum().reset_index()

    # Convert to polars long format with explicit column order matching old_combined
    new_combined = (
        pl.from_pandas(new_annual_pd)
        .unpivot(index='Year', on=_ALL_REGIONS, variable_name='Region', value_name='Value')
        .select(['Region', 'Year', 'Value'])   # match old_combined column order
        .sort(['Region', 'Year'])
    )


    # -- R3. Concatenate, add metadata, filter to 2000–2024 -----------------------

    refinery_annual = (
        pl.concat([old_combined, new_combined])
        .filter((pl.col('Year') >= 2000) & (pl.col('Year') <= LAST_DATA_YEAR["stat_can_crude_new"]))
        .with_columns(
            pl.lit('Petroleum Refining').alias('Variable'),
            pl.lit('m3').alias('Unit'),
        )
        .select(['Region', 'Variable', 'Unit', 'Year', 'Value'])
        .sort(['Region', 'Year'])
    )

    refinery_output = refinery_annual


    # -- R4. Map Stats Can province names to CIMS codes ---------------------------

    region_map = pl.read_csv(REGION_MAP_FILE, columns=["CIMS", "Stats Can"])

    refinery_output = (
        refinery_output
        .join(region_map, left_on="Region", right_on="Stats Can", how="left")
        .with_columns(pl.col("CIMS").fill_null(pl.col("Region")).alias("Region"))
        .drop("CIMS")
        .sort(["Region", "Year"])
    )


    # -- R5. Extend to 2100 via CAGR ----------------------------------------------
    #
    #   Raw CAGR from CAGR_START to CAGR_END (direct point values), dampened
    #   across three periods loaded from the assumptions CSV.
    #   Regions with no data at CAGR_END (e.g. NWT) are not projected forward.

    last_year = refinery_output["Year"].max()

    future_rows_list = []
    for (region, variable, unit), grp in (
        refinery_output
        .to_pandas()
        .groupby(["Region", "Variable", "Unit"])
    ):
        series = grp.set_index("Year")["Value"]

        if CAGR_END not in series.index or series.loc[CAGR_END] <= 0:
            continue

        base_val = float(series.loc[CAGR_END])
        raw_cagr = compute_cagr(series, CAGR_START, CAGR_END)

        ext = extend_cagr_periods(
            base_val=base_val,
            raw_cagr=raw_cagr,
            periods=CAGR_PERIODS,
            override=CAGR_OVERRIDES.get(region),
        )
        for yr, val in ext.items():
            if yr <= last_year:
                continue
            future_rows_list.append({
                "Region":   region,
                "Variable": variable,
                "Unit":     unit,
                "Year":     int(yr),
                "Value":    float(val),
            })

    future_rows = pl.DataFrame(future_rows_list).cast({"Year": pl.Int64})

    refinery_output = (
        pl.concat([refinery_output, future_rows])
        .sort(["Region", "Year"])
        .with_columns(
            pl.when(pl.col("Year") <= LAST_DATA_YEAR["stat_can_crude_new"])
            .then(pl.lit("Stats Can"))
            .otherwise(pl.lit("Assumptions"))
            .alias("Source")
        )
        .select(["Region", "Variable", "Unit", "Source", "Year", "Value"])
    )


    # -- R6. Save -----------------------------------------------------------------

    print(f'\n✅ Petroleum refining complete')
    print(f'   Total rows:          {len(refinery_output):,}')
    print(f'   Regions processed:   {refinery_output["Region"].n_unique()}')
    print(f'   Regions:             {sorted(refinery_output["Region"].unique().to_list())}')
    print(f'   Years covered:       {refinery_output["Year"].min()} – {refinery_output["Year"].max()}')

    refinery_path = OUTPUT_DIR / 'petroleum_refining.csv'
    refinery_path.parent.mkdir(parents=True, exist_ok=True)
    refinery_output.write_csv(refinery_path)
    print(f'   Saved to:            {refinery_path}')

    return refinery_output


if __name__ == '__main__':
    main()
