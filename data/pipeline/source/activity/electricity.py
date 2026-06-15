"""
Electricity Activity and Load Fractions by Region
==================================================

OUTPUT
------
A long CSV with columns:
    Region | Variable | Unit | Source | Year | Value

Variables:
    Electricity.Utility Generation              (MWh, all years 2000-2100)
    Electricity.Utility Generation.Base Load    (% MWh, all years)
    Electricity.Utility Generation.Shoulder Load
    Electricity.Utility Generation.Peak Load

MWh activity comes from electricity_activity_jcims.csv.
Load fractions are derived by nearest-neighbour matching against existing JCIMS
provinces, using Stats Can generation-mix vectors as the similarity metric.
Fractions are held constant across all projection years (consistent with JCIMS).

INPUT FILES
-----------
1.  25100015.csv  -- Statistics Canada Table 25-10-0015-01
    "Electric power generation, monthly", January 2008 - present

2.  electricity_activity_jcims.csv
    a) Existing load fractions (Base/Shoulder/Peak) for reference provinces
    b) Total utility generation MWh for all regions, 2000-2100

METHODOLOGY
-----------
The JCIMS load fractions reflect the *demand* load-duration curve -- NOT the
generation mix. Stats Can generation-by-fuel data is used only as a similarity
metric for nearest-neighbour matching; it is not output directly.

For each missing region (NB, NS, PE, NL, YT, NT, NU), the reference province
(BC, AB, SK, MB, ON, QC) with the most similar Stats Can generation-mix vector
(minimum Euclidean distance) is found, and that province's JCIMS fractions are
assigned. TR is a generation-weighted average of YT, NT, and NU.

JCIMS fractions sum to slightly more than 1.0 (typically 1.07-1.14) because
they express generation-per-unit-of-demand, covering both demand and
transmission losses.
"""

import sys
import math
import polars as pl
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# -- File paths ----------------------------------------------------------------
PATH_SC15  = BASE_PATH / 'raw_data/stats_can/activity/25100015.csv'
PATH_JCIMS = BASE_PATH / 'raw_data/clean_fuel_production/electricity_activity_jcims.csv'
REGION_MAP = BASE_PATH / 'mappings_conversions/region_map.csv'
OUTPUT     = BASE_PATH / 'processed_data/activity/electricity.csv'

# Year columns present in the JCIMS CSV
JCIMS_YEAR_COLS = [str(y) for y in range(2000, 2101)]

# Stats Can generation category → load segment (used for NN matching only)
LOAD_MAP_RECORDS = [
    ('Hydraulic turbine',                      'Base Load'),
    ('Nuclear steam turbine',                  'Base Load'),
    ('Wind power turbine',                     'Base Load'),
    ('Solar',                                  'Base Load'),
    ('Conventional steam turbine',             'Shoulder Load'),
    ('Other types of electricity generation',  'Shoulder Load'),
    ('Combustion turbine',                     'Peak Load'),
    ('Internal combustion turbine',            'Peak Load'),
]
LOAD_SEGMENTS = ['Base Load', 'Shoulder Load', 'Peak Load']

ATL_REGIONS = ['NB', 'NS', 'PE', 'NL']
TER_REGIONS = ['YT', 'NT', 'NU']

# Individual provinces available for nearest-neighbour matching.
# AT is excluded because it is a JCIMS aggregate (NB+NS+PE+NL), not a
# single Stats Can region; it is used separately for the Atlantic provinces.
NN_CANDIDATES = ['BC', 'AB', 'SK', 'MB', 'ON', 'QC']

# JCIMS target strings for the three load segments
JCIMS_FRAC_TARGETS = {
    '.Electricity.Utility Generation.Base Load':     'Base Load',
    '.Electricity.Utility Generation.Shoulder Load': 'Shoulder Load',
    '.Electricity.Utility Generation.Peak Load':     'Peak Load',
}
# Target string for total utility generation activity
JCIMS_ACT_TARGET = '.Electricity.Utility Generation'


# -- JCIMS CSV parsers --------------------------------------------------------

def _read_jcims_service_request(path: Path) -> pl.DataFrame:
    """Read electricity_activity_jcims.csv, returning only service_request rows.

    Selects the 13 metadata columns plus year columns 2000-2100; drops the
    trailing empty column and INDEX column so there are no name conflicts.
    """
    raw = pl.read_csv(path, infer_schema_length=0, null_values=[''])
    keep = ['Branch', 'Type', 'region', 'sector', 'Service', 'technology',
            'Parameter', 'Context', 'Sub_Context', 'Target', 'Source',
            'Comments', 'Unit'] + JCIMS_YEAR_COLS
    available = set(raw.columns)
    return (
        raw
        .select([c for c in keep if c in available])
        .filter(pl.col('Parameter') == 'service_request')
    )


def load_jcims_fractions(path: Path) -> dict[str, dict[str, float]]:
    """Parse existing load fractions from electricity_activity_jcims.csv.

    Returns dict[region] -> dict[load_segment] -> float.
    Fractions are constant across years; the value from column '2000' is used.
    """
    df = _read_jcims_service_request(path).filter(
        (pl.col('Branch') == '.Electricity.Utility Generation') &
        pl.col('Target').is_in(list(JCIMS_FRAC_TARGETS.keys())) &
        pl.col('region').is_not_null()
    )
    fractions: dict[str, dict[str, float]] = {}
    for row in df.to_dicts():
        region = row.get('region')
        seg    = JCIMS_FRAC_TARGETS.get(row.get('Target', ''))
        if not region or not seg:
            continue
        # Fractions are constant across all years; take the first non-null year
        value: float | None = None
        for y in JCIMS_YEAR_COLS:
            raw_val = row.get(y)
            if raw_val is not None:
                try:
                    value = float(raw_val)
                    break
                except (TypeError, ValueError):
                    continue
        if value is not None:
            fractions.setdefault(region, {})[seg] = value
    return fractions


def load_jcims_activity(path: Path) -> pl.DataFrame:
    """Parse total utility generation MWh from electricity_activity_jcims.csv.

    Returns long-format DataFrame: Region | Year | MWh
    covering all regions and years 2000-2100 that have non-null values.
    """
    df = _read_jcims_service_request(path).filter(
        pl.col('Branch').is_null() &
        (pl.col('Target') == JCIMS_ACT_TARGET) &
        pl.col('region').is_not_null()
    )
    year_cols_present = [y for y in JCIMS_YEAR_COLS if y in df.columns]
    return (
        df
        .unique(subset=['region'], keep='first')
        .select(['region'] + year_cols_present)
        .unpivot(index='region', variable_name='year_str', value_name='value_str')
        .with_columns([
            pl.col('year_str').cast(pl.Int32).alias('Year'),
            pl.col('value_str').cast(pl.Float64, strict=False).alias('MWh'),
        ])
        .filter(pl.col('MWh').is_not_null() & (pl.col('MWh') > 0))
        .select([pl.col('region').alias('Region'), 'Year', 'MWh'])
        .sort(['Region', 'Year'])
    )


# -- Helpers ------------------------------------------------------------------

def euclidean(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt(sum((a.get(s, 0) - b.get(s, 0)) ** 2 for s in LOAD_SEGMENTS))


def weighted_avg_fractions(
    fractions_mean: pl.DataFrame,
    avg_totals: pl.DataFrame,
    regions: list[str],
) -> dict[str, float]:
    """Generation-weighted average of raw fractions across sub-regions."""
    joined = (
        fractions_mean
        .filter(pl.col('CIMS').is_in(regions))
        .join(avg_totals, on='CIMS', how='left')
        .filter(pl.col('avg_MWh').is_not_null() & (pl.col('avg_MWh') > 0))
        .with_columns(
            (pl.col('raw_fraction') * pl.col('avg_MWh')).alias('weighted')
        )
    )
    if joined.is_empty():
        return {s: 0.0 for s in LOAD_SEGMENTS}
    result = (
        joined
        .group_by('Load_Segment')
        .agg(
            (pl.col('weighted').sum() / pl.col('avg_MWh').sum()).alias('raw_fraction')
        )
    )
    return dict(zip(result['Load_Segment'].to_list(), result['raw_fraction'].to_list()))


def main() -> pl.DataFrame:
    # -- Load JCIMS reference data ---------------------------------------------
    print("Loading electricity_activity_jcims.csv...")
    known_jcims   = load_jcims_fractions(PATH_JCIMS)
    jcims_activity = load_jcims_activity(PATH_JCIMS)
    print(f"  Fractions parsed for regions: {sorted(known_jcims)}")
    print(f"  Activity rows: {jcims_activity.height:,}  "
          f"(regions: {sorted(jcims_activity['Region'].unique().to_list())})")

    print("Loading Stats Can 25-10-0015-01...")

    # -- Load and filter -------------------------------------------------------
    sc = (
        pl.scan_csv(PATH_SC15)
        .with_columns([
            pl.col('REF_DATE').str.slice(0, 4).cast(pl.Int32).alias('year'),
            pl.col('VALUE').cast(pl.Float64, strict=False),
        ])
        .filter(
            pl.col('Class of electricity producer') == 'Total all classes of electricity producer'
        )
        .select(['year', 'REF_DATE', 'GEO', 'Type of electricity generation', 'VALUE'])
        .unique(subset=['REF_DATE', 'GEO', 'Type of electricity generation'])
        .collect()
    )

    last_year = sc['year'].max()
    print(f"Data coverage: 2008-{last_year}  ({last_year - 2008 + 1} years)")

    # Map Stats Can GEO names to CIMS region codes
    rmap = pl.read_csv(REGION_MAP).select(['CIMS', 'Stats Can'])
    sc = (
        sc
        .join(rmap, left_on='GEO', right_on='Stats Can', how='inner')
        .filter(pl.col('CIMS') != 'CAN')
        .with_columns(pl.col('VALUE').fill_null(0.0))
    )

    # -- Annual totals per region-year -----------------------------------------
    totals = (
        sc
        .filter(pl.col('Type of electricity generation') == 'Total all types of electricity generation')
        .group_by(['CIMS', 'year'])
        .agg(pl.col('VALUE').sum().alias('total_MWh'))
        .filter(pl.col('total_MWh') > 0)
    )

    # -- Generation by load segment (for NN matching) --------------------------
    load_map_df = pl.DataFrame({
        'tech':         [r[0] for r in LOAD_MAP_RECORDS],
        'Load_Segment': [r[1] for r in LOAD_MAP_RECORDS],
    })

    by_seg = (
        sc
        .join(load_map_df, left_on='Type of electricity generation', right_on='tech', how='inner')
        .group_by(['CIMS', 'year', 'Load_Segment'])
        .agg(pl.col('VALUE').sum().alias('seg_MWh'))
    )

    # Unclassified residual (suppressed "Other types" rows) → Shoulder Load
    classified_totals = (
        by_seg
        .group_by(['CIMS', 'year'])
        .agg(pl.col('seg_MWh').sum().alias('classified_MWh'))
    )
    unclassified = (
        totals
        .join(classified_totals, on=['CIMS', 'year'], how='left')
        .with_columns(pl.col('classified_MWh').fill_null(0.0))
        .with_columns(
            (pl.col('total_MWh') - pl.col('classified_MWh')).clip(lower_bound=0).alias('seg_MWh')
        )
        .with_columns(pl.lit('Shoulder Load').alias('Load_Segment'))
        .select(['CIMS', 'year', 'Load_Segment', 'seg_MWh'])
    )
    by_seg = pl.concat([by_seg, unclassified])

    # -- Compute annual raw fractions (sum to 1.0 by construction) -------------
    fractions_annual = (
        by_seg
        .join(totals, on=['CIMS', 'year'], how='inner')
        .with_columns(
            (pl.col('seg_MWh') / pl.col('total_MWh')).alias('raw_fraction')
        )
    )

    # Ensure all three segments present for every region (fill absent with 0)
    fractions_mean = (
        fractions_annual
        .group_by(['CIMS', 'Load_Segment'])
        .agg([
            pl.col('raw_fraction').mean().alias('raw_fraction'),
            pl.col('year').min().alias('year_from'),
            pl.col('year').max().alias('year_to'),
        ])
    )
    all_combos = (
        fractions_mean.select('CIMS').unique()
        .join(pl.DataFrame({'Load_Segment': LOAD_SEGMENTS}), how='cross')
    )
    fractions_mean = (
        all_combos
        .join(fractions_mean, on=['CIMS', 'Load_Segment'], how='left')
        .with_columns(pl.col('raw_fraction').fill_null(0.0))
    )

    # Average total MWh per region (for weighting aggregates)
    avg_totals = (
        totals
        .group_by('CIMS')
        .agg(pl.col('total_MWh').mean().alias('avg_MWh'))
    )

    # Helper: get raw fractions dict for one CIMS region
    def raw_dict(region: str) -> dict[str, float]:
        r = fractions_mean.filter(pl.col('CIMS') == region)
        return dict(zip(r['Load_Segment'].to_list(), r['raw_fraction'].to_list()))

    # AT weighted average (NB+NS+PE+NL) for cross-validation
    atl_raw = weighted_avg_fractions(fractions_mean, avg_totals, ATL_REGIONS)

    # -- Cross-validation ------------------------------------------------------
    print()
    print("Cross-validation: Stats Can generation mix vs. JCIMS load-duration fractions")
    print("NOTE: Divergence is expected for thermal provinces (AB, SK); the")
    print("      Stats Can mix is a generation-type proxy, not a demand-duration curve.")
    print()
    hdr = (f"{'Reg':>4}  "
           f"{'SC Base':>8}  {'SC Shldr':>8}  {'SC Peak':>8}  "
           f"{'JC Base':>8}  {'JC Shldr':>8}  {'JC Peak':>8}  "
           f"{'JC Sum':>7}  Note")
    print(hdr)
    print("-" * len(hdr))

    ref_raw: dict[str, dict[str, float]] = {}
    for code in NN_CANDIDATES:
        rd = raw_dict(code)
        ref_raw[code] = rd
        jc = known_jcims.get(code, {})
        jcs = sum(jc.values()) if jc else float('nan')
        note = ""
        if jc and abs(rd.get('Base Load', 0) - jc.get('Base Load', 0) / jcs) > 0.25:
            note = "<-- generation mix diverges from demand-duration curve"
        print(
            f"{code:>4}  "
            f"{rd.get('Base Load', 0):8.4f}  {rd.get('Shoulder Load', 0):8.4f}  "
            f"{rd.get('Peak Load', 0):8.4f}  "
            f"{jc.get('Base Load', float('nan')):8.4f}  "
            f"{jc.get('Shoulder Load', float('nan')):8.4f}  "
            f"{jc.get('Peak Load', float('nan')):8.4f}  {jcs:7.4f}  {note}"
        )
    # AT as combined Atlantic reference
    jc_at = known_jcims.get('AT', {})
    print(
        f"{'AT':>4}  "
        f"{atl_raw.get('Base Load', 0):8.4f}  {atl_raw.get('Shoulder Load', 0):8.4f}  "
        f"{atl_raw.get('Peak Load', 0):8.4f}  "
        f"{jc_at['Base Load']:8.4f}  {jc_at['Shoulder Load']:8.4f}  "
        f"{jc_at['Peak Load']:8.4f}  {sum(jc_at.values()):7.4f}  (AT = wtd avg NB+NS+PE+NL)"
    )

    # -- Nearest-neighbour matching for missing regions ------------------------
    # Build the reference space from SC raw fractions of NN_CANDIDATES
    missing_regions = ATL_REGIONS + TER_REGIONS

    # Also include SC raw for individual AT provinces for display
    all_missing_raws: dict[str, dict[str, float]] = {r: raw_dict(r) for r in missing_regions}

    print()
    print("Nearest-neighbour matching (reference = SC generation-mix space):")
    print(f"{'Region':>6}  {'SC Base':>8}  {'SC Shldr':>8}  {'SC Peak':>8}  "
          f"  {'Match':>4}  {'Distance':>8}")
    print("-" * 60)

    matched_fractions: dict[str, dict[str, float]] = {}
    matched_to: dict[str, str] = {}

    for region in missing_regions:
        rd = all_missing_raws[region]
        best_match = min(NN_CANDIDATES, key=lambda c: euclidean(rd, ref_raw[c]))
        dist = euclidean(rd, ref_raw[best_match])
        matched_to[region] = best_match
        matched_fractions[region] = known_jcims[best_match]
        print(
            f"{region:>6}  "
            f"{rd.get('Base Load', 0):8.4f}  {rd.get('Shoulder Load', 0):8.4f}  "
            f"{rd.get('Peak Load', 0):8.4f}  "
            f"  {best_match:>4}  {dist:8.4f}"
        )

    # Atlantic provinces: also note AT aggregate as alternative
    print()
    print("NOTE for NB/NS/PE/NL: the AT aggregate fractions already exist in JCIMS.")
    print("  Individual province fractions above are nearest-neighbour estimates.")
    print("  Consider using AT fractions if the model uses AT as the unit of analysis.")

    # TR = generation-weighted average of YT + NT + NU matched fractions
    ter_raw = weighted_avg_fractions(fractions_mean, avg_totals, TER_REGIONS)
    ter_jcims_avg = (
        jcims_activity
        .filter(
            pl.col('Region').is_in(TER_REGIONS) &
            (pl.col('Year') <= int(last_year))
        )
        .group_by('Region')
        .agg(pl.col('MWh').mean().alias('avg_MWh'))
    )
    ter_mwh_dict = dict(zip(
        ter_jcims_avg['Region'].to_list(),
        ter_jcims_avg['avg_MWh'].to_list(),
    ))
    total_ter_mwh = sum(ter_mwh_dict.values())
    tr_fractions: dict[str, float] = {s: 0.0 for s in LOAD_SEGMENTS}
    if total_ter_mwh > 0:
        for region in TER_REGIONS:
            w = ter_mwh_dict.get(region, 0.0) / total_ter_mwh
            for s in LOAD_SEGMENTS:
                tr_fractions[s] += w * matched_fractions[region][s]
    matched_to['TR'] = 'wtd avg YT+NT+NU'

    # -- Final summary ---------------------------------------------------------
    # Reference provinces use JCIMS fractions directly; matched regions use NN result
    # AT and TR aggregates are excluded from output
    final_fractions: dict[str, dict[str, float]] = {}
    for r in NN_CANDIDATES:
        final_fractions[r] = known_jcims[r]
    for r, f in matched_fractions.items():
        if r != 'TR':
            final_fractions[r] = f

    print()
    print("=" * 80)
    print("Final fractions (for electricity_activity_jcims.csv):")
    print("=" * 80)
    print(f"{'Region':>6}  {'Base Load':>10}  {'Shldr Load':>10}  {'Peak Load':>10}  "
          f"{'Sum':>7}  {'Source':>30}")
    print("-" * 80)

    for region, f in final_fractions.items():
        base, shoulder, peak = f['Base Load'], f['Shoulder Load'], f['Peak Load']
        total = base + shoulder + peak
        if region in NN_CANDIDATES:
            source = "JCIMS (existing)"
        else:
            source = f"nearest neighbour: {matched_to.get(region, '?')}"
        print(f"{region:>6}  {base:10.6f}  {shoulder:10.6f}  {peak:10.6f}  "
              f"{total:7.4f}  {source}")

    # -- Build long-format output ----------------------------------------------
    rows = []

    # Load fractions (constant across all years)
    sc_last = LAST_DATA_YEAR['stat_can_electricity']
    for region, f in final_fractions.items():
        for seg in LOAD_SEGMENTS:
            value = round(f[seg], 9)
            for year in range(DATA_START, PROJECTION_END + 1):
                source = 'StatsCan/JCIMS' if year <= sc_last else 'assumptions'
                rows.append({
                    'Region':   region,
                    'Variable': f'Electricity.Utility Generation.{seg}',
                    'Unit':     '% MWh',
                    'Source':   source,
                    'Year':     year,
                    'Value':    value,
                })

    frac_df = pl.DataFrame(rows)

    # Total utility generation activity (MWh) from JCIMS CSV, individual provinces only
    act_df = (
        jcims_activity
        .filter(~pl.col('Region').is_in(['AT', 'TR']))
        .with_columns([
            pl.lit('Electricity.Utility Generation').alias('Variable'),
            pl.lit('MWh').alias('Unit'),
            pl.lit('electricity_activity_jcims.csv').alias('Source'),
        ])
        .rename({'MWh': 'Value'})
        .select(['Region', 'Variable', 'Unit', 'Source', 'Year', 'Value'])
    )

    output_df = pl.concat([
        frac_df,
        act_df.with_columns(pl.col('Year').cast(pl.Int64)),
    ]).sort(['Region', 'Variable', 'Year'])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(OUTPUT)
    print(f"Saved: {OUTPUT}")
    print(f"Rows: {output_df.height:,}  |  Regions: {sorted(output_df['Region'].unique().to_list())}")
    print(f"Variables: {sorted(output_df['Variable'].unique().to_list())}")

    return output_df


if __name__ == '__main__':
    main()
