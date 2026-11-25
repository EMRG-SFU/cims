"""
Natural Gas Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

There are three distinct file types produced:

1. Provincial files (AB, BC, SK) — full production hierarchy with dynamic
   service_request rows from gas_production.py.

2. Secondary region files (MB, ON, QC, NB, NS, PE, NL, YT, NT, NU) — same
   fixed-data hierarchy/template as AB/BC/SK (technology economics etc.),
   but with the region-unique service_request rows (Natural Gas total,
   extraction splits, Processing, LNG Compression, drilling intensity,
   fugitive factors) baked directly into each region's fixed CSV from
   natural_gas_annual_data_jcims.csv instead of gas_production.py — since
   gas_production.py's CER-based activity data only covers AB/BC/SK.
   multiplier_price rows are still added dynamically here (below), since
   energy_price_multipliers.py already computes them for all 13 regions.

3. Pass-through file (RoW) — all rows come from fixed data only, passed
   through unchanged.

4. Market file (natural_gas_market/natural_gas_market_can.csv) — the
   national Natural Gas Market, as its own optional sector,
   'Natural Gas Market'. See "Natural Gas Market switch" below.

Sources
-------
Fixed structural parameters
    extracted/legacy/fixed-data/natural_gas/natural_gas_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Natural Gas hierarchy (AB, BC, SK):
      Natural Gas (Sector)  [service_request: gross total production]
        └── Production (Service, Fixed Ratio)
              ├── Supply (Service, Fixed Ratio)
              │     ├── Extraction (Service, Fixed Ratio)
              │     │     ├── Conventional, Tight, Shale, Coalbed Methane
              │     │     ├── Drilling
              │     │     ├── Compression
              │     │     └── Off road transport
              │     ├── Processing
              │     ├── LNG Compression (BC only)
              │     ├── Direct Heat, Direct Drive Small/Large
              │     ├── Controls, Pumping
              │     ├── Testing and Maintenance (Blowdowns, Liquid Unloading, Well Tests)
              │     ├── Formation CO2, Flaring, Venting (Diffuse, Point), Fugitive
              │     └── CCS
              ├── Transmission (with Compression)
              └── Distribution
      Natural Gas.OG Diesel/Gasoline/Methane Blend  (utility services)
    
    Decisions/notes:
    1. SK, AB, BC only because other regions so small and we take data from CER supply data
    and they don't report in EF work.


    All service_provide rows in the fixed data have null values and are
    retained unchanged. Activity data is inserted as service_request rows.
    service_request rows for Natural Gas, extraction
    splits, Processing, and LNG Compression are absent from fixed data and
    generated entirely from the activity script. Drilling intensity
    service_request rows (Wells per 1000m3) exist in the fixed data with
    region-specific values and are retained unchanged.

    The RoW file (natural_gas_RoW.csv) is fully static and passes through
    unchanged. The CAN file (natural_gas_CAN.csv) holds the market, which
    moves to its own sector (below).

Natural Gas Market switch
    The fixed data routes each region's production through the national
    market: CIMS.CAN.{r}.Natural Gas.Production requests
    CIMS.CAN.Natural Gas Market (1.0), and the market's Supply,
    Transmission, and Distribution nodes split the national total back
    across the regions by fixed shares. The model's sector filter can only
    add rows for a sector that is turned on, so the switch works like this:

    - Regional files hold the market-off path: Production requests its own
      region's Supply, Transmission, and Distribution directly (1.0 each).
      Each region then produces what its own region node requests.
    - The market file holds the market and, for every region, overrides
      that path: Production requests the market (1.0) and its own three
      nodes at 0. All its rows carry Sector 'Natural Gas Market', so they
      load only when that sector is in the notebook's sector_list.
    - The market file must load after the natural_gas files: list
      'natural_gas_market' after 'natural_gas' in the notebook's sector_req.
      service_request rows replace earlier rows with the same Target.
    - The market's Supply shares come from the regions' own production
      requests (CIMS.CAN.{r} -> CIMS.CAN.{r}.Natural Gas), as each region's
      share of the national total in each year, replacing the JCIMS shares.
      With the market on, each region then supplies what it requests, as
      it does with the market off. Transmission and Distribution keep their
      fixed-data shares.
    - With the market off, consumers still pay the CIMS.Generic Fuels.Natural
      Gas price (a Supply - Fixed Price node in fuels). The market never set
      that price; it only redistributes production between regions.

Activity data  (from gas_production.py, Region/Variable/Unit/Source/Year/Value format)
    Variable mappings:
      'Natural Gas'              → service_request at CIMS.CAN.{r}.Natural Gas, Target: CIMS.CAN.{r}.Natural Gas (1000m3)
      'Extraction.Conventional'  → service_request at Extraction → Conventional (%)
      'Extraction.Shale'         → service_request at Extraction → Shale (%)
      'Extraction.Tight'         → service_request at Extraction → Tight (%)
      'Extraction.Coalbed Methane' → service_request at Extraction → Coalbed Methane (%)
      'Processing'               → service_request at Supply → Processing (1000m3/1000m3)
      'LNG Compression'          → service_request at Supply → LNG Compression (%, BC only)
    ('Natural Gas.Extraction' is not used in model inputs)
    Extraction split, Processing, and LNG Compression rows are dropped per
    region when 0 for the entire 2000–2100 series (drop_zero_activity_regions)
    — e.g. LNG Compression is BC-only, so other provincial regions get no
    orphan service_request for it. The sector-level 'Natural Gas' total is
    not filtered this way.

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Computed for all 13 provinces/territories; applied to provincial (AB, BC,
    SK) and secondary region files alike. Not applicable to CAN/RoW.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order — provincial files (AB, BC, SK)
--------------------------------------------
1. competition      — from fixed data (Sector level)
2. multiplier_price — from energy_price_multipliers
3. service_request  — Natural Gas (from gas_production)
4. service_request  — extraction splits, Processing, LNG (from gas_production)
5. rest of fixed data (service_provide null rows and drilling intensity rows retained)

Output order — secondary region and RoW files
-----------------------------------------------------
1. fixed data pass-through (secondary regions already include their
   region-unique service_request rows, baked in from jcims)
2. multiplier_price — from energy_price_multipliers (secondary regions only)
"""


import pandas as pd
import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_folder

import CIMS.data_processing.source.activity.gas_production as _ng_mod

import CIMS.data_processing.source.energy_prices.energy_price_multipliers as _energy_price_mod

import CIMS.data_processing.source.cer.cer_resd_demand as _cer_resd_mod

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.currency import write_model_csv
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years
from CIMS.data_processing.utils.drop_zero_activity import drop_zero_activity_regions
from CIMS.data_processing.utils.feedstock_demand import build_feedstock_rows_all_regions

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = paths.FIXED_DATA / 'natural_gas'
OUTPUT_DIR      = paths.MODEL / 'natural_gas'
MARKET_OUTPUT_DIR = paths.MODEL / 'natural_gas_market'

# The national market, an optional sector (see "Natural Gas Market switch")
MARKET_SECTOR = 'Natural Gas Market'
MARKET_BRANCH = 'CIMS.CAN.Natural Gas Market'
MARKET_REGION = 'CAN'
PRODUCTION_STAGES = ['Supply', 'Transmission', 'Distribution']

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

PROVINCIAL_REGIONS = ['AB', 'BC', 'SK']

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel',
    'Ethanol', 'Hydrogen',
}

# Extraction split variables from gas_production.py and their target sub-service names
EXTRACTION_SPLITS = {
    'Extraction.Conventional':    'Conventional',
    'Extraction.Shale':           'Shale',
    'Extraction.Tight':           'Tight',
    'Extraction.Coalbed Methane': 'Coalbed Methane',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Natural Gas sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Natural Gas')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Natural Gas')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Natural Gas').alias('Sector'),
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


def _build_activity_rows(ng: pl.DataFrame) -> pl.DataFrame:
    """
    service_request row for the Natural Gas sector.
    Value = gross total production (1000m3) from gas_production.
    Replaces the null-valued service_request row at the sector level
    in the fixed data. service_provide rows remain null.
    """
    return (
        ng.filter(pl.col('Variable') == 'Natural Gas')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
            pl.lit('Region').alias('Type'),
            pl.col('Region'),
            pl.lit('Natural Gas').alias('Sector'),
            pl.lit('').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Natural Gas')).alias('Target'),
            pl.col('Source'),
            pl.col('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ])
    )


def _build_extraction_split_rows(ng: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Extraction service level.
    One block per sub-type (Conventional, Shale, Tight, Coalbed Methane);
    value = %
    These rows are absent from the fixed data and generated entirely here.
    Target: CIMS.CAN.{region}.Natural Gas.Production.Supply.Extraction.{subtype}
    """
    parts = []
    for variable, subtype in EXTRACTION_SPLITS.items():
        df = ng.filter(pl.col('Variable') == variable)
        df = drop_zero_activity_regions(df)
        if df.is_empty():
            continue
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Natural Gas.Production.Supply.Extraction')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Natural Gas').alias('Sector'),
            pl.lit('Extraction').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Natural Gas.Production.Supply.Extraction.{subtype}')).alias('Target'),
            pl.col('Source'),
            pl.col('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


def _build_processing_rows(ng: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Supply service level for Processing.
    Value = ratio (1000m3/1000m3) from gas_production.
    Absent from the fixed data; generated entirely here.
    Target: CIMS.CAN.{region}.Natural Gas.Production.Supply.Processing
    """
    df = ng.filter(pl.col('Variable') == 'Processing')
    df = drop_zero_activity_regions(df)
    if df.is_empty():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Natural Gas.Production.Supply')).alias('Branch'),
        pl.lit('Service').alias('Type'),
        pl.col('Region'),
        pl.lit('Natural Gas').alias('Sector'),
        pl.lit('Natural Gas Supply').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Natural Gas.Production.Supply.Processing')).alias('Target'),
        pl.col('Source'),
        pl.col('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_lng_rows(ng: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Supply service level for LNG Compression (BC only).
    Value = %
    Absent from the fixed data; generated entirely here.
    Target: CIMS.CAN.BC.Natural Gas.Production.Supply.LNG Compression
    """
    df = ng.filter(pl.col('Variable') == 'LNG Compression')
    df = drop_zero_activity_regions(df)
    if df.is_empty():
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Natural Gas.Production.Supply')).alias('Branch'),
        pl.lit('Service').alias('Type'),
        pl.col('Region'),
        pl.lit('Natural Gas').alias('Sector'),
        pl.lit('Natural Gas Supply').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region')
         + pl.lit('.Natural Gas.Production.Supply.LNG Compression')).alias('Target'),
        pl.col('Source'),
        pl.col('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _region_anchor(fixed_provincial: pl.DataFrame, mask: pl.Expr) -> pl.DataFrame:
    """One row per Region: the _order of the (single) fixed row matching mask."""
    return (
        fixed_provincial.filter(mask)
        .group_by('Region')
        .agg(pl.col('_order').first().alias('_anchor_order'))
    )


def _order_after_region_anchor(rows: pl.DataFrame, anchors: pl.DataFrame, offset: float) -> pl.DataFrame:
    """
    Attach an _order column to `rows` (which must have a Region column),
    positioned just after each row's region-specific anchor _order,
    preserving the rows' own relative order within each region.
    """
    if rows.is_empty():
        return rows.with_columns(pl.lit(None, dtype=pl.Float64).alias('_order'))
    rows = rows.with_row_index('_seq')
    region_base = rows.group_by('Region').agg(pl.col('_seq').min().alias('_base'))
    return (
        rows
        .join(region_base, on='Region', how='left')
        .join(anchors, on='Region', how='left')
        .with_columns(
            (pl.col('_anchor_order') + offset + (pl.col('_seq') - pl.col('_base')) * 1e-6)
            .alias('_order')
        )
        .drop(['_seq', '_base', '_anchor_order'])
    )


def _build_feedstock_rows(ng: pl.DataFrame) -> pl.DataFrame:
    """
    Feedstock service rows for the Natural Gas sector (CER vFsDmd-CIMS.csv,
    via cer_resd_demand.load_feedstock_demand()), tied to the same
    'Natural Gas' production driver _build_activity_rows() already uses.

    Only built for PROVINCIAL_REGIONS (AB/BC/SK): gas_production.py's
    activity data -- and therefore a real 2000-2100 scale driver -- only
    exists for those three; every other region's own Natural Gas ->
    Production service_request is a flat structural placeholder ("1" for
    every year) with no genuine per-year signal to divide by. NL in
    particular has real, substantial CER feedstock demand with no usable
    driver anywhere in this pipeline -- a known gap, left for a follow-up
    decision on a proxy driver rather than guessed at here.
    """
    feedstock_demand = _cer_resd_mod.load_feedstock_demand()
    feedstock_demand = feedstock_demand[feedstock_demand['Node'] == '.Natural Gas']
    if feedstock_demand.empty:
        return pl.DataFrame()

    driver = (
        ng.filter((pl.col('Variable') == 'Natural Gas') & pl.col('Region').is_in(PROVINCIAL_REGIONS))
        .with_columns(pl.col('Year').cast(pl.Int64), pl.col('Value').cast(pl.Float64))
    )
    scale_unit = driver['Unit'][0] if len(driver) else '1000m3'
    scale_by_region = {
        region: pd.Series(sub['Value'].to_list(), index=sub['Year'].to_list())
        for region, sub in driver.to_pandas().groupby('Region')
    }

    return build_feedstock_rows_all_regions(
        sector_name='Natural Gas',
        feedstock_demand=feedstock_demand,
        scale_by_region=scale_by_region,
        scale_unit=scale_unit,
    )


def _market_supply_shares(provincial: pl.DataFrame, passthrough: pl.DataFrame) -> pl.DataFrame:
    """
    Replace the market's Supply shares with each region's share of the
    national production request, year by year.

    The production request is the CIMS.CAN.{r} -> CIMS.CAN.{r}.Natural Gas
    service_request (gas_production.py for AB, BC, and SK; fixed data for
    the other regions). A constant row (no Year) applies to every year
    without its own row. Returns `passthrough` with the market's Supply
    service_request rows replaced.
    """
    regional = pl.concat([provincial, passthrough], how='diagonal_relaxed')
    is_request = (
        (pl.col('Parameter') == 'service_request')
        & (pl.col('Branch') == pl.lit('CIMS.CAN.') + pl.col('Region'))
        & (pl.col('Target') == pl.col('Branch') + pl.lit('.Natural Gas'))
    )
    requests = regional.filter(is_request).select(
        'Region',
        pl.col('Year').cast(pl.String).replace('', None).alias('Year'),
        pl.col('Value').cast(pl.Float64).alias('Value'),
    )
    years = sorted(requests['Year'].drop_nulls().unique().to_list())

    by_region = {}
    for region in requests['Region'].unique().sort().to_list():
        rows = requests.filter(pl.col('Region') == region)
        constant = rows.filter(pl.col('Year').is_null())['Value']
        values = {y: (constant[-1] if len(constant) else 0.0) for y in years}
        values.update(dict(zip(rows.filter(pl.col('Year').is_not_null())['Year'].to_list(),
                               rows.filter(pl.col('Year').is_not_null())['Value'].to_list())))
        by_region[region] = values
    totals = {y: sum(v[y] for v in by_region.values()) for y in years}

    is_supply_share = (
        (pl.col('Branch') == f'{MARKET_BRANCH}.Supply')
        & (pl.col('Parameter') == 'service_request')
    )
    template = passthrough.filter(is_supply_share).head(1)
    if template.is_empty():
        raise ValueError(f'No {MARKET_BRANCH}.Supply service_request rows found to replace.')
    template = template.row(0, named=True)

    share_rows = [
        {**template,
         'Region': MARKET_REGION,
         'Target': f'CIMS.CAN.{region}.Natural Gas.Production.Supply',
         'Source': 'Share of regional production requests',
         'Year': year,
         'Value': str(values[year] / totals[year] if totals[year] else 0.0)}
        for region, values in by_region.items()
        for year in years
    ]
    shares = pl.DataFrame(share_rows, schema=passthrough.schema)

    # Put the new rows where the old ones were
    passthrough = passthrough.with_row_index('_i')
    first = passthrough.filter(is_supply_share)['_i'].min()
    passthrough = pl.concat([
        passthrough.filter(~is_supply_share & (pl.col('_i') < first)).drop('_i'),
        shares,
        passthrough.filter(~is_supply_share & (pl.col('_i') >= first)).drop('_i'),
    ])
    return passthrough


def _split_market(regional: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Separate the Natural Gas Market from the regional rows.

    Returns (regional rows with the market-off path, market file rows).

    Regional rows: every Production -> market request is replaced, in place,
    by Production -> own Supply, Transmission, and Distribution (1.0 each,
    only for nodes that exist in that region). The CAN market rows are
    removed.

    Market file rows: the CAN market rows, then for each region the original
    Production -> market request and a 0 request for each of the direct
    links. Every market file row gets Sector 'Natural Gas Market'.
    """
    to_market = (
        (pl.col('Parameter') == 'service_request')
        & (pl.col('Target') == MARKET_BRANCH)
        & pl.col('Branch').str.ends_with('.Natural Gas.Production')
    )
    is_market_node = pl.col('Region') == MARKET_REGION

    regional = regional.with_row_index('_i').with_columns(pl.col('_i').cast(pl.Float64))
    links = regional.filter(to_market)
    existing_branches = set(regional['Branch'].drop_nulls().to_list())

    def _direct(value: str) -> pl.DataFrame:
        parts = []
        for k, stage in enumerate(PRODUCTION_STAGES):
            part = links.with_columns(
                (pl.col('Branch') + pl.lit(f'.{stage}')).alias('Target'),
                pl.lit(value).alias('Value'),
                pl.lit('').alias('Source'),
                (pl.col('_i') + (k + 1) * 0.1).alias('_i'),
            )
            parts.append(part.filter(pl.col('Target').is_in(list(existing_branches))))
        return pl.concat(parts)

    regional_out = (
        pl.concat([regional.filter(~to_market & ~is_market_node), _direct('1.0')])
        .sort('_i')
        .drop('_i')
    )

    market_nodes = regional.filter(is_market_node & ~to_market)
    switch_rows = pl.concat([links, _direct('0')]).sort('_i')
    market_out = (
        pl.concat([market_nodes, switch_rows])
        .drop('_i')
        .with_columns(pl.lit(MARKET_SECTOR).alias('Sector'))
    )
    return regional_out, market_out


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Natural Gas model inputs and write one CSV per region."""
    print('=' * 60)
    print('NATURAL GAS MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = read_fixed_data_folder(FIXED_INPUT_DIR).with_row_index('_order')
    print(f'  Rows: {len(fixed):,}')

    print('Loading gas production activity data...')
    ng = _ng_mod.main()

    print('Building activity service_request rows...')
    activity_rows = _build_activity_rows(ng)
    print(f'  Rows: {len(activity_rows):,}')

    print('Building extraction split service_request rows...')
    extraction_rows = _build_extraction_split_rows(ng)
    print(f'  Rows: {len(extraction_rows):,}')

    print('Building Processing service_request rows...')
    processing_rows = _build_processing_rows(ng)
    print(f'  Rows: {len(processing_rows):,}')

    print('Building LNG Compression service_request rows...')
    lng_rows = _build_lng_rows(ng)
    print(f'  Rows: {len(lng_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building feedstock rows...')
    feedstock_rows = _build_feedstock_rows(ng)
    print(f'  Rows: {len(feedstock_rows):,}')

    print('Combining...')

    # ── provincial files (AB, BC, SK) ──────────────────────────────────────────
    # Dynamic rows are absent from the fixed data entirely, so each block is
    # anchored to the _order of the matching branch's own competition row
    # (matching the reference CIMS model's branch-hierarchy row order):
    #   - top-level activity: before everything (parent Region branch)
    #   - multiplier_price: right after the Sector-level competition row
    #   - extraction splits: right after the Extraction service's own competition row
    #   - Processing / LNG Compression: right after the Supply service's own competition row
    _sector_branch = pl.col('Branch').str.ends_with('.Natural Gas')
    _extraction_branch = pl.col('Branch').str.ends_with('.Extraction')
    _supply_branch = pl.col('Branch').str.ends_with('.Supply')
    _is_competition = pl.col('Parameter').fill_null('') == 'competition'

    fixed_provincial = fixed.filter(pl.col('Region').is_in(PROVINCIAL_REGIONS))

    sector_competition_anchor = _region_anchor(fixed_provincial, _sector_branch & _is_competition)
    extraction_competition_anchor = _region_anchor(fixed_provincial, _extraction_branch & _is_competition)
    supply_competition_anchor = _region_anchor(fixed_provincial, _supply_branch & _is_competition)
    region_fixed_start = fixed_provincial.group_by('Region').agg(
        pl.col('_order').min().alias('_anchor_order')
    )

    activity_rows_ordered = _order_after_region_anchor(activity_rows, region_fixed_start, offset=-1000.0)
    price_rows_provincial = price_rows.filter(pl.col('Region').is_in(PROVINCIAL_REGIONS))
    price_rows_ordered = _order_after_region_anchor(price_rows_provincial, sector_competition_anchor, offset=0.5)
    extraction_rows_ordered = _order_after_region_anchor(extraction_rows, extraction_competition_anchor, offset=0.5)
    processing_rows_ordered = _order_after_region_anchor(processing_rows, supply_competition_anchor, offset=0.5)
    lng_rows_ordered = _order_after_region_anchor(lng_rows, supply_competition_anchor, offset=0.6)
    feedstock_rows_ordered = _order_after_region_anchor(feedstock_rows, sector_competition_anchor, offset=0.7)

    provincial_output = (
        pl.concat(
            [fixed_provincial,
             activity_rows_ordered, price_rows_ordered, feedstock_rows_ordered,
             extraction_rows_ordered, processing_rows_ordered, lng_rows_ordered],
            how='diagonal_relaxed',
        )
        .sort('_order')
        .select(OUTPUT_COLS)
    )

    # ── secondary regions — fixed data pass-through, plus multiplier_price
    #    rows for any secondary region present in the energy price
    #    multipliers (mirrors the provincial treatment without touching it).
    #    CAN and RoW are not real production regions and never appear in the
    #    multipliers, but are excluded explicitly so those two files stay
    #    pure fixed-data pass-through regardless ─────────────────────────────
    NON_REGION_FILES = {'CAN', 'RoW'}
    secondary_price_rows = price_rows.filter(
        ~pl.col('Region').is_in(PROVINCIAL_REGIONS) & ~pl.col('Region').is_in(NON_REGION_FILES)
    )
    passthrough_output = pl.concat(
        [fixed.filter(~pl.col('Region').is_in(PROVINCIAL_REGIONS)), secondary_price_rows],
        how='diagonal_relaxed',
    ).select(OUTPUT_COLS)

    # ── Natural Gas Market: its own sector and file ───────────────────────────
    passthrough_output = _market_supply_shares(provincial_output, passthrough_output)
    provincial_output, provincial_market = _split_market(provincial_output)
    passthrough_output, passthrough_market = _split_market(passthrough_output)
    market_output = pl.concat(
        [passthrough_market.filter(pl.col('Region') == MARKET_REGION),
         provincial_market,
         passthrough_market.filter(pl.col('Region') != MARKET_REGION)],
        how='diagonal_relaxed',
    ).select(OUTPUT_COLS)
    print(f'Natural Gas Market rows: {len(market_output):,}')

    # ── write outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_outputs = []

    # The CAN market now lives in its own folder; remove the old file so the
    # natural_gas folder can't load the market twice.
    (OUTPUT_DIR / f'natural_gas_{MARKET_REGION.lower()}.csv').unlink(missing_ok=True)

    for region in PROVINCIAL_REGIONS:
        region_df = provincial_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        write_model_csv(region_df, out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        all_outputs.append(region_df)

    for region in passthrough_output['Region'].drop_nulls().unique().sort().to_list():
        region_df = passthrough_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        write_model_csv(region_df, out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        all_outputs.append(region_df)

    MARKET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market_path = MARKET_OUTPUT_DIR / f'natural_gas_market_{MARKET_REGION.lower()}.csv'
    market_output = collapse_constant_years(market_output)
    write_model_csv(market_output, market_path)
    print(f'  Wrote {len(market_output):,} rows → natural_gas_market/{market_path.name}')
    all_outputs.append(market_output)

    output = pl.concat(all_outputs, how='diagonal_relaxed')
    print(f'\n✅ Natural Gas model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(all_outputs)}')

    return output


if __name__ == '__main__':
    main()
