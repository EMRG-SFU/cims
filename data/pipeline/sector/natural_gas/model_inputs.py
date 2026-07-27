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

3. Market/pass-through files (CAN, RoW) — all rows come from fixed data
   only, passed through unchanged.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/natural_gas/natural_gas_{region}.csv
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

    CAN file (natural_gas_CAN.csv) and RoW file (natural_gas_RoW.csv) are
    fully static and pass through unchanged.

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

Output order — secondary region, CAN, and RoW files
-----------------------------------------------------
1. fixed data pass-through (secondary regions already include their
   region-unique service_request rows, baked in from jcims)
2. multiplier_price — from energy_price_multipliers (secondary regions only)
"""

import sys
import tempfile
import importlib.util
from pathlib import Path

import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flatten_mod)

_ng_spec = importlib.util.spec_from_file_location(
    'gas_production',
    _PIPELINE_ROOT / 'source' / 'activity' / 'gas_production.py',
)
_ng_mod = importlib.util.module_from_spec(_ng_spec)
_ng_spec.loader.exec_module(_ng_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years
from utils.drop_zero_activity import drop_zero_activity_regions

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/natural_gas'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/natural_gas'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

PROVINCIAL_REGIONS = ['AB', 'BC', 'SK']

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}

# Extraction split variables from gas_production.py and their target sub-service names
EXTRACTION_SPLITS = {
    'Extraction.Conventional':    'Conventional',
    'Extraction.Shale':           'Shale',
    'Extraction.Tight':           'Tight',
    'Extraction.Coalbed Methane': 'Coalbed Methane',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Natural Gas fixed CSVs and return combined DataFrame."""
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
    return (
        pl.concat(frames, how='diagonal_relaxed')
        .cast(pl.String)
        .with_row_index('_order')
    )


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


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Natural Gas model inputs and write one CSV per region."""
    print('=' * 60)
    print('NATURAL GAS MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
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

    provincial_output = (
        pl.concat(
            [fixed_provincial,
             activity_rows_ordered, price_rows_ordered,
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

    # ── write outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_outputs = []

    for region in PROVINCIAL_REGIONS:
        region_df = provincial_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        all_outputs.append(region_df)

    for region in passthrough_output['Region'].drop_nulls().unique().sort().to_list():
        region_df = passthrough_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        all_outputs.append(region_df)

    output = pl.concat(all_outputs, how='diagonal_relaxed')
    print(f'\n✅ Natural Gas model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(all_outputs)}')

    return output


if __name__ == '__main__':
    main()
