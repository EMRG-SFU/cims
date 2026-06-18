"""
Natural Gas Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

There are two distinct file types produced:

1. Provincial files (AB, BC, SK) — full production hierarchy with dynamic
   service_request rows from gas_production.py.

2. Market/pass-through files (CAN, RoW) — all rows come from fixed data
   only, passed through unchanged.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Natural Gas/natural_gas_{region}.csv
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
      'Extraction.Conventional'  → service_request at Extraction → Conventional (% of 1000m3)
      'Extraction.Shale'         → service_request at Extraction → Shale (% of 1000m3)
      'Extraction.Tight'         → service_request at Extraction → Tight (% of 1000m3)
      'Extraction.Coalbed Methane' → service_request at Extraction → Coalbed Methane (% of 1000m3)
      'Processing'               → service_request at Supply → Processing (1000m3/1000m3)
      'LNG Compression'          → service_request at Supply → LNG Compression (% of 1000m3, BC only)
    ('Natural Gas.Extraction' is not used in model inputs)

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())
    Applied to provincial files (AB, BC, SK) only.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order — provincial files (AB, BC, SK)
--------------------------------------------
1. competition      — from fixed data (Sector level)
2. is_supply        — generated (TRUE)
3. multiplier_price — from energy_price_multipliers
4. service_request  — Natural Gas (from gas_production)
5. service_request  — extraction splits, Processing, LNG (from gas_production)
6. rest of fixed data (service_provide null rows and drilling intensity rows retained)

Output order — CAN and RoW files
---------------------------------
1. fixed data pass-through only
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

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Natural Gas'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/natural gas'

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
    return pl.concat(frames, how='diagonal_relaxed').cast(pl.String)


def _build_is_supply_rows(regions: list[str]) -> pl.DataFrame:
    """Generate a single is_supply=TRUE row per region for the sector header."""
    return pl.DataFrame([
        {
            'Branch':      f'CIMS.CAN.{r}.Natural Gas',
            'Type':        'Sector',
            'Region':      r,
            'Sector':      'Natural Gas',
            'Service':     '',
            'Technology':  '',
            'Parameter':   'is_supply',
            'Context':     'TRUE',
            'Sub_Context': '',
            'Target':      '',
            'Source':      '',
            'Unit':        '',
            'Year':        '',
            'Value':       '',
        }
        for r in regions
    ])


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
    value = % of 1000m3 from gas_production.
    These rows are absent from the fixed data and generated entirely here.
    Target: CIMS.CAN.{region}.Natural Gas.Production.Supply.Extraction.{subtype}
    """
    parts = []
    for variable, subtype in EXTRACTION_SPLITS.items():
        df = ng.filter(pl.col('Variable') == variable)
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
    Value = % of 1000m3 from gas_production.
    Absent from the fixed data; generated entirely here.
    Target: CIMS.CAN.BC.Natural Gas.Production.Supply.LNG Compression
    """
    df = ng.filter(pl.col('Variable') == 'LNG Compression')
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
    _sector_branch = pl.col('Branch').str.ends_with('.Natural Gas')

    fixed_provincial = fixed.filter(pl.col('Region').is_in(PROVINCIAL_REGIONS))

    # competition at sector level — repositioned before is_supply
    fixed_sector_competition = fixed_provincial.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # competition (repositioned). All other fixed rows pass through including
    # sub-service null service_provide rows and drilling intensity rows.
    fixed_provincial_rest = fixed_provincial.filter(
        ~(_sector_branch & pl.col('Parameter').fill_null('').is_in(['competition']))
    )

    is_supply_rows = _build_is_supply_rows(PROVINCIAL_REGIONS)

    provincial_output = (
        pl.concat(
            [fixed_sector_competition, is_supply_rows, price_rows,
             activity_rows, extraction_rows, processing_rows, lng_rows,
             fixed_provincial_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    # ── CAN and RoW files — pass-through only ──────────────────────────────────
    passthrough_output = fixed.filter(
        ~pl.col('Region').is_in(PROVINCIAL_REGIONS)
    ).select(OUTPUT_COLS)

    # ── write outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_outputs = []

    for region in PROVINCIAL_REGIONS:
        region_df = provincial_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        all_outputs.append(region_df)

    for region in passthrough_output['Region'].drop_nulls().unique().sort().to_list():
        region_df = passthrough_output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'natural_gas_{region}.csv'
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
