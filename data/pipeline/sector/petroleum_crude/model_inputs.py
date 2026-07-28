"""
Petroleum Crude Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/petroleum_crude/petroleum_crude_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Encodes the full Petroleum Crude hierarchy:
      Petroleum Crude (Sector)
        ├── Production (Service, Fixed Ratio)
        │     ├── Exploration  (service_request: Wells per m3 — fixed, region-specific)
        │     ├── Light Medium (Service, Fixed Ratio)
        │     │     ├── Onshore   (all regions except NL)
        │     │     └── Offshore  (NL only)
        │     ├── Heavy        (Service, Fixed Ratio — BC, SK, NL, AB)
        │     └── Bitumen      (Service, Fixed Ratio — AB only)
        │           ├── In-Situ
        │           ├── Mining
        │           └── Upgrading
        ├── Industrial Engines, Steam, Still Gas Steam,
        │   Hydrogen, CCS  (shared utility services)
    
    Decisions/notes:
    1. Kept exploration in fixed data because not in activity script
    2. Added heavy oil to NL and BC (copied from AB)
    3. Removed onshore from NL
    4. Removed offshore from regions without
    5. Used MB as proxy for NT (revisit - likely more exploration in NT)

Total production  (service_request at sector level)
    pipeline/source/activity/oil_production.py  (called directly via main())
    Variable: 'Total'  (m3, per province, 2000–2100)

Production sub-type splits  (service_request rows at Production service level)
    pipeline/source/activity/oil_production.py  (called directly via main())
    Variables: 'Bitumen', 'Light Medium', 'Heavy'  (%)
    Targets: CIMS.CAN.{region}.Petroleum Crude.Production.{subtype}
    Regions where a subtype is 0% for the entire 2000–2100 series are
    dropped (drop_zero_activity_regions) — e.g. MB/ON/NT have no Bitumen or
    Heavy production, so no service_request is written for those branches.

Bitumen sub-splits  (service_request rows at Bitumen service level — AB only)
    pipeline/source/activity/oil_production.py  (called directly via main())
    Variables: 'Bitumen.In-Situ', 'Bitumen.Mining', 'Bitumen.Upgrading'  (m3 ratios)
    Targets: CIMS.CAN.{region}.Petroleum Crude.Production.Bitumen.{subtype}
    Same zero-region drop as above — non-AB regions have no Bitumen branch
    in the fixed hierarchy, so their all-zero rows are dropped rather than
    written as orphan service_requests.

Light Medium sub-splits  (service_request rows at Light Medium service level)
    pipeline/source/activity/oil_production.py  (called directly via main())
    Variables: 'Light Medium.Onshore', 'Light Medium.Offshore'  (%)
    Targets: CIMS.CAN.{region}.Petroleum Crude.Production.Light Medium.{subtype}
    Same zero-region drop — e.g. NL is 0% Onshore, all other regions are
    0% Offshore.

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py  (called directly via main())

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. service_request  — total production (from oil_production)
2. competition      — from fixed data (Sector level)
3. multiplier_price — from energy_price_multipliers
4. service_request  — Production → Bitumen, Light Medium, Heavy (from oil_production)
5. service_request  — Bitumen → In-Situ, Mining, Upgrading (from oil_production, AB only)
6. service_request  — Light Medium → Onshore, Offshore (from oil_production)
7. rest of fixed data
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

_op_spec = importlib.util.spec_from_file_location(
    'oil_production',
    _PIPELINE_ROOT / 'source' / 'activity' / 'oil_production.py',
)
_oil_production_mod = importlib.util.module_from_spec(_op_spec)
_op_spec.loader.exec_module(_oil_production_mod)

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
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/petroleum_crude'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/petroleum_crude'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Level 1 sub-types at the Production service level
PRODUCTION_SUBTYPES = ['Bitumen', 'Light Medium', 'Heavy']

# Level 2 sub-splits at the Bitumen service level (AB only)
BITUMEN_SUBTYPES = ['Bitumen.In-Situ', 'Bitumen.Mining', 'Bitumen.Upgrading']

# Level 2 sub-splits at the Light Medium service level
LIGHT_MEDIUM_SUBTYPES = ['Light Medium.Onshore', 'Light Medium.Offshore']

REGION_SPECIFIC_ENERGIES = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed() -> pl.DataFrame:
    """Flatten all Petroleum Crude fixed CSVs and return combined DataFrame."""
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


def _build_total_rows(oil: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows for the Petroleum Crude sector.
    Value = total oil production m3 from oil_production.
    """
    df = oil.filter(pl.col('Variable') == 'Total')
    return df.select([
        (pl.lit('CIMS.CAN.') + pl.col('Region')).alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.col('Region'),
        pl.lit('Petroleum Crude').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Petroleum Crude')).alias('Target'),
        pl.col('Source'),
        pl.lit('m3').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Value').cast(pl.String).alias('Value'),
    ])


def _build_price_rows(multipliers: pl.DataFrame) -> pl.DataFrame:
    """multiplier_price rows for the Petroleum Crude sector."""
    return (
        multipliers
        .filter(pl.col('Sector') == 'Petroleum Crude')
        .select([
            (pl.lit('CIMS.CAN.') + pl.col('Region') + pl.lit('.Petroleum Crude')).alias('Branch'),
            pl.lit('Sector').alias('Type'),
            pl.col('Region').alias('Region'),
            pl.lit('Petroleum Crude').alias('Sector'),
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


def _build_production_split_rows(oil: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Production service level.
    One block per sub-type (Bitumen, Light Medium, Heavy); value = %
    Target: CIMS.CAN.{region}.Petroleum Crude.Production.{subtype}
    """
    parts = []
    for sub in PRODUCTION_SUBTYPES:
        df = oil.filter(pl.col('Variable') == sub)
        df = drop_zero_activity_regions(df)
        if df.is_empty():
            continue
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Petroleum Crude.Production')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Petroleum Crude').alias('Sector'),
            pl.lit('Production').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Petroleum Crude.Production.{sub}')).alias('Target'),
            pl.col('Source'),
            pl.lit('%').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


def _build_bitumen_split_rows(oil: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Bitumen service level (AB only).
    One block per sub-type (Bitumen.In-Situ, .Mining, .Upgrading); value = %
    These coexist with the fixed data Bitumen service_request rows, which are intensity
    coefficients (m3 of service per m3 of bitumen) — a different thing entirely.
    Target: CIMS.CAN.{region}.Petroleum Crude.Production.Bitumen.{subtype}
    """
    parts = []
    for sub in BITUMEN_SUBTYPES:
        df = oil.filter(pl.col('Variable') == sub)
        df = drop_zero_activity_regions(df)
        if df.is_empty():
            continue
        subtype = sub.split('.', 1)[1]   # e.g. 'In-Situ'
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Petroleum Crude.Production.Bitumen')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Petroleum Crude').alias('Sector'),
            pl.lit('Bitumen').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Petroleum Crude.Production.Bitumen.{subtype}')).alias('Target'),
            pl.col('Source'),
            pl.col('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


def _build_light_medium_split_rows(oil: pl.DataFrame) -> pl.DataFrame:
    """
    service_request rows at the Light Medium service level.
    One block per sub-type (Onshore, Offshore); value = %
    NL is entirely Offshore; all other regions are entirely Onshore.
    Target: CIMS.CAN.{region}.Petroleum Crude.Production.Light Medium.{subtype}
    """
    parts = []
    for sub in LIGHT_MEDIUM_SUBTYPES:
        df = oil.filter(pl.col('Variable') == sub)
        df = drop_zero_activity_regions(df)
        if df.is_empty():
            continue
        subtype = sub.split('.', 1)[1]   # e.g. 'Onshore'
        parts.append(df.select([
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit('.Petroleum Crude.Production.Light Medium')).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.col('Region'),
            pl.lit('Petroleum Crude').alias('Sector'),
            pl.lit('Light Medium').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            (pl.lit('CIMS.CAN.') + pl.col('Region')
             + pl.lit(f'.Petroleum Crude.Production.Light Medium.{subtype}')).alias('Target'),
            pl.col('Source'),
            pl.lit('%').alias('Unit'),
            pl.col('Year').cast(pl.String).alias('Year'),
            pl.col('Value').cast(pl.String).alias('Value'),
        ]))
    if not parts:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return pl.concat(parts)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble Petroleum Crude model inputs and write one CSV per region."""
    print('=' * 60)
    print('PETROLEUM CRUDE MODEL INPUTS')
    print('=' * 60)

    print('\nFlattening fixed structural data...')
    fixed = _read_flattened_fixed()
    print(f'  Rows: {len(fixed):,}')

    print('Loading oil production activity data...')
    oil = _oil_production_mod.main()

    print('Building total production rows...')
    total_rows = _build_total_rows(oil)
    print(f'  Rows: {len(total_rows):,}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    price_rows = _build_price_rows(multipliers)
    print(f'  Rows: {len(price_rows):,}')

    print('Building production sub-type split rows...')
    prod_split_rows = _build_production_split_rows(oil)
    print(f'  Rows: {len(prod_split_rows):,}')

    print('Building bitumen sub-split rows...')
    bitumen_split_rows = _build_bitumen_split_rows(oil)
    print(f'  Rows: {len(bitumen_split_rows):,}')

    print('Building light medium sub-split rows...')
    lm_split_rows = _build_light_medium_split_rows(oil)
    print(f'  Rows: {len(lm_split_rows):,}')

    print('Combining...')

    # Sector-level branch: ends with '.Petroleum Crude' (no further sub-path)
    _sector_branch = pl.col('Branch').str.ends_with('.Petroleum Crude')

    # Keep only competition from fixed data at sector level;
    fixed_sector_competition = fixed.filter(
        _sector_branch & (pl.col('Parameter').fill_null('') == 'competition')
    )

    # Drop sector-level competition.
    # The Bitumen-level service_request rows in the fixed data are intensity
    # coefficients (m3 of service per m3 of bitumen) and are NOT replaced by
    # the pipeline bitumen split rows — those are %,
    # a different quantity that coexists with the fixed intensity rows.
    # Everything else passes through: Production sub-tree with Exploration,
    # Light Medium, Heavy, Bitumen intensity rows, utility services, etc.
    fixed_rest = fixed.filter(
        ~(_sector_branch & pl.col('Parameter').fill_null('').is_in(['competition']))
    )

    regions = total_rows['Region'].unique().sort().to_list()

    output = (
        pl.concat(
            [total_rows, fixed_sector_competition,
             price_rows, prod_split_rows, bitumen_split_rows,
             lm_split_rows, fixed_rest],
            how='diagonal_relaxed',
        )
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        out_path = OUTPUT_DIR / f'petroleum_crude_{region.lower()}.csv'
        region_df = collapse_constant_years(region_df)
        region_df = region_df.filter(pl.col('Parameter') != 'technology')
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Petroleum Crude model inputs complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
