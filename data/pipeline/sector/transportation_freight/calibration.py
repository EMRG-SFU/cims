"""
Extract Transportation Freight calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total_cumul_net)
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

import sys
import importlib.util
from pathlib import Path

import polars as pl
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_PIPELINE_ROOT = Path(__file__).parent.parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_crosswalk_mod = _load_module(
    'nir_crosswalk_tables_cims',
    _PIPELINE_ROOT / 'source/eccc/nir/nir_crosswalk_tables_cims.py',
)
_nir_mod = _load_module(
    'nir_to_cims',
    _PIPELINE_ROOT / 'source/eccc/nir/nir_to_cims.py',
)
_cer_mod = _load_module(
    'cer_resd_demand',
    _PIPELINE_ROOT / 'source/cer/cer_resd_demand.py',
)
_tf_mod = _load_module(
    'transportation_freight',
    _PIPELINE_ROOT / 'source/nrcan/ceud/transportation_freight/transportation_freight.py',
)

from utils.controls_conversions import BASE_PATH

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/transportation_freight'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

_REGION_MAP: dict[str, str] = {
    'British Columbia':          'BC',
    'Alberta':                   'AB',
    'Saskatchewan':              'SK',
    'Manitoba':                  'MB',
    'Ontario':                   'ON',
    'Quebec':                    'QC',
    'New Brunswick':             'NB',
    'Nova Scotia':               'NS',
    'Prince Edward Island':      'PE',
    'Newfoundland and Labrador': 'NL',
    'Yukon':                     'YT',
    'Northwest Territories':     'NT',
    'Nunavut':                   'NU',
}

_REGIONAL_FUELS = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen',
}

# Pipeline fuel category → Light Medium technology name for the calibration year
LM_CAT_TO_TECH: dict[str, str] = {
    'Diesel':   'Diesel Existing',
    'Gasoline': 'Gasoline Existing',
    'Propane':  'Propane',
}

# (pipeline variable, service_name, branch suffix after sector, cat_to_tech, year_max)
# year_max=None → all historical years
_TECH_SHARE_SERVICES: list[tuple[str, str, str, dict[str, str] | None, int | None]] = [
    ('Light Medium',       'Light Medium', '.Freight.Land.Light Medium', LM_CAT_TO_TECH, None),
    ('Freight.Land.Heavy', 'Heavy',        '.Freight.Land.Heavy',        None,           None),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _branch_meta(branch: str) -> dict:
    """Infer Type, Region, Sector, Service from a CIMS branch string."""
    parts = branch.split('.')
    if len(parts) < 3:
        return {'Type': '', 'Region': '', 'Sector': '', 'Service': ''}
    region = parts[2]
    if len(parts) == 3:
        return {'Type': 'Region', 'Region': region, 'Sector': '', 'Service': ''}
    sector = parts[3]
    if len(parts) == 4:
        return {'Type': 'Sector', 'Region': region, 'Sector': sector, 'Service': ''}
    service = parts[4]
    return {'Type': 'Service', 'Region': region, 'Sector': sector, 'Service': service}


def _fuel_target(region: str, fuel: str) -> str:
    if fuel in _REGIONAL_FUELS:
        return f'CIMS.CAN.{region}.{fuel}'
    return f'CIMS.Generic Fuels.{fuel}'


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── energy demand builder ─────────────────────────────────────────────────────

def _build_cer_energy(cer_df: pd.DataFrame) -> pl.DataFrame:
    """Filter cer_resd_demand output to Transportation Freight CIMS nodes."""
    tf = cer_df[cer_df['Node'].str.startswith('.Transportation Freight')].copy()
    if tf.empty:
        return _empty_df()

    rows = []
    for _, row in tf.iterrows():
        region = str(row['Region'])
        node   = str(row['Node'])
        fuel   = str(row['Variable'])
        branch = f'CIMS.CAN.{region}{node}'
        meta   = _branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      region,
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_quantity_requested',
            'Context':     '',
            'Sub_Context': '',
            'Target':      _fuel_target(region, fuel),
            'Source':      str(row.get('Source', 'CER')),
            'Unit':        str(row.get('Unit', 'GJ')),
            'Year':        str(int(row['Year'])),
            'Value':       str(row['Value']),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── emission builders ─────────────────────────────────────────────────────────

def _build_crosswalk_emissions(crosswalk_df: pl.DataFrame) -> pl.DataFrame:
    """Filter nir_crosswalk_tables_cims output to Transportation Freight CIMS branches."""
    tf = crosswalk_df.filter(pl.col('CIMS_Branch').str.contains(r'\.Transportation Freight'))
    if tf.is_empty():
        return _empty_df()

    rows = []
    for row in tf.to_dicts():
        branch = row['CIMS_Branch']
        meta   = _branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      meta['Region'],
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_emissions_total_cumul_net',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      str(row.get('Source', 'NIR')),
            'Unit':        str(row.get('Unit', 'tCO2e')),
            'Year':        str(row['Year']),
            'Value':       str(row['Value']),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _build_nir_emissions(nir_df: pl.DataFrame) -> pl.DataFrame:
    """Extract per-gas NIR emissions for Transportation Freight branches."""
    known_regions = set(_REGION_MAP.keys())
    tf = nir_df.filter(
        pl.col('CIMS Branch').str.contains(r'\.Transportation Freight')
        & pl.col('Region').is_in(known_regions)
    )
    if tf.is_empty():
        return _empty_df()

    rows = []
    for row in tf.to_dicts():
        full_region = row['Region']
        abbr        = _REGION_MAP[full_region]
        branch      = row['CIMS Branch'].replace(
            f'CIMS.CAN.{full_region}.', f'CIMS.CAN.{abbr}.'
        )
        meta = _branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      abbr,
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_emissions_total_cumul_net',
            'Context':     str(row['Variable']),
            'Sub_Context': '',
            'Target':      '',
            'Source':      'NIR',
            'Unit':        str(row['Unit']),
            'Year':        str(row['Year']),
            'Value':       str(row['Value']),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


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
    mask = (pl.col('variable') == variable) & (pl.col('parameter') == 'calibration_market_share_total')
    if year_max is not None:
        mask = mask & (pl.col('year') <= year_max)
    data = tf.filter(mask)
    if data.is_empty():
        return _empty_df()

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
        return _empty_df()
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
    cer_rows = _build_cer_energy(cer_df)
    print(f'  Rows: {len(cer_rows):,}')

    print('Building crosswalk emission rows...')
    crosswalk_rows = _build_crosswalk_emissions(crosswalk_df)
    print(f'  Rows: {len(crosswalk_rows):,}')

    print('Building NIR annual emission rows (tCO2e via AR5 GWP100)...')
    nir_rows = _build_nir_emissions(nir_df)
    print(f'  Rows: {len(nir_rows):,}')

    print('Building technology market share rows...')
    tech_frames: list[pl.DataFrame] = []
    for variable, service_name, branch_suffix, cat_to_tech, year_max in _TECH_SHARE_SERVICES:
        frame = _build_tf_tech_shares(tf, variable, service_name, branch_suffix, cat_to_tech, year_max)
        tech_frames.append(frame)
        print(f'  {service_name}: {len(frame):,} rows')
    tech_rows = pl.concat(tech_frames, how='diagonal_relaxed') if tech_frames else _empty_df()
    print(f'  Tech share total: {len(tech_rows):,} rows')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows, tech_rows], how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    written = 0
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        if not (region_df['Value'].cast(pl.Float64, strict=False).fill_null(0) != 0).any():
            continue
        out_path = OUTPUT_DIR / f'transportation_freight_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows -> {out_path.name}')
        written += 1

    print(f'\n Transportation Freight calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {written} (one per region)')

    return output


if __name__ == '__main__':
    main()
