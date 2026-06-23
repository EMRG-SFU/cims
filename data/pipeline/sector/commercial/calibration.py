"""
Extract commercial calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total_cumul_net)
    nir_crosswalk_tables_cims.py  → total tCO2e per commercial CIMS branch
                                    5-year intervals (2000–2020);
                                    abbreviation regions (AB, BC, …)
    nir_to_cims.py                → per-gas kt per commercial CIMS branch,
                                    annual resolution (2000–latest NIR year);
                                    summed to tCO2e using AR5 GWP100 factors;
                                    full province names mapped to abbreviations

Energy demand  (calibration_quantity_requested)
    cer_resd_demand.py            → energy demand in PJ by fuel and CIMS node;
                                    abbreviation regions

Hot water and HVAC technologies  (market_share_total)
    commercial.py                 → CEUD-derived market shares with projections;
                                    one DataFrame per region/province/territory;
                                    BC exports both Marine and Cold HVAC data

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

from source.nrcan.ceud.commercial.commercial import main as _ceud_main
from utils.controls_conversions import BASE_PATH

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/commercial'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# NIR full province name → CIMS abbreviation (excludes Canada)
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

# Fuels that have region-specific CIMS branches
_REGIONAL_FUELS = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _branch_meta(branch: str) -> dict:
    """Infer Type, Region, Sector, Service from a CIMS branch string.

    Branch structure: CIMS.CAN.{Region}[.{Sector}[.{Service}[...]]]
    """
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
    """Build CIMS branch for a fuel (mirrors model_inputs.py price_mult logic)."""
    if fuel in _REGIONAL_FUELS:
        return f'CIMS.CAN.{region}.{fuel}'
    return f'CIMS.Generic Fuels.{fuel}'


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _get_series(df: pl.DataFrame, variable: str, category: str = '') -> dict:
    """Extract {year: value} from a long-format Polars DataFrame."""
    mask = pl.col('variable') == variable
    if category:
        mask = mask & (pl.col('category') == category)
    subset = df.filter(mask)
    if len(subset) == 0:
        return {}
    years  = subset.get_column('year').cast(pl.Int64).to_list()
    values = subset.get_column('value').cast(pl.Float64).to_list()
    return {int(y): float(v) for y, v in zip(years, values) if v is not None}


def _get_categories(df: pl.DataFrame, variable: str) -> list[str]:
    """Return sorted unique category values for a given variable."""
    subset = df.filter(pl.col('variable') == variable)
    return sorted(subset.get_column('category').unique().to_list())


# ── energy demand builder ─────────────────────────────────────────────────────

def _build_cer_energy(cer_df: pd.DataFrame) -> pl.DataFrame:
    """Filter cer_resd_demand output to Commercial CIMS nodes."""
    com = cer_df[cer_df['Node'].str.startswith('.Commercial')].copy()
    if com.empty:
        return _empty_df()

    rows = []
    for _, row in com.iterrows():
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
    """Filter nir_crosswalk_tables_cims output to Commercial CIMS branches."""
    com = crosswalk_df.filter(pl.col('CIMS_Branch').str.contains(r'\.Commercial'))
    if com.is_empty():
        return _empty_df()

    rows = []
    for row in com.to_dicts():
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
    """Extract per-gas NIR emissions for Commercial branches."""
    known_regions = set(_REGION_MAP.keys())
    com = nir_df.filter(
        pl.col('CIMS Branch').str.contains(r'\.Commercial')
        & pl.col('Region').is_in(known_regions)
    )
    if com.is_empty():
        return _empty_df()

    rows = []
    for row in com.to_dicts():
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


# ── technology builders ───────────────────────────────────────────────────────

def _build_hot_water_techs(ceud_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Extract hot water technology market shares from CEUD data."""
    rows: list[dict] = []

    for region_code, df in ceud_results.items():
        region = region_code.upper()

        for tech in _get_categories(df, 'hot_water_tech'):
            for year, value in _get_series(df, 'hot_water_tech', tech).items():
                rows.append({
                    'Branch':      f'CIMS.CAN.{region}.Commercial.Buildings.Hot Water',
                    'Type':        'Service',
                    'Region':      region,
                    'Sector':      'Commercial',
                    'Service':     'Hot Water',
                    'Technology':  tech,
                    'Parameter':   'market_share_total',
                    'Context':     '',
                    'Sub_Context': '',
                    'Target':      '',
                    'Source':      'CEUD',
                    'Unit':        '%',
                    'Year':        str(year),
                    'Value':       str(value),
                })

    if not rows:
        return _empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _build_hvac_techs(ceud_results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Extract HVAC technology market shares from CEUD data.

    BC exports both Marine and Cold climate; all other regions export Cold only.
    """
    rows: list[dict] = []

    for region_code, df in ceud_results.items():
        region = region_code.upper()
        is_bc  = region == 'BC'

        climates = (
            [('hvac_marine', '(Marine)'), ('hvac_cold', '(Cold)')]
            if is_bc else
            [('hvac_cold', '(Cold)')]
        )

        for climate_var, climate_label in climates:
            for tech in _get_categories(df, climate_var):
                for year, value in _get_series(df, climate_var, tech).items():
                    rows.append({
                        'Branch':      f'CIMS.CAN.{region}.Commercial.HVAC {climate_label}',
                        'Type':        'Service',
                        'Region':      region,
                        'Sector':      'Commercial',
                        'Service':     'HVAC',
                        'Technology':  tech,
                        'Parameter':   'market_share_total',
                        'Context':     '',
                        'Sub_Context': '',
                        'Target':      '',
                        'Source':      'CEUD',
                        'Unit':        '%',
                        'Year':        str(year),
                        'Value':       str(value),
                    })

    if not rows:
        return _empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble commercial calibration data and write one CSV per region."""
    print('=' * 60)
    print('COMMERCIAL CALIBRATION')
    print('=' * 60)

    print('\nRunning NIR crosswalk (nir_crosswalk_tables_cims)...')
    crosswalk_df = pl.from_pandas(_crosswalk_mod.main())

    print('\nRunning NIR to CIMS (nir_to_cims)...')
    nir_df = _nir_mod.main()

    print('\nRunning CER demand (cer_resd_demand)...')
    cer_df = _cer_mod.main()

    print('\nRunning CEUD commercial pipeline (with projections)...')
    ceud_results = _ceud_main(apply_projections=True, export_csv=False)

    print('\nBuilding CER energy demand rows...')
    cer_rows = _build_cer_energy(cer_df)
    print(f'  Rows: {len(cer_rows):,}')

    print('Building crosswalk emission rows...')
    crosswalk_rows = _build_crosswalk_emissions(crosswalk_df)
    print(f'  Rows: {len(crosswalk_rows):,}')

    print('Building NIR annual emission rows (tCO2e via AR5 GWP100)...')
    nir_rows = _build_nir_emissions(nir_df)
    print(f'  Rows: {len(nir_rows):,}')

    print('Building hot water technology rows...')
    hw_rows = _build_hot_water_techs(ceud_results)
    print(f'  Rows: {len(hw_rows):,}')

    print('Building HVAC technology rows...')
    hvac_rows = _build_hvac_techs(ceud_results)
    print(f'  Rows: {len(hvac_rows):,}')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows, hw_rows, hvac_rows],
                  how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        if not (region_df['Value'].cast(pl.Float64, strict=False).fill_null(0) != 0).any():
            continue
        out_path  = OUTPUT_DIR / f'commercial_{region}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Commercial calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
