"""
Extract Transportation Passenger calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total from crosswalk; calibration_emissions_by_type from nir_to_cims)
    nir_crosswalk_tables_cims.py  → total tCO2e per transportation passenger CIMS branch
                                    5-year intervals (2000–2020);
                                    abbreviation regions (AB, BC, …)
    nir_to_cims.py                → per-gas kt per transportation passenger CIMS branch,
                                    annual resolution (2000–latest NIR year);
                                    summed to tCO2e using AR5 GWP100 factors;
                                    full province names mapped to abbreviations

Energy demand  (calibration_quantity_requested)
    cer_resd_demand.py            → energy demand in PJ by fuel and CIMS node;
                                    abbreviation regions

Technology market shares  (calibration_market_share_new)
    transportation_passenger.py   → CEUD-derived market shares (2000–last CEUD year)
                                    for Urban, Intercity Land, Intercity Air modes;
                                    Passenger Vehicles, Passenger Vehicle Motors;
                                    Transit Public Bus, Intercity Bus, Intercity Rail
    stats_can/passenger_transportation.py
                                  → Build Passenger Vehicle Motors calibration_market_share_new 
                                  rows from StatCan vehicle sales and EPA engine-package data.
                                  → StatCan vehicle sales provides total annual vehicle sales by fuel type.
                                  → EPA table_export.csv provides U.S. production data
                                    used to calculate Low, Medium, and High efficiency market shares
                                    for gasoline vehicles. Efficiency shares are calculated by:
                                    1. Mapping engine packages to Low, Medium, and High efficiency classes.
                                    2. Calculating annual market shares for the three efficiency classes.
                                    4. Applying a centered 3-year moving average smoothing method.
                                    Given lack of data, gasoline shares as used as a proxy for diesel shares.
                                    More detail on method can be found in Renate's "Gasoline Market Shares" workbook

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import sys
import ast
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
_tp_mod = _load_module(
    'transportation_passenger',
    _PIPELINE_ROOT / 'source/nrcan/ceud/transportation_passenger/transportation_passenger.py',
)
_statcan_tp_mod = _load_module(
    'stats_can_passenger_transportation',
    _PIPELINE_ROOT / 'source/stats_can/passenger_transportation.py',
)

from utils.controls_conversions import BASE_PATH, load_sector_regions, filter_excluded_branches

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/transportation_passenger'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

SECTOR_NAME = 'Transportation Passenger'

STATCAN_MARKET_SHARE_START_YEAR = 2000
STATCAN_MARKET_SHARE_SERVICE = 'Passenger Vehicle Motors'
STATCAN_MARKET_SHARE_BRANCH_SUFFIX = '.Passenger Vehicle Motors'
STATCAN_MARKET_SHARE_SOURCE = 'StatCan/EPA'

EPA_FILE = Path (
r"C:\cims\data\raw_data\epa\table_export.csv"
)


# EPA engine-package mapping used to split gasoline into efficiency classes.
# Diesel is assigned the same Low / Medium / High shares as gasoline.
EPA_ENGINE_PACKAGE_TO_EFFICIENCY = {
    'Carb, Fixed Valve Timing, Two-Valve': 'Low',
    'Carb, Fixed Valve Timing, Multi-Valve': 'Low',
    'TBI, Fixed Valve Timing, Two-Valve': 'Low',
    'TBI, Fixed Valve Timing, Multi-Valve': 'Low',
    'Port, Fixed Valve Timing, Two-Valve': 'Low',
    'Port, Fixed Valve Timing, Multi-Valve': 'Low',
    'GDPI, Fixed Valve Timing, Multi-Valve': 'Medium',
    'GDPI, Variable Valve Timing, Multi-Valve': 'Medium',
    'GDI, Fixed Valve Timing, Multi-Valve': 'Medium',
    'Port, Variable Valve Timing, Multi-Valve': 'Medium',
    'Port, Variable Valve Timing, Two-Valve': 'Medium',
    'GDI, Variable Valve Timing, Multi-Valve': 'High',
    'GDI, Variable Valve Timing, Two-Valve': 'High',
}

EPA_EXCLUDED_ENGINE_PACKAGES = {'PHEV', 'All', 'Diesel', 'BEV'}
EFFICIENCY_LEVELS = ('Low', 'Medium', 'High')

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

# Pipeline category → CIMS Technology for Urban mode tech shares
_URBAN_CAT_TO_TECH: dict[str, str] = {
    'Walk Cycle':            'Walk Cycle Urban',
    'Passenger Vehicle SOV': 'Passenger Vehicle Urban SOV',
    'Passenger Vehicle HOV': 'Passenger Vehicle Urban HOV',
    'Public Transit':        'Public Transit Urban',
}

# Pipeline category → CIMS Technology for Intercity Land mode tech shares
_INTERCITY_LAND_CAT_TO_TECH: dict[str, str] = {
    'Bus Intercity':     'Bus Intercity',
    'Rail Intercity':    'Rail Intercity',
    'Passenger Vehicle': 'Passenger Vehicle Intercity',
}

# (pipeline variable, CIMS service name, branch suffix after sector, category mapping, year_max)
# year_max=None → all historical years; year_max=N → years up to and including N
_TECH_SHARE_SERVICES: list[tuple[str, str, str, dict[str, str] | None, int | None]] = [
    ('Mode.Urban',               'Urban',                    '.Mode.Urban',               _URBAN_CAT_TO_TECH,          None),
    ('Mode.Intercity Land',      'Intercity Land',           '.Mode.Intercity Land',      _INTERCITY_LAND_CAT_TO_TECH, None),
    ('Mode.Intercity Air',       'Intercity Air',            '.Mode.Intercity Air',       None,                        2000),
    ('Passenger Vehicles',       'Passenger Vehicles',       '.Passenger Vehicles',       None,                        None),
    ('Passenger Vehicle Motors', 'Passenger Vehicle Motors', '.Passenger Vehicle Motors', None,                        2000),
    ('Transit.Public Bus',       'Public Bus',               '.Transit.Public Bus',       None,                        None),
    ('Intercity Bus',            'Intercity Bus',            '.Intercity Bus',            None,                        None),
    ('Intercity Rail',           'Intercity Rail',           '.Intercity Rail',           None,                        2000),
]


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


def _load_controls() -> dict:
    """Read CONTROLS from control.py without executing the Marimo app."""
    candidates = [
        _PIPELINE_ROOT / 'control.py',
        _PIPELINE_ROOT / 'utils/control.py',
        _PIPELINE_ROOT / 'source/control.py',
        _PIPELINE_ROOT / 'source/utils/control.py',
        Path(__file__).with_name('control.py'),
        Path.cwd() / 'control.py',
    ]

    # Also search upward from this file and the current working directory. This
    # handles running the calibration script from a different folder than the
    # repository root.
    for base in [Path(__file__).resolve(), Path.cwd().resolve()]:
        for parent in [base.parent, *base.parents]:
            candidates.append(parent / 'control.py')

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'CONTROLS':
                        return ast.literal_eval(node.value)
    searched = ', '.join(str(p) for p in seen) or 'no existing candidate paths'
    raise FileNotFoundError(
        f'Could not find a control.py file containing CONTROLS. Searched: {searched}'
    )


def _last_data_year(source_name: str, default: int) -> int:
    """Return last_data_year[source_name] from control.py, falling back to default."""
    try:
        controls = _load_controls()
        return int(controls.get('last_data_year', {}).get(source_name, default))
    except Exception as exc:
        print(f'  Warning: could not read control.py for {source_name!r}; using {default}. ({exc})')
        return int(default)


def _region_abbr(region: str) -> str:
    """Convert full province/territory names from StatCan to CIMS abbreviations."""
    if region in set(_REGION_MAP.values()):
        return region
    return _REGION_MAP.get(region, region)


# ── energy demand builder ─────────────────────────────────────────────────────

def _build_cer_energy(cer_df: pd.DataFrame) -> pl.DataFrame:
    """Filter cer_resd_demand output to Transportation Passenger CIMS nodes."""
    tp = cer_df[cer_df['Node'].str.startswith('.Transportation Passenger')].copy()
    if tp.empty:
        return _empty_df()

    rows = []
    for _, row in tp.iterrows():
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
    """Filter nir_crosswalk_tables_cims output to Transportation Passenger CIMS branches."""
    tp = crosswalk_df.filter(pl.col('CIMS_Branch').str.contains(r'\.Transportation Passenger'))
    if tp.is_empty():
        return _empty_df()

    rows = []
    for row in tp.to_dicts():
        branch = row['CIMS_Branch']
        meta   = _branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      meta['Region'],
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_emissions_total',
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
    """Extract per-gas NIR emissions for Transportation Passenger branches."""
    known_regions = set(_REGION_MAP.keys())
    tp = nir_df.filter(
        pl.col('CIMS Branch').str.contains(r'\.Transportation Passenger')
        & pl.col('Region').is_in(known_regions)
    )
    if tp.is_empty():
        return _empty_df()

    rows = []
    for row in tp.to_dicts():
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
            'Parameter':   'calibration_emissions_by_type',
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

def _build_tp_tech_shares(
    tp: pl.DataFrame,
    variable: str,
    service_name: str,
    branch_suffix: str,
    cat_to_tech: dict[str, str] | None = None,
    year_max: int | None = None,
) -> pl.DataFrame:
    """Extract calibration_market_share_new rows for one transportation passenger service.

    year_max=None outputs all historical years; year_max=N caps at year N inclusive.
    """
    mask = (pl.col('variable') == variable) & (pl.col('parameter') == 'calibration_market_share_new')
    if year_max is not None:
        mask = mask & (pl.col('year') <= year_max)
    data = tp.filter(mask)
    if data.is_empty():
        return _empty_df()

    rows: list[dict] = []
    for r in data.iter_rows(named=True):
        region   = r['province']
        category = r['category']
        tech     = cat_to_tech.get(category, category) if cat_to_tech else category
        branch   = f'CIMS.CAN.{region}.Transportation Passenger{branch_suffix}'
        rows.append({
            'Branch':      branch,
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Passenger',
            'Service':     service_name,
            'Technology':  tech,
            'Parameter':   'calibration_market_share_new',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      'CEUD',
            'Unit':        '%',
            'Year':        str(r['year']),
            'Value':       str(r['value']),
        })
    if not rows:
        return _empty_df()
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    """Return the first matching column, ignoring case and surrounding spaces."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f'Could not find {label} column. Expected one of {candidates}; '
        f'available columns are {list(df.columns)}.'
    )


def _load_efficiency_shares(epa_file: Path | str, epa_last_year: int) -> pd.DataFrame:
    """Calculate smoothed Low/Medium/High gasoline shares from EPA engine packages.

    The method mirrors the workbook Shares tab:
      1. sum production by year and efficiency class;
      2. divide each class by total mapped gasoline production;
      3. keep the first year unsmoothed;
      4. use a centred three-year average for interior years; and
      5. average the previous/current values for the final year.

    Diesel later receives the same smoothed efficiency shares.
    """
    epa_file = Path(epa_file)
    if epa_file.suffix.lower() in {'.xlsx', '.xlsm', '.xls'}:
        try:
            epa = pd.read_excel(epa_file, sheet_name='EPA data')
        except ValueError:
            epa = pd.read_excel(epa_file)
    else:
        epa = pd.read_csv(epa_file)

    engine_col = _find_column(
        epa,
        ('Engine Package', 'engine_package', 'engine package'),
        'EPA engine package',
    )
    year_col = _find_column(
        epa,
        ('Model Year', 'year', 'model_year', 'model year'),
        'EPA model year',
    )
    production_col = _find_column(
        epa,
        ('Production (000)', 'production', 'production_000', 'Production Share'),
        'EPA production',
    )

    epa = epa[[engine_col, year_col, production_col]].copy()
    epa['engine_package'] = epa[engine_col].astype(str).str.strip()
    epa['year'] = pd.to_numeric(epa[year_col], errors='coerce')
    epa['production'] = pd.to_numeric(
        epa[production_col].replace({'-': None, '—': None, '': None}),
        errors='coerce',
    )
    epa['efficiency'] = epa['engine_package'].map(EPA_ENGINE_PACKAGE_TO_EFFICIENCY)

    unknown = sorted(
        set(epa.loc[epa['efficiency'].isna(), 'engine_package'].dropna())
        - EPA_EXCLUDED_ENGINE_PACKAGES
        - {'nan', 'None'}
    )
    if unknown:
        print(
            '  Warning: excluding unmapped EPA engine packages: '
            + ', '.join(unknown)
        )

    mapped = epa[
        epa['efficiency'].notna()
        & epa['year'].notna()
        & epa['production'].notna()
        & (epa['year'] <= epa_last_year)
    ].copy()
    if mapped.empty:
        raise ValueError('No mapped gasoline EPA production data were found.')

    production = (
        mapped.groupby(['year', 'efficiency'], as_index=False)['production'].sum()
        .pivot(index='year', columns='efficiency', values='production')
        .reindex(columns=EFFICIENCY_LEVELS, fill_value=0.0)
        .sort_index()
        .fillna(0.0)
    )
    totals = production.sum(axis=1)
    if (totals <= 0).any():
        bad_years = production.index[totals <= 0].astype(int).tolist()
        raise ValueError(f'EPA mapped gasoline production is zero for years {bad_years}.')

    raw_shares = production.div(totals, axis=0)
    smoothed = raw_shares.rolling(window=3, center=True, min_periods=1).mean()
    smoothed.iloc[0] = raw_shares.iloc[0]  # Shares tab leaves the first year unchanged.
    smoothed = smoothed.div(smoothed.sum(axis=1), axis=0)

    return (
        smoothed.rename_axis(index='year', columns='efficiency')
        .reset_index()
        .assign(year=lambda x: x['year'].astype(int))
    )


def _expand_efficiency_technologies(
    annual: pd.DataFrame,
    efficiency_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Split gasoline and diesel rows into Low/Medium/High technologies."""
    year_col = _find_column(annual, ('year', 'Year'), 'StatCan year')
    fuel_col = _find_column(
        annual,
        ('fuel_type', 'fuel type', 'Fuel Type', 'fuel'),
        'StatCan fuel type',
    )

    shares_long = efficiency_shares.melt(
        id_vars='year',
        value_vars=list(EFFICIENCY_LEVELS),
        var_name='efficiency',
        value_name='_efficiency_share',
    )
    out = annual.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors='coerce').astype('Int64')
    out['_base_fuel'] = out[fuel_col].astype(str).str.strip().str.lower()

    split_mask = out['_base_fuel'].isin({'gasoline', 'diesel'})
    split = out.loc[split_mask].merge(
        shares_long,
        left_on=year_col,
        right_on='year',
        how='inner',
        suffixes=('', '_share'),
    )
    unsplit = out.loc[~split_mask].copy()

    # Detect the additive sales/registration field used by calculate_market_shares().
    measure_candidates = (
        'annual_sales', 'sales', 'registrations', 'registration',
        'vehicles', 'count', 'value',
    )
    measure_col = _find_column(out, measure_candidates, 'StatCan sales/registration')
    split[measure_col] = (
        pd.to_numeric(split[measure_col], errors='coerce')
        * split['_efficiency_share']
    )
    split[fuel_col] = (
        split['_base_fuel'].str.title()
        + '_'
        + split['efficiency']
        + ' Efficiency'
    )

    helper_cols = ['_base_fuel', '_efficiency_share', 'efficiency', 'year_share']
    split = split.drop(columns=[c for c in helper_cols if c in split.columns])
    unsplit = unsplit.drop(columns=['_base_fuel'])
    split = split.reindex(columns=unsplit.columns)
    return pd.concat([unsplit, split], ignore_index=True)


def _build_statcan_vehicle_motor_shares() -> pl.DataFrame:
    """Build Passenger Vehicle Motors market shares from StatCan and EPA data.

    EPA engine packages determine gasoline Low/Medium/High efficiency shares.
    The gasoline shares are smoothed using the Shares-tab method and then applied
    identically to diesel. Other StatCan fuel technologies remain unchanged.
    """
    statcan_last_year = _last_data_year(
        'stat_can_market_shares',
        getattr(_statcan_tp_mod, 'LAST_OBSERVED_YEAR', 2025),
    )
    epa_last_year = _last_data_year(
        'epa',
        getattr(_statcan_tp_mod, 'LAST_OBSERVED_YEAR', 2025),
    )

    _statcan_tp_mod.LAST_OBSERVED_YEAR = statcan_last_year
    annual = _statcan_tp_mod.read_statscan_vehicle_sales(
        _statcan_tp_mod.DEFAULT_STATCAN_FILE,
        first_observed_year=_statcan_tp_mod.FIRST_OBSERVED_YEAR,
        last_observed_year=statcan_last_year,
    )
    annual = _statcan_tp_mod.apply_region_proxies(annual, _statcan_tp_mod.PROXY_REGIONS)
    annual = _statcan_tp_mod.add_backcast_history(annual)

    EPA_FILE = r"C:\cims\data\raw_data\epa\table_export.csv"

    efficiency_shares = _load_efficiency_shares(
        EPA_FILE,
        epa_last_year,
    )
    tech_sales = _expand_efficiency_technologies(annual, efficiency_shares)
    market_shares = _statcan_tp_mod.calculate_market_shares(tech_sales)
    market_shares = market_shares[
        market_shares['year'].between(STATCAN_MARKET_SHARE_START_YEAR, statcan_last_year)
    ].copy()
    if market_shares.empty:
        return _empty_df()

    rows: list[dict] = []
    for r in market_shares.to_dict('records'):
        region = _region_abbr(str(r['region']))
        branch = f'CIMS.CAN.{region}.Transportation Passenger{STATCAN_MARKET_SHARE_BRANCH_SUFFIX}'
        rows.append({
            'Branch':      branch,
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Passenger',
            'Service':     STATCAN_MARKET_SHARE_SERVICE,
            'Technology':  str(r['fuel_type']),
            'Parameter':   'calibration_market_share_new',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      STATCAN_MARKET_SHARE_SOURCE,
            'Unit':        '%',
            'Year':        str(int(r['year'])),
            'Value':       f"{float(r['market_share']):.12g}",
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> pl.DataFrame:
    """Assemble transportation passenger calibration data and write one CSV per region."""
    print('=' * 60)
    print('TRANSPORTATION PASSENGER CALIBRATION')
    print('=' * 60)

    print('\nRunning NIR crosswalk (nir_crosswalk_tables_cims)...')
    crosswalk_df = pl.from_pandas(_crosswalk_mod.main())

    print('\nRunning NIR to CIMS (nir_to_cims)...')
    nir_df = _nir_mod.main()

    print('\nRunning CER demand (cer_resd_demand)...')
    cer_df = _cer_mod.main()

    print('\nRunning transportation passenger pipeline (CEUD)...')
    tp_results = _tp_mod.main(export_csv=False)
    tp = (
        pl.concat(list(tp_results.values()), how='diagonal_relaxed')
        .filter(pl.col('year') <= _tp_mod.LAST_HIST_YEAR)
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
        frame = _build_tp_tech_shares(tp, variable, service_name, branch_suffix, cat_to_tech, year_max)
        tech_frames.append(frame)
        print(f'  {service_name}: {len(frame):,} rows')
    tech_rows = pl.concat(tech_frames, how='diagonal_relaxed') if tech_frames else _empty_df()
    print(f'  CEUD tech share total: {len(tech_rows):,} rows')

    print('Building StatCan/EPA Passenger Vehicle Motors market share rows...')
    statcan_market_share_rows = _build_statcan_vehicle_motor_shares()
    print(f'  StatCan/EPA Passenger Vehicle Motors: {len(statcan_market_share_rows):,} rows')
    statcan_market_share_end_year = _last_data_year(
        'stat_can_market_shares',
        getattr(_statcan_tp_mod, 'LAST_OBSERVED_YEAR', 2025),
    )

    # Replace overlapping CEUD Passenger Vehicle Motors rows from 2000 through
    # the latest StatCan market-share year with the more detailed StatCan/EPA
    # split so the output does not contain duplicate calibration_market_share_new records
    # for the same branch/technology/year.
    tech_rows = tech_rows.filter(
        ~(
            (pl.col('Service') == STATCAN_MARKET_SHARE_SERVICE)
            & (
                pl.col('Year')
                .cast(pl.Int64, strict=False)
                .is_between(STATCAN_MARKET_SHARE_START_YEAR, statcan_market_share_end_year)
            )
        )
    )
    tech_rows = pl.concat([tech_rows, statcan_market_share_rows], how='diagonal_relaxed')
    print(f'  Tech share total after StatCan/EPA merge: {len(tech_rows):,} rows')

    print('Combining...')
    output = (
        pl.concat([cer_rows, crosswalk_rows, nir_rows, tech_rows], how='diagonal_relaxed')
        .select(OUTPUT_COLS)
    )

    print('Filtering to regions with Transportation Passenger (see sector_region_map.csv)...')
    allowed_regions = load_sector_regions().get(SECTOR_NAME)
    if allowed_regions:
        before_count = len(output)
        output = output.filter(pl.col('Region').is_in(list(allowed_regions)))
        dropped_count = before_count - len(output)
        if dropped_count:
            print(f'  Dropped {dropped_count:,} rows for regions without Transportation Passenger')

    output = filter_excluded_branches(output)

    # Final output normalization:
    # - Remove any leading apostrophes that can make numeric-looking values
    #   appear as text markers in the generated calibration CSVs.
    # - Force every calibration_market_share_new row to use Unit = '%', including rows
    #   that originate from upstream sources with Unit values like 'fraction'.
    output = output.with_columns(
        pl.col(pl.Utf8).str.replace(r"^'", "")
    ).with_columns(
        pl.when(pl.col('Parameter') == 'calibration_market_share_new')
        .then(pl.lit('%'))
        .otherwise(pl.col('Unit'))
        .alias('Unit')
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = output['Region'].drop_nulls().unique().sort().to_list()
    for region in regions:
        region_df = output.filter(pl.col('Region') == region)
        if not (region_df['Value'].cast(pl.Float64, strict=False).fill_null(0) != 0).any():
            continue
        out_path = OUTPUT_DIR / f'transportation_passenger_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')

    print(f'\n✅ Transportation Passenger calibration complete')
    print(f'   Total rows:  {len(output):,}')
    print(f'   Files:       {len(regions)} (one per region)')

    return output


if __name__ == '__main__':
    main()
