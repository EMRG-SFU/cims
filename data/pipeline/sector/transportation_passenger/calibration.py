"""
Extract Transportation Passenger calibration data and save to CIMS-formatted CSV files.

Sources
-------
Emissions  (calibration_emissions_total_cumul_net)
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

Technology market shares  (market_share_total)
    transportation_passenger.py   → CEUD-derived market shares (2000–last CEUD year)
                                    for Urban, Intercity Land, Intercity Air modes;
                                    Passenger Vehicles, Passenger Vehicle Motors;
                                    Transit Public Bus, Intercity Bus, Intercity Rail
    stats_can/passenger_transportation.py
                                  → StatCan/EPA-derived Passenger Vehicle Motors
                                    market shares for 2000–latest StatCan market share year

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

from utils.controls_conversions import BASE_PATH

# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_PATH / 'calibration/transportation_passenger'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

STATCAN_MARKET_SHARE_START_YEAR = 2000
STATCAN_MARKET_SHARE_SERVICE = 'Passenger Vehicle Motors'
STATCAN_MARKET_SHARE_BRANCH_SUFFIX = '.Passenger Vehicle Motors'
STATCAN_MARKET_SHARE_SOURCE = 'StatCan/EPA'

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

def _build_tp_tech_shares(
    tp: pl.DataFrame,
    variable: str,
    service_name: str,
    branch_suffix: str,
    cat_to_tech: dict[str, str] | None = None,
    year_max: int | None = None,
) -> pl.DataFrame:
    """Extract market_share_total rows for one transportation passenger service.

    year_max=None outputs all historical years; year_max=N caps at year N inclusive.
    """
    mask = (pl.col('variable') == variable) & (pl.col('parameter') == 'market_share_total')
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
            'Parameter':   'market_share_total',
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


def _build_statcan_vehicle_motor_shares() -> pl.DataFrame:
    """Build Passenger Vehicle Motors market_share_total rows from StatCan/EPA data.

    The StatCan source provides vehicle registrations by fuel type; the EPA source
    provides gasoline/diesel standard vs. efficient splits. control.py provides the
    latest data years through last_data_year['stat_can_market_shares'] and
    last_data_year['epa'].
    """
    statcan_last_year = _last_data_year(
        'stat_can_market_shares',
        getattr(_statcan_tp_mod, 'LAST_OBSERVED_YEAR', 2025),
    )
    epa_last_year = _last_data_year(
        'epa',
        getattr(_statcan_tp_mod, 'LAST_OBSERVED_YEAR', 2025),
    )

    # The source module uses LAST_OBSERVED_YEAR inside add_backcast_history(), so
    # set it from control.py before generating historical/backcast rows.
    _statcan_tp_mod.LAST_OBSERVED_YEAR = statcan_last_year

    annual = _statcan_tp_mod.read_statscan_vehicle_sales(
        _statcan_tp_mod.DEFAULT_STATCAN_FILE,
        first_observed_year=_statcan_tp_mod.FIRST_OBSERVED_YEAR,
        last_observed_year=statcan_last_year,
    )
    annual = _statcan_tp_mod.apply_region_proxies(annual, _statcan_tp_mod.PROXY_REGIONS)
    annual = _statcan_tp_mod.add_backcast_history(annual)

    gas_diesel_shares = _statcan_tp_mod.load_gasoline_technology_shares(
        _statcan_tp_mod.DEFAULT_EPA_FILE
    )
    gas_diesel_shares = gas_diesel_shares[gas_diesel_shares['year'] <= epa_last_year]
    if gas_diesel_shares.empty:
        raise ValueError('No EPA gasoline/diesel technology shares available after applying control.py last_data_year["epa"].')

    tech_sales = _statcan_tp_mod.expand_vehicle_technologies(annual, gas_diesel_shares)
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
            'Parameter':   'market_share_total',
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
    # split so the output does not contain duplicate market_share_total records
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

    # Final output normalization:
    # - Remove any leading apostrophes that can make numeric-looking values
    #   appear as text markers in the generated calibration CSVs.
    # - Force every market_share_total row to use Unit = '%', including rows
    #   that originate from upstream sources with Unit values like 'fraction'.
    output = output.with_columns(
        pl.col(pl.Utf8).str.replace(r"^'", "")
    ).with_columns(
        pl.when(pl.col('Parameter') == 'market_share_total')
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
