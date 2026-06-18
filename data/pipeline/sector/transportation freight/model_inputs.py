"""
Transportation Freight Pipeline — Model Inputs

Combines fixed structural parameters with CEUD pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Transportation Freight/transportation freight_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.

Total k*tkm  (service_request rows, Region level)
    pipeline → variable = 'total_ktkm'
    Placed before the Transportation Freight sector block.

Energy price multipliers  (multiplier_price rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Inserted after the sector header (service_provide / competition).

Off-Road service_request  (% of k*tkm)
    pipeline → variable = 'Freight.Off-Road'
    Inserted after the sector's fixed service_request to the Freight sub-node.

Freight mode service_requests  (Land / Marine / Air shares)
    pipeline → variables 'Freight.Land', 'Freight.Marine', 'Freight.Air'
    Inserted after the Freight service competition row.

Land sub-service service_requests  (Light Medium / Heavy shares)
    pipeline → variables 'Freight.Land.Light Medium', 'Freight.Land.Heavy'
    Inserted after the Land service competition row.

Heavy service Trucks/Rail market_share_total  (year 2000)
    pipeline → variable 'Freight.Land.Heavy', parameter='market_share_total'
    Inserted after the Heavy service's Trucks and Rail service_request rows.

Light Medium tech market_share_total  (year 2000)
    pipeline → variable 'Light Medium', fuel categories
    Spliced after the 'lifetime' row for each LM technology.

Light Medium tech output  (all years)
    pipeline → variable 'Light Medium', category='' (fleet-average per-vehicle)
    Duplicated for every LM technology; placed at lifetime + 0.6.

Trucks tech output  (all years)
    pipeline → variable 'Heavy Trucks', category='' (fleet-average per-vehicle)
    Duplicated for every Trucks technology; placed after its fixed market_share_total.

Rail / Marine / Air / Off Road outputs
    Already present in fixed data — not modified.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
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

_flatten_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_flatten_spec)
_flatten_spec.loader.exec_module(_flatten_mod)

_tf_spec = importlib.util.spec_from_file_location(
    'transportation_freight',
    _PIPELINE_ROOT / 'source' / 'nrcan' / 'ceud' / 'transportation_freight'
    / 'transportation_freight.py',
)
_tf_mod = importlib.util.module_from_spec(_tf_spec)
_tf_spec.loader.exec_module(_tf_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Transportation Freight'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/transportation freight'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

FIXED_TEMPLATE: dict[str, str] = {
    'AB': 'AB', 'BC': 'BC', 'MB': 'MB', 'NB': 'NB', 'NL': 'NL',
    'NS': 'NS', 'NT': 'NT', 'NU': 'NU', 'ON': 'ON', 'PE': 'PE',
    'QC': 'QC', 'SK': 'SK', 'YT': 'YT',
}

REGION_SPECIFIC_ENERGIES: set[str] = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}

# Pipeline fuel category → LM technology representing the 2000 installed base.
# Technologies absent from this map receive market_share_total = 0.
LM_CAT_TO_TECH: dict[str, str] = {
    'Diesel':   'Diesel Existing',
    'Gasoline': 'Gasoline Existing',
    'Propane':  'Propane',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    """Flatten one fixed transportation freight CSV and return as a row-indexed DataFrame."""
    fixed_path = FIXED_INPUT_DIR / f'transportation freight_{region}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'transportation freight_{region}.csv'
        _flatten_mod.process_file(
            input_path=fixed_path,
            output_path=out_file,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR['cer'],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)
    return df.with_row_index('_order')


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})


def _find_max_order(df: pl.DataFrame, service: str, parameter: str) -> float | None:
    """Return the max _order value for rows matching service + parameter."""
    mask = (
        (pl.col('Service').fill_null('') == service) &
        (pl.col('Parameter').fill_null('') == parameter)
    )
    subset = df.filter(mask)
    return float(subset['_order'].max()) if len(subset) > 0 else None


def _tech_lifetime_orders(fixed: pl.DataFrame, service_name: str) -> dict[str, float]:
    """Return dict mapping technology name → max _order of its lifetime row in service."""
    mask = (
        (pl.col('Service').fill_null('') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology').fill_null('') != '') &
        (pl.col('Parameter').fill_null('') == 'lifetime')
    )
    result: dict[str, float] = {}
    for r in fixed.filter(mask).select(['Technology', '_order']).iter_rows(named=True):
        t, o = r['Technology'], float(r['_order'])
        if t not in result or o > result[t]:
            result[t] = o
    return result


def _techs_with_existing_output(fixed: pl.DataFrame, service_name: str) -> set[str]:
    """Return technology names that already have an output row in the fixed data."""
    mask = (
        (pl.col('Service').fill_null('') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology').fill_null('') != '') &
        (pl.col('Parameter').fill_null('') == 'output')
    )
    return set(fixed.filter(mask)['Technology'].unique().to_list())


def _tech_mst_orders(fixed: pl.DataFrame, service_name: str) -> dict[str, float]:
    """Return dict mapping technology name → max _order of its market_share_total row."""
    mask = (
        (pl.col('Service').fill_null('') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology').fill_null('') != '') &
        (pl.col('Parameter').fill_null('') == 'market_share_total')
    )
    result: dict[str, float] = {}
    for r in fixed.filter(mask).select(['Technology', '_order']).iter_rows(named=True):
        t, o = r['Technology'], float(r['_order'])
        if t not in result or o > result[t]:
            result[t] = o
    return result


def _tech_param_orders(
    fixed: pl.DataFrame, service_name: str, parameter: str
) -> dict[str, float]:
    """Return dict mapping technology name → max _order for a given service+parameter."""
    mask = (
        (pl.col('Service').fill_null('') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology').fill_null('') != '') &
        (pl.col('Parameter').fill_null('') == parameter)
    )
    result: dict[str, float] = {}
    for r in fixed.filter(mask).select(['Technology', '_order']).iter_rows(named=True):
        t, o = r['Technology'], float(r['_order'])
        if t not in result or o > result[t]:
            result[t] = o
    return result


# ── row builders ───────────────────────────────────────────────────────────────

def _build_total_ktkm_rows(
    tf: pl.DataFrame, region: str, start_order: float
) -> pl.DataFrame:
    """Region-level service_request rows from the total_ktkm pipeline variable."""
    data = (
        tf.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'total_ktkm') &
            (pl.col('parameter') == 'service_request')
        )
        .sort('year')
    )
    n = len(data)
    if n == 0:
        return _empty_frame()
    return data.select([
        pl.lit(f'CIMS.CAN.{region}').alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Transportation Freight').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit(f'CIMS.CAN.{region}.Transportation Freight').alias('Target'),
        pl.col('source').alias('Source'),
        pl.col('unit').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('value').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _build_price_mult_rows(
    multipliers: pl.DataFrame, region: str, start_order: float
) -> pl.DataFrame:
    """multiplier_price rows for the Transportation Freight sector."""
    data = (
        multipliers
        .filter(
            (pl.col('Sector') == 'Transportation Freight') &
            (pl.col('Region') == region)
        )
        .sort('Energy', 'Year')
    )
    n = len(data)
    if n == 0:
        return _empty_frame()
    return data.select([
        pl.lit(f'CIMS.CAN.{region}.Transportation Freight').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Transportation Freight').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('multiplier_price').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.when(pl.col('Energy').is_in(list(REGION_SPECIFIC_ENERGIES)))
        .then(pl.lit(f'CIMS.CAN.{region}.') + pl.col('Energy'))
        .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('Energy'))
        .alias('Target'),
        pl.col('Source').alias('Source'),
        pl.lit('').alias('Unit'),
        pl.col('Year').cast(pl.String).alias('Year'),
        pl.col('Multiplier').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i * 1e-4 for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _build_sr_rows(
    tf: pl.DataFrame,
    region: str,
    variables: list[str],
    branch: str,
    branch_type: str,
    service: str,
    var_to_target: dict[str, str],
    start_order: float,
) -> pl.DataFrame:
    """
    Build service_request rows at a given branch for a list of pipeline variables.

    Rows are ordered by variable (in the order given) then year, all packed
    consecutively starting at start_order with 1e-4 spacing.
    """
    data = (
        tf.filter(
            (pl.col('province') == region) &
            pl.col('variable').is_in(variables) &
            (pl.col('parameter') == 'service_request') &
            (pl.col('category').fill_null('') == '')
        )
        .sort(['variable', 'year'])
    )
    if len(data) == 0:
        return _empty_frame()

    rows: list[dict] = []
    counter = 0
    for var in variables:
        target = var_to_target.get(var)
        if target is None:
            continue
        for r in data.filter(pl.col('variable') == var).iter_rows(named=True):
            rows.append({
                'Branch':      branch,
                'Type':        branch_type,
                'Region':      region,
                'Sector':      'Transportation Freight',
                'Service':     service,
                'Technology':  '',
                'Parameter':   'service_request',
                'Context':     '',
                'Sub_Context': '',
                'Target':      target,
                'Source':      r['source'],
                'Unit':        r['unit'],
                'Year':        str(r['year']),
                'Value':       str(r['value']),
                '_order':      start_order + counter * 1e-4,
            })
            counter += 1
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_heavy_mst_rows(
    tf: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    branch: str,
) -> pl.DataFrame:
    """
    market_share_total (year 2000) for Trucks and Rail at the Heavy service level.

    The Heavy service has Trucks and Rail as competing "technologies" (routing
    demand to each sub-service). These don't have lifetime rows, so insertion
    is keyed off each technology's 'technology' parameter row (which has no year
    and is always a single row), placing market_share_total before the
    service_request rows.
    """
    tech_tech_orders = _tech_param_orders(fixed, 'Heavy', 'technology')
    if not tech_tech_orders:
        return _empty_frame()

    data = tf.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'Freight.Land.Heavy') &
        (pl.col('parameter') == 'market_share_total') &
        (pl.col('year') == 2000)
    )
    pipeline_by_cat: dict[str, tuple[float, str, str]] = {}
    default_unit = '%'
    for r in data.iter_rows(named=True):
        pipeline_by_cat[r['category']] = (float(r['value']), r['source'], r['unit'])
        default_unit = r['unit'] or default_unit

    rows: list[dict] = []
    for tech, tech_order in tech_tech_orders.items():
        entry = pipeline_by_cat.get(tech, (0.0, 'CEUD', default_unit))
        val, src, unit = entry
        rows.append({
            'Branch':      branch,
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Freight',
            'Service':     'Heavy',
            'Technology':  tech,
            'Parameter':   'market_share_total',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      src,
            'Unit':        unit,
            'Year':        '2000',
            'Value':       str(val),
            '_order':      tech_order + 0.5,
        })
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_lm_mst_rows(
    tf: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    branch: str,
) -> pl.DataFrame:
    """
    market_share_total (year 2000) for every Light Medium technology.

    Fuel categories from the pipeline are mapped to LM_CAT_TO_TECH for the
    base-year installed fleet; all other technologies receive value 0.
    """
    tech_lifetime_max = _tech_lifetime_orders(fixed, 'Light Medium')
    if not tech_lifetime_max:
        return _empty_frame()

    tech_to_cat: dict[str, str] = {v: k for k, v in LM_CAT_TO_TECH.items()}

    data = tf.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'Light Medium') &
        (pl.col('parameter') == 'market_share_total') &
        (pl.col('year') == 2000)
    )
    pipeline_by_cat: dict[str, tuple[float, str, str]] = {}
    default_unit = '%'
    for r in data.iter_rows(named=True):
        pipeline_by_cat[r['category']] = (float(r['value']), r['source'], r['unit'])
        default_unit = r['unit'] or default_unit

    rows: list[dict] = []
    for tech, lifetime_max in tech_lifetime_max.items():
        cat = tech_to_cat.get(tech, tech)
        entry = pipeline_by_cat.get(cat, (0.0, 'CEUD', default_unit))
        val, src, unit = entry
        rows.append({
            'Branch':      branch,
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Freight',
            'Service':     'Light Medium',
            'Technology':  tech,
            'Parameter':   'market_share_total',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      src,
            'Unit':        unit,
            'Year':        '2000',
            'Value':       str(val),
            '_order':      lifetime_max + 0.5,
        })
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_lm_output_rows(
    tf: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    branch: str,
) -> pl.DataFrame:
    """
    output rows (all years) for all Light Medium technologies.

    The fleet-average per-vehicle productivity (variable='Light Medium',
    category='') is duplicated for every LM technology.  Technologies that
    already carry an output row in the fixed data are skipped.
    """
    tech_lifetime_max = _tech_lifetime_orders(fixed, 'Light Medium')
    existing_output   = _techs_with_existing_output(fixed, 'Light Medium')
    techs_to_fill = {t: o for t, o in tech_lifetime_max.items()
                     if t not in existing_output}
    if not techs_to_fill:
        return _empty_frame()

    svc_data = (
        tf.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'Light Medium') &
            (pl.col('parameter') == 'output') &
            (pl.col('category').fill_null('') == '')
        )
        .sort('year')
    )
    if len(svc_data) == 0:
        return _empty_frame()

    unit     = svc_data['unit'][0] or 'k*tkm'
    src      = svc_data['source'][0] or 'CEUD'
    svc_list = list(svc_data.iter_rows(named=True))

    rows: list[dict] = []
    for tech, lifetime_max in techs_to_fill.items():
        for i, r in enumerate(svc_list):
            rows.append({
                'Branch':      branch,
                'Type':        'Service',
                'Region':      region,
                'Sector':      'Transportation Freight',
                'Service':     'Light Medium',
                'Technology':  tech,
                'Parameter':   'output',
                'Context':     '',
                'Sub_Context': '',
                'Target':      '',
                'Source':      src,
                'Unit':        unit,
                'Year':        str(r['year']),
                'Value':       str(r['value']),
                '_order':      lifetime_max + 0.6 + i * 1e-4,
            })
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_trucks_output_rows(
    tf: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    branch: str,
) -> pl.DataFrame:
    """
    output rows (all years) for all Trucks technologies.

    The fleet-average per-vehicle productivity (variable='Heavy Trucks',
    category='') is duplicated for every Trucks technology.  Rows are ordered
    just after each technology's existing market_share_total row in the fixed data.
    Technologies that already carry an output row are skipped.
    """
    tech_mst_max    = _tech_mst_orders(fixed, 'Trucks')
    existing_output = _techs_with_existing_output(fixed, 'Trucks')
    techs_to_fill   = {t: o for t, o in tech_mst_max.items()
                       if t not in existing_output}
    if not techs_to_fill:
        return _empty_frame()

    svc_data = (
        tf.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'Heavy Trucks') &
            (pl.col('parameter') == 'output') &
            (pl.col('category').fill_null('') == '')
        )
        .sort('year')
    )
    if len(svc_data) == 0:
        return _empty_frame()

    unit     = svc_data['unit'][0] or 'k*tkm'
    src      = svc_data['source'][0] or 'CEUD'
    svc_list = list(svc_data.iter_rows(named=True))

    rows: list[dict] = []
    for tech, mst_order in techs_to_fill.items():
        for i, r in enumerate(svc_list):
            rows.append({
                'Branch':      branch,
                'Type':        'Service',
                'Region':      region,
                'Sector':      'Transportation Freight',
                'Service':     'Trucks',
                'Technology':  tech,
                'Parameter':   'output',
                'Context':     '',
                'Sub_Context': '',
                'Target':      '',
                'Source':      src,
                'Unit':        unit,
                'Year':        str(r['year']),
                'Value':       str(r['value']),
                '_order':      mst_order + 0.5 + i * 1e-4,
            })
    return pl.DataFrame(rows) if rows else _empty_frame()


# ── region assembler ───────────────────────────────────────────────────────────

def _assemble_region(
    fixed: pl.DataFrame,
    tf: pl.DataFrame,
    multipliers: pl.DataFrame,
    region: str,
) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with pipeline-derived rows at the correct positions.
    """
    # ── branch strings ─────────────────────────────────────────────────────────
    b_sector  = f'CIMS.CAN.{region}.Transportation Freight'
    b_freight = f'{b_sector}.Freight'
    b_land    = f'{b_freight}.Land'
    b_lm      = f'{b_land}.Light Medium'
    b_heavy   = f'{b_land}.Heavy'
    b_trucks  = f'{b_heavy}.Trucks'

    # ── insertion-point discovery ──────────────────────────────────────────────
    # sector level: competition and service_request rows (Service='')
    sector_competition_max = _find_max_order(fixed, '', 'competition') or 0.0
    sector_sr_max          = _find_max_order(fixed, '', 'service_request') or 0.0

    # Freight / Land / Heavy service competition rows
    freight_competition_max = _find_max_order(fixed, 'Freight', 'competition') or 0.0
    land_competition_max    = _find_max_order(fixed, 'Land', 'competition') or 0.0

    # ── build pipeline rows ────────────────────────────────────────────────────

    # 1. Total k*tkm: before all fixed data
    total_ktkm_rows = _build_total_ktkm_rows(
        tf, region,
        start_order=float(fixed['_order'].min()) - 1000.0,
    )

    # 2. Price multipliers: just after sector-level competition row
    price_rows = _build_price_mult_rows(
        multipliers, region,
        start_order=sector_competition_max + 0.5,
    )

    # 3. Off-Road service_request: right after the sector's service_request to Freight
    offroad_sr_rows = _build_sr_rows(
        tf, region,
        variables=['Freight.Off-Road'],
        branch=b_sector,
        branch_type='Sector',
        service='',
        var_to_target={'Freight.Off-Road': f'{b_sector}.Off Road'},
        start_order=sector_sr_max + 0.5,
    )

    # 4. Freight sub-service shares: Land, Marine, Air — after Freight competition
    freight_mode_sr_rows = _build_sr_rows(
        tf, region,
        variables=['Freight.Land', 'Freight.Marine', 'Freight.Air'],
        branch=b_freight,
        branch_type='Service',
        service='Freight',
        var_to_target={
            'Freight.Land':   f'{b_freight}.Land',
            'Freight.Marine': f'{b_freight}.Marine',
            'Freight.Air':    f'{b_freight}.Air',
        },
        start_order=freight_competition_max + 0.5,
    )

    # 5. Land sub-service shares: Light Medium, Heavy — after Land competition
    land_sr_rows = _build_sr_rows(
        tf, region,
        variables=['Freight.Land.Light Medium', 'Freight.Land.Heavy'],
        branch=b_land,
        branch_type='Service',
        service='Land',
        var_to_target={
            'Freight.Land.Light Medium': b_lm,
            'Freight.Land.Heavy':        b_heavy,
        },
        start_order=land_competition_max + 0.5,
    )

    # 6. Heavy service Trucks/Rail competition (market_share_total year 2000)
    heavy_mst_rows = _build_heavy_mst_rows(tf, fixed, region, branch=b_heavy)

    # 7. Light Medium tech market_share_total (year 2000)
    lm_mst_rows = _build_lm_mst_rows(tf, fixed, region, branch=b_lm)

    # 8. Light Medium tech output (all years, fleet-average per-vehicle productivity)
    lm_out_rows = _build_lm_output_rows(tf, fixed, region, branch=b_lm)

    # 9. Trucks tech output (all years, fleet-average per-vehicle productivity)
    trucks_out_rows = _build_trucks_output_rows(tf, fixed, region, branch=b_trucks)

    # ── combine and sort ───────────────────────────────────────────────────────
    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        total_ktkm_rows,
        price_rows,
        offroad_sr_rows,
        freight_mode_sr_rows,
        land_sr_rows,
        heavy_mst_rows,
        lm_mst_rows,
        lm_out_rows,
        trucks_out_rows,
    ]

    combined = pl.concat(
        [f for f in all_frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble transportation freight model inputs and write one CSV per region."""
    print('=' * 60)
    print('TRANSPORTATION FREIGHT MODEL INPUTS')
    print('=' * 60)

    print('\nLoading pipeline data...')
    _tf_results = _tf_mod.main(export_csv=False)
    tf = pl.concat(list(_tf_results.values()), how='diagonal_relaxed')
    tf = tf.with_columns(
        pl.when(pl.col('year') <= _tf_mod.LAST_HIST_YEAR)
        .then(pl.lit('CEUD'))
        .otherwise(pl.lit('Assumptions'))
        .alias('source')
    )

    print('  Loading energy price multipliers...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    print(f'  Freight pipeline: {len(tf):,} rows, '
          f'regions: {sorted(tf["province"].unique().to_list())}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'transportation freight_{template}.csv'
        if not fixed_path.exists():
            print(f'  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue
        if region not in tf['province'].unique().to_list():
            print(f'  Skipping {region} — no pipeline data for this region')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            fixed = _read_flattened_fixed(template)

            print('  Assembling...')
            output = _assemble_region(fixed, tf, multipliers, region)

            out_path = OUTPUT_DIR / f'transportation freight_{region}.csv'
            output.write_csv(str(out_path))
            print(f'  Wrote {len(output):,} rows -> {out_path.name}')
            results[region] = output

        except Exception as exc:
            print(f'  ERROR: {exc}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Regions complete: {len(results)}/{len(FIXED_TEMPLATE)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
