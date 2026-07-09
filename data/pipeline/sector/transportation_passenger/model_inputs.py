"""
Transportation Passenger Pipeline — Model Inputs

Combines fixed structural parameters with CEUD pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Transportation Personal/transportation passenger_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Every region has its own fixed-data file.

Total k*pkm  (service_request rows, Region level)
    processed via transportation_passenger pipeline  →  variable = 'total_kpkm'
    Placed before the Transportation Passenger sector block.

Energy price multipliers  (multiplier_price rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Inserted after the sector header (service_provide / competition).

Mode service_request  (Urban / Intercity Land / Intercity Air fractions)
    pipeline  →  variables 'Mode.Urban', 'Mode.Intercity Land', 'Mode.Intercity Air'
    Inserted after the Mode service header (service_provide / competition).

Urban / Intercity Land / Intercity Air tech market_share_total  (year 2000 only)
    Spliced after the 'lifetime' row for each technology in those services.

Passenger Vehicles tech market_share_total (year 2000) + output (all years)
    Spliced after the 'lifetime' row for each technology.

Passenger Vehicle Motors tech market_share_total (year 2000) + output (all years)
    market_share_total per technology; output is the fleet-average distance
    (service-level from pipeline, duplicated for each motor technology).

Transit.Public Bus tech market_share_total (year 2000) + output (all years)
    Fuel-share market_share_total per technology (defaulting to 0 for techs
    absent from CEUD); avg k*pkm-per-bus output duplicated for each technology.
    Technologies that already carry an output row in the fixed data are skipped.

Intercity Bus tech market_share_total (year 2000) + output (all years)
    Same pattern as Public Bus.

Intercity Rail tech market_share_total (year 2000 only)
    Rail tech defaults inserted after each technology's lifetime row.
    Output is already present in the fixed data.

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

_spec = importlib.util.spec_from_file_location(
    'flatten_fixed_data',
    _PIPELINE_ROOT / 'utils' / 'flatten_fixed_data.py',
)
_flatten_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_flatten_mod)

_tp_spec = importlib.util.spec_from_file_location(
    'transportation_passenger',
    _PIPELINE_ROOT / 'source' / 'nrcan' / 'ceud' / 'transportation_passenger'
    / 'transportation_passenger.py',
)
_tp_mod = importlib.util.module_from_spec(_tp_spec)
_tp_spec.loader.exec_module(_tp_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/transportation_passenger'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/transportation_passenger'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Every region maps to its own fixed-data file.
FIXED_TEMPLATE: dict[str, str] = {
    'AB': 'AB', 'BC': 'BC', 'MB': 'MB', 'NB': 'NB', 'NL': 'NL',
    'NS': 'NS', 'NT': 'NT', 'NU': 'NU', 'ON': 'ON', 'PE': 'PE',
    'QC': 'QC', 'SK': 'SK', 'YT': 'YT',
}

# Energies whose price target is region-specific (CIMS.CAN.{region}.{energy})
REGION_SPECIFIC_ENERGIES: set[str] = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}

# Pipeline category → fixed-data Technology name for the Urban service.
URBAN_CAT_TO_TECH: dict[str, str] = {
    'Walk Cycle':            'Walk Cycle Urban',
    'Passenger Vehicle SOV': 'Passenger Vehicle Urban SOV',
    'Passenger Vehicle HOV': 'Passenger Vehicle Urban HOV',
    'Public Transit':        'Public Transit Urban',
}

# Pipeline category → fixed-data Technology name for the Intercity Land service.
INTERCITY_LAND_CAT_TO_TECH: dict[str, str] = {
    'Bus Intercity':     'Bus Intercity',
    'Rail Intercity':    'Rail Intercity',
    'Passenger Vehicle': 'Passenger Vehicle Intercity',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    """Flatten one fixed transportation passenger CSV and return as a row-indexed DataFrame."""
    fixed_path = FIXED_INPUT_DIR / f'transportation_passenger_{region.lower()}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'transportation_passenger_{region.lower()}.csv'
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
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


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
        t = r['Technology']
        o = float(r['_order'])
        if t not in result or o > result[t]:
            result[t] = o
    return result


def _techs_with_existing_output(fixed: pl.DataFrame, service_name: str) -> set[str]:
    """Return the set of technology names that already have an output row in the fixed data."""
    mask = (
        (pl.col('Service').fill_null('') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology').fill_null('') != '') &
        (pl.col('Parameter').fill_null('') == 'output')
    )
    return set(fixed.filter(mask)['Technology'].unique().to_list())


def _build_total_kpkm_rows(tp: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """Region-level service_request rows from the total_kpkm pipeline variable."""
    data = (
        tp.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'total_kpkm') &
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
        pl.lit('Transportation Passenger').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit(f'CIMS.CAN.{region}.Transportation Passenger').alias('Target'),
        pl.col('source').alias('Source'),
        pl.col('unit').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('value').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _build_price_mult_rows(multipliers: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """multiplier_price rows for the Transportation Passenger sector."""
    data = (
        multipliers
        .filter((pl.col('Sector') == 'Transportation Passenger') & (pl.col('Region') == region))
        .sort('Energy', 'Year')
    )
    n = len(data)
    if n == 0:
        return _empty_frame()
    return data.select([
        pl.lit(f'CIMS.CAN.{region}.Transportation Passenger').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Transportation Passenger').alias('Sector'),
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


def _build_mode_sr_rows(tp: pl.DataFrame, region: str,
                         start_order: float) -> pl.DataFrame:
    """
    service_request rows for Urban, Intercity Land, and Intercity Air mode shares.

    These rows come from the Mode service and point to each transport-mode sub-service.
    Rows are sorted by variable then year so all Urban years precede all Intercity years.
    """
    var_to_target = {
        'Mode.Urban':
            f'CIMS.CAN.{region}.Transportation Passenger.Mode.Urban',
        'Mode.Intercity Land':
            f'CIMS.CAN.{region}.Transportation Passenger.Mode.Intercity Land',
        'Mode.Intercity Air':
            f'CIMS.CAN.{region}.Transportation Passenger.Mode.Intercity Air',
    }
    data = (
        tp.filter(
            (pl.col('province') == region) &
            pl.col('variable').is_in(list(var_to_target.keys())) &
            (pl.col('parameter') == 'service_request')
        )
        .sort(['variable', 'year'])
    )
    if len(data) == 0:
        return _empty_frame()

    rows: list[dict] = []
    for i, r in enumerate(data.iter_rows(named=True)):
        rows.append({
            'Branch':      f'CIMS.CAN.{region}.Transportation Passenger.Mode',
            'Type':        'Service',
            'Region':      region,
            'Sector':      'Transportation Passenger',
            'Service':     'Mode',
            'Technology':  '',
            'Parameter':   'service_request',
            'Context':     '',
            'Sub_Context': '',
            'Target':      var_to_target[r['variable']],
            'Source':      r['source'],
            'Unit':        r['unit'],
            'Year':        str(r['year']),
            'Value':       str(r['value']),
            '_order':      start_order + i * 1e-4,
        })
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_transit_sr_rows(tp: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """
    service_request rows for Public Bus and Rapid Transit from the Transit service.

    These replace the fixed-data Transit service_request (which only covers Rapid
    Transit with hardcoded benchmark values).  Pipeline provides dynamic annual
    values for both PB and RT for historical years; the last historical value is
    held constant through PROJECTION_END so CIMS has coverage for all years.
    """
    cat_to_target = {
        'Public Bus':    f'CIMS.CAN.{region}.Transportation Passenger.Transit.Public Bus',
        'Rapid Transit': f'CIMS.CAN.{region}.Transportation Passenger.Transit.Rapid Transit',
    }
    data = (
        tp.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'Transit') &
            (pl.col('parameter') == 'service_request') &
            pl.col('category').is_in(list(cat_to_target.keys()))
        )
        .sort(['category', 'year'])
    )
    if len(data) == 0:
        return _empty_frame()

    all_years = list(range(DATA_START, PROJECTION_END + 1))
    rows: list[dict] = []
    counter = 0
    for cat, target in cat_to_target.items():
        cat_rows = data.filter(pl.col('category') == cat)
        if len(cat_rows) == 0:
            continue
        yr_map: dict[int, dict] = {r['year']: r for r in cat_rows.iter_rows(named=True)}
        last_r = cat_rows.row(-1, named=True)
        unit = last_r['unit'] or '%'
        for yr in all_years:
            if yr in yr_map:
                r = yr_map[yr]
                val, src = r['value'], r['source']
            else:
                val, src = last_r['value'], 'Assumptions'
            rows.append({
                'Branch':      f'CIMS.CAN.{region}.Transportation Passenger.Transit',
                'Type':        'Service',
                'Region':      region,
                'Sector':      'Transportation Passenger',
                'Service':     'Transit',
                'Technology':  '',
                'Parameter':   'service_request',
                'Context':     '',
                'Sub_Context': '',
                'Target':      target,
                'Source':      src,
                'Unit':        unit,
                'Year':        str(yr),
                'Value':       str(val),
                '_order':      start_order + counter * 1e-4,
            })
            counter += 1
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_mst_rows(
    tp: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    variable: str,
    service_name: str,
    branch: str,
    cat_to_tech: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    market_share_total rows (year 2000 only) for every technology in service_name.

    Each row is assigned _order = that technology's lifetime max + 0.5 so it
    lands immediately after lifetime and before any subsequent fixed-data rows.
    Technologies absent from the pipeline data receive value 0.
    """
    tech_lifetime_max = _tech_lifetime_orders(fixed, service_name)
    if not tech_lifetime_max:
        return _empty_frame()

    # Invert cat_to_tech so we can look up pipeline category by tech name.
    tech_to_cat: dict[str, str] = (
        {v: k for k, v in cat_to_tech.items()} if cat_to_tech else {}
    )

    data = tp.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == variable) &
        (pl.col('parameter') == 'market_share_total') &
        (pl.col('year') == 2000)
    )

    # Build category → (value, source, unit) lookup from pipeline.
    # Key is the RAW pipeline category (before any mapping).
    pipeline_by_cat: dict[str, tuple[float, str, str]] = {}
    default_unit = '%'
    for r in data.iter_rows(named=True):
        pipeline_by_cat[r['category']] = (float(r['value']), r['source'], r['unit'])
        default_unit = r['unit'] or default_unit

    rows: list[dict] = []
    for tech, lifetime_max in tech_lifetime_max.items():
        # Map tech → pipeline category.
        cat = tech_to_cat.get(tech, tech)
        entry = pipeline_by_cat.get(cat, (0.0, 'CEUD', default_unit))
        val, src, unit = entry
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
            'Source':      src,
            'Unit':        unit,
            'Year':        '2000',
            'Value':       str(val),
            '_order':      lifetime_max + 0.5,
        })
    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_output_rows(
    tp: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    variable: str,
    service_name: str,
    branch: str,
    per_tech: bool = False,
    cat_to_tech: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    output rows (all years) for technologies in service_name.

    per_tech=True  — each technology gets its own pipeline values (matched by
                     category == tech name, or via cat_to_tech inversion).
    per_tech=False — a single service-level pipeline series (category == '') is
                     duplicated for every technology that lacks an output row in
                     the fixed data.

    In both cases, technologies that already have an output row in the fixed
    data are skipped to avoid duplicates.

    Rows are ordered at lifetime_max + 0.6 + i * 1e-4 (after market_share_total
    at + 0.5) for each technology.
    """
    tech_lifetime_max = _tech_lifetime_orders(fixed, service_name)
    existing_output = _techs_with_existing_output(fixed, service_name)
    techs_to_process = {t: o for t, o in tech_lifetime_max.items()
                        if t not in existing_output}
    if not techs_to_process:
        return _empty_frame()

    tech_to_cat: dict[str, str] = (
        {v: k for k, v in cat_to_tech.items()} if cat_to_tech else {}
    )

    data = tp.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == variable) &
        (pl.col('parameter') == 'output')
    ).sort('year')

    rows: list[dict] = []

    if per_tech:
        for tech, lifetime_max in techs_to_process.items():
            cat = tech_to_cat.get(tech, tech)
            tech_data = data.filter(pl.col('category') == cat)
            if len(tech_data) == 0:
                continue
            unit = tech_data['unit'][0] or ''
            src  = tech_data['source'][0] or 'CEUD'
            for i, r in enumerate(tech_data.iter_rows(named=True)):
                rows.append({
                    'Branch':      branch,
                    'Type':        'Service',
                    'Region':      region,
                    'Sector':      'Transportation Passenger',
                    'Service':     service_name,
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
    else:
        # Service-level output duplicated across all eligible technologies.
        svc_data = data.filter(pl.col('category') == '')
        if len(svc_data) == 0:
            return _empty_frame()
        unit = svc_data['unit'][0] or ''
        src  = svc_data['source'][0] or 'CEUD'
        svc_rows = svc_data.iter_rows(named=True)
        svc_list = list(svc_rows)

        for tech, lifetime_max in techs_to_process.items():
            for i, r in enumerate(svc_list):
                rows.append({
                    'Branch':      branch,
                    'Type':        'Service',
                    'Region':      region,
                    'Sector':      'Transportation Passenger',
                    'Service':     service_name,
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


def _assemble_region(
    fixed: pl.DataFrame,
    tp: pl.DataFrame,
    multipliers: pl.DataFrame,
    region: str,
) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with pipeline-derived rows at the correct positions.
    """
    # ── branch strings ─────────────────────────────────────────────────────────
    b_mode = f'CIMS.CAN.{region}.Transportation Passenger.Mode'
    b_urban = f'{b_mode}.Urban'
    b_il    = f'{b_mode}.Intercity Land'
    b_ia    = f'{b_mode}.Intercity Air'
    b_pv    = f'CIMS.CAN.{region}.Transportation Passenger.Passenger Vehicles'
    b_pvm   = f'CIMS.CAN.{region}.Transportation Passenger.Passenger Vehicle Motors'
    b_pb    = f'CIMS.CAN.{region}.Transportation Passenger.Transit.Public Bus'
    b_ib    = f'CIMS.CAN.{region}.Transportation Passenger.Intercity Bus'
    b_ir    = f'CIMS.CAN.{region}.Transportation Passenger.Intercity Rail'

    # Remove fixed-data Transit service_request rows — replaced by pipeline values
    # that cover both Public Bus and Rapid Transit for all years 2000–2100.
    fixed = fixed.filter(
        ~(
            (pl.col('Service').fill_null('') == 'Transit') &
            (pl.col('Parameter').fill_null('') == 'service_request') &
            (pl.col('Technology').fill_null('') == '')
        )
    )

    # ── insertion-point discovery ──────────────────────────────────────────────
    sector_competition_max  = _find_max_order(fixed, '', 'competition') or 0.0
    mode_competition_max    = _find_max_order(fixed, 'Mode', 'competition') or 0.0
    transit_competition_max = _find_max_order(fixed, 'Transit', 'competition') or 0.0

    # ── build pipeline rows ────────────────────────────────────────────────────
    # 1. Total k*pkm: before all fixed data
    total_kpkm_rows = _build_total_kpkm_rows(
        tp, region,
        start_order=float(fixed['_order'].min()) - 1000.0,
    )

    # 2. Price multipliers: just after sector-level competition row
    price_rows = _build_price_mult_rows(
        multipliers, region,
        start_order=sector_competition_max + 0.5,
    )

    # 3. Mode service_request: just after Mode competition row
    mode_sr_rows = _build_mode_sr_rows(
        tp, region,
        start_order=mode_competition_max + 0.5,
    )

    # 3b. Transit service_request (Public Bus + Rapid Transit): just after Transit competition row
    transit_sr_rows = _build_transit_sr_rows(
        tp, region,
        start_order=transit_competition_max + 0.5,
    )

    # 4–6. Mode sub-service tech market_share_total (year 2000)
    urban_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Mode.Urban', service_name='Urban', branch=b_urban,
        cat_to_tech=URBAN_CAT_TO_TECH,
    )
    il_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Mode.Intercity Land', service_name='Intercity Land', branch=b_il,
        cat_to_tech=INTERCITY_LAND_CAT_TO_TECH,
    )
    ia_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Mode.Intercity Air', service_name='Intercity Air', branch=b_ia,
    )

    # 7. Passenger Vehicles: market_share_total + per-technology output
    pv_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Passenger Vehicles', service_name='Passenger Vehicles', branch=b_pv,
    )
    pv_out = _build_output_rows(
        tp, fixed, region,
        variable='Passenger Vehicles', service_name='Passenger Vehicles', branch=b_pv,
        per_tech=True,
    )

    # 8. Passenger Vehicle Motors: market_share_total + service-level output
    pvm_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Passenger Vehicle Motors', service_name='Passenger Vehicle Motors',
        branch=b_pvm,
    )
    pvm_out = _build_output_rows(
        tp, fixed, region,
        variable='Passenger Vehicle Motors', service_name='Passenger Vehicle Motors',
        branch=b_pvm,
        per_tech=False,
    )

    # 9. Transit.Public Bus: market_share_total + service-level output
    pb_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Transit.Public Bus', service_name='Public Bus', branch=b_pb,
    )
    pb_out = _build_output_rows(
        tp, fixed, region,
        variable='Transit.Public Bus', service_name='Public Bus', branch=b_pb,
        per_tech=False,
    )

    # 10. Intercity Bus: market_share_total + service-level output
    ib_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Intercity Bus', service_name='Intercity Bus', branch=b_ib,
    )
    ib_out = _build_output_rows(
        tp, fixed, region,
        variable='Intercity Bus', service_name='Intercity Bus', branch=b_ib,
        per_tech=False,
    )

    # 11. Intercity Rail: market_share_total only (output already in fixed data)
    ir_mst = _build_mst_rows(
        tp, fixed, region,
        variable='Intercity Rail', service_name='Intercity Rail', branch=b_ir,
    )

    # ── combine and sort ───────────────────────────────────────────────────────
    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        total_kpkm_rows,
        price_rows,
        mode_sr_rows,
        transit_sr_rows,
        urban_mst,
        il_mst,
        ia_mst,
        pv_mst, pv_out,
        pvm_mst, pvm_out,
        pb_mst, pb_out,
        ib_mst, ib_out,
        ir_mst,
    ]

    combined = pl.concat(
        [f for f in all_frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble transportation passenger model inputs and write one CSV per region."""
    print('=' * 60)
    print('TRANSPORTATION PASSENGER MODEL INPUTS')
    print('=' * 60)

    print('\nLoading pipeline data...')
    _tp_results = _tp_mod.main(export_csv=False)
    tp = pl.concat(list(_tp_results.values()), how='diagonal_relaxed')
    tp = tp.with_columns(
        pl.when(pl.col('year') <= _tp_mod.LAST_HIST_YEAR)
        .then(pl.lit('CEUD'))
        .otherwise(pl.lit('Assumptions'))
        .alias('source')
    )

    multipliers = pl.from_pandas(_energy_price_mod.main())
    print(f'  Transportation passenger data: {len(tp):,} rows, '
          f'regions: {sorted(tp["province"].unique().to_list())}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'transportation passenger_{template}.csv'
        if not fixed_path.exists():
            print(f'  ⚠  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue
        if region not in tp['province'].unique().to_list():
            print(f'  ⚠  Skipping {region} — no pipeline data for this region')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            fixed = _read_flattened_fixed(template)

            print('  Assembling...')
            output = _assemble_region(fixed, tp, multipliers, region)

            out_path = OUTPUT_DIR / f'transportation_passenger_{region.lower()}.csv'
            output = collapse_constant_years(output)
            output.write_csv(str(out_path))
            print(f'  Wrote {len(output):,} rows → {out_path.name}')
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
