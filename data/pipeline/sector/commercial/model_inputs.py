"""
Commercial Pipeline — Model Inputs

Combines fixed structural parameters with CEUD pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/commercial/commercial_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    AT is used as template for NL/PE/NS/NB; BC is used for YT/NT/NU.

Total floorspace  (service_request rows)
    processed_data/nrcan/ceud/commercial.csv  →  variable = 'total_floorspace'
    Placed as Region-level service_request before the Commercial sector block.

Energy price multipliers  (multiplier_price rows)
    processed_data/energy_prices/energy_price_multipliers.csv
    Inserted after the Commercial sector header (service_provide / competition).

Building shell shares  (market_share_total, year 2000 only)
    processed_data/nrcan/ceud/commercial.csv  →  variable = 'building_shell_shares'
    Inserted after the Shell service header rows (service_provide, competition,
    intercept_retirement), before the Shell activity sub-service sections.
    BC splits each activity 80 % cold / 20 % marine.

HVAC Cold / Marine and Hot Water market_share_total  (year 2000 only)
    processed_data/nrcan/ceud/commercial.csv  →  variables hvac_cold / hvac_marine / hot_water_tech
    Spliced between the 'lifetime' and 'output' parameter blocks for each
    technology within those service sections.

HVAC and Cooling service_request  (see _build_hvac_service_request_rows)
    processed_data/nrcan/ceud/commercial.csv  →  variables hvac_service_request /
    hvac_cooling_service_request / buildings_hvac_service_request /
    buildings_cooling_service_request
    Two demand sources per target, covering mutually exclusive year ranges:
    Buildings -> HVAC (Cold)/(Marine) and Buildings -> Cooling carry exact
    CEUD-derived historical demand (through commercial.py's
    BUILDINGS_HVAC_HISTORICAL_CUTOFF); Shell.<Activity> -> HVAC (per shell
    technology) and HVAC -> Cooling (per HVAC technology) are zero through
    that same cutoff and take over from SHELL_HVAC_COMPETITION_START_YEAR
    onward, once Shell's own Std/LEED Silver/LEED Platinum competition
    becomes live. See commercial.py's compute_hvac_service_requests for the
    full rationale.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import sys
import tempfile
import importlib.util
from pathlib import Path

import pandas as pd
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

_comm_spec = importlib.util.spec_from_file_location(
    'commercial',
    _PIPELINE_ROOT / 'source' / 'nrcan' / 'ceud' / 'commercial' / 'commercial.py',
)
_commercial_mod = importlib.util.module_from_spec(_comm_spec)
_comm_spec.loader.exec_module(_commercial_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

_cer_spec = importlib.util.spec_from_file_location(
    'cer_resd_demand',
    _PIPELINE_ROOT / 'source' / 'cer' / 'cer_resd_demand.py',
)
_cer_resd_mod = importlib.util.module_from_spec(_cer_spec)
_cer_spec.loader.exec_module(_cer_resd_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years
from utils.feedstock_demand import build_feedstock_rows

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/commercial'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/commercial'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Each pipeline region maps directly to its own fixed-data file.
FIXED_TEMPLATE: dict[str, str] = {
    'AB': 'AB', 'BC': 'BC', 'MB': 'MB', 'NB': 'NB', 'NL': 'NL',
    'NS': 'NS', 'NT': 'NT', 'NU': 'NU', 'ON': 'ON', 'PE': 'PE',
    'QC': 'QC', 'SK': 'SK', 'YT': 'YT',
}

BC_COLD_FRACTION   = 0.80
BC_MARINE_FRACTION = 0.20
BC_TERRITORY_CODES = {'YT', 'NT', 'NU'}

# How many of the most recent historical years' feedstock-per-floorspace
# ratio to average when projecting feedstock demand beyond LAST_HIST_YEAR.
FEEDSTOCK_RATIO_YEARS = 5

# Pipeline building_shell_shares category → CIMS Shell sub-service name
CAT_TO_COLD_SVC: dict[str, str] = {
    'Wholesale':                         'Wholesale (Cold)',
    'Retail':                            'Retail (Cold)',
    'Transportation and Warehousing':    'Transportation and Warehousing (Cold)',
    'Information and Cultural':          'Information and Cultural (Cold)',
    'Offices':                           'Offices (Cold)',
    'Educational':                       'Educational (Cold)',
    'Healthcare and Social Assistance':  'Healthcare and Social Assistance (Cold)',
    'Arts Entertainment and Recreation': 'Arts Entertainment and Recreation (Cold)',
    'Accommodation and Food Services':   'Accommodation and Food services (Cold)',
    'Other Services':                    'Other Services (Cold)',
}
CAT_TO_MARINE_SVC: dict[str, str] = {
    k: v.replace('(Cold)', '(Marine)') for k, v in CAT_TO_COLD_SVC.items()
}

# Energies whose price target is region-specific (CIMS.CAN.{region}.{energy})
REGION_SPECIFIC_ENERGIES: set[str] = {
    'Electricity', 'Biodiesel',
    'Ethanol', 'Hydrogen',
}

# Parameter ordering within HVAC / Hot Water technology blocks
_PARAMS_BEFORE_MST = {'technology', 'available', 'unavailable', 'lifetime'}
_PARAMS_AFTER_MST  = {'output', 'fcc', 'capital_recovery', 'fom',
                      'service_request', 'market_share_new_max'}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(template_region: str, output_region: str) -> pl.DataFrame:
    """
    Flatten one fixed commercial CSV and return as a row-indexed DataFrame.

    When output_region differs from template_region (AT sub-regions, BC
    territories) the region code is substituted throughout Branch / Target /
    Region, and Marine rows are dropped for BC territory regions.
    """
    fixed_path = FIXED_INPUT_DIR / f'commercial_{template_region}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'commercial_{template_region}.csv'
        _flatten_mod.process_file(
            input_path=fixed_path,
            output_path=out_file,
            year_min=DATA_START,
            year_max=LAST_DATA_YEAR["cer"],
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)

    return df.with_row_index('_order')


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})


def _build_floorspace_rows(commercial: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """Region-level service_request rows from the total_floorspace pipeline data."""
    data = (
        commercial
        .filter((pl.col('region') == region) & (pl.col('variable') == 'total_floorspace'))
        .sort('year')
    )
    n = len(data)
    return data.select([
        pl.lit(f'CIMS.CAN.{region}').alias('Branch'),
        pl.lit('Region').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Commercial').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit(f'CIMS.CAN.{region}.Commercial').alias('Target'),
        pl.col('source').alias('Source'),
        pl.col('unit').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('value').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _build_price_mult_rows(multipliers: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """
    multiplier_price rows for the Commercial sector.

    All rows receive _order values clustered tightly at start_order with a
    step of 1e-4 so they sort as a single block between the two adjacent
    fixed-data rows (competition and the first service_request).
    """
    data = (
        multipliers
        .filter((pl.col('Sector') == 'Commercial') & (pl.col('Region') == region))
        .sort('Energy', 'Year')
    )
    n = len(data)
    return data.select([
        pl.lit(f'CIMS.CAN.{region}.Commercial').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Commercial').alias('Sector'),
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



# Buildings -> sub-service targets whose fixed-data service_request rows are
# now computed from CEUD (see commercial.py:compute_enduse_service_requests)
# rather than hardcoded. Maps the pipeline variable name to the CIMS
# sub-service branch name.
ENDUSE_VARIABLE_TARGETS: dict[str, str] = {
    'lighting_service_request':      'Lighting',
    'refrigeration_service_request': 'Refrigeration',
    'cooking_service_request':       'Cooking',
    'hot_water_service_request':     'Hot Water',
    'plug_load_service_request':     'Plug Load',
}


def _build_enduse_service_request_rows(commercial: pl.DataFrame, region: str,
                                        start_order: float) -> pl.DataFrame:
    """
    Buildings -> {Lighting, Refrigeration, Cooking, Hot Water, Plug Load}
    service_request rows (all years), computed from CEUD end-use energy in
    commercial.py's compute_enduse_service_requests(). Replaces the
    hand-fixed constants _assemble_region() strips out of `fixed` for these
    five sub-services.

    _order values are packed into a narrow band just above start_order (the
    surviving Buildings -> Shell row) so they land in the same spot the
    stripped fixed-data rows used to occupy, in the same Lighting /
    Refrigeration / Cooking / Hot Water / Plug Load order.
    """
    branch = f'CIMS.CAN.{region}.Commercial.Buildings'
    frames = []

    for var_idx, (variable, suffix) in enumerate(ENDUSE_VARIABLE_TARGETS.items()):
        data = (
            commercial
            .filter((pl.col('region') == region) & (pl.col('variable') == variable))
            .sort('year')
        )
        n = len(data)
        if n == 0:
            continue
        frames.append(data.select([
            pl.lit(branch).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.lit(region).alias('Region'),
            pl.lit('Commercial').alias('Sector'),
            pl.lit('Buildings').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.lit(f'{branch}.{suffix}').alias('Target'),
            pl.col('source').alias('Source'),
            pl.col('unit').alias('Unit'),
            pl.col('year').cast(pl.String).alias('Year'),
            pl.col('value').cast(pl.String).alias('Value'),
            pl.Series('_order', [start_order + var_idx * 0.01 + j * 1e-4 for j in range(n)],
                      dtype=pl.Float64).alias('_order'),
        ]))

    return pl.concat(frames, how='diagonal_relaxed') if frames else _empty_frame()


def _build_shell_share_rows(commercial: pl.DataFrame, region: str,
                             insert_order: float) -> pl.DataFrame:
    """
    service_request rows (all years) for each Shell activity sub-service.

    The building_shell_shares from the pipeline give the fraction of total
    Shell demand that flows to each activity sub-service each year.  These
    are emitted as service_request rows from the Shell service, with Target
    pointing to the corresponding Shell sub-service branch.
    BC splits each activity 80 % cold / 20 % marine.
    """
    data = (
        commercial
        .filter(
            (pl.col('region') == region) &
            (pl.col('variable') == 'building_shell_shares')
        )
        .sort('category', 'year')
    )
    branch = f'CIMS.CAN.{region}.Commercial.Buildings.Shell'
    is_bc  = (region == 'BC')
    rows: list[dict] = []

    for r in data.iter_rows(named=True):
        cat, val, yr = r['category'], r['value'], str(r['year'])
        cold_svc   = CAT_TO_COLD_SVC.get(cat)
        marine_svc = CAT_TO_MARINE_SVC.get(cat) if is_bc else None

        if cold_svc:
            rows.append({
                'Branch': branch, 'Type': 'Service', 'Region': region,
                'Sector': 'Commercial', 'Service': 'Shell', 'Technology': '',
                'Parameter': 'service_request', 'Context': '', 'Sub_Context': '',
                'Target': f'{branch}.{cold_svc}',
                'Source': r['source'], 'Unit': '%',
                'Year': yr,
                'Value': str(val * BC_COLD_FRACTION if is_bc else val),
                '_order': insert_order,
            })
        if marine_svc:
            rows.append({
                'Branch': branch, 'Type': 'Service', 'Region': region,
                'Sector': 'Commercial', 'Service': 'Shell', 'Technology': '',
                'Parameter': 'service_request', 'Context': '', 'Sub_Context': '',
                'Target': f'{branch}.{marine_svc}',
                'Source': r['source'], 'Unit': '%',
                'Year': yr, 'Value': str(val * BC_MARINE_FRACTION),
                '_order': insert_order,
            })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_tech_mst_rows(
    commercial: pl.DataFrame,
    fixed: pl.DataFrame,
    region: str,
    variable: str,
    service_name: str,
    branch_suffix: str,
) -> pl.DataFrame:
    """
    Build year-2000 market_share_total rows for HVAC Cold / Marine or Hot Water.

    Each technology's row is assigned an _order value of (that technology's last
    'lifetime' _order + 0.5) so it lands between the lifetime and output blocks
    for that specific technology in the sorted output.
    """
    data = commercial.filter(
        (pl.col('region') == region) &
        (pl.col('variable') == variable) &
        (pl.col('year') == 2000)
    )
    branch = f'CIMS.CAN.{region}.{branch_suffix}'
    rows: list[dict] = []

    # Pre-compute per-technology lifetime max _order from the fixed data
    service_fixed = fixed.filter(
        (pl.col('Service') == service_name) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology') != '') &
        (pl.col('Parameter') == 'lifetime')
    )
    tech_lifetime_max: dict[str, float] = {}
    for r in service_fixed.select(['Technology', '_order']).iter_rows(named=True):
        t = r['Technology']
        o = float(r['_order'])
        if t not in tech_lifetime_max or o > tech_lifetime_max[t]:
            tech_lifetime_max[t] = o

    # Build lookups from pipeline: category → year-2000 value and source
    pipeline_vals: dict[str, float] = {}
    pipeline_sources: dict[str, str] = {}
    for r in data.iter_rows(named=True):
        pipeline_vals[r['category']] = r['value']
        pipeline_sources[r['category']] = r['source']
    pipeline_unit = data['unit'][0] if len(data) > 0 else '%'

    # Emit a market_share_total row for every technology in the fixed data,
    # using 0 for any technology absent from the pipeline.
    for tech, lifetime_max in tech_lifetime_max.items():
        val = pipeline_vals.get(tech, 0.0)
        # A technology can be PRESENT in the pipeline with an undefined (NaN)
        # share -- e.g. a fuel with zero measured demand across the whole
        # disaggregation group for that year -- and .get()'s default only
        # covers a technology missing entirely. Treat NaN the same as
        # missing rather than writing a literal "nan" into the model_inputs
        # CSV, which silently drops out of any downstream sum.
        if val is None or val != val:
            val = 0.0
        rows.append({
            'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Commercial', 'Service': service_name,
            'Technology': tech,
            'Parameter': 'market_share_total', 'Context': '', 'Sub_Context': '',
            'Target': '', 'Source': pipeline_sources.get(tech, 'CEUD'), 'Unit': pipeline_unit,
            'Year': '2000', 'Value': str(val),
            '_order': lifetime_max + 0.5,
        })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _find_max_order(df: pl.DataFrame, service: str, parameter: str,
                    require_tech: bool = False) -> float | None:
    """Return the max _order value for rows matching service + parameter."""
    mask = (pl.col('Service') == service) & (pl.col('Parameter') == parameter)
    if require_tech:
        mask = mask & pl.col('Technology').is_not_null() & (pl.col('Technology') != '')
    subset = df.filter(mask)
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


def _find_tech_param_max_order(df: pl.DataFrame, service: str, technology: str,
                               parameter: str) -> float | None:
    """Return the max _order value for rows matching service + technology + parameter."""
    subset = df.filter(
        (pl.col('Service') == service) & (pl.col('Technology') == technology) &
        (pl.col('Parameter') == parameter)
    )
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


def _build_hvac_service_request_rows(commercial: pl.DataFrame, fixed: pl.DataFrame,
                                      region: str) -> pl.DataFrame:
    """
    Builds four sets of rows, all computed from CEUD Space Heating/Cooling
    energy in commercial.py's compute_hvac_service_requests(), replacing the
    hand-fixed constants _assemble_region() strips out of `fixed` for these
    targets:
      - Buildings.Shell.<Activity> -> HVAC (Cold/Marine) service_request
        (all years, all shell technologies -- zero before commercial.py's
        SHELL_HVAC_COMPETITION_START_YEAR)
      - HVAC (Cold/Marine) technologies' own service_request to Cooling
        (same zero-before-competition-start pattern)
      - Buildings -> HVAC (Cold/Marine) direct historical service_request
        (years through BUILDINGS_HVAC_HISTORICAL_CUTOFF only)
      - Buildings -> Cooling direct historical service_request (same cutoff)

    Each new row is anchored to sit immediately after its technology's
    surviving 'fom' row (or, for the two Buildings -> ... direct requests,
    the surviving Buildings -> Shell row) -- the same slot the stripped
    fixed-data service_request row used to occupy.
    """
    frames = []

    hvac_data = commercial.filter(
        (pl.col('region') == region) & (pl.col('variable') == 'hvac_service_request')
    )
    for category in hvac_data['category'].unique().to_list():
        activity, climate, tech = category.split('|')
        shell_svc = (CAT_TO_COLD_SVC if climate == 'Cold' else CAT_TO_MARINE_SVC).get(activity)
        if shell_svc is None:
            continue

        anchor = _find_tech_param_max_order(fixed, shell_svc, tech, 'fom')
        if anchor is None:
            continue

        data = hvac_data.filter(pl.col('category') == category).sort('year')
        n = len(data)
        if n == 0:
            continue

        branch = f'CIMS.CAN.{region}.Commercial.Buildings.Shell.{shell_svc}'
        target = f'CIMS.CAN.{region}.Commercial.HVAC ({climate})'
        frames.append(data.select([
            pl.lit(branch).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.lit(region).alias('Region'),
            pl.lit('Commercial').alias('Sector'),
            pl.lit(shell_svc).alias('Service'),
            pl.lit(tech).alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.lit(target).alias('Target'),
            pl.col('source').alias('Source'),
            pl.col('unit').alias('Unit'),
            pl.col('year').cast(pl.String).alias('Year'),
            pl.col('value').cast(pl.String).alias('Value'),
            pl.Series('_order', [anchor + 0.5 + j * 1e-4 for j in range(n)],
                      dtype=pl.Float64).alias('_order'),
        ]))

    buildings_hvac_data = commercial.filter(
        (pl.col('region') == region) & (pl.col('variable') == 'buildings_hvac_service_request')
    )
    buildings_hvac_anchor = _find_max_order(fixed, 'Buildings', 'service_request', require_tech=False)
    for climate_idx, climate in enumerate(buildings_hvac_data['category'].unique().to_list()):
        if buildings_hvac_anchor is None:
            break
        data = buildings_hvac_data.filter(pl.col('category') == climate).sort('year')
        n = len(data)
        if n == 0:
            continue
        branch = f'CIMS.CAN.{region}.Commercial.Buildings'
        target = f'CIMS.CAN.{region}.Commercial.HVAC ({climate})'
        frames.append(data.select([
            pl.lit(branch).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.lit(region).alias('Region'),
            pl.lit('Commercial').alias('Sector'),
            pl.lit('Buildings').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.lit(target).alias('Target'),
            pl.col('source').alias('Source'),
            pl.col('unit').alias('Unit'),
            pl.col('year').cast(pl.String).alias('Year'),
            pl.col('value').cast(pl.String).alias('Value'),
            pl.Series('_order', [buildings_hvac_anchor + 0.6 + climate_idx * 0.01 + j * 1e-4
                                 for j in range(n)], dtype=pl.Float64).alias('_order'),
        ]))

    buildings_cooling_data = commercial.filter(
        (pl.col('region') == region) & (pl.col('variable') == 'buildings_cooling_service_request')
    )
    for climate_idx, climate in enumerate(buildings_cooling_data['category'].unique().to_list()):
        if buildings_hvac_anchor is None:
            break
        data = buildings_cooling_data.filter(pl.col('category') == climate).sort('year')
        n = len(data)
        if n == 0:
            continue
        branch = f'CIMS.CAN.{region}.Commercial.Buildings'
        target = f'CIMS.CAN.{region}.Commercial.Cooling'
        frames.append(data.select([
            pl.lit(branch).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.lit(region).alias('Region'),
            pl.lit('Commercial').alias('Sector'),
            pl.lit('Buildings').alias('Service'),
            pl.lit('').alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.lit(target).alias('Target'),
            pl.col('source').alias('Source'),
            pl.col('unit').alias('Unit'),
            pl.col('year').cast(pl.String).alias('Year'),
            pl.col('value').cast(pl.String).alias('Value'),
            pl.Series('_order', [buildings_hvac_anchor + 0.7 + climate_idx * 0.01 + j * 1e-4
                                 for j in range(n)], dtype=pl.Float64).alias('_order'),
        ]))

    hvac_cooling_data = commercial.filter(
        (pl.col('region') == region) & (pl.col('variable') == 'hvac_cooling_service_request')
    )
    for category in hvac_cooling_data['category'].unique().to_list():
        climate, tech = category.split('|')
        service = f'HVAC ({climate})'

        anchor = _find_tech_param_max_order(fixed, service, tech, 'fom')
        if anchor is None:
            continue

        data = hvac_cooling_data.filter(pl.col('category') == category).sort('year')
        n = len(data)
        if n == 0:
            continue

        branch = f'CIMS.CAN.{region}.Commercial.{service}'
        target = f'CIMS.CAN.{region}.Commercial.Cooling'
        frames.append(data.select([
            pl.lit(branch).alias('Branch'),
            pl.lit('Service').alias('Type'),
            pl.lit(region).alias('Region'),
            pl.lit('Commercial').alias('Sector'),
            pl.lit(service).alias('Service'),
            pl.lit(tech).alias('Technology'),
            pl.lit('service_request').alias('Parameter'),
            pl.lit('').alias('Context'),
            pl.lit('').alias('Sub_Context'),
            pl.lit(target).alias('Target'),
            pl.col('source').alias('Source'),
            pl.col('unit').alias('Unit'),
            pl.col('year').cast(pl.String).alias('Year'),
            pl.col('value').cast(pl.String).alias('Value'),
            pl.Series('_order', [anchor + 0.5 + j * 1e-4 for j in range(n)],
                      dtype=pl.Float64).alias('_order'),
        ]))

    return pl.concat(frames, how='diagonal_relaxed') if frames else _empty_frame()


def _assemble_region(
    fixed: pl.DataFrame,
    commercial: pl.DataFrame,
    multipliers: pl.DataFrame,
    feedstock: pd.DataFrame,
    feedstock_last_hist_year: int,
    region: str,
    template_region: str,
) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with pipeline-derived rows at the correct positions.
    """
    # ── insertion-point discovery ──────────────────────────────────────────────
    # 1. After Commercial sector header (service_provide / competition)
    comm_header_max = float(
        fixed.filter(
            (pl.col('Branch') == f'CIMS.CAN.{region}.Commercial') &
            pl.col('Parameter').is_in(['service_provide', 'competition'])
        )['_order'].max()
    )

    # 2. After Shell service header (intercept_retirement — no Technology on these rows)
    shell_interc_max = _find_max_order(fixed, 'Shell', 'intercept_retirement',
                                       require_tech=False)
    if shell_interc_max is None:
        shell_interc_max = comm_header_max  # fallback

    # 3–5. Per-technology lifetime maxima are resolved inside _build_tech_mst_rows
    is_bc_full = (region == 'BC')

    # Strip the fixed-data Buildings -> {Lighting, Refrigeration, Cooking,
    # Hot Water, Plug Load} service_request rows: these five are now computed
    # from CEUD (see ENDUSE_VARIABLE_TARGETS / _build_enduse_service_request_rows)
    # instead of being hand-fixed constants. Buildings -> Shell is untouched.
    enduse_suffixes = list(ENDUSE_VARIABLE_TARGETS.values())
    is_stripped_enduse_row = (
        (pl.col('Service') == 'Buildings') &
        (pl.col('Parameter') == 'service_request') &
        pl.any_horizontal([pl.col('Target').str.ends_with(f'.{s}') for s in enduse_suffixes])
    )
    fixed = fixed.filter(~is_stripped_enduse_row)

    # Strip the fixed-data Shell.<Activity> -> HVAC (Cold/Marine)
    # service_request rows (per shell technology): these are now computed
    # from CEUD (see compute_hvac_service_requests / _build_hvac_service_request_rows)
    # instead of being hand-fixed constants. HVAC's own fuel-technology
    # service_request rows (-> Methane Blend / Electricity / Motive Power)
    # are untouched.
    #
    # NOTE: the Cooling NODE's own Std -> Electricity rate is intentionally
    # LEFT AS THE ORIGINAL fixed_data constant here (not stripped) -- that's
    # Cooling's own efficiency conversion, a separate concern from how much
    # Cooling gets requested in the first place.
    #
    # HVAC (Cold/Marine) technologies' own service_request to Cooling *is*
    # stripped: this used to be a flat "1" for every technology (Cooling
    # tracking heat 1:1), now replaced by the CEUD-derived cooling/heat ratio
    # (zero before SHELL_HVAC_COMPETITION_START_YEAR, since the Buildings ->
    # Cooling direct request carries historical demand instead -- see
    # compute_hvac_service_requests / _hvac_cooling_to_heat_ratio).
    is_stripped_hvac_row = (
        (pl.col('Parameter') == 'service_request') &
        (pl.col('Target').str.ends_with('.HVAC (Cold)') |
         pl.col('Target').str.ends_with('.HVAC (Marine)') |
         ((pl.col('Branch').str.ends_with('.HVAC (Cold)') |
           pl.col('Branch').str.ends_with('.HVAC (Marine)')) &
          pl.col('Target').str.ends_with('.Cooling')))
    )
    fixed = fixed.filter(~is_stripped_hvac_row)

    # Anchor for the new end-use rows: right after the surviving Buildings ->
    # Shell row (where the stripped rows used to sit).
    buildings_shell_max = _find_max_order(fixed, 'Buildings', 'service_request',
                                          require_tech=False)
    if buildings_shell_max is None:
        buildings_shell_max = comm_header_max  # fallback

    # ── build pipeline rows with fractional _order values ─────────────────────
    # Floorspace: before everything (negative orders)
    floorspace_rows = _build_floorspace_rows(
        commercial, region, start_order=float(fixed['_order'].min()) - 1000.0
    )

    # Buildings end-use service_request rows: just after Buildings -> Shell
    enduse_rows = _build_enduse_service_request_rows(
        commercial, region, start_order=buildings_shell_max + 0.5
    )

    # Shell -> HVAC and Cooling -> Electricity service_request rows: each
    # anchored to its own surviving technology's 'fom' row (see
    # _build_hvac_service_request_rows).
    hvac_rows = _build_hvac_service_request_rows(commercial, fixed, region)

    # Price multipliers: just after Commercial header
    price_rows = _build_price_mult_rows(
        multipliers, region, start_order=comm_header_max + 0.5
    )

    # Shell shares: just after Shell intercept_retirement
    shell_share_rows = _build_shell_share_rows(
        commercial, region, insert_order=shell_interc_max + 0.5
    )

    # Hot Water market_share_total — per-technology insertion between lifetime and output
    hw_mst = _build_tech_mst_rows(
        commercial, fixed, region, 'hot_water_tech', 'Hot Water',
        'Commercial.Buildings.Hot Water',
    )

    # HVAC Cold market_share_total
    hvac_cold_mst = _build_tech_mst_rows(
        commercial, fixed, region, 'hvac_cold', 'HVAC (Cold)',
        'Commercial.HVAC (Cold)',
    )

    # HVAC Marine market_share_total (BC full province only)
    hvac_marine_mst = (
        _build_tech_mst_rows(
            commercial, fixed, region, 'hvac_marine', 'HVAC (Marine)',
            'Commercial.HVAC (Marine)',
        )
        if is_bc_full else _empty_frame()
    )

    # Feedstock: after everything else in this region's fixed data (mirrors
    # floorspace's "before everything" -1000.0 anchor, but at the other end).
    floorspace_series = (
        commercial
        .filter((pl.col('region') == region) & (pl.col('variable') == 'total_floorspace'))
        .sort('year')
    )
    scale_series = pd.Series(
        floorspace_series['value'].to_list(),
        index=floorspace_series['year'].to_list(),
    )
    feedstock_region = feedstock.loc[
        feedstock['Region'] == region, ['Variable', 'Year', 'Value']
    ]
    feedstock_rows = build_feedstock_rows(
        sector_branch=f'CIMS.CAN.{region}.Commercial',
        sector_name='Commercial',
        region=region,
        feedstock=feedstock_region,
        scale_series=scale_series,
        scale_unit='m2',
        last_hist_year=feedstock_last_hist_year,
        ratio_window=FEEDSTOCK_RATIO_YEARS,
        start_order=float(fixed['_order'].max()) + 1000.0,
    )

    # ── combine and sort ───────────────────────────────────────────────────────
    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        floorspace_rows,
        enduse_rows,
        hvac_rows,
        price_rows,
        shell_share_rows,
        hw_mst,
        hvac_cold_mst,
        hvac_marine_mst,
        feedstock_rows,
    ]

    combined = pl.concat(
        [f for f in all_frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble commercial model inputs and write one CSV per region."""
    print('=' * 60)
    print('COMMERCIAL MODEL INPUTS')
    print('=' * 60)

    print('\nLoading pipeline data...')
    _commercial_results = _commercial_mod.main()
    commercial = (
        pl.concat(list(_commercial_results.values()), how='diagonal_relaxed')
        .with_columns(
            pl.when(pl.col('year') <= _commercial_mod.LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
    )
    multipliers = pl.from_pandas(_energy_price_mod.main())
    print(f'  Commercial data: {len(commercial):,} rows, '
          f'regions: {sorted(commercial["region"].unique().to_list())}')

    feedstock_demand = _cer_resd_mod.load_feedstock_demand()
    feedstock_demand = feedstock_demand[feedstock_demand['Node'] == '.Commercial']
    # CER's own last historical year (vFsDmd-CIMS.csv currently runs through
    # 2024) rather than _commercial_mod.LAST_HIST_YEAR (CEUD's cutoff, 2023):
    # feedstock demand is CER-sourced, and its last actual year should still
    # be reported as real CER/RESD data, not folded into the post-historical
    # trailing-average projection a year early.
    feedstock_last_hist_year = (
        int(feedstock_demand['Year'].max())
        if len(feedstock_demand) else _commercial_mod.LAST_HIST_YEAR
    )
    print(f'  Feedstock data: {len(feedstock_demand):,} rows, '
          f'fuels: {sorted(feedstock_demand["Variable"].unique())}, '
          f'last historical year: {feedstock_last_hist_year}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region, template in sorted(FIXED_TEMPLATE.items()):
        fixed_path = FIXED_INPUT_DIR / f'commercial_{template}.csv'
        if not fixed_path.exists():
            print(f'  ⚠  Skipping {region} — fixed data template not found: {fixed_path.name}')
            continue
        if region not in commercial['region'].unique().to_list():
            print(f'  ⚠  Skipping {region} — no pipeline data for this region')
            continue

        try:
            print(f'\n{region} (template: {template}):')
            print('  Flattening fixed data...')
            fixed = _read_flattened_fixed(template, region)

            print('  Assembling...')
            output = _assemble_region(
                fixed, commercial, multipliers, feedstock_demand,
                feedstock_last_hist_year, region, template,
            )

            out_path = OUTPUT_DIR / f'commercial_{region.lower()}.csv'
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
