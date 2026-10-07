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
    BC splits each activity 25 % cold / 75 % marine.

HVAC Cold / Marine and Hot Water market_share_total  (year 2000 only)
    processed_data/nrcan/ceud/commercial.csv  →  variables hvac_cold / hvac_marine / hot_water_tech
    Spliced between the 'lifetime' and 'output' parameter blocks for each
    technology within those service sections.

Shell -> HVAC service_request  (see _build_hvac_intensity_rows)
    processed_data/nrcan/ceud/commercial.csv  ->  variable = 'hvac_intensity'
    Replaces the fixed-data Buildings.Shell.<Activity> -> HVAC (Cold)/(Marine)
    service_request rows (one per shell technology) with the CEUD space-heating
    intensity, averaged over HVAC_INTENSITY_YEARS.  Each shell tier keeps its
    fixed-data ratio to Std, and BC's Cold/Marine nodes keep their climate
    split, so only the overall level is recalibrated.  HVAC's own technology
    service_request rows (-> fuels / Motive Power / Cooling) are untouched
    fixed data.

Weather nodes  (WEATHER_NODES; service_provide / competition / service_request)
    processed_data/nrcan/ceud/commercial.csv  ->  variable = 'hdd_index'
    A Fixed Ratio "Weather (Cold|Marine)" node is added under each
    Buildings.Shell.<Activity> node, and that activity's technologies request
    it instead of HVAC -- so the only service_request into HVAC comes from the
    Shell subtree and HVAC stays a pure technology-competition node (same
    arrangement as residential's '<bin> Bldg Code.Weather (<climate>)').  Its
    service_request to HVAC is the CEUD Heating Degree-Day Index (historical
    years; mean of the last HDD_PROJECTION_YEARS thereafter), and the Shell
    intensity above is the weather-normalised mean.  A node without
    technologies isn't vintage-weighted, so the year-to-year weather signal
    reaches all floor space rather than only new stock.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
import CIMS.data_processing.utils.flatten_fixed_data as _flatten_mod

import CIMS.data_processing.source.nrcan.ceud.commercial.commercial as _commercial_mod

import CIMS.data_processing.source.energy_prices.energy_price_multipliers as _energy_price_mod

import CIMS.data_processing.source.cer.cer_resd_demand as _cer_resd_mod

from CIMS.data_processing.utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years
from CIMS.data_processing.utils.feedstock_demand import build_feedstock_rows

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

# Share of BC's commercial floor space in each climate zone, applied to every
# activity's CEUD floorspace share (_build_shell_share_rows). CEUD carries no
# climate-zone split of its own, so this is a modelling assumption, not a
# measured quantity -- it is the only statement of it in the pipeline, and
# commercial_calibration.py mirrors it so the two stay consistent. 75/25 lines
# up with the split implied by residential's own BC fixed data, where each
# Bldg Code node's Marine and Cold heating requests sit at a uniform 0.63249
# Marine share -- 75/25 floor space once that figure's embedded climate
# correction (MARINE_TO_COLD_RATIO, 0.5725) is backed out.
BC_COLD_FRACTION   = 0.25
BC_MARINE_FRACTION = 0.75
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

# Shell.<Activity> -> HVAC service_request intensity. CIMS vintage-weights a
# technology's service_request, so the year-to-year signal in a per-shell-tech
# rate can't reach the model intact -- old floorspace keeps its base-year
# value and new stock is diluted toward the existing mix. A single historical
# mean is used instead, and the annual weather signal is carried by the
# Weather node below, which has no technologies to vintage-weight.
HVAC_INTENSITY_YEARS: tuple[int, int] = (DATA_START, _commercial_mod.LAST_HIST_YEAR)
HVAC_SERVICES: tuple[str, ...] = ('HVAC (Cold)', 'HVAC (Marine)')
# Shell technology carrying 100 % of year-2000 market share (LEED Silver /
# Platinum only become available later), so it anchors the rescaling the same
# way 'Reference' does on the residential side.
HVAC_ANCHOR_TECH = 'Std'

# Weather nodes: a Fixed Ratio "Weather (<climate>)" node under each Shell
# activity node, between that activity's technologies and the HVAC node,
# carrying the CEUD heating degree-day index. Its node-level service_request
# isn't vintage-weighted.
WEATHER_NODES: bool = True
# Projection-year weather factor = mean HDD index over the last N CEUD years.
HDD_PROJECTION_YEARS: int = 10


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
    BC splits each activity 25 % cold / 75 % marine.
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


def _hdd_index(commercial: pl.DataFrame, region: str) -> pd.Series:
    """CEUD HDD index for one region as a year-indexed Series (empty if absent)."""
    data = (
        commercial
        .filter((pl.col('region') == region) & (pl.col('variable') == 'hdd_index'))
        .sort('year')
    )
    return pd.Series(data['value'].to_list(),
                     index=[int(y) for y in data['year'].to_list()], dtype=float)


def _weather_factors(hdd: pd.Series) -> dict[int, float]:
    """{year: weather factor} for DATA_START..PROJECTION_END from the HDD index."""
    proj = float(hdd.iloc[-HDD_PROJECTION_YEARS:].mean())
    return {y: float(hdd.get(y, proj)) for y in range(DATA_START, PROJECTION_END + 1)}


def _hvac_intensity_means(commercial: pl.DataFrame, region: str,
                          hdd: pd.Series | None = None) -> dict[str, float]:
    """
    {activity: mean GJ of heat per m2 over HVAC_INTENSITY_YEARS} for one
    region, from commercial.py's compute_hvac_intensity(). With `hdd`, each
    year is divided by its HDD index first (weather-normalised intensity).
    """
    lo, hi = HVAC_INTENSITY_YEARS
    data = commercial.filter(
        (pl.col('region') == region) &
        (pl.col('variable') == 'hvac_intensity') &
        pl.col('year').is_between(lo, hi)
    )
    if hdd is not None:
        data = data.with_columns(
            pl.col('value') / pl.col('year').cast(pl.Int64).replace_strict(
                {int(y): float(v) for y, v in hdd.items()}, default=None,
                return_dtype=pl.Float64,
            )
        ).drop_nulls('value')
    return {
        r['category']: float(r['value'])
        for r in data.group_by('category').agg(pl.col('value').mean()).iter_rows(named=True)
    }


def _climate_of(hvac_target: str) -> str:
    """'...Commercial.HVAC (Marine)' -> 'Marine'."""
    return hvac_target.rsplit('(', 1)[-1].rstrip(')')


def _weather_node_name(shell_branch: str, climate: str) -> str:
    """
    '...Buildings.Shell.Offices (Cold)' + 'Cold'
        -> '...Buildings.Shell.Offices (Cold).Weather (Cold)'

    The Weather node is a child of the shell activity node whose technologies
    request it, mirroring residential, where each Bldg Code node carries its
    own '<bin> Bldg Code.Weather (<climate>)'. That keeps the only
    service_request into HVAC inside the Shell subtree -- HVAC itself stays a
    pure technology-competition node.
    """
    return f'{shell_branch}.Weather ({climate})'


def _build_weather_node_rows(fixed: pl.DataFrame, shell_branch: str,
                             hvac_branch: str, region: str,
                             factors: dict[int, float]) -> list[dict]:
    """
    Fixed Ratio Weather node rows for one shell activity, positioned straight
    after that activity's own block and requesting the sector-level HVAC node.

    One node per shell activity (and per climate), not one shared sector-wide:
    the request into HVAC has to come from the Shell subtree, so each activity
    needs its own.
    """
    shell_rows = fixed.filter(pl.col('Branch') == shell_branch)
    order = float(shell_rows['_order'].max()) + 0.4
    hvac_rows = fixed.filter(pl.col('Branch') == hvac_branch)
    sp = hvac_rows.filter(pl.col('Parameter') == 'service_provide')
    unit = sp['Unit'][0] if len(sp) else 'GJ'
    climate = _climate_of(hvac_branch)
    branch = _weather_node_name(shell_branch, climate)
    service = f'Weather ({climate})'
    base = {'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Commercial', 'Service': service, 'Technology': '',
            'Context': '', 'Sub_Context': ''}
    rows = [
        {**base, 'Parameter': 'service_provide', 'Target': '', 'Source': '',
         'Unit': unit, 'Year': '', 'Value': '', '_order': order},
        {**base, 'Parameter': 'competition', 'Target': '', 'Source': '',
         'Unit': '', 'Year': '', 'Value': 'Fixed Ratio', '_order': order + 1e-3},
    ]
    # One Source for every year: collapse_constant_years groups on Source, and
    # splitting history from projection would let a repeated historical value
    # become a second blank-Year default.
    for i, (year, f) in enumerate(factors.items()):
        rows.append({**base, 'Parameter': 'service_request', 'Target': hvac_branch,
                     'Source': 'CEUD HDD index',
                     'Unit': 'GJ', 'Year': str(year), 'Value': str(f),
                     '_order': order + 2e-3 + i * 1e-6})
    return rows


def _build_hvac_intensity_rows(commercial: pl.DataFrame, fixed: pl.DataFrame,
                               region: str) -> tuple[pl.DataFrame, list[int]]:
    """
    Buildings.Shell.<Activity> -> HVAC service_request rows (all years, per
    shell technology) from the CEUD space-heating intensity, replacing the
    fixed-data constants.

    Each fixed-data (shell technology, climate) rate keeps its ratio to the
    activity's floorspace-weighted Std rate -- LEED Silver / Platinum stay at
    their JCIMS fraction of Std, and BC's (Cold) / (Marine) activity nodes
    keep the climate correction already embedded in fixed_data -- and the
    whole activity is rescaled so that its floorspace-weighted mean rate
    reproduces the CEUD intensity. BC's two climate nodes are weighted by
    BC_COLD_FRACTION / BC_MARINE_FRACTION, the same floorspace split
    _build_shell_share_rows applies, so Cold + Marine partition the activity's
    heat rather than each reproducing all of it.

    With WEATHER_NODES (and an HDD index for the region) the intensity is
    weather-normalised, the rows target a Weather node instead of the HVAC
    node, and the Weather node's own rows are added -- so the flat mean here
    carries the structural level while the Weather node makes HVAC demand
    track real year-to-year weather.

    Returns the new rows (placed at the replaced rows' positions) and the
    _order values of the fixed rows they replace. Activities with no CEUD
    intensity keep their fixed-data rows.
    """
    hdd = _hdd_index(commercial, region) if WEATHER_NODES else None
    if hdd is not None and hdd.empty:
        print(f'  No CEUD HDD index for {region}; no Weather nodes')
        hdd = None
    factors = _weather_factors(hdd) if hdd is not None else None
    means = _hvac_intensity_means(commercial, region, hdd)
    lo, hi = HVAC_INTENSITY_YEARS
    source = f'CEUD mean {lo}-{hi}' + (' / HDD index' if hdd is not None else '')

    climate_fraction = (
        {'Cold': BC_COLD_FRACTION, 'Marine': BC_MARINE_FRACTION} if region == 'BC'
        else {'Cold': 1.0, 'Marine': 1.0}
    )
    svc_to_activity = {v: k for k, v in CAT_TO_COLD_SVC.items()}
    svc_to_activity.update({v: k for k, v in CAT_TO_MARINE_SVC.items()})

    sr = fixed.filter(
        (pl.col('Parameter') == 'service_request') &
        pl.col('Branch').str.contains('.Commercial.Buildings.Shell.', literal=True) &
        pl.any_horizontal([pl.col('Target').str.ends_with(f'.{s}') for s in HVAC_SERVICES])
    ).with_columns(
        pl.col('Service').replace_strict(svc_to_activity, default=None).alias('_activity'),
        pl.col('Value').cast(pl.Float64, strict=False).alias('_v'),
    ).drop_nulls('_activity')

    rows: list[dict] = []
    replaced: list[int] = []
    weather_done: set[str] = set()
    for (activity,), g in sr.group_by('_activity', maintain_order=True):
        if activity not in means:
            print(f'  No CEUD HVAC intensity for {activity}; keeping fixed data')
            continue
        intensity = means[activity]

        mean_v = {
            (branch, tech, target): float(sub['_v'].mean())
            for (branch, tech, target), sub
            in g.group_by(['Branch', 'Technology', 'Target'], maintain_order=True)
        }
        anchor_total = sum(
            climate_fraction.get(_climate_of(target), 0.0) * v
            for (_, tech, target), v in mean_v.items() if tech == HVAC_ANCHOR_TECH
        )
        if not anchor_total:
            print(f'  No {HVAC_ANCHOR_TECH} HVAC request for {activity}; keeping fixed data')
            continue

        for (branch, tech, target), v in mean_v.items():
            first = g.filter(
                (pl.col('Branch') == branch) & (pl.col('Technology') == tech) &
                (pl.col('Target') == target)
            )
            order = float(first['_order'].min())
            value = str(intensity * v / anchor_total)
            if factors is not None:
                weather_branch = _weather_node_name(branch, _climate_of(target))
                if weather_branch not in weather_done:
                    rows.extend(_build_weather_node_rows(
                        fixed, branch, target, region, factors))
                    weather_done.add(weather_branch)
                target = weather_branch
            for i, year in enumerate(range(DATA_START, PROJECTION_END + 1)):
                rows.append({
                    'Branch': branch, 'Type': 'Service', 'Region': region,
                    'Sector': 'Commercial', 'Service': first['Service'][0],
                    'Technology': tech,
                    'Parameter': 'service_request',
                    'Context': '', 'Sub_Context': '',
                    'Target': target,
                    'Source': source,
                    'Unit': first['Unit'][0],
                    'Year': str(year), 'Value': value,
                    '_order': order + i * 1e-6,
                })
        replaced.extend(g['_order'].to_list())

    return (pl.DataFrame(rows) if rows else _empty_frame()), replaced


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

    # Shell.<Activity> -> HVAC service_request rows: the CEUD-derived
    # historical mean intensity in place of the fixed-data constants, routed
    # through a Weather node (see _build_hvac_intensity_rows).
    hvac_rows, hvac_replaced = _build_hvac_intensity_rows(commercial, fixed, region)
    if hvac_replaced:
        fixed = fixed.filter(~pl.col('_order').is_in(hvac_replaced))

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
