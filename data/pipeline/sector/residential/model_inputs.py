"""
Residential Pipeline — Model Inputs

Combines fixed structural parameters with CEUD pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/residential/residential_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    One file per region — no template substitution required.

Housing stock  (service_request rows)
    Inserted before all fixed data as a Region-level service_request
    from CIMS.CAN.{region} to CIMS.CAN.{region}.Residential.

Energy price multipliers  (multiplier_price rows)
    Inserted after the Residential sector header (service_provide / competition).

Appliances per household  (service_request rows, all years)
    Inserted after the Dwellings service_request → Building Type row.

Building type market shares  (market_share_total, year 2000)
    Inserted after each Building Type technology row.

Floorspace per building  (service_request rows, all years)
    Inserted after the building type market_share_total rows.

Vintage bin shares  (market_share_total, year 2000)
    Inserted after each Vintage technology row for High Density
    and LowMed Density separately.

Heating market shares  (market_share_total, year 2000)
    Inserted after the lifetime rows of each Heating (Cold) technology
    in every vintage × bldg-code context.  For BC only, the same is done
    for Heating (Marine).

Cooling shares  (service_request rows, all years)
    Inserted after the inheritance row of each density's Cooling service.

Water Heating density split  (service_request rows, all years)
    Inserted after the Water Heating competition row.

Water Heating technology shares  (market_share_total, year 2000)
    Inserted after the lifetime rows of each WH technology.

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

_res_spec = importlib.util.spec_from_file_location(
    'residential',
    _PIPELINE_ROOT / 'source' / 'nrcan' / 'ceud' / 'residential' / 'residential.py',
)
_residential_mod = importlib.util.module_from_spec(_res_spec)
_res_spec.loader.exec_module(_residential_mod)

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/residential'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/residential'

REGIONS = [
    'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU',
    'ON', 'PE', 'QC', 'SK', 'YT',
]

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Energies whose price target is region-specific (CIMS.CAN.{region}.{energy})
REGION_SPECIFIC_ENERGIES: set[str] = {
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
}

# Pipeline building-type category → CIMS Building Type technology name
PIPELINE_TO_CIMS_BUILDING: dict[str, str] = {
    'Apartments':      'Apartment',
    'Single Detached': 'Detached',
    'Single Attached': 'Attached',
    'Mobile Homes':    'Mobile',
}

# CIMS Building Type technology → density sub-service name
CIMS_BUILDING_TO_DENSITY: dict[str, str] = {
    'Apartment': 'High Density',
    'Detached':  'LowMed Density',
    'Attached':  'LowMed Density',
    'Mobile':    'LowMed Density',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    fixed_path = FIXED_INPUT_DIR / f'residential_{region.lower()}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'residential_{region.lower()}.csv'
        _flatten_mod.process_file(
            input_path=fixed_path,
            output_path=out_file,
            year_min=DATA_START,
            year_max=PROJECTION_END,
            target_start=DATA_START,
            target_end=PROJECTION_END,
            target_step=1,
        )
        df = pl.read_csv(out_file, infer_schema_length=0)
    return df.with_row_index('_order')


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']}
    )


def _find_max_order(df: pl.DataFrame, service: str, parameter: str,
                    require_tech: bool = False) -> float | None:
    mask = (pl.col('Service') == service) & (pl.col('Parameter') == parameter)
    if require_tech:
        mask = mask & pl.col('Technology').is_not_null() & (pl.col('Technology') != '')
    subset = df.filter(mask)
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


def _build_housing_rows(residential: pl.DataFrame, region: str,
                         start_order: float) -> pl.DataFrame:
    """Region-level service_request rows from housing_thousand pipeline data."""
    data = (
        residential
        .filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'housing_thousand')
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
        pl.lit('Residential').alias('Sector'),
        pl.lit('').alias('Service'),
        pl.lit('').alias('Technology'),
        pl.lit('service_request').alias('Parameter'),
        pl.lit('').alias('Context'),
        pl.lit('').alias('Sub_Context'),
        pl.lit(f'CIMS.CAN.{region}.Residential').alias('Target'),
        pl.col('source').alias('Source'),
        pl.col('unit').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('value').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _build_price_mult_rows(multipliers: pl.DataFrame, region: str,
                            start_order: float) -> pl.DataFrame:
    """multiplier_price rows for the Residential sector."""
    data = (
        multipliers
        .filter(
            (pl.col('Sector') == 'Residential') &
            (pl.col('Region') == region)
        )
        .sort('Energy', 'Year')
    )
    n = len(data)
    if n == 0:
        return _empty_frame()
    return data.select([
        pl.lit(f'CIMS.CAN.{region}.Residential').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Residential').alias('Sector'),
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


def _build_appliance_rows(residential: pl.DataFrame, region: str,
                           insert_order: float) -> pl.DataFrame:
    """service_request rows (all years) from Dwellings to each appliance sub-service."""
    data = (
        residential
        .filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'appliances_per_household')
        )
        .sort('category', 'year')
    )
    if len(data) == 0:
        return _empty_frame()
    branch = f'CIMS.CAN.{region}.Residential.Dwellings'
    rows: list[dict] = []
    for r in data.iter_rows(named=True):
        rows.append({
            'Branch': branch,
            'Type': 'Service',
            'Region': region,
            'Sector': 'Residential',
            'Service': 'Dwellings',
            'Technology': '',
            'Parameter': 'service_request',
            'Context': '',
            'Sub_Context': '',
            'Target': f'{branch}.{r["category"]}',
            'Source': r['source'],
            'Unit': r['unit'],
            'Year': str(r['year']),
            'Value': str(r['value']),
            '_order': insert_order,
        })
    return pl.DataFrame(rows)


def _build_building_type_rows(residential: pl.DataFrame, fixed: pl.DataFrame,
                               region: str) -> pl.DataFrame:
    """
    Per Building Type technology: market_share_total (all years) at tech_order+0.3
    and service_request floorspace rows (all years) at tech_order+0.6.
    """
    bt_branch = f'CIMS.CAN.{region}.Residential.Dwellings.Building Type'

    # Pipeline building shares — all years, keyed by CIMS tech name
    bs_data = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'building_shares')
    ).sort('category', 'year')
    bs_by_tech: dict[str, list] = {}
    for r in bs_data.iter_rows(named=True):
        cims_tech = PIPELINE_TO_CIMS_BUILDING.get(r['category'], r['category'])
        bs_by_tech.setdefault(cims_tech, []).append(
            (r['year'], r['value'], r['source'])
        )
    bs_unit = bs_data['unit'][0] if len(bs_data) > 0 else '%'

    # Pipeline floorspace lookup: {cims_tech: [(year, value, source, unit)]}
    fs_data = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'floorspace_per_building')
    ).sort('category', 'year')
    fs_by_tech: dict[str, list] = {}
    for r in fs_data.iter_rows(named=True):
        cims_tech = PIPELINE_TO_CIMS_BUILDING.get(r['category'], r['category'])
        fs_by_tech.setdefault(cims_tech, []).append(
            (r['year'], r['value'], r['source'], r['unit'])
        )

    # Find Building Type technology rows in fixed data
    bt_tech_rows = fixed.filter(
        (pl.col('Service') == 'Building Type') &
        (pl.col('Parameter') == 'technology') &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology') != '')
    )

    rows: list[dict] = []
    for r in bt_tech_rows.iter_rows(named=True):
        tech = r['Technology']
        tech_order = float(r['_order'])
        density = CIMS_BUILDING_TO_DENSITY.get(tech, 'LowMed Density')
        density_target = f'{bt_branch}.{density}'

        # market_share_total — all years at tech_order + 0.3
        for i, (year, value, source) in enumerate(bs_by_tech.get(tech, [])):
            rows.append({
                'Branch': bt_branch, 'Type': 'Service', 'Region': region,
                'Sector': 'Residential', 'Service': 'Building Type',
                'Technology': tech,
                'Parameter': 'market_share_total',
                'Context': '', 'Sub_Context': '', 'Target': '',
                'Source': source,
                'Unit': bs_unit, 'Year': str(year),
                'Value': str(value),
                '_order': tech_order + 0.3 + i * 1e-4,
            })

        # service_request (all years) at tech_order + 0.6
        for year, value, source, unit in fs_by_tech.get(tech, []):
            rows.append({
                'Branch': bt_branch, 'Type': 'Service', 'Region': region,
                'Sector': 'Residential', 'Service': 'Building Type',
                'Technology': tech,
                'Parameter': 'service_request',
                'Context': '', 'Sub_Context': '',
                'Target': density_target,
                'Source': source, 'Unit': unit,
                'Year': str(year), 'Value': str(value),
                '_order': tech_order + 0.6,
            })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_vintage_bin_rows(residential: pl.DataFrame, fixed: pl.DataFrame,
                             region: str) -> pl.DataFrame:
    """market_share_total (year 2000) after each Vintage technology row."""
    rows: list[dict] = []

    for variable, density in [
        ('vintage_bins_high',   'High Density'),
        ('vintage_bins_lowmed', 'LowMed Density'),
    ]:
        data = residential.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == variable) &
            (pl.col('year') == 2000)
        )
        pipe_vals: dict[str, float] = {}
        pipe_sources: dict[str, str] = {}
        for r in data.iter_rows(named=True):
            pipe_vals[r['category']] = r['value']
            pipe_sources[r['category']] = r['source']
        pipe_unit = data['unit'][0] if len(data) > 0 else '%'

        vint_tech_rows = fixed.filter(
            (pl.col('Service') == 'Vintage') &
            (pl.col('Parameter') == 'technology') &
            pl.col('Branch').str.contains(f'{density}.Vintage') &
            pl.col('Technology').is_not_null() &
            (pl.col('Technology') != '')
        )

        for r in vint_tech_rows.iter_rows(named=True):
            tech = r['Technology']
            if tech not in pipe_vals:
                continue  # >2035 bin has no pipeline data
            rows.append({
                'Branch': r['Branch'], 'Type': 'Service', 'Region': region,
                'Sector': 'Residential', 'Service': 'Vintage',
                'Technology': tech,
                'Parameter': 'market_share_total',
                'Context': '', 'Sub_Context': '', 'Target': '',
                'Source': pipe_sources.get(tech, 'CEUD'),
                'Unit': pipe_unit, 'Year': '2000',
                'Value': str(pipe_vals[tech]),
                '_order': float(r['_order']) + 0.5,
            })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_heating_mst_rows(residential: pl.DataFrame, fixed: pl.DataFrame,
                              region: str, variable: str,
                              service_name: str, density: str) -> pl.DataFrame:
    """
    market_share_total (year 2000) after each heating technology's lifetime rows.

    One MST row is inserted per (Branch, Technology) pair — positioned at the
    max _order of that pair's lifetime rows + 0.5, placing it after the last
    annual lifetime row and before the output block.
    """
    data = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == variable) &
        (pl.col('year') == 2000)
    )
    pipe_vals: dict[str, float] = {}
    pipe_sources: dict[str, str] = {}
    for r in data.iter_rows(named=True):
        pipe_vals[r['category']] = r['value']
        pipe_sources[r['category']] = r['source']
    pipe_unit = data['unit'][0] if len(data) > 0 else '%'

    lifetime_rows = fixed.filter(
        (pl.col('Service') == service_name) &
        (pl.col('Parameter') == 'lifetime') &
        pl.col('Branch').str.contains(density) &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology') != '')
    )

    # Max _order per (Branch, Technology) — one MST insertion per context
    branch_tech_max: dict[tuple, float] = {}
    for r in lifetime_rows.iter_rows(named=True):
        key = (r['Branch'], r['Technology'])
        o = float(r['_order'])
        if key not in branch_tech_max or o > branch_tech_max[key]:
            branch_tech_max[key] = o

    rows: list[dict] = []
    for (branch, tech), max_order in branch_tech_max.items():
        rows.append({
            'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Residential', 'Service': service_name,
            'Technology': tech,
            'Parameter': 'market_share_total',
            'Context': '', 'Sub_Context': '', 'Target': '',
            'Source': pipe_sources.get(tech, 'CEUD'),
            'Unit': pipe_unit, 'Year': '2000',
            'Value': str(pipe_vals.get(tech, 0.0)),
            '_order': max_order + 0.5,
        })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_cooling_rows(residential: pl.DataFrame, fixed: pl.DataFrame,
                         region: str, density: str) -> pl.DataFrame:
    """service_request rows (all years) after the Cooling inheritance row."""
    data = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'cooling_share_data')
    ).sort('category', 'year')

    if len(data) == 0:
        return _empty_frame()

    cool_branch = (
        f'CIMS.CAN.{region}.Residential.Dwellings'
        f'.Building Type.{density}.Cooling'
    )
    inherit_rows = fixed.filter(
        (pl.col('Service') == 'Cooling') &
        (pl.col('Parameter') == 'inheritance') &
        (pl.col('Branch') == cool_branch)
    )
    if len(inherit_rows) == 0:
        return _empty_frame()
    insert_order = float(inherit_rows['_order'].max()) + 0.5

    rows: list[dict] = []
    for i, r in enumerate(data.iter_rows(named=True)):
        rows.append({
            'Branch': cool_branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Residential', 'Service': 'Cooling',
            'Technology': '',
            'Parameter': 'service_request',
            'Context': '', 'Sub_Context': '',
            'Target': f'{cool_branch}.{r["category"]}',
            'Source': r['source'], 'Unit': r['unit'],
            'Year': str(r['year']), 'Value': str(r['value']),
            '_order': insert_order + i * 1e-4,
        })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_wh_split_rows(residential: pl.DataFrame, region: str,
                          insert_order: float) -> pl.DataFrame:
    """
    service_request rows (all years) splitting Water Heating demand between
    LowMed Density and High Density sub-services.
    """
    wh_branch = f'CIMS.CAN.{region}.Residential.Water Heating'
    rows: list[dict] = []

    for variable, target_suffix in [
        ('wh_lowmed', 'LowMed Density'),
        ('wh_high',   'High Density'),
    ]:
        data = residential.filter(
            (pl.col('province') == region) &
            (pl.col('variable') == variable)
        ).sort('year')
        for r in data.iter_rows(named=True):
            rows.append({
                'Branch': wh_branch, 'Type': 'Service', 'Region': region,
                'Sector': 'Residential', 'Service': 'Water Heating',
                'Technology': '',
                'Parameter': 'service_request',
                'Context': '', 'Sub_Context': '',
                'Target': f'{wh_branch}.{target_suffix}',
                'Source': r['source'], 'Unit': r['unit'],
                'Year': str(r['year']), 'Value': str(r['value']),
                '_order': insert_order,
            })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _build_wh_tech_mst_rows(residential: pl.DataFrame, fixed: pl.DataFrame,
                              region: str, variable: str,
                              wh_service: str) -> pl.DataFrame:
    """
    market_share_total (year 2000) after each WH technology's lifetime rows.

    Filters Branch to 'Water Heating' to avoid matching Building Type density
    services that share the same Service name.

    Shares are renormalized to sum to 1 within each branch: CEUD's raw shares
    can include minor fuels (e.g. kerosene, LPG, solid biomass) that have no
    corresponding fixed_data technology to land on, which would otherwise
    leave the defined technologies' total short of 1 (e.g. NS/PE LowMed
    Density summing to ~0.9948 instead of 1).
    """
    data = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == variable) &
        (pl.col('year') == 2000)
    )
    pipe_vals: dict[str, float] = {}
    pipe_sources: dict[str, str] = {}
    for r in data.iter_rows(named=True):
        pipe_vals[r['category']] = r['value']
        pipe_sources[r['category']] = r['source']
    pipe_unit = data['unit'][0] if len(data) > 0 else '%'

    lifetime_rows = fixed.filter(
        (pl.col('Service') == wh_service) &
        (pl.col('Parameter') == 'lifetime') &
        pl.col('Branch').str.contains('Water Heating') &
        pl.col('Technology').is_not_null() &
        (pl.col('Technology') != '')
    )

    branch_tech_max: dict[tuple, float] = {}
    for r in lifetime_rows.iter_rows(named=True):
        key = (r['Branch'], r['Technology'])
        o = float(r['_order'])
        if key not in branch_tech_max or o > branch_tech_max[key]:
            branch_tech_max[key] = o

    branch_totals: dict[str, float] = {}
    for branch, tech in branch_tech_max:
        branch_totals[branch] = branch_totals.get(branch, 0.0) + pipe_vals.get(tech, 0.0)

    rows: list[dict] = []
    for (branch, tech), max_order in branch_tech_max.items():
        total = branch_totals.get(branch, 0.0)
        raw_value = pipe_vals.get(tech, 0.0)
        value = raw_value / total if total > 0 else raw_value
        rows.append({
            'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Residential', 'Service': wh_service,
            'Technology': tech,
            'Parameter': 'market_share_total',
            'Context': '', 'Sub_Context': '', 'Target': '',
            'Source': pipe_sources.get(tech, 'CEUD'),
            'Unit': pipe_unit, 'Year': '2000',
            'Value': str(value),
            '_order': max_order + 0.5,
        })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _assemble_region(fixed: pl.DataFrame, residential: pl.DataFrame,
                      multipliers: pl.DataFrame, region: str) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with pipeline-derived rows at the correct positions.
    """
    # 1. Housing before all fixed data
    housing = _build_housing_rows(
        residential, region,
        start_order=float(fixed['_order'].min()) - 1000.0,
    )

    # 2. Price multipliers after Residential sector header
    res_header_max = float(
        fixed.filter(
            (pl.col('Branch') == f'CIMS.CAN.{region}.Residential') &
            pl.col('Parameter').is_in(['service_provide', 'competition'])
        )['_order'].max()
    )
    prices = _build_price_mult_rows(multipliers, region, res_header_max + 0.5)

    # 3. Appliances after Dwellings service_request → Building Type
    bt_sr_rows = fixed.filter(
        (pl.col('Service') == 'Dwellings') &
        (pl.col('Parameter') == 'service_request') &
        pl.col('Target').str.ends_with('.Building Type')
    )
    bt_sr_order = (
        float(bt_sr_rows['_order'].max()) if len(bt_sr_rows) > 0
        else res_header_max + 2
    )
    appliances = _build_appliance_rows(residential, region, bt_sr_order + 0.5)

    # 4. Building type market_share_total + service_request per technology
    bt_rows = _build_building_type_rows(residential, fixed, region)

    # 5. Vintage bin market_share_total (year 2000)
    vb_rows = _build_vintage_bin_rows(residential, fixed, region)

    # 6. Heating market_share_total (year 2000): Cold for all; Marine for BC only
    heat_cold_high = _build_heating_mst_rows(
        residential, fixed, region,
        'heating_high_cold', 'Heating (Cold)', 'High Density',
    )
    heat_cold_lm = _build_heating_mst_rows(
        residential, fixed, region,
        'heating_lowmed_cold', 'Heating (Cold)', 'LowMed Density',
    )
    heat_mar_high = _empty_frame()
    heat_mar_lm   = _empty_frame()
    if region == 'BC':
        heat_mar_high = _build_heating_mst_rows(
            residential, fixed, region,
            'heating_high_marine', 'Heating (Marine)', 'High Density',
        )
        heat_mar_lm = _build_heating_mst_rows(
            residential, fixed, region,
            'heating_lowmed_marine', 'Heating (Marine)', 'LowMed Density',
        )

    # 7. Cooling service_request after each density's inheritance row
    cool_high = _build_cooling_rows(residential, fixed, region, 'High Density')
    cool_lm   = _build_cooling_rows(residential, fixed, region, 'LowMed Density')

    # 8. Water Heating density split after Water Heating competition
    wh_comp_order = _find_max_order(fixed, 'Water Heating', 'competition')
    if wh_comp_order is None:
        wh_comp_order = res_header_max + 10
    wh_split = _build_wh_split_rows(residential, region, wh_comp_order + 0.5)

    # 9. WH technology market_share_total (year 2000)
    wh_mst_lm   = _build_wh_tech_mst_rows(
        residential, fixed, region, 'wh_tech_lowmed', 'LowMed Density'
    )
    wh_mst_high = _build_wh_tech_mst_rows(
        residential, fixed, region, 'wh_tech_high', 'High Density'
    )

    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        housing, prices, appliances, bt_rows, vb_rows,
        heat_cold_high, heat_cold_lm, heat_mar_high, heat_mar_lm,
        cool_high, cool_lm, wh_split, wh_mst_lm, wh_mst_high,
    ]

    combined = pl.concat(
        [f for f in all_frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble residential model inputs and write one CSV per region."""
    print('=' * 60)
    print('RESIDENTIAL MODEL INPUTS')
    print('=' * 60)

    print('\nLoading pipeline data...')
    _residential_results = _residential_mod.main(export_csv=False)
    residential = (
        pl.concat(list(_residential_results.values()), how='diagonal_relaxed')
        .with_columns(
            pl.when(pl.col('year') <= _residential_mod.LAST_HIST_YEAR)
            .then(pl.lit('CEUD'))
            .otherwise(pl.lit('Assumptions'))
            .alias('source')
        )
    )
    multipliers = pl.from_pandas(_energy_price_mod.main())
    print(
        f'  Residential data: {len(residential):,} rows, '
        f'regions: {sorted(residential["province"].unique().to_list())}'
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region in sorted(REGIONS):
        fixed_path = FIXED_INPUT_DIR / f'residential_{region.lower()}.csv'
        if not fixed_path.exists():
            print(f'  Skipping {region} — fixed data not found: {fixed_path.name}')
            continue
        if region not in residential['province'].unique().to_list():
            print(f'  Skipping {region} — no pipeline data for this region')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            fixed = _read_flattened_fixed(region)

            print('  Assembling...')
            output = _assemble_region(fixed, residential, multipliers, region)
            output = collapse_constant_years(output)

            out_path = OUTPUT_DIR / f'residential_{region.lower()}.csv'
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
    print(f'Regions complete: {len(results)}/{len(REGIONS)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
