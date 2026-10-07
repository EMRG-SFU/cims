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

Space-heating intensity  (service_request rows, all years)
    Replaces the fixed-data Reference / Retrofit service_request rows from
    each Vintage "<bin> Bldg Code" node to its Heating node(s) with the CEUD
    intensity from residential_heating_intensity.py (GJ of heat per m2),
    averaged over HEATING_INTENSITY_YEARS.  Retrofit techs keep their
    fixed-data ratios to Reference.  BC's Cold/Marine split is set by
    BC_MARINE_HEAT_SHARE (None keeps the fixed-data split, ~63% Marine).

Weather nodes  (WEATHER_NODES; service_provide / competition / service_request)
    A Fixed Ratio "Weather (Cold|Marine)" node is inserted before each
    Heating node, and the Bldg Code technologies request it instead of the
    Heating node.  Its service_request to Heating is the CEUD Heating
    Degree-Day Index (historical years; mean of the last
    HDD_PROJECTION_YEARS thereafter), and the Bldg Code intensity is the
    weather-normalised mean (CEUD intensity / HDD index).  A node without
    technologies isn't vintage-weighted, so the year-to-year weather signal
    reaches all floor space rather than only new stock.

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

import tempfile
from pathlib import Path
from typing import Optional

import polars as pl

# ── path setup ─────────────────────────────────────────────────────────────────
import CIMS.data_processing.utils.flatten_fixed_data as _flatten_mod

import CIMS.data_processing.source.nrcan.ceud.residential.residential as _residential_mod

import CIMS.data_processing.source.energy_prices.energy_price_multipliers as _energy_price_mod

import CIMS.data_processing.source.nrcan.ceud.residential.residential_heating_intensity as _heating_mod

from CIMS.data_processing.utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/residential'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/residential'
LIGHTING_ESTIMATED_MS_DIR = BASE_PATH / 'calibration/calibration_estimated/residential'

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
    'Electricity', 'Biodiesel',
    'Ethanol', 'Hydrogen',
}

# Space-heating intensity (Vintage Bldg Code -> Heating service_request).
# CIMS vintage-weights technology service_request, so the pre-2001 vintages
# (base stock only, never any new stock) always use their base-year value and
# later bins average over their stock vintages -- year-by-year values can't
# reach the model. A single historical mean is used instead.
HEATING_INTENSITY_YEARS: tuple[int, int] = (DATA_START, _heating_mod.LAST_HIST_YEAR)
HEATING_SERVICES: tuple[str, ...] = ('Heating (Cold)', 'Heating (Marine)')
# BC share of each Bldg Code technology's heating request that goes to
# Heating (Marine); the rest goes to Heating (Cold). 0.75 reflects ~79% of
# BC's population in the coastal (Marine) zone, weighted by the Cold zone's
# higher HDD and floor space per person. None keeps the fixed-data (JCIMS)
# split, ~63% Marine.
BC_MARINE_HEAT_SHARE: Optional[float] = 0.75
DENSITY_TO_INTENSITY_VARIABLE: dict[str, str] = {
    'High Density':   'heating_intensity_high',
    'LowMed Density': 'heating_intensity_lowmed',
}

# Weather nodes: Fixed Ratio "Weather (<climate>)" node between each Bldg Code
# node and its Heating node, carrying the CEUD HDD index (see module docstring).
# Its node-level service_request isn't vintage-weighted.
WEATHER_NODES: bool = True
# Projection-year weather factor = mean HDD index over the last N CEUD years.
HDD_PROJECTION_YEARS: int = 10

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


def _ceud_series(residential: pl.DataFrame, region: str, variable: str) -> pl.DataFrame:
    """All-years rows for a scalar (category-less) CEUD variable, sorted by year."""
    return (
        residential
        .filter(
            (pl.col('province') == region) &
            (pl.col('variable') == variable)
        )
        .sort('year')
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


def _build_appliance_rows(residential: pl.DataFrame, fixed: pl.DataFrame, region: str,
                           insert_order: float) -> pl.DataFrame:
    """
    service_request rows (all years) from Dwellings to each appliance sub-service.

    Minor Appliances is special-cased: CEUD's raw appliance-count index
    (Table 31 "Other Appliances", ~13-20 "units"/household) isn't a literal
    quantity compatible with the fixed_data Existing-technology GJ/unit
    factor (a flat, nationally-fixed JCIMS assumption) -- multiplying the two
    overstates minor-appliance energy by roughly 20x. Instead, the quantity
    sent is CEUD's actual Minor Appliances energy use (GJ/household, from
    Table 13) divided by that same Existing-tech factor, which reproduces
    CEUD's real historical total under the model's year-2000 benchmark
    convention (Existing = 100% share in the base year).
    """
    data = (
        residential
        .filter(
            (pl.col('province') == region) &
            (pl.col('variable') == 'appliances_per_household') &
            (pl.col('category') != 'Minor Appliances')
        )
        .sort('category', 'year')
    )

    minor_gj = _ceud_series(residential, region, 'minor_appliance_gj')
    minor_rows = pl.DataFrame()
    if len(minor_gj) > 0:
        existing = fixed.filter(
            (pl.col('Service') == 'Minor Appliances') &
            (pl.col('Technology') == 'Existing') &
            (pl.col('Parameter') == 'service_request')
        )
        if len(existing) > 0:
            gj_per_unit = float(existing['Value'][0])
            minor_rows = minor_gj.with_columns(
                (pl.col('value') / gj_per_unit).alias('value'),
                pl.lit('unit/building').alias('unit'),
                pl.lit('Minor Appliances').alias('category'),
            )

    combined = (
        pl.concat([data, minor_rows], how='diagonal_relaxed')
        if len(minor_rows) > 0 else data
    )
    if len(combined) == 0:
        return _empty_frame()

    branch = f'CIMS.CAN.{region}.Residential.Dwellings'
    rows: list[dict] = []
    for r in combined.sort(['category', 'year']).iter_rows(named=True):
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
                              wh_service: str, year: int = 2000) -> pl.DataFrame:
    """
    market_share_total (at `year`, 2000 by default) after each WH
    technology's lifetime rows.

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
        (pl.col('year') == year)
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
            'Unit': pipe_unit, 'Year': str(year),
            'Value': str(value),
            '_order': max_order + 0.5,
        })

    return pl.DataFrame(rows) if rows else _empty_frame()


def _annual_rows_with_continuation(branch: str, service: str, technology: str,
                                    target: str, region: str, source: str, unit: str,
                                    order: float, values_by_year: dict[int, float]) -> pl.DataFrame:
    """
    Build one service_request row per year from `values_by_year`'s earliest
    year through PROJECTION_END, holding flat at the last available (CEUD-
    covered) year's value for every year beyond it -- `collapse_constant_years`
    later folds that flat tail back into a single default row, matching the
    convention used throughout this module for CEUD-derived series.
    """
    last_hist_year = max(values_by_year.keys())
    rows: list[dict] = []
    for i, year in enumerate(range(min(values_by_year.keys()), PROJECTION_END + 1)):
        value = values_by_year.get(year, values_by_year[last_hist_year])
        rows.append({
            'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Residential', 'Service': service,
            'Technology': technology or '',
            'Parameter': 'service_request',
            'Context': '', 'Sub_Context': '',
            'Target': target,
            'Source': source, 'Unit': unit,
            'Year': str(year), 'Value': str(value),
            '_order': order + i * 1e-4,
        })
    return pl.DataFrame(rows)


def _density_floorspace_series(residential: pl.DataFrame, region: str) -> dict[str, dict[int, float]]:
    """
    Density -> year -> total floor space (m2), built the same way the CIMS
    engine itself accumulates it onto Building Type.{Density}: each building
    type's floorspace_per_building x building_shares x total households,
    summed over the building types that map to that density.
    """
    fs = {
        (r['category'], int(r['year'])): float(r['value'])
        for r in residential.filter(
            (pl.col('province') == region) & (pl.col('variable') == 'floorspace_per_building')
        ).iter_rows(named=True)
    }
    bs = {
        (r['category'], int(r['year'])): float(r['value'])
        for r in residential.filter(
            (pl.col('province') == region) & (pl.col('variable') == 'building_shares')
        ).iter_rows(named=True)
    }
    households = {
        int(r['year']): float(r['value'])
        for r in residential.filter(
            (pl.col('province') == region) & (pl.col('variable') == 'housing_thousand')
        ).iter_rows(named=True)
    }

    totals: dict[str, dict[int, float]] = {'High Density': {}, 'LowMed Density': {}}
    for pipeline_cat, cims_tech in PIPELINE_TO_CIMS_BUILDING.items():
        density = CIMS_BUILDING_TO_DENSITY[cims_tech]
        for year, hh in households.items():
            fsv = fs.get((pipeline_cat, year))
            bsv = bs.get((pipeline_cat, year))
            if fsv is None or bsv is None:
                continue
            totals[density][year] = totals[density].get(year, 0.0) + fsv * bsv * hh
    return totals


def _replace_cooling_intensity(residential: pl.DataFrame, fixed: pl.DataFrame,
                                region: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Recalibrate the flat fixed_data Cooling intensity constant (Building
    Type.{Density}.Cooling service_request, identical for both densities) so
    that Cooling's TOTAL demand -- summed across both densities' floor space,
    split Room/Central, and shrunk by each system type's own technology mix
    -- matches CEUD's real annual "Total Space Cooling Energy Use" (Table 4).
    Same reconciliation approach as Water Heating (see `_replace_wh_intensity`):
    the original per-m2 constant was set without this downstream shrinkage in
    mind, so matching CEUD's per-m2 intensity directly at this node
    systematically misses the target (sometimes high, sometimes low --
    the original constants aren't consistently calibrated against it).
    """
    bt_branch = f'CIMS.CAN.{region}.Residential.Dwellings.Building Type'
    old_by_density: dict[str, pl.DataFrame] = {}
    drop_mask = pl.lit(False)
    for density in ['High Density', 'LowMed Density']:
        branch = f'{bt_branch}.{density}'
        target = f'{branch}.Cooling'
        mask = (
            (pl.col('Branch') == branch) &
            (pl.col('Parameter') == 'service_request') &
            (pl.col('Target') == target) &
            (pl.col('Unit') == 'GJ')
        )
        old = fixed.filter(mask)
        if len(old) > 0:
            old_by_density[density] = old
            drop_mask = drop_mask | mask
    if not old_by_density:
        return fixed, _empty_frame()

    room_factor = _weighted_tech_factor_2000(fixed, 'Room', branch_contains='Cooling')
    central_factor = _weighted_tech_factor_2000(fixed, 'Central', branch_contains='Cooling')
    if room_factor is None or central_factor is None:
        return fixed, _empty_frame()

    cooling_total_pj = _ceud_series(residential, region, 'cooling_total_pj')
    cooling_shares = _ceud_series(residential, region, 'cooling_share_data')
    room_share = cooling_shares.filter(pl.col('category') == 'Room')
    central_share = cooling_shares.filter(pl.col('category') == 'Central')
    floorspace = _density_floorspace_series(residential, region)

    def _lookup(df: pl.DataFrame, year: int) -> float:
        row = df.filter(pl.col('year') == year)
        return float(row['value'][0]) if len(row) > 0 else 0.0

    years = [y for y in cooling_total_pj['year'].to_list() if y <= _residential_mod.LAST_HIST_YEAR]
    const_by_year: dict[int, float] = {}
    for year in years:
        downstream_factor = (
            _lookup(room_share, year) * room_factor + _lookup(central_share, year) * central_factor
        )
        total_floorspace = floorspace['High Density'].get(year, 0.0) + floorspace['LowMed Density'].get(year, 0.0)
        if downstream_factor <= 0 or total_floorspace <= 0:
            continue
        target_gj = _lookup(cooling_total_pj, year) * 1e6
        const_by_year[year] = target_gj / (total_floorspace * downstream_factor)

    if not const_by_year:
        return fixed, _empty_frame()

    frames: list[pl.DataFrame] = []
    for density, old in old_by_density.items():
        branch = f'{bt_branch}.{density}'
        target = f'{branch}.Cooling'
        frames.append(_annual_rows_with_continuation(
            branch, density, '', target, region,
            old['Source'][0], old['Unit'][0], float(old['_order'].min()), const_by_year,
        ))

    fixed = fixed.filter(~drop_mask)
    return fixed, pl.concat(frames, how='diagonal_relaxed')


def _estimated_lighting_factor_by_year(fixed: pl.DataFrame, region: str) -> Optional[dict[int, float]]:
    """
    Year -> weighted-average GJ/unit conversion factor for Lighting, built
    from the Incandescent/CFL/LED `estimated_market_share_total` series in
    `LIGHTING_ESTIMATED_MS_DIR` (an external bulb-penetration estimate --
    CEUD itself doesn't survey lighting by technology, so this is a stand-in
    for the real annual mix) weighted against each technology's fixed_data
    GJ/unit at the Lighting node. Returns None if no estimated-mix file
    exists for this region, so the caller can fall back to
    `_weighted_tech_factor_2000`.
    """
    ms_path = LIGHTING_ESTIMATED_MS_DIR / f'estimated_market_share_total_{region.lower()}.csv'
    if not ms_path.exists():
        return None

    # GJ/unit is a flat constant per technology in fixed_data, so after
    # flatten_fixed_data's constant-year collapsing it lands on a single row
    # with Year == None rather than an explicit '2000' -- prefer an exact
    # '2000' row if one exists (mirrors _weighted_tech_factor_2000's lookup),
    # but fall back to whatever row is there instead of matching nothing.
    sr_rows = fixed.filter(
        (pl.col('Service') == 'Lighting') &
        (pl.col('Parameter') == 'service_request') &
        pl.col('Technology').is_not_null() & (pl.col('Technology') != '')
    )
    gj_per_unit: dict[str, float] = {}
    for tech in sr_rows['Technology'].unique().to_list():
        tech_rows = sr_rows.filter(pl.col('Technology') == tech)
        exact = tech_rows.filter(pl.col('Year') == '2000')
        row = exact if len(exact) > 0 else tech_rows
        gj_per_unit[tech] = float(row['Value'][0])
    if not gj_per_unit:
        return None

    ms = pl.read_csv(ms_path, infer_schema_length=0)
    factor_by_year: dict[int, float] = {}
    for row in ms.iter_rows(named=True):
        gj = gj_per_unit.get(row['Technology'])
        if gj is None:
            continue
        year = int(row['Year'])
        factor_by_year[year] = factor_by_year.get(year, 0.0) + float(row['Value']) * gj
    return factor_by_year or None


def _replace_lighting_total(residential: pl.DataFrame, fixed: pl.DataFrame,
                             region: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Recalibrate the flat fixed_data Lighting constant (Building
    Type.{Density}.Lighting service_request, identical for both densities and
    nationally uniform) so that Lighting's TOTAL demand -- summed across both
    densities' floor space and shrunk by the Incandescent/CFL/LED technology
    mix -- matches CEUD's real annual "Total Lighting Energy Use" (Table 3).

    CEUD doesn't survey lighting by bulb technology, so the technology-mix
    side of the conversion uses `_estimated_lighting_factor_by_year` (an
    external per-year Incandescent/CFL/LED estimate) where available,
    falling back to `_weighted_tech_factor_2000`'s single 2000-mix snapshot
    for any region missing that file. Either way, the resulting constant is
    still recalibrated per-year, since floor space and household counts grow
    over the historical period even independent of the technology mix.
    """
    bt_branch = f'CIMS.CAN.{region}.Residential.Dwellings.Building Type'
    lighting_target = f'CIMS.CAN.{region}.Residential.Dwellings.Lighting'
    old_by_density: dict[str, pl.DataFrame] = {}
    drop_mask = pl.lit(False)
    for density in ['High Density', 'LowMed Density']:
        branch = f'{bt_branch}.{density}'
        mask = (
            (pl.col('Branch') == branch) &
            (pl.col('Parameter') == 'service_request') &
            (pl.col('Target') == lighting_target)
        )
        old = fixed.filter(mask)
        if len(old) > 0:
            old_by_density[density] = old
            drop_mask = drop_mask | mask
    if not old_by_density:
        return fixed, _empty_frame()

    lighting_factor_by_year = _estimated_lighting_factor_by_year(fixed, region)
    fallback_factor = _weighted_tech_factor_2000(fixed, 'Lighting')
    if not lighting_factor_by_year and fallback_factor is None:
        return fixed, _empty_frame()

    def _factor_for(year: int) -> Optional[float]:
        if lighting_factor_by_year:
            if year in lighting_factor_by_year:
                return lighting_factor_by_year[year]
            nearest = min(lighting_factor_by_year, key=lambda yy: abs(yy - year))
            return lighting_factor_by_year[nearest]
        return fallback_factor

    lighting_total_pj = _ceud_series(residential, region, 'lighting_total_pj')
    floorspace = _density_floorspace_series(residential, region)

    def _lookup(df: pl.DataFrame, year: int) -> float:
        row = df.filter(pl.col('year') == year)
        return float(row['value'][0]) if len(row) > 0 else 0.0

    years = [y for y in lighting_total_pj['year'].to_list() if y <= _residential_mod.LAST_HIST_YEAR]
    const_by_year: dict[int, float] = {}
    for year in years:
        total_floorspace = floorspace['High Density'].get(year, 0.0) + floorspace['LowMed Density'].get(year, 0.0)
        factor = _factor_for(year)
        if total_floorspace <= 0 or not factor:
            continue
        target_gj = _lookup(lighting_total_pj, year) * 1e6
        const_by_year[year] = target_gj / (total_floorspace * factor)

    if not const_by_year:
        return fixed, _empty_frame()

    frames: list[pl.DataFrame] = []
    for density, old in old_by_density.items():
        branch = f'{bt_branch}.{density}'
        frames.append(_annual_rows_with_continuation(
            branch, density, '', lighting_target, region,
            old['Source'][0], old['Unit'][0], float(old['_order'].min()), const_by_year,
        ))

    fixed = fixed.filter(~drop_mask)
    return fixed, pl.concat(frames, how='diagonal_relaxed')


def _reanchored_factor(factor_by_anchor: dict[int, float], year: int) -> Optional[float]:
    """
    Re-anchored downstream factor for `year`: its own value if it was
    computed, otherwise the nearest year that has one (e.g. a year CEUD
    reports as suppressed/missing for that particular technology mix).

    `factor_by_anchor` is keyed by every year being reconciled, so this re-
    derives the technology-mix-weighted downstream factor annually instead
    of freezing it at year 2000 (or any single anchor) forever -- a factor
    frozen at one year goes increasingly stale as the (separately, correctly)
    calibrated tech competition shifts the real technology mix over time --
    e.g. a fuel's efficiency-tier split can flip almost entirely within
    15-20 years -- which otherwise makes computed demand drift further from
    CEUD's real total every year even when the base year matched exactly.
    """
    if year in factor_by_anchor:
        return factor_by_anchor[year]
    if not factor_by_anchor:
        return None
    return factor_by_anchor[min(factor_by_anchor, key=lambda a: abs(a - year))]


def _weighted_tech_factor_2000(fixed: pl.DataFrame, service: str,
                                branch_contains: Optional[str] = None,
                                target_contains: Optional[str] = None,
                                shares: Optional[dict[str, float]] = None,
                                anchor_year: int = 2000) -> Optional[float]:
    """
    Weighted-average GJ/unit conversion factor for a Tech-Compete service's
    `anchor_year` technology mix: sum(market_share_total(tech) *
    service_request(tech)). A technology can send separate service_request
    rows to several different targets (e.g. a dishwasher sends both to
    Electricity and to Water Heating) -- pass `target_contains` to restrict
    to the row(s) relevant to one of them.

    `shares` lets a caller supply technology -> market_share_total directly
    instead of reading it from `fixed` -- needed for the Water Heating
    density nodes, whose fuel-technology market shares aren't in the raw
    fixed_data CSV at all; they're pipeline-inserted from CEUD
    (`_build_wh_tech_mst_rows`) rather than hand-curated.
    """
    mask = (pl.col('Service') == service)
    if branch_contains is not None:
        mask = mask & pl.col('Branch').str.contains(branch_contains, literal=True)
    base = fixed.filter(
        mask & pl.col('Technology').is_not_null() & (pl.col('Technology') != '')
    )
    if shares is None:
        shares = {
            r['Technology']: float(r['Value'])
            for r in base.filter(pl.col('Parameter') == 'market_share_total').iter_rows(named=True)
        }
    if not shares:
        return None

    sr = base.filter(pl.col('Parameter') == 'service_request')
    if target_contains is not None:
        sr = sr.filter(pl.col('Target').str.contains(target_contains, literal=True))

    weighted = 0.0
    for tech, share in shares.items():
        tech_rows = sr.filter(pl.col('Technology') == tech)
        if len(tech_rows) == 0:
            continue
        exact = tech_rows.filter(pl.col('Year') == str(anchor_year))
        row = exact if len(exact) > 0 else tech_rows
        weighted += share * float(row['Value'][0])
    return weighted if weighted > 0 else None


def _fixed_value_by_year(fixed: pl.DataFrame, branch: str, target: str,
                          years: list[int]) -> dict[int, float]:
    """
    Year -> value for a fixed_data service_request row (Branch, Target), for
    each year in `years`. Handles both a flat/constant row (a single blank-
    Year default applied to every year) and a genuine multi-year trend
    (Dishwashing's Machine/Non-machine split moves from 60/40 in 2000 to
    77/23 by 2050) the same way, falling back to the nearest explicit year
    if a requested year has neither an exact match nor a default.
    """
    rows = fixed.filter(
        (pl.col('Branch') == branch) &
        (pl.col('Parameter') == 'service_request') &
        (pl.col('Target') == target)
    )
    by_year = {
        int(r['Year']): float(r['Value'])
        for r in rows.iter_rows(named=True) if r['Year'] is not None
    }
    default = next(
        (float(r['Value']) for r in rows.iter_rows(named=True) if r['Year'] is None), None
    )

    out: dict[int, float] = {}
    for y in years:
        if y in by_year:
            out[y] = by_year[y]
        elif default is not None:
            out[y] = default
        elif by_year:
            out[y] = by_year[min(by_year, key=lambda yy: abs(yy - y))]
    return out


def _replace_wh_intensity(residential: pl.DataFrame, fixed: pl.DataFrame,
                           region: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Recalibrate the flat fixed_data 'unit' quantity sent from Dwellings to
    Non-appliance Hot Water so that Water Heating's TOTAL demand matches
    CEUD's real annual "Total Water Heating Energy Use" (Table 10).

    Water Heating actually has three independent contributors -- Non-
    appliance Hot Water, Dishwashing, and Clothes Washing -- which all get
    summed at the Water Heating node before it splits by density and each
    density's fuel-technology mix shrinks the total again. Matching CEUD's
    number at Non-appliance Hot Water alone (ignoring the other two, and
    ignoring the downstream density/fuel-tech shrinkage) systematically
    misses the target. Dishwashing and Clothes Washing's own GJ-per-cycle
    assumptions are fixed engineering inputs (not something CEUD reports),
    so per user direction they're left untouched -- Non-appliance Hot Water
    alone is solved for so that:

        (NAHW_contribution + Dishwashing_contribution + Clothes_Washing_contribution)
            x downstream_density_and_fueltech_factor
            == CEUD's actual annual Total Water Heating Energy Use (Table 10)

    downstream_density_and_fueltech_factor (the NAHW Std-use/Low-flow split
    and each density's fuel-technology mix) is re-anchored every year instead
    of frozen at 2000 (`_reanchored_factor`): that mix evolves via the
    same calibrated tech competition that sets market_share_total everywhere
    else (e.g. NG boiler efficiency tiers can flip almost entirely within
    15-20 years), so a factor frozen at year 2000 understates or overstates
    the real one by a growing amount every year past it, even in years this
    function otherwise reconciles exactly to CEUD's target.
    """
    dwellings_branch = f'CIMS.CAN.{region}.Residential.Dwellings'
    nahw_target = f'{dwellings_branch}.Non-appliance Hot Water'
    nahw_mask = (
        (pl.col('Branch') == dwellings_branch) &
        (pl.col('Parameter') == 'service_request') &
        (pl.col('Target') == nahw_target)
    )
    old = fixed.filter(nahw_mask)
    if len(old) == 0:
        return fixed, _empty_frame()

    dw_machine_factor = _weighted_tech_factor_2000(
        fixed, 'Machine', branch_contains='Dishwashing', target_contains='Water Heating') or 0.0
    dw_nonmachine_factor = _weighted_tech_factor_2000(
        fixed, 'Non-machine', branch_contains='Dishwashing', target_contains='Water Heating') or 0.0
    cw_factor = _weighted_tech_factor_2000(
        fixed, 'Clothes Washing', target_contains='Water Heating') or 0.0

    wh_total_pj = _ceud_series(residential, region, 'wh_total_pj')
    wh_lowmed = _ceud_series(residential, region, 'wh_lowmed')
    wh_high = _ceud_series(residential, region, 'wh_high')
    # 'housing_thousand' is a misleading name -- extract_housing_stock() already
    # multiplies the raw CEUD thousands figure by 1000, so its stored value is
    # the absolute household count already.
    households = _ceud_series(residential, region, 'housing_thousand')
    dw_count = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'appliances_per_household') &
        (pl.col('category') == 'Dishwashing')
    )
    cw_count = residential.filter(
        (pl.col('province') == region) &
        (pl.col('variable') == 'appliances_per_household') &
        (pl.col('category') == 'Clothes Washing')
    )

    years = sorted(
        set(wh_total_pj['year'].to_list()) & set(wh_lowmed['year'].to_list()) &
        set(wh_high['year'].to_list()) & set(households['year'].to_list()) &
        set(dw_count['year'].to_list()) & set(cw_count['year'].to_list())
    )
    years = [y for y in years if y <= _residential_mod.LAST_HIST_YEAR]
    if not years:
        return fixed, _empty_frame()

    # Re-anchored every year rather than frozen at 2000: the NAHW "Std
    # use"/"Low flow devices" split and each density's fuel-technology mix
    # (e.g. NG boiler efficiency tiers) both evolve via the same calibrated
    # tech competition that sets market_share_total everywhere else, so a
    # factor frozen at year 2000 goes increasingly stale -- see
    # `_reanchored_factor`.
    nahw_factor_by_anchor: dict[int, float] = {}
    f_lowmed_by_anchor: dict[int, float] = {}
    f_high_by_anchor: dict[int, float] = {}
    for anchor in years:
        nf = _weighted_tech_factor_2000(fixed, 'Non-appliance Hot Water', anchor_year=anchor)
        if nf is not None:
            nahw_factor_by_anchor[anchor] = nf

        lm_mst_rows = _build_wh_tech_mst_rows(
            residential, fixed, region, 'wh_tech_lowmed', 'LowMed Density', year=anchor)
        high_mst_rows = _build_wh_tech_mst_rows(
            residential, fixed, region, 'wh_tech_high', 'High Density', year=anchor)
        lm_shares = ({r['Technology']: float(r['Value']) for r in lm_mst_rows.iter_rows(named=True)}
                     if len(lm_mst_rows) > 0 else None)
        high_shares = ({r['Technology']: float(r['Value']) for r in high_mst_rows.iter_rows(named=True)}
                       if len(high_mst_rows) > 0 else None)

        fl = _weighted_tech_factor_2000(
            fixed, 'LowMed Density', branch_contains='Water Heating', shares=lm_shares, anchor_year=anchor)
        fh = _weighted_tech_factor_2000(
            fixed, 'High Density', branch_contains='Water Heating', shares=high_shares, anchor_year=anchor)
        if fl is not None:
            f_lowmed_by_anchor[anchor] = fl
        if fh is not None:
            f_high_by_anchor[anchor] = fh

    if not nahw_factor_by_anchor or not f_lowmed_by_anchor or not f_high_by_anchor:
        return fixed, _empty_frame()

    dishwashing_branch = f'{dwellings_branch}.Dishwashing'
    machine_split = _fixed_value_by_year(
        fixed, dishwashing_branch, f'{dishwashing_branch}.Machine', years)
    nonmachine_split = _fixed_value_by_year(
        fixed, dishwashing_branch, f'{dishwashing_branch}.Non-machine', years)

    def _lookup(df: pl.DataFrame, year: int) -> float:
        row = df.filter(pl.col('year') == year)
        return float(row['value'][0]) if len(row) > 0 else 0.0

    unit_by_year: dict[int, float] = {}
    for year in years:
        hh = _lookup(households, year)
        f_lowmed = _reanchored_factor(f_lowmed_by_anchor, year)
        f_high = _reanchored_factor(f_high_by_anchor, year)
        nahw_factor = _reanchored_factor(nahw_factor_by_anchor, year)
        if f_lowmed is None or f_high is None or nahw_factor is None:
            continue
        downstream_factor = _lookup(wh_lowmed, year) * f_lowmed + _lookup(wh_high, year) * f_high
        if hh <= 0 or downstream_factor <= 0:
            continue

        target_gj = _lookup(wh_total_pj, year) * 1e6
        dw_gj_per_unit = (
            machine_split.get(year, 0.0) * dw_machine_factor +
            nonmachine_split.get(year, 0.0) * dw_nonmachine_factor
        )
        dw_contribution = _lookup(dw_count, year) * hh * dw_gj_per_unit
        cw_contribution = _lookup(cw_count, year) * hh * cw_factor

        needed_assessed_demand = target_gj / downstream_factor - dw_contribution - cw_contribution
        unit_by_year[year] = needed_assessed_demand / (nahw_factor * hh)

    if not unit_by_year:
        return fixed, _empty_frame()

    new_rows = _annual_rows_with_continuation(
        dwellings_branch, 'Dwellings', '', nahw_target, region,
        old['Source'][0], old['Unit'][0], float(old['_order'].min()), unit_by_year,
    )
    fixed = fixed.filter(~nahw_mask)
    return fixed, new_rows


def _hdd_index(heating, region: str):
    """CEUD HDD index for one region as a year-indexed Series (empty if absent)."""
    data = heating[(heating['Region'] == region) & (heating['Variable'] == 'hdd_index')]
    return data.set_index('Year')['Value'].astype(float).sort_index()


def _weather_factors(hdd) -> dict[int, float]:
    """{year: weather factor} for DATA_START..PROJECTION_END from the HDD index."""
    proj = float(hdd.iloc[-HDD_PROJECTION_YEARS:].mean())
    return {y: float(hdd.get(y, proj)) for y in range(DATA_START, PROJECTION_END + 1)}


def _heating_intensity_means(heating, region: str,
                             hdd=None) -> dict[tuple[str, str], float]:
    """
    {(density, vintage bin): mean GJ/m2 over HEATING_INTENSITY_YEARS} for one
    region. With `hdd`, each year is divided by its HDD index first
    (weather-normalised intensity).
    """
    lo, hi = HEATING_INTENSITY_YEARS
    data = heating[
        (heating['Region'] == region) &
        heating['Variable'].isin(DENSITY_TO_INTENSITY_VARIABLE.values()) &
        heating['Year'].between(lo, hi)
    ]
    if hdd is not None:
        data = data.assign(Value=data['Value'] / data['Year'].map(hdd))
    var_to_density = {v: d for d, v in DENSITY_TO_INTENSITY_VARIABLE.items()}
    return {
        (var_to_density[var], cat): float(g['Value'].mean())
        for (var, cat), g in data.groupby(['Variable', 'Category'])
    }


def _weather_node_name(heating_branch: str) -> str:
    """'...Bldg Code.Heating (Cold)' -> '...Bldg Code.Weather (Cold)'."""
    return heating_branch.replace('.Heating (', '.Weather (')


def _build_weather_node_rows(fixed: pl.DataFrame, heating_branch: str, region: str,
                             factors: dict[int, float]) -> list[dict]:
    """Fixed Ratio Weather node rows, positioned just before the Heating node."""
    heat_rows = fixed.filter(pl.col('Branch') == heating_branch)
    order = float(heat_rows['_order'].min()) - 0.5
    sp = heat_rows.filter(pl.col('Parameter') == 'service_provide')
    unit = sp['Unit'][0] if len(sp) else 'GJ of heat'
    branch = _weather_node_name(heating_branch)
    service = branch.split('.')[-1]
    base = {'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Residential', 'Service': service, 'Technology': '',
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
        rows.append({**base, 'Parameter': 'service_request', 'Target': heating_branch,
                     'Source': 'CEUD HDD index',
                     'Unit': 'GJ', 'Year': str(year), 'Value': str(f),
                     '_order': order + 2e-3 + i * 1e-6})
    return rows


def _apply_bc_climate_split(mean_v: dict, marine_share: float) -> dict:
    """Re-split each technology's Cold + Marine heating request so Marine
    gets marine_share. Each technology's total is unchanged, so Retrofit
    Average / Deep keep their fraction of Reference."""
    tech_total: dict[str, float] = {}
    for (tech, _), v in mean_v.items():
        tech_total[tech] = tech_total.get(tech, 0.0) + v
    out = {}
    for (tech, tgt), v in mean_v.items():
        if tgt.endswith('.Heating (Marine)'):
            out[(tech, tgt)] = tech_total[tech] * marine_share
        elif tgt.endswith('.Heating (Cold)'):
            out[(tech, tgt)] = tech_total[tech] * (1 - marine_share)
        else:
            out[(tech, tgt)] = v
    return out


def _build_heating_intensity_rows(heating, fixed: pl.DataFrame,
                                   region: str) -> tuple[pl.DataFrame, list[int]]:
    """
    Reference / Retrofit service_request rows (all years) from each Vintage
    "<bin> Bldg Code" node to its Heating node(s), from CEUD intensity.

    Each fixed-data (technology, target) row keeps its ratio to the node's
    total Reference request -- Retrofit Average / Deep stay at their JCIMS
    fraction of Reference -- and is rescaled to the CEUD mean intensity.
    BC's Heating (Cold) / (Marine) targets are first re-split to
    BC_MARINE_HEAT_SHARE (see _apply_bc_climate_split).

    With WEATHER_NODES (and an HDD index for the region) the intensity is
    weather-normalised, the rows target a Weather node instead of the
    Heating node, and the Weather node's own rows are added.

    Returns the new rows (placed at the replaced rows' positions) and the
    _order values of the fixed rows they replace. Bldg Code nodes with no
    CEUD intensity keep their fixed-data rows.
    """
    hdd = _hdd_index(heating, region) if WEATHER_NODES else None
    if hdd is not None and hdd.empty:
        print(f'  No CEUD HDD index for {region}; no Weather nodes')
        hdd = None
    factors = _weather_factors(hdd) if hdd is not None else None
    means = _heating_intensity_means(heating, region, hdd)
    lo, hi = HEATING_INTENSITY_YEARS
    source = f'CEUD mean {lo}-{hi}' + (' / HDD index' if hdd is not None else '')
    sr = fixed.filter(
        (pl.col('Parameter') == 'service_request') &
        pl.col('Service').str.ends_with(' Bldg Code') &
        pl.any_horizontal([pl.col('Target').str.ends_with(f'.{s}') for s in HEATING_SERVICES])
    )

    rows: list[dict] = []
    replaced: list[int] = []
    weather_done: set[str] = set()
    for (branch,), g in sr.group_by('Branch', maintain_order=True):
        parts = branch.split('.')
        density, vintage_bin = parts[-3], parts[-1].removesuffix(' Bldg Code')
        if (density, vintage_bin) not in means:
            print(f'  No CEUD heating intensity for {density} {vintage_bin}; keeping fixed data')
            continue
        intensity = means[(density, vintage_bin)]

        g = g.with_columns(pl.col('Value').cast(pl.Float64, strict=False).alias('_v'))
        mean_v = {
            (t, tgt): float(sub['_v'].mean())
            for (t, tgt), sub in g.group_by(['Technology', 'Target'], maintain_order=True)
        }
        if region == 'BC' and BC_MARINE_HEAT_SHARE is not None:
            mean_v = _apply_bc_climate_split(mean_v, BC_MARINE_HEAT_SHARE)
        ref_total = sum(v for (t, _), v in mean_v.items() if t == 'Reference')
        if not ref_total:
            print(f'  No Reference heating request at {branch}; keeping fixed data')
            continue

        for (tech, target), v in mean_v.items():
            first = g.filter((pl.col('Technology') == tech) & (pl.col('Target') == target))
            order = float(first['_order'].min())
            value = str(intensity * v / ref_total)
            if factors is not None:
                if target not in weather_done:
                    rows.extend(_build_weather_node_rows(fixed, target, region, factors))
                    weather_done.add(target)
                target = _weather_node_name(target)
            for i, year in enumerate(range(DATA_START, PROJECTION_END + 1)):
                rows.append({
                    'Branch': branch, 'Type': 'Service', 'Region': region,
                    'Sector': 'Residential', 'Service': first['Service'][0],
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


def _assemble_region(fixed: pl.DataFrame, residential: pl.DataFrame,
                      multipliers: pl.DataFrame, heating, region: str) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with pipeline-derived rows at the correct positions.
    """
    # 0. Recalculate end-use demand quantities from CEUD's real annual data,
    #    in place of the flat/smoothed constants baked into fixed_data.
    fixed, cooling_intensity_rows = _replace_cooling_intensity(residential, fixed, region)
    fixed, wh_intensity_rows = _replace_wh_intensity(residential, fixed, region)
    fixed, lighting_total_rows = _replace_lighting_total(residential, fixed, region)

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
    appliances = _build_appliance_rows(residential, fixed, region, bt_sr_order + 0.5)

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

    # 10. Space-heating intensity: Vintage Bldg Code -> Heating service_request
    heat_sr, replaced = _build_heating_intensity_rows(heating, fixed, region)
    fixed = fixed.filter(~pl.col('_order').is_in(replaced))

    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        housing, prices, appliances, bt_rows, vb_rows,
        heat_cold_high, heat_cold_lm, heat_mar_high, heat_mar_lm,
        cool_high, cool_lm, wh_split, wh_mst_lm, wh_mst_high, heat_sr,
        cooling_intensity_rows, wh_intensity_rows, lighting_total_rows,
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
    heating = _heating_mod.main(export_csv=False)
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
            output = _assemble_region(fixed, residential, multipliers, heating, region)
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
