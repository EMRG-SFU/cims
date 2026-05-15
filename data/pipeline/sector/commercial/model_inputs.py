"""
Commercial Pipeline — Model Inputs

Combines fixed structural parameters with CEUD pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Commercial/commercial_{region}.csv
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

from utils.controls_conversions import DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
BASE_PATH       = Path('C:/cims/data')
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Commercial'
COMMERCIAL_CSV  = BASE_PATH / 'processed_data/nrcan/ceud/commercial.csv'
MULTIPLIERS_CSV = BASE_PATH / 'processed_data/energy_prices/energy_price_multipliers.csv'
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
    'Electricity', 'Biodiesel', 'Renewable Diesel',
    'Ethanol', 'Renewable Gasoline', 'Hydrogen', 'Renewable Natural Gas',
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
        .filter((pl.col('sector') == 'Commercial') & (pl.col('region') == region))
        .sort('energy', 'year')
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
        pl.when(pl.col('energy').is_in(list(REGION_SPECIFIC_ENERGIES)))
        .then(pl.lit(f'CIMS.CAN.{region}.') + pl.col('energy'))
        .otherwise(pl.lit('CIMS.Generic Fuels.') + pl.col('energy'))
        .alias('Target'),
        pl.col('source').alias('Source'),
        pl.lit('').alias('Unit'),
        pl.col('year').cast(pl.String).alias('Year'),
        pl.col('multiplier').cast(pl.String).alias('Value'),
        pl.Series('_order', [start_order + i * 1e-4 for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


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
                'Source': 'CEUD', 'Unit': '% of m2',
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
                'Source': 'CEUD', 'Unit': '% of m2',
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

    # Build a lookup from pipeline: category → year-2000 value
    pipeline_vals: dict[str, float] = {
        r['category']: r['value']
        for r in data.iter_rows(named=True)
    }
    pipeline_unit = data['unit'][0] if len(data) > 0 else '% of GJ'

    # Emit a market_share_total row for every technology in the fixed data,
    # using 0 for any technology absent from the pipeline.
    for tech, lifetime_max in tech_lifetime_max.items():
        val = pipeline_vals.get(tech, 0.0)
        rows.append({
            'Branch': branch, 'Type': 'Service', 'Region': region,
            'Sector': 'Commercial', 'Service': service_name,
            'Technology': tech,
            'Parameter': 'market_share_total', 'Context': '', 'Sub_Context': '',
            'Target': '', 'Source': 'CEUD', 'Unit': pipeline_unit,
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


def _assemble_region(
    fixed: pl.DataFrame,
    commercial: pl.DataFrame,
    multipliers: pl.DataFrame,
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

    # ── build pipeline rows with fractional _order values ─────────────────────
    # Floorspace: before everything (negative orders)
    floorspace_rows = _build_floorspace_rows(
        commercial, region, start_order=float(fixed['_order'].min()) - 1000.0
    )

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

    # ── combine and sort ───────────────────────────────────────────────────────
    all_frames = [
        fixed.cast({'_order': pl.Float64}),
        floorspace_rows,
        price_rows,
        shell_share_rows,
        hw_mst,
        hvac_cold_mst,
        hvac_marine_mst,
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
    commercial = pl.read_csv(
        COMMERCIAL_CSV,
        schema_overrides={'year': pl.Int64, 'value': pl.Float64},
    )
    multipliers = pl.read_csv(
        MULTIPLIERS_CSV,
        schema_overrides={'year': pl.Int64, 'multiplier': pl.Float64},
    )
    print(f'  Commercial data: {len(commercial):,} rows, '
          f'regions: {sorted(commercial["region"].unique().to_list())}')

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
            output = _assemble_region(fixed, commercial, multipliers, region, template)

            out_path = OUTPUT_DIR / f'commercial_{region}.csv'
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
