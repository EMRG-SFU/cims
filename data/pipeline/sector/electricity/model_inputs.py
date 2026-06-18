"""
Electricity Pipeline — Model Inputs

Combines fixed structural parameters with electricity activity data and energy
price multipliers into CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/Electricity/electricity_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.
    Province files (AB, BC, MB, NB, NL, NS, ON, PE, QC, SK) include
    Base Load / Shoulder Load / Peak Load sub-services.
    Territory files (NT, NU, YT) use a flat Utility Generation structure.

Electricity activity / load fractions
    processed_data/activity/electricity.csv
    Variables:
        Electricity.Utility Generation.Base Load    → % MWh → Utility Generation service_request
        Electricity.Utility Generation.Shoulder Load → % MWh → Utility Generation service_request
        Electricity.Utility Generation.Peak Load    → % MWh → Utility Generation service_request

    Load-fraction service_requests are only emitted for regions whose fixed data
    contains Base Load / Shoulder Load / Peak Load sub-services (provinces).

Energy price multipliers  (multiplier_price rows)
    pipeline/source/energy_prices/energy_price_multipliers.py

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value

Output order per region
-----------------------
1. Electricity sector service_provide + competition
2. multiplier_price (Electricity sector level)
3. Fixed data — Utility Generation service_provide / competition
4. service_request (Utility Generation) — Base Load / Shoulder Load / Peak Load % MWh
   (provinces only)
5. Remainder of fixed data (technologies, Storage, Transmission, CCS)
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

_ep_spec = importlib.util.spec_from_file_location(
    'energy_price_multipliers',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_price_multipliers.py',
)
_energy_price_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_price_mod)

_elec_spec = importlib.util.spec_from_file_location(
    'electricity_activity',
    _PIPELINE_ROOT / 'source' / 'activity' / 'electricity.py',
)
_elec_mod = importlib.util.module_from_spec(_elec_spec)
_elec_spec.loader.exec_module(_elec_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/Electricity'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/electricity'

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

# Activity variable → load sub-service name used in the Target branch path
_LOAD_VAR_TO_SERVICE: dict[str, str] = {
    'Electricity.Utility Generation.Base Load':     'Base Load',
    'Electricity.Utility Generation.Shoulder Load': 'Shoulder Load',
    'Electricity.Utility Generation.Peak Load':     'Peak Load',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened_fixed(region: str) -> pl.DataFrame:
    """Flatten one Electricity fixed CSV and return a row-indexed DataFrame."""
    fixed_path = FIXED_INPUT_DIR / f'electricity_{region}.csv'
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f'electricity_{region}.csv'
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
    frame = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS})
    return frame.with_columns(pl.Series('_order', [], dtype=pl.Float64))


def _find_max_order(df: pl.DataFrame, service: str, parameter: str,
                    no_tech: bool = False) -> float | None:
    """Return the max _order matching service + parameter."""
    mask = (
        (pl.col('Service').fill_null('') == service) &
        (pl.col('Parameter').fill_null('') == parameter)
    )
    if no_tech:
        mask = mask & (pl.col('Technology').fill_null('') == '')
    subset = df.filter(mask)
    if len(subset) == 0:
        return None
    return float(subset['_order'].max())


def _has_load_subservices(fixed: pl.DataFrame) -> bool:
    """True when fixed data contains Base / Shoulder / Peak Load sub-services."""
    return len(fixed.filter(pl.col('Service').fill_null('') == 'Base Load')) > 0


def _build_sector_rows(region: str, start_order: float) -> pl.DataFrame:
    """Electricity sector service_provide and competition rows (structural, no year values)."""
    branch = f'CIMS.CAN.{region}.Electricity'
    base = {'Branch': branch, 'Type': 'Sector', 'Region': region, 'Sector': 'Electricity',
            'Service': '', 'Technology': '', 'Sub_Context': '', 'Target': '',
            'Source': '', 'Unit': '', 'Year': '', 'Value': ''}
    rows = [
        {**base, 'Parameter': 'service_provide', 'Context': '', 'Source': 'JCIMS',
         'Unit': 'GJ', '_order': start_order},
        {**base, 'Parameter': 'competition', 'Context': 'Sector',
         '_order': start_order + 1e-4},
    ]
    return pl.DataFrame(rows)


def _build_price_rows(multipliers: pl.DataFrame, region: str,
                       start_order: float) -> pl.DataFrame:
    """multiplier_price rows for the Electricity sector."""
    data = (
        multipliers
        .filter(
            (pl.col('Sector') == 'Electricity') & (pl.col('Region') == region)
        )
        .sort('Energy', 'Year')
    )
    if len(data) == 0:
        return _empty_frame()
    n = len(data)
    result = data.select([
        pl.lit(f'CIMS.CAN.{region}.Electricity').alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.lit('Electricity').alias('Sector'),
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
    ])
    return result.with_columns(
        pl.Series('_order', [start_order + i * 1e-4 for i in range(n)], dtype=pl.Float64)
    )


def _build_load_fraction_rows(activity: pl.DataFrame, region: str,
                               start_order: float) -> pl.DataFrame:
    """
    service_request rows (% MWh) from Utility Generation to each load sub-service.
    Inserted after the Utility Generation competition row.
    Order: Base Load years, Shoulder Load years, Peak Load years.
    """
    rows: list[dict] = []
    counter = 0
    for var, svc in _LOAD_VAR_TO_SERVICE.items():
        target = f'CIMS.CAN.{region}.Electricity.Utility Generation.{svc}'
        var_data = (
            activity
            .filter(
                (pl.col('Region') == region) &
                (pl.col('Variable') == var)
            )
            .sort('Year')
        )
        for r in var_data.iter_rows(named=True):
            rows.append({
                'Branch':      f'CIMS.CAN.{region}.Electricity.Utility Generation',
                'Type':        'Service',
                'Region':      region,
                'Sector':      'Electricity',
                'Service':     'Utility Generation',
                'Technology':  '',
                'Parameter':   'service_request',
                'Context':     '',
                'Sub_Context': '',
                'Target':      target,
                'Source':      r['Source'],
                'Unit':        r['Unit'],
                'Year':        str(r['Year']),
                'Value':       str(r['Value']),
                '_order':      start_order + counter * 1e-4,
            })
            counter += 1
    return pl.DataFrame(rows) if rows else _empty_frame()


# ── assembly ───────────────────────────────────────────────────────────────────

def _assemble_region(
    fixed: pl.DataFrame,
    activity: pl.DataFrame,
    multipliers: pl.DataFrame,
    region: str,
) -> pl.DataFrame:
    """Build the complete model-inputs DataFrame for one region."""
    has_load = _has_load_subservices(fixed)
    min_fixed = float(fixed['_order'].min())

    # Insertion points
    ug_comp_max = (
        _find_max_order(fixed, 'Utility Generation', 'competition', no_tech=True)
        or min_fixed
    )

    # Pipeline rows
    sector_rows = _build_sector_rows(region, start_order=min_fixed - 1500.0)
    price_rows = _build_price_rows(
        multipliers, region, start_order=min_fixed - 1000.0,
    )
    load_rows = (
        _build_load_fraction_rows(activity, region, start_order=ug_comp_max + 0.5)
        if has_load else _empty_frame()
    )

    combined = pl.concat(
        [f for f in [fixed.cast({'_order': pl.Float64}), sector_rows, price_rows, load_rows]
         if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble Electricity model inputs and write one CSV per region."""
    print('=' * 60)
    print('ELECTRICITY MODEL INPUTS')
    print('=' * 60)

    print('\nBuilding electricity activity data...')
    activity = _elec_mod.main()
    act_regions = sorted(activity['Region'].unique().to_list())
    print(f'  Rows: {len(activity):,}  regions: {act_regions}')

    print('Building energy price multiplier rows...')
    multipliers = pl.from_pandas(_energy_price_mod.main())
    print(f'  Rows: {len(multipliers):,}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region in sorted(FIXED_TEMPLATE):
        fixed_path = FIXED_INPUT_DIR / f'electricity_{region}.csv'
        if not fixed_path.exists():
            print(f'\n⚠  Skipping {region} — fixed data not found')
            continue
        if region not in act_regions:
            print(f'\n⚠  Skipping {region} — no activity data')
            continue

        print(f'\n{region}:')
        try:
            fixed = _read_flattened_fixed(region)
            has_load = _has_load_subservices(fixed)
            print(f'  Fixed rows: {len(fixed):,}  '
                  f'load sub-services: {"yes" if has_load else "no (territory)"}')

            output = _assemble_region(fixed, activity, multipliers, region)

            out_path = OUTPUT_DIR / f'electricity_{region}.csv'
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
