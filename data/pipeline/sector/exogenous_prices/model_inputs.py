"""
Exogenous Prices Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs (one per region).

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/exogenous_prices/exogenous prices_{region}.csv
    Flattened from wide (2000–2050 year columns) to long format.

Energy prices  (lcc_financial rows)
    source/energy_prices/energy_prices.py — regional production costs in 2025 C$/GJ.
    Inserted after each fuel's is_supply, TRUE row.

Emission factors  (emissions / emissions_biomass rows)
    source/emission_factors/emission_factors.py — tGas/GJ from Canada NIR Annex 6.
    Inserted after each fuel's lcc_financial rows.
    Fuels with zero EF (Electricity, Hydrogen) produce no emission factor rows.

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
import pandas as pd

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

_ep_spec = importlib.util.spec_from_file_location(
    'energy_prices',
    _PIPELINE_ROOT / 'source' / 'energy_prices' / 'energy_prices.py',
)
_energy_prices_mod = importlib.util.module_from_spec(_ep_spec)
_ep_spec.loader.exec_module(_energy_prices_mod)

_ef_spec = importlib.util.spec_from_file_location(
    'emission_factors',
    _PIPELINE_ROOT / 'source' / 'emission_factors' / 'emission_factors.py',
)
_ef_mod = importlib.util.module_from_spec(_ef_spec)
_ef_spec.loader.exec_module(_ef_mod)

from utils.controls_conversions import BASE_PATH, DATA_START, PROJECTION_END, LAST_DATA_YEAR
from utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/exogenous_prices'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/exogenous_prices'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

REGIONS = [
    'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT',
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened(fixed_path: Path) -> pl.DataFrame:
    """Flatten one fixed CSV and return as a row-indexed DataFrame."""
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / fixed_path.name
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
    df = df.with_columns(
        pl.when(pl.col('Parameter') == 'is_supply').then(pl.lit('')).otherwise(pl.col('Context')).alias('Context'),
        pl.when(pl.col('Parameter') == 'is_supply').then(pl.lit('TRUE')).otherwise(pl.col('Value')).alias('Value'),
    )
    return df.with_row_index('_order')


def _find_is_supply_orders(fixed: pl.DataFrame) -> dict[str, float]:
    """Return {Service: _order} for each fuel's is_supply row."""
    orders: dict[str, float] = {}
    for row in fixed.filter(pl.col('Parameter').fill_null('') == 'is_supply').iter_rows(named=True):
        service = (row.get('Service') or '').strip()
        if service:
            orders[service] = float(row['_order'])
    return orders


def _get_branch_map(fixed: pl.DataFrame) -> dict[str, str]:
    """Return {Service: Branch} read from the is_supply rows of the fixed data."""
    branch_map: dict[str, str] = {}
    for row in fixed.filter(pl.col('Parameter').fill_null('') == 'is_supply').iter_rows(named=True):
        service = (row.get('Service') or '').strip()
        branch  = (row.get('Branch')  or '').strip()
        if service and service not in branch_map:
            branch_map[service] = branch
    return branch_map


def _build_lcc_rows(
    service: str,
    branch: str,
    region: str,
    prices_df: pd.DataFrame,
    start_order: float,
) -> pl.DataFrame:
    """
    Build lcc_financial rows for one fuel from energy_prices output.
    Uses the region-specific price (Electricity, Biodiesel, Ethanol, Hydrogen
    are all regional in the energy_prices source).
    """
    data = prices_df[
        (prices_df['Energy'] == service) &
        (prices_df['Region'] == region)
    ].sort_values('Year').reset_index(drop=True)

    if data.empty:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})

    rows = [
        {
            'Branch':      branch,
            'Type':        'Sector',
            'Region':      region,
            'Sector':      '',
            'Service':     service,
            'Technology':  '',
            'Parameter':   'lcc_financial',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      str(r['Source']),
            'Unit':        str(r['Unit']),
            'Year':        str(int(r['Year'])),
            'Value':       str(r['Price']),
            '_order':      start_order + i * 1e-5,
        }
        for i, (_, r) in enumerate(data.iterrows())
    ]
    return pl.DataFrame(rows)


def _build_ef_rows(
    service: str,
    branch: str,
    region: str,
    ef_df: pl.DataFrame,
    start_order: float,
) -> pl.DataFrame:
    """
    Build emission factor rows for one fuel from build_cims_table() output.
    Branch, Type, Service, and Region are overridden to match the fixed data context.
    Returns an empty frame if no emission factors exist for this fuel.
    """
    data = ef_df.filter(pl.col('Service') == service)

    if len(data) == 0:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})

    n = len(data)
    return data.select([
        pl.lit(branch).alias('Branch'),
        pl.lit('Sector').alias('Type'),
        pl.lit(region).alias('Region'),
        pl.col('Sector'),
        pl.lit(service).alias('Service'),
        pl.col('Technology'),
        pl.col('Parameter'),
        pl.col('Context'),
        pl.col('Sub_Context'),
        pl.col('Target'),
        pl.col('Source'),
        pl.col('Unit'),
        pl.col('Year').cast(pl.String),
        pl.col('Value').cast(pl.String),
        pl.Series('_order', [start_order + i * 1e-5 for i in range(n)],
                  dtype=pl.Float64).alias('_order'),
    ])


def _assemble_region(
    fixed: pl.DataFrame,
    region: str,
    prices_df: pd.DataFrame,
    ef_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for one region by interleaving
    fixed structural data with energy prices (lcc_financial) and emission
    factors at the correct positions.

    Injection order per fuel:
        is_supply, TRUE   ← fixed data
        lcc_financial     ← energy_prices (regional production costs)
        emissions         ← emission_factors (NIR Annex 6)
    """
    is_supply_orders = _find_is_supply_orders(fixed)
    branch_map       = _get_branch_map(fixed)

    frames: list[pl.DataFrame] = [fixed.cast({'_order': pl.Float64})]

    for service, is_supply_order in is_supply_orders.items():
        branch = branch_map.get(service, f'CIMS.CAN.{region}.{service}')

        lcc_rows = _build_lcc_rows(service, branch, region, prices_df,
                                    start_order=is_supply_order + 0.3)
        if len(lcc_rows) > 0:
            frames.append(lcc_rows)

        ef_rows = _build_ef_rows(service, branch, region, ef_df,
                                  start_order=is_supply_order + 0.6)
        if len(ef_rows) > 0:
            frames.append(ef_rows)

    combined = pl.concat(
        [f for f in frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    return combined.select(OUTPUT_COLS)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble exogenous prices model inputs and write one CSV per region."""
    print('=' * 60)
    print('EXOGENOUS PRICES MODEL INPUTS')
    print('=' * 60)

    print('\nLoading energy prices...')
    prices_df = _energy_prices_mod.main()

    print('\nBuilding emission factors CIMS table...')
    ef_records = _ef_mod.build_records()
    _excluded  = {f['fuel_name'] for f in _ef_mod.FUELS if f.get('exclude_from_output')}
    ef_out     = ef_records.filter(~pl.col('fuel').is_in(pl.Series(list(_excluded))))
    ef_df      = _ef_mod.build_cims_table(ef_out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    for region in REGIONS:
        fixed_path = FIXED_INPUT_DIR / f'exogenous_prices_{region.lower()}.csv'
        if not fixed_path.exists():
            print(f'\n  ⚠  Skipping {region} — file not found: {fixed_path.name}')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            fixed = _read_flattened(fixed_path)

            print('  Assembling with energy prices and emission factors...')
            output = _assemble_region(fixed, region, prices_df, ef_df)

            out_path = OUTPUT_DIR / f'exogenous_prices_{region.lower()}.csv'
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
    print(f'Regions complete: {len(results)}/{len(REGIONS)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
