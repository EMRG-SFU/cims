"""
Fuels Pipeline — Model Inputs

Combines fixed structural parameters with pipeline data into
CIMS-formatted CSVs.

Sources
-------
Fixed structural parameters
    raw_data/fixed_data/fuels/fuels_{region}.csv  (one per region, flattened as-is)
    raw_data/fixed_data/fuels/fuels_CIMS.csv      (flattened + energy prices + emission factors)

Energy prices  (lcc_financial rows)
    source/energy_prices/energy_prices.py — generic production costs in 2025 C$/GJ.
    Inserted after each fuel's is_supply, TRUE row in fuels_CIMS.csv.

Emission factors  (emissions / emissions_biomass rows)
    source/emission_factors/emission_factors.py — tGas/GJ from Canada NIR Annex 6.
    Inserted after each fuel's lcc_financial rows in fuels_CIMS.csv.

Base-year transportation blend shares  (market_share_total, DATA_START year)
    sector/fuels_calibration.py — CER Passenger/Freight shares of each technology in
    Fuel Blends.Diesel_Transportation / Gasoline_Transportation. Replace the fixed-data
    base-year values in the regional files, so the model starts from the same shares the
    calibration targets.

Output columns
--------------
Branch, Type, Region, Sector, Service, Technology, Parameter,
Context, Sub_Context, Target, Source, Unit, Year, Value
"""

from pathlib import Path

import polars as pl
import pandas as pd

# ── path setup ─────────────────────────────────────────────────────────────────
from CIMS.data_processing.utils.flatten_fixed_data import read_fixed_data_file

import CIMS.data_processing.source.energy_prices.energy_prices as _energy_prices_mod

import CIMS.data_processing.source.emission_factors.emission_factors as _ef_mod

import CIMS.data_processing.sector.fuels_calibration as _fuels_cal_mod

from CIMS.data_processing.utils.controls_conversions import BASE_PATH, DATA_START
from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

# ── configuration ──────────────────────────────────────────────────────────────
FIXED_INPUT_DIR = BASE_PATH / 'raw_data/fixed_data/fuels'
OUTPUT_DIR      = BASE_PATH / 'model_inputs/model/fuels'

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Each region maps 1:1 to its own fixed-data file (no template sharing).
REGIONAL_FILES: dict[str, str] = {
    'AB': 'AB', 'BC': 'BC', 'MB': 'MB', 'NB': 'NB', 'NL': 'NL',
    'NS': 'NS', 'NT': 'NT', 'NU': 'NU', 'ON': 'ON', 'PE': 'PE',
    'QC': 'QC', 'SK': 'SK', 'YT': 'YT',
}

# Service name in fuels_CIMS.csv → fuel name used in energy_prices / emission_factors
# where the two differ.
_SERVICE_TO_ENERGY: dict[str, str] = {
    'Waste Fuel': 'Waste',
}

# Fuels whose energy_prices rows are regional-only (no 'generic' entry).
# ON prices are used as the generic proxy — the price is the same across all regions.
_USE_ON_AS_GENERIC: frozenset[str] = frozenset({
    'Renewable Diesel',
    'Renewable Gasoline',
})

# ── helpers ────────────────────────────────────────────────────────────────────

def _read_flattened(fixed_path: Path) -> pl.DataFrame:
    """Flatten one fixed CSV, force is_supply rows to TRUE, and return a row-indexed DataFrame."""
    df = read_fixed_data_file(fixed_path)
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
            branch_map[service] = branch or f'CIMS.Generic Fuels.{service}'
    return branch_map


def _apply_base_year_blend_shares(output: pl.DataFrame, base_shares: pl.DataFrame) -> pl.DataFrame:
    """Overwrite base-year market_share_total for blend technologies with CER shares."""
    if len(base_shares) == 0:
        return output
    key = ['Branch', 'Technology', 'Year']
    is_target = (pl.col('Parameter') == 'market_share_total') & (pl.col('Year') == str(DATA_START))
    replaced = (
        output.with_row_index('_row')
        .join(base_shares.select(key + [pl.col('Value').alias('_cer_value')]), on=key, how='left')
        .with_columns(
            pl.when(is_target & pl.col('_cer_value').is_not_null())
            .then(pl.col('_cer_value')).otherwise(pl.col('Value')).alias('Value'),
            pl.when(is_target & pl.col('_cer_value').is_not_null())
            .then(pl.lit('CER')).otherwise(pl.col('Source')).alias('Source'),
        )
        .sort('_row')
    )
    n = len(replaced.filter(is_target & pl.col('_cer_value').is_not_null()))
    if n:
        print(f'  Set {n} base-year blend market shares from CER')
    return replaced.select(output.columns)


def _build_lcc_rows(
    service: str,
    branch: str,
    prices_df: pd.DataFrame,
    start_order: float,
) -> pl.DataFrame:
    """
    Build lcc_financial rows for one fuel from energy_prices output.
    Only generic-region prices are used (all CIMS-level fuels are non-regional).
    Returns an empty frame if no generic price exists for this fuel.
    """
    energy_name = _SERVICE_TO_ENERGY.get(service, service)
    region = 'ON' if energy_name in _USE_ON_AS_GENERIC else 'generic'
    data = prices_df[
        (prices_df['Energy'] == energy_name) &
        (prices_df['Region'] == region)
    ].sort_values('Year').reset_index(drop=True)

    if data.empty:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})

    rows = [
        {
            'Branch':      branch,
            'Type':        'Service',
            'Region':      'CIMS',
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
    ef_df: pl.DataFrame,
    start_order: float,
) -> pl.DataFrame:
    """
    Build emission factor rows for one fuel from build_cims_table() output.
    Branch and Service are overridden to match the fixed data's naming.
    Returns an empty frame if no emission factors exist for this fuel.
    """
    energy_name = _SERVICE_TO_ENERGY.get(service, service)
    data = ef_df.filter(pl.col('Service') == energy_name)

    if len(data) == 0:
        return pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in OUTPUT_COLS + ['_order']})

    n = len(data)
    return data.select([
        pl.lit(branch).alias('Branch'),
        pl.col('Type'),
        pl.col('Region'),
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


def _assemble_cims(
    fixed: pl.DataFrame,
    prices_df: pd.DataFrame,
    ef_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build the complete model-inputs DataFrame for fuels_CIMS.csv by
    interleaving fixed structural data with energy prices (lcc_financial)
    and emission factors at the correct positions.

    Injection order per fuel:
        is_supply, TRUE   ← fixed data
        lcc_financial     ← energy_prices (generic production costs)
        emissions         ← emission_factors (NIR Annex 6)
    """
    is_supply_orders = _find_is_supply_orders(fixed)
    branch_map       = _get_branch_map(fixed)

    frames: list[pl.DataFrame] = [fixed.cast({'_order': pl.Float64})]

    for service, is_supply_order in is_supply_orders.items():
        branch = branch_map.get(service, f'CIMS.Generic Fuels.{service}')

        lcc_rows = _build_lcc_rows(service, branch, prices_df,
                                    start_order=is_supply_order + 0.3)
        if len(lcc_rows) > 0:
            frames.append(lcc_rows)

        ef_rows = _build_ef_rows(service, branch, ef_df,
                                  start_order=is_supply_order + 0.6)
        if len(ef_rows) > 0:
            frames.append(ef_rows)

    combined = pl.concat(
        [f for f in frames if len(f) > 0],
        how='diagonal_relaxed',
    ).sort('_order')

    # lcc_financial rows must keep an explicit row for every year even when
    # the value is constant across the whole projection — collapse_constant_years
    # is skipped for just this Parameter in the CIMS output.
    keep_mask  = pl.col('Parameter') == 'lcc_financial'
    full_years = combined.filter(keep_mask)
    rest       = combined.filter(~keep_mask)

    collapsed = collapse_constant_years(rest.select(OUTPUT_COLS + ['_order']))

    return (
        pl.concat([collapsed, full_years.select(OUTPUT_COLS + ['_order'])], how='diagonal_relaxed')
        .sort('_order')
        .select(OUTPUT_COLS)
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> dict[str, pl.DataFrame]:
    """Assemble fuels model inputs and write one CSV per region plus CIMS."""
    print('=' * 60)
    print('FUELS MODEL INPUTS')
    print('=' * 60)

    print('\nLoading energy prices...')
    prices_df = _energy_prices_mod.main()

    print('\nBuilding emission factors CIMS table...')
    ef_records = _ef_mod.build_records()
    _excluded  = {f['fuel_name'] for f in _ef_mod.FUELS if f.get('exclude_from_output')}
    ef_out     = ef_records.filter(~pl.col('fuel').is_in(list(_excluded)))
    ef_df      = _ef_mod.build_cims_table(ef_out)

    print('\nBuilding base-year transportation blend shares from CER...')
    base_shares = _fuels_cal_mod.blend_shares().filter(pl.col('Year') == str(DATA_START))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pl.DataFrame] = {}

    # ── Regional files: flatten only ──────────────────────────────────────────
    for region, template in sorted(REGIONAL_FILES.items()):
        fixed_path = FIXED_INPUT_DIR / f'fuels_{template.lower()}.csv'
        if not fixed_path.exists():
            print(f'\n  ⚠  Skipping {region} — file not found: {fixed_path.name}')
            continue

        try:
            print(f'\n{region}:')
            print('  Flattening fixed data...')
            df = _read_flattened(fixed_path)
            output = df.select(OUTPUT_COLS)
            output = _apply_base_year_blend_shares(
                output, base_shares.filter(pl.col('Region') == region))

            out_path = OUTPUT_DIR / f'fuels_{region.lower()}.csv'
            output = collapse_constant_years(output)
            output.write_csv(str(out_path))
            print(f'  Wrote {len(output):,} rows → {out_path.name}')
            results[region] = output

        except Exception as exc:
            print(f'  ERROR: {exc}')
            import traceback
            traceback.print_exc()

    # ── CIMS file: flatten + energy prices + emission factors ─────────────────
    cims_path = FIXED_INPUT_DIR / 'fuels_cims.csv'
    if cims_path.exists():
        try:
            print('\nCIMS:')
            print('  Flattening fixed data...')
            fixed = _read_flattened(cims_path)

            print('  Assembling with energy prices and emission factors...')
            output = _assemble_cims(fixed, prices_df, ef_df)

            out_path = OUTPUT_DIR / 'fuels_cims.csv'
            output.write_csv(str(out_path))
            print(f'  Wrote {len(output):,} rows → {out_path.name}')
            results['CIMS'] = output

        except Exception as exc:
            print(f'  ERROR: {exc}')
            import traceback
            traceback.print_exc()
    else:
        print(f'\n  ⚠  fuels_CIMS.csv not found at {cims_path}')

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    total_expected = len(REGIONAL_FILES) + 1  # +1 for CIMS
    print(f'Files complete: {len(results)}/{total_expected}')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
