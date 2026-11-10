"""
Shared helpers for the sector ``*_calibration.py`` scripts.

Every sector calibration script builds rows in the same output shape, derives
the same branch metadata and fuel target branches, and filters the same three
sources (CER energy demand, NIR crosswalk totals, NIR per-gas emissions) down
to its own sector. Those pieces live here so the sector scripts only hold
their sector-specific logic.
"""

from collections.abc import Callable

import pandas as pd
import polars as pl

OUTPUT_COLS = [
    'Branch', 'Type', 'Region', 'Sector', 'Service', 'Technology',
    'Parameter', 'Context', 'Sub_Context', 'Target', 'Source', 'Unit',
    'Year', 'Value',
]

# Fuels that have region-specific CIMS branches
REGIONAL_FUELS = {
    'Electricity', 'Biodiesel',
    'Ethanol', 'Hydrogen',
}


def branch_meta(branch: str) -> dict:
    """Infer Type, Region, Sector, Service from a CIMS branch string.

    Branch structure: CIMS.CAN.{Region}[.{Sector}[.{Service}[...]]]
    """
    parts = branch.split('.')
    if len(parts) < 3:
        return {'Type': '', 'Region': '', 'Sector': '', 'Service': ''}
    region = parts[2]
    if len(parts) == 3:
        return {'Type': 'Region', 'Region': region, 'Sector': '', 'Service': ''}
    sector = parts[3]
    if len(parts) == 4:
        return {'Type': 'Sector', 'Region': region, 'Sector': sector, 'Service': ''}
    service = parts[4]
    return {'Type': 'Service', 'Region': region, 'Sector': sector, 'Service': service}


def fuel_target(region: str, fuel: str) -> str:
    """Build CIMS branch for a fuel (mirrors model_inputs.py price_mult logic)."""
    if fuel in REGIONAL_FUELS:
        return f'CIMS.CAN.{region}.{fuel}'
    return f'CIMS.Generic Fuels.{fuel}'


def empty_df() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})


# NIR full province name → CIMS abbreviation (excludes Canada)
REGION_MAP: dict[str, str] = {
    'British Columbia':          'BC',
    'Alberta':                   'AB',
    'Saskatchewan':              'SK',
    'Manitoba':                  'MB',
    'Ontario':                   'ON',
    'Quebec':                    'QC',
    'New Brunswick':             'NB',
    'Nova Scotia':               'NS',
    'Prince Edward Island':      'PE',
    'Newfoundland and Labrador': 'NL',
    'Yukon':                     'YT',
    'Northwest Territories':     'NT',
    'Nunavut':                   'NU',
}


# ── energy demand builder ─────────────────────────────────────────────────────

def build_cer_energy(
    cer_df: pd.DataFrame,
    sector: str,
    scale: Callable[[str, str], float] | None = None,
) -> pl.DataFrame:
    """Filter cer_resd_demand output to the CIMS nodes of `sector`.

    `scale`, if given, is called as ``scale(region, node)`` and each row's
    value is multiplied by the result (e.g. to split a total that the CER
    mapping assigns to more than one node).
    """
    sector_df = cer_df[cer_df['Node'].str.startswith(f'.{sector}')].copy()
    if sector_df.empty:
        return empty_df()

    rows = []
    for _, row in sector_df.iterrows():
        region = str(row['Region'])
        node   = str(row['Node'])
        fuel   = str(row['Variable'])
        branch = f'CIMS.CAN.{region}{node}'
        meta   = branch_meta(branch)
        value  = row['Value']
        if scale is not None:
            value = float(value) * scale(region, node)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      region,
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_quantity_requested',
            'Context':     '',
            'Sub_Context': '',
            'Target':      fuel_target(region, fuel),
            'Source':      str(row.get('Source', 'CER')),
            'Unit':        str(row.get('Unit', 'GJ')),
            'Year':        str(int(row['Year'])),
            'Value':       str(value),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


# ── emission builders ─────────────────────────────────────────────────────────

def build_crosswalk_emissions(crosswalk_df: pl.DataFrame, sector: str) -> pl.DataFrame:
    """Filter nir_crosswalk_tables_cims output to the CIMS branches of `sector`."""
    sector_df = crosswalk_df.filter(pl.col('CIMS_Branch').str.contains(rf'\.{sector}'))
    if sector_df.is_empty():
        return empty_df()

    rows = []
    for row in sector_df.to_dicts():
        branch = row['CIMS_Branch']
        meta   = branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      meta['Region'],
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_emissions_total',
            'Context':     '',
            'Sub_Context': '',
            'Target':      '',
            'Source':      str(row.get('Source', 'NIR')),
            'Unit':        str(row.get('Unit', 'tCO2e')),
            'Year':        str(row['Year']),
            'Value':       str(row['Value']),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})


def build_nir_emissions(nir_df: pl.DataFrame, sector: str) -> pl.DataFrame:
    """Extract per-gas NIR emissions for the CIMS branches of `sector`."""
    known_regions = set(REGION_MAP.keys())
    sector_df = nir_df.filter(
        pl.col('CIMS Branch').str.contains(rf'\.{sector}')
        & pl.col('Region').is_in(known_regions)
    )
    if sector_df.is_empty():
        return empty_df()

    rows = []
    for row in sector_df.to_dicts():
        full_region = row['Region']
        abbr        = REGION_MAP[full_region]
        branch      = row['CIMS Branch'].replace(
            f'CIMS.CAN.{full_region}.', f'CIMS.CAN.{abbr}.'
        )
        meta = branch_meta(branch)
        rows.append({
            'Branch':      branch,
            'Type':        meta['Type'],
            'Region':      abbr,
            'Sector':      meta['Sector'],
            'Service':     meta['Service'],
            'Technology':  '',
            'Parameter':   'calibration_emissions_by_type',
            'Context':     str(row['Variable']),
            'Sub_Context': '',
            'Target':      '',
            'Source':      'NIR',
            'Unit':        str(row['Unit']),
            'Year':        str(row['Year']),
            'Value':       str(row['Value']),
        })
    return pl.DataFrame(rows, schema={c: pl.Utf8 for c in OUTPUT_COLS})
