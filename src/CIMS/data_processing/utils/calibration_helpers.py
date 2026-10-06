"""
Shared helpers for the sector ``*_calibration.py`` scripts.

Every sector calibration script builds rows in the same output shape and
derives the same branch metadata and fuel target branches; those pieces live
here so the sector scripts only hold their sector-specific logic.
"""

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
