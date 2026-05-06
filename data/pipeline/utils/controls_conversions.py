"""
Utility functions for energy price processing.
Uses Polars for data processing with pandas for Excel reading.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Configuration
BASE_PATH = Path('C:/cims/data')
MAPPINGS_PATH    = BASE_PATH / 'mappings_conversions'
CONTROL_FILE     = MAPPINGS_PATH / 'control.py'
ENERGY_MAP_FILE  = MAPPINGS_PATH / 'energy_map.csv'
REGION_MAP_FILE  = MAPPINGS_PATH / 'region_map.csv'
SECTOR_MAP_FILE  = MAPPINGS_PATH / 'sector_map.csv'
CONVERSIONS_FILE = MAPPINGS_PATH / 'energy_conversions.csv'

def load_control_config() -> Dict:
    """
    Load control settings from the CONTROLS dict in control.py.

    Returns
    -------
    dict
        Control key/value pairs (e.g. currency years, scenario name).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('control', CONTROL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONTROLS


def load_energy_conversions() -> Dict:
    """
    Load energy conversion factors from energy_conversions.csv.

    Parses the 'Approximate Energy Content' section for GJ-per-m³ values
    and the general volume/energy rows for unit-to-unit scalars used by
    the pipeline (bbl_to_m3, mmbtu_to_gj, and per-fuel gj_per_m3 keys).

    Returns
    -------
    dict
        Flat dict of conversion factor keys to float values.
    """
    df = pd.read_csv(CONVERSIONS_FILE, header=None, dtype=str)

    # --- Parse 'Approximate Energy Content' section --------------------------
    # Find the header row for that section
    content_start = None
    for i, row in df.iterrows():
        if str(row.iloc[0]).strip() == 'Approximate Energy Content':
            content_start = i + 1   # next row is column headers
            break

    gj_per_m3: Dict[str, float] = {}
    if content_start is not None:
        for i in range(content_start + 1, len(df)):
            row = df.iloc[i]
            energy_name = str(row.iloc[0]).strip()
            unit        = str(row.iloc[1]).strip()
            equiv       = str(row.iloc[2]).strip()
            if not energy_name or energy_name in ('nan', ''):
                break
            # Only capture rows whose unit is "1.0 Cubic metres (m³)" → GJ
            if 'Cubic metres' in unit and 'Gigajoules' in equiv:
                try:
                    gj_val = float(equiv.split()[0].replace(',', ''))
                    gj_per_m3[energy_name.lower()] = gj_val
                except ValueError:
                    pass

    # --- Known scalar conversions (from the volume/energy rows) --------------
    # 1 bbl = 0.159 m³  →  bbl_to_m3 = 0.159
    # 1 MMBtu = 1.0551 GJ  →  mmbtu_to_gj = 1.0551
    scalars = {
        'bbl_to_m3':   0.159,
        'mmbtu_to_gj': 1.0551,
    }

    # Map energy names in the CSV to the keys expected by the pipeline
    name_to_key = {
        'petrochemical feedstock': 'petrochemical_feedstock_gj_per_m3',
        'naphtha specialties':     'naphtha_specialties_gj_per_m3',
        'asphalt':                 'asphalt_gj_per_m3',
        'lubes and greases':       'lubricants_gj_per_m3',
        'other products':          'other_non_energy_products_gj_per_m3',
        'ethanol':                 'ethanol_gj_per_m3',
        'biodiesel':               'biodiesel_gj_per_m3',
        'renewable diesel':        'renewable_diesel_gj_per_m3',
    }

    for csv_name, key in name_to_key.items():
        if csv_name in gj_per_m3:
            scalars[key] = gj_per_m3[csv_name]

    return scalars

def load_energy_mapping() -> pd.DataFrame:
    """
    Load the CIMS↔JCIMS↔CER energy mapping from energy_map.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(ENERGY_MAP_FILE)

def load_sector_mapping() -> pd.DataFrame:
    """
    Load the sector map.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(SECTOR_MAP_FILE)

def load_region_mapping() -> pd.DataFrame:
    """
    Load the region map.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(REGION_MAP_FILE)

def load_macro_indicators(filepath: str, scenario: str) -> pl.DataFrame:
    """Load macro indicators (deflators and exchange rates)."""
    # Read CSV with Polars
    df = pl.read_csv(filepath)
    df = df.filter(pl.col('Scenario') == scenario)
    
    # Pivot to wide format for easier access
    pivot = df.pivot(
        values='Value',
        index='Year',
        columns='Variable'
    )
    
    return pivot


def get_deflator(macro_df: pl.DataFrame, from_year: int, to_year: int) -> pl.Series:
    """Calculate deflator to convert prices from one year to another."""
    deflator_col = 'Gross Domestic Product Deflator (2017=100)'
    
    if deflator_col not in macro_df.columns:
        raise ValueError(f"GDP Deflator column not found in macro data")
    
    # Base deflator value for target year
    base_deflator = macro_df.filter(pl.col('Year') == to_year)[deflator_col][0]
    
    # Calculate deflator for each year
    deflator = base_deflator / macro_df[deflator_col]
    
    return deflator


def get_exchange_rate(macro_df: pl.DataFrame, from_currency: str) -> pl.Series:
    """Get exchange rate from foreign currency to CAD."""
    if from_currency == 'CAD':
        # CAD to CAD, no conversion needed
        return pl.Series('exchange_rate', [1.0] * len(macro_df))
    
    # For USD, use the exchange rate column
    if from_currency == 'USD':
        rate_col = 'Canada-US Exchange Rate (C$/US$)'
        if rate_col not in macro_df.columns:
            raise ValueError(f"Exchange rate column for USD not found")
        
        return macro_df[rate_col]
    
    # For other currencies, assume 1.0
    return pl.Series('exchange_rate', [1.0] * len(macro_df))


def convert_currency_polars(
    prices: pl.Series,
    from_year: int,
    to_year: int,
    from_currency: str,
    to_currency: str,
    macro_df: pl.DataFrame,
    constant_dollars: bool = False
) -> pl.Series:
    """
    Convert prices from one currency/year to another (Polars version).
    
    Args:
        prices: Price series with 'Year' column
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Polars DataFrame with macro indicators
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    if constant_dollars:
        # All prices are in constant from_year dollars
        # Apply single conversion factor to all values
        deflator_col = 'Gross Domestic Product Deflator (2017=100)'
        deflator_from = macro_df.filter(pl.col('Year') == from_year)[deflator_col][0]
        deflator_to = macro_df.filter(pl.col('Year') == to_year)[deflator_col][0]
        conversion_factor = deflator_to / deflator_from
        
        # Exchange rate (use from_year rate for constant dollars)
        if from_currency != to_currency:
            exchange_rate = get_exchange_rate(macro_df, from_currency)
            year_mask = macro_df['Year'] == from_year
            if year_mask.sum() > 0:
                er_value = exchange_rate.filter(year_mask)[0]
            else:
                er_value = 1.0
        else:
            er_value = 1.0
        
        converted = prices * conversion_factor * er_value
        
    else:
        # Nominal prices: each year's price is in that year's dollars
        # This needs to work with pandas Series for compatibility
        # Convert to dict, process, convert back
        raise NotImplementedError("Nominal currency conversion not yet implemented for Polars. Use pandas version.")
    
    return converted


def convert_currency(
    prices: pd.Series,
    from_year: int,
    to_year: int,
    from_currency: str,
    to_currency: str,
    macro_df: pl.DataFrame,
    constant_dollars: bool = False
) -> pd.Series:
    """
    Convert prices from one currency/year to another.

    Queries macro_df directly with Polars — no to_pandas() call, so no
    pyarrow dependency regardless of which columns are present.

    Args:
        prices: Price series indexed by year
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Polars DataFrame with macro indicators (pivoted, numeric cols)
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    DEFLATOR_COL  = 'Gross Domestic Product Deflator (2017=100)'
    EXCHANGE_COL  = 'Canada-US Exchange Rate (C$/US$)'

    def _scalar(col: str, year: int) -> float:
        """Return a single float from macro_df for (col, year). No pyarrow needed."""
        vals = macro_df.filter(pl.col('Year') == year).get_column(col).to_list()
        if not vals:
            raise KeyError(f"No macro data for year {year}, column '{col}'")
        return float(vals[0])

    if constant_dollars:
        # Single conversion factor applied to all years
        deflator_from = _scalar(DEFLATOR_COL, from_year)
        deflator_to   = _scalar(DEFLATOR_COL, to_year)
        conversion_factor = deflator_to / deflator_from

        er_value = _scalar(EXCHANGE_COL, from_year) if from_currency != to_currency else 1.0

        converted = prices * conversion_factor * er_value

    else:
        # Nominal: convert year-by-year
        deflator_to = _scalar(DEFLATOR_COL, to_year)

        # Pre-build a lookup dict for the years we need — one Polars filter per
        # unique year is much faster than converting the whole DataFrame.
        needed_years = [int(y) for y in prices.index]

        deflator_lookup: dict[int, float] = {}
        exchange_lookup: dict[int, float] = {}
        for year in needed_years:
            row = macro_df.filter(pl.col('Year') == year)
            if len(row) == 0:
                continue
            deflator_lookup[year] = float(row.get_column(DEFLATOR_COL).to_list()[0])
            if from_currency != to_currency:
                exchange_lookup[year] = float(row.get_column(EXCHANGE_COL).to_list()[0])

        converted_dict: dict[int, float] = {}
        for year, price in prices.items():
            year = int(year)
            if year not in deflator_lookup:
                continue
            price_deflated = price * (deflator_to / deflator_lookup[year])
            er = exchange_lookup.get(year, 1.0)
            converted_dict[year] = price_deflated * er

        converted = pd.Series(converted_dict)

    return converted
