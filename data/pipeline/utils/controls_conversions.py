"""
Utility functions for energy price processing.
Uses Polars for data processing with pandas for Excel reading.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_control_config(filepath: str) -> Dict[str, str]:
    """Load configuration from Control and Mappings file."""
    # Use pandas for Excel reading (better Excel support)
    df = pd.read_excel(filepath, sheet_name='control', header=None)
    
    config = {}
    for _, row in df.iterrows():
        if pd.notna(row[0]):
            key = str(row[0]).strip()
            value = str(row[1]).strip() if pd.notna(row[1]) else None
            config[key] = value
    
    return config


def load_energy_conversions(filepath: str) -> Dict[str, float]:
    """Load energy conversion factors from Control and Mappings file."""
    # Use pandas for Excel reading
    df = pd.read_excel(filepath, sheet_name='energy conversions', header=None)
    
    conversions = {}
    conversions['bbl_to_m3'] = 6.2898
    conversions['mmbtu_to_gj'] = 1.055056
    
    energy_content_map = {
        139: 'asphalt',
        146: 'lubricants', 
        148: 'naphtha_specialties',
        149: 'petrochemical_feedstock',
        152: 'other_non_energy_products',
        153: 'ethanol',
        156: 'biodiesel',
        157: 'renewable_diesel',
        158: 'renewable_gasoline'
    }
    
    for row_idx, energy_key in energy_content_map.items():
        value_str = str(df.iloc[row_idx, 2])
        gj_value = float(value_str.split()[0])
        conversions[f'{energy_key}_gj_per_m3'] = gj_value
    
    return conversions


def load_energy_mapping(filepath: str) -> pl.DataFrame:
    """Load CIMS to JCIMS energy mapping."""
    # Use pandas to read Excel, then convert to Polars
    df_pd = pd.read_excel(filepath, sheet_name='energy')
    return pl.from_pandas(df_pd)


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
    Uses pandas Series for compatibility with both pandas and polars workflows.
    
    Args:
        prices: Price series indexed by year
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Polars DataFrame with macro indicators
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    # Convert macro_df to pandas for easier indexing
    macro_pd = macro_df.to_pandas()
    
    if constant_dollars:
        # All prices are in constant from_year dollars
        # Apply single conversion factor to all values
        deflator_col = 'Gross Domestic Product Deflator (2017=100)'
        deflator_from = macro_pd[macro_pd['Year'] == from_year][deflator_col].values[0]
        deflator_to = macro_pd[macro_pd['Year'] == to_year][deflator_col].values[0]
        conversion_factor = deflator_to / deflator_from
        
        # Exchange rate (use from_year rate for constant dollars)
        if from_currency != to_currency:
            rate_col = 'Canada-US Exchange Rate (C$/US$)'
            er_value = macro_pd[macro_pd['Year'] == from_year][rate_col].values[0]
        else:
            er_value = 1.0
        
        converted = prices * conversion_factor * er_value
        
    else:
        # Nominal prices: each year's price is in that year's dollars
        # Need to convert each year separately
        deflator_col = 'Gross Domestic Product Deflator (2017=100)'
        deflator_to = macro_pd[macro_pd['Year'] == to_year][deflator_col].values[0]
        
        converted_dict = {}
        for year, price in prices.items():
            year_data = macro_pd[macro_pd['Year'] == year]
            if len(year_data) == 0:
                continue
            
            deflator_year = year_data[deflator_col].values[0]
            
            # Deflate to target year
            price_deflated = price * (deflator_to / deflator_year)
            
            # Exchange rate for this year
            if from_currency != to_currency:
                exchange_rate_col = 'Canada-US Exchange Rate (C$/US$)'
                exchange_rate = year_data[exchange_rate_col].values[0]
            else:
                exchange_rate = 1.0
            
            converted_dict[year] = price_deflated * exchange_rate
        
        converted = pd.Series(converted_dict)
    
    return converted
