"""
Data Extension Functions

This module contains utility functions for extending data projections for data frames
from the base year (2022) to future years (2101) or backfilling historical data. These functions support different
projection and backfill methods. 
"""
import pandas as pd


# ================================================================================
# Dataframe Functions
# ================================================================================

def backfill_extend(series: pd.Series, start_year: int = 2000, end_year: int = 2100) -> pd.Series:
    """Backfill and forward fill a series to cover full year range."""
    full_range = pd.Series(index=range(start_year, end_year + 1), dtype=float)
    full_range.update(series)
    
    first_valid = series.loc[series.first_valid_index()] if len(series) > 0 and series.first_valid_index() is not None else None
    last_valid = series.loc[series.last_valid_index()] if len(series) > 0 and series.last_valid_index() is not None else None
    
    if first_valid is not None:
        first_year = series.first_valid_index()
        full_range.loc[:first_year] = full_range.loc[:first_year].fillna(first_valid)
    
    if last_valid is not None:
        last_year = series.last_valid_index()
        full_range.loc[last_year:] = full_range.loc[last_year:].fillna(last_valid)
    
    return full_range


def interpolate_5year_to_annual(series: pd.Series) -> pd.Series:
    """Interpolate 5-year interval data to annual."""
    min_year = series.index.min()
    max_year = series.index.max()
    annual_index = pd.Series(index=range(min_year, max_year + 1), dtype=float)
    annual_index.update(series)
    annual_index = annual_index.interpolate(method='linear')
    return annual_index