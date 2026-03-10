"""
Data Extension Functions

This module contains utility functions for extending data projections
from the base year (2022) to future years (2101) or backfilling historical data. These functions support different
projection and backfill methods. 
"""
import pandas as pd

# ================================================================================
# Dictionary Functions
# ================================================================================

def extend_data_constant(data: dict, base_year: int = 2022, end_year: int = 2101):
    """
    Extend data by holding the base year value constant through all future years.
    
    This method is used for data that is expected to remain stable over time.
    
    Parameters
    ----------
    data : dict
        Dictionary with year keys and numeric values
    base_year : int, optional
        Year to start extending from (default: 2022)
    end_year : int, optional
        Final year to extend to (default: 2101)
    
    Returns
    -------
    dict
        Extended data dictionary with constant values from base_year onward
        
    Examples
    --------
    >>> data = {2020: 100, 2021: 105, 2022: 110}
    >>> extend_data_constant(data)
    {2020: 100, 2021: 105, 2022: 110, 2023: 110, ..., 2101: 110}
    """
    extended_constant = {int(k): v for k, v in data.items()}
    if base_year in extended_constant and extended_constant[base_year] is not None:
        val = extended_constant[base_year]
        for year in range(base_year + 1, end_year):
            extended_constant[year] = val
    return extended_constant


def extend_data_linear(data: dict, base_year: int = 2022, periods=None):
    """
    Extend data with linear growth applied over specified periods.
    
    Parameters
    ----------
    data : dict
        Dictionary with year keys and numeric values
    base_year : int, optional
        Year to start extending from (default: 2022)
    periods : list of lists, optional
        List of [start_year, end_year, growth_rate] for each period.
        Default: [[2023, 2101, 0.01]] (1% annual growth)
    
    Returns
    -------
    dict
        Extended data dictionary with linear growth applied
        
    Examples
    --------
    >>> data = {2020: 1000, 2021: 1020, 2022: 1040}
    >>> periods = [[2023, 2051, 0.012], [2051, 2101, 0.006]]
    >>> extend_data_linear(data, periods=periods)
    {2020: 1000, 2021: 1020, 2022: 1040, 2023: 1052.48, ..., 2100: ...}
    """
    if periods is None:
        periods = [(2023, 2101, 0.01)]
    extended_linear = {int(k): v for k, v in data.items()}
    if base_year in extended_linear and extended_linear[base_year] is not None:
        val = extended_linear[base_year]
        for start_year, end_year, growth_rate in periods:
            for year in range(start_year, end_year):
                val = val * (1 + growth_rate)
                extended_linear[year] = val
    return extended_linear


def extend_trend_decline(data: dict, base_year: int = 2022, trend_start: int = 2000, 
                        trend_end: int = 2022, trend_period=(2023, 2031),
                        decrease_periods=None):
    """
    Extend data with historical trend followed by decline periods.
    
    1. Historical trend (2000-2022) is calculated
    2. Trend continues for a short period (2023-2031)
    3. Then gradual decline occurs over subsequent periods
    
    Parameters
    ----------
    data : dict
        Dictionary with year keys and numeric values
    base_year : int, optional
        Year to start extending from (default: 2022)
    trend_start : int, optional
        First year for trend calculation (default: 2000)
    trend_end : int, optional
        Last year for trend calculation (default: 2022)
    trend_period : tuple, optional
        (start_year, end_year) for continuing the trend (default: (2023, 2031))
    decrease_periods : list of lists, optional
        List of [start_year, end_year, pct_decrease] for each decline period.
        Default: [[2031, 2051, 0.05], [2051, 2101, 0.10]]
        
    Returns
    -------
    dict
        Extended data dictionary with trend and decline applied
        
    Examples
    --------
    >>> data = {2000: 100, 2010: 95, 2020: 90, 2022: 89}
    >>> extend_trend_decline(data)
    {2000: 100, ..., 2022: 89, 2023: 88.5, ..., 2050: ..., 2100: ...}
    """
    if decrease_periods is None:
        decrease_periods = [(2031, 2051, 0.05), (2051, 2101, 0.10)]

    def calculate_trend(data_inner: dict, start_yr: int, end_yr: int):
        """Calculate linear trend slope from historical data."""
        years = []
        values = []
        for y in range(start_yr, end_yr + 1):
            if y in data_inner and data_inner[y] is not None:
                years.append(y)
                values.append(data_inner[y])

        if len(years) < 2:
            return 0

        n = len(years)
        sum_x = sum(years)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(years, values))
        sum_x2 = sum(x * x for x in years)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    extended_trend_decline = {int(k): v for k, v in data.items()}
    trend = calculate_trend(extended_trend_decline, trend_start, trend_end)

    if base_year in extended_trend_decline and extended_trend_decline[base_year] is not None:
        val = extended_trend_decline[base_year]

        # Apply trend period
        trend_start_year, trend_end_year = trend_period
        for year in range(trend_start_year, trend_end_year):
            val = val + trend
            extended_trend_decline[year] = val

        # Apply decrease periods
        for start_year, end_year, pct_decrease in decrease_periods:
            val_start = extended_trend_decline[start_year - 1] if start_year - 1 in extended_trend_decline else val
            num_years = end_year - start_year
            annual_decrease = (val_start - val_start * (1 - pct_decrease)) / num_years

            for year in range(start_year, end_year):
                val = val - annual_decrease
                extended_trend_decline[year] = val

    return extended_trend_decline

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