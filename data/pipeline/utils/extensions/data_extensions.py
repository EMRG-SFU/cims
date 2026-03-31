"""
Data Extension Functions

Utility functions for extending data projections from a base year (2022) to
future years (2101), or backfilling historical data to 2000.

All functions operate on pandas Series (indexed by year) or Polars DataFrames.
The old dict-based extension functions (extend_data_constant, extend_data_linear,
extend_trend_decline) have been replaced with their Series equivalents below.
"""

import pandas as pd
import polars as pl
import numpy as np
from scipy import stats


# ==============================================================================
# PANDAS SERIES — backfill / forward-fill
# ==============================================================================

def backfill_constant(series: pd.Series, start_year: int = 2000) -> pd.Series:
    """
    Backfill a Series to start_year by repeating the first valid value.

    Parameters
    ----------
    series : pd.Series
        Year-indexed numeric series.
    start_year : int
        Earliest year to include (default 2000).

    Returns
    -------
    pd.Series
        Series covering start_year … series.index.max(), with all years
        before the first valid observation filled with that observation.
    """
    full_range = pd.Series(index=range(start_year, series.index.max() + 1), dtype=float)
    full_range.update(series)

    first_idx = series.first_valid_index()
    if first_idx is not None:
        first_valid = series.loc[first_idx]
        full_range.loc[:first_idx] = full_range.loc[:first_idx].fillna(first_valid)

    return full_range


def extend_constant(series: pd.Series, end_year: int = 2100) -> pd.Series:
    """
    Forward-fill a Series to end_year by repeating the last valid value.

    Parameters
    ----------
    series : pd.Series
        Year-indexed numeric series.
    end_year : int
        Latest year to include (default 2100).

    Returns
    -------
    pd.Series
        Series covering series.index.min() … end_year, with all years after
        the last valid observation filled with that observation.
    """
    full_range = pd.Series(index=range(series.index.min(), end_year + 1), dtype=float)
    full_range.update(series)

    last_idx = series.last_valid_index()
    if last_idx is not None:
        last_valid = series.loc[last_idx]
        full_range.loc[last_idx:] = full_range.loc[last_idx:].fillna(last_valid)

    return full_range


def interpolate_5year_to_annual(series: pd.Series) -> pd.Series:
    """
    Linearly interpolate a 5-year interval Series to annual frequency.

    Parameters
    ----------
    series : pd.Series
        Year-indexed series with values every 5 years.

    Returns
    -------
    pd.Series
        Annual series spanning the same year range as the input.
    """
    min_year = series.index.min()
    max_year = series.index.max()
    annual = pd.Series(index=range(min_year, max_year + 1), dtype=float)
    annual.update(series)
    annual = annual.interpolate(method='linear')
    return annual


# ==============================================================================
# PANDAS SERIES — projection extensions
# (replaces dict-based extend_data_constant / extend_data_linear /
#  extend_trend_decline from the old dict_data_extensions.py)
# ==============================================================================

def extend_series_constant(series: pd.Series, base_year: int = 2022,
                            end_year: int = 2101) -> pd.Series:
    """
    Hold the base-year value constant through all future years.

    This is the Series equivalent of the old ``extend_data_constant``.

    Parameters
    ----------
    series : pd.Series
        Year-indexed series containing at least the base_year.
    base_year : int
        Year whose value is held constant (default 2022).
    end_year : int
        Final year (exclusive) of the output (default 2101).

    Returns
    -------
    pd.Series
        Original series extended to end_year - 1 with a constant value.

    Examples
    --------
    >>> s = pd.Series({2020: 100.0, 2021: 105.0, 2022: 110.0})
    >>> extend_series_constant(s)
    # 2023 … 2100 all equal 110.0
    """
    out = series.copy().astype(float)
    if base_year in out.index and pd.notna(out[base_year]):
        val = float(out[base_year])
        for year in range(base_year + 1, end_year):
            out[year] = val
    return out.sort_index()


def extend_series_linear(series: pd.Series, base_year: int = 2022,
                          periods=None) -> pd.Series:
    """
    Apply compound linear (CAGR-style) growth over one or more periods.

    This is the Series equivalent of the old ``extend_data_linear``.

    Parameters
    ----------
    series : pd.Series
        Year-indexed series containing at least the base_year.
    base_year : int
        Year to start extending from (default 2022).
    periods : list of (start_year, end_year, growth_rate), optional
        Each tuple defines a growth-rate period. End year is exclusive.
        Default: [(2023, 2101, 0.01)]  — 1 % annual growth.

    Returns
    -------
    pd.Series
        Original series extended through the last period's end year.

    Examples
    --------
    >>> s = pd.Series({2020: 1000.0, 2021: 1020.0, 2022: 1040.0})
    >>> extend_series_linear(s, periods=[(2023, 2051, 0.012), (2051, 2101, 0.006)])
    """
    if periods is None:
        periods = [(2023, 2101, 0.01)]

    out = series.copy().astype(float)
    if base_year not in out.index or pd.isna(out[base_year]):
        return out

    val = float(out[base_year])
    for start, end, rate in periods:
        for year in range(start, end):
            val = val * (1 + rate)
            out[year] = val
    return out.sort_index()


def extend_series_trend_decline(series: pd.Series, base_year: int = 2022,
                                  trend_start: int = 2000, trend_end: int = 2022,
                                  trend_period=(2023, 2031),
                                  decrease_periods=None) -> pd.Series:
    """
    Extend data by continuing a historical trend for a short period, then
    applying linear annual declines over subsequent periods.

    This is the Series equivalent of the old ``extend_trend_decline``.

    Steps
    -----
    1. Fit a linear trend to the historical data between trend_start and
       trend_end.
    2. Continue that trend from trend_period[0] to trend_period[1].
    3. Apply each decrease_period as a straight-line annual reduction
       (total pct decrease spread evenly across the sub-period years).

    Parameters
    ----------
    series : pd.Series
        Year-indexed series with historical values.
    base_year : int
        Last historical year (default 2022).
    trend_start : int
        First year used to fit the historical trend (default 2000).
    trend_end : int
        Last year used to fit the historical trend (default 2022).
    trend_period : tuple of (int, int)
        (start, end) years for the trend-continuation phase (default
        (2023, 2031)). End is exclusive.
    decrease_periods : list of (start_year, end_year, pct_decrease), optional
        Each tuple defines a decline phase. pct_decrease is a fraction
        (e.g. 0.05 = 5 % total decline over the sub-period). End is exclusive.
        Default: [(2031, 2051, 0.05), (2051, 2101, 0.10)]

    Returns
    -------
    pd.Series
        Extended series through the last decrease period.

    Examples
    --------
    >>> s = pd.Series({2000: 100.0, 2010: 95.0, 2020: 90.0, 2022: 89.0})
    >>> extend_series_trend_decline(s)
    """
    if decrease_periods is None:
        decrease_periods = [(2031, 2051, 0.05), (2051, 2101, 0.10)]

    out = series.copy().astype(float)
    if base_year not in out.index or pd.isna(out[base_year]):
        return out

    # -- Fit historical trend --------------------------------------------------
    hist_years, hist_vals = [], []
    for y in range(trend_start, trend_end + 1):
        if y in out.index and pd.notna(out[y]):
            hist_years.append(y)
            hist_vals.append(float(out[y]))

    if len(hist_years) >= 2:
        slope = stats.linregress(hist_years, hist_vals).slope
    else:
        slope = 0.0

    val = float(out[base_year])

    # -- Trend continuation phase ----------------------------------------------
    t_start, t_end = trend_period
    for year in range(t_start, t_end):
        val = val + slope
        out[year] = val

    # -- Decline phases --------------------------------------------------------
    for d_start, d_end, pct in decrease_periods:
        # Use the value at the end of the previous phase as the starting point
        prev_year = d_start - 1
        val_start = float(out[prev_year]) if prev_year in out.index and pd.notna(out.get(prev_year)) else val
        n_years = d_end - d_start
        annual_decrease = (val_start * pct) / n_years
        val = val_start
        for year in range(d_start, d_end):
            val = val - annual_decrease
            out[year] = val

    return out.sort_index()


# ==============================================================================
# POLARS DATAFRAME — backward trend extrapolation
# ==============================================================================

def trend_backwards(
    df: pl.DataFrame,
    group_cols: list[str],
    year_col: str,
    value_col: str,
    start_year: int,
    fit_start_year: int,
    fit_end_year: int,
) -> pl.DataFrame:
    """
    Extrapolate values backwards using a linear trend anchored at fit_start_year.

    For each unique combination of group_cols, a linear regression is fitted
    to the values between fit_start_year and fit_end_year, and that trend is
    projected back to start_year. Values are floored at 0.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with at least group_cols, year_col, and value_col.
    group_cols : list of str
        Columns that identify each group (e.g. ['Region', 'Variable']).
    year_col : str
        Name of the integer year column.
    value_col : str
        Name of the numeric value column.
    start_year : int
        Earliest year to back-extrapolate to.
    fit_start_year : int
        First year of the window used to fit the trend.
    fit_end_year : int
        Last year of the window used to fit the trend.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with back-extrapolated rows prepended and sorted.

    Examples
    --------
    >>> trend_backwards(df, ['Region', 'Variable'], 'Year', 'Value_1000m3',
    ...                 start_year=2000, fit_start_year=2005, fit_end_year=2010)
    """
    early_years = list(range(start_year, fit_start_year))
    if not early_years:
        return df

    # Filter to the fitting window — stay in Polars, no to_pandas()
    fit_window_pl = df.filter(pl.col(year_col).is_between(fit_start_year, fit_end_year))

    # Get unique group combinations as aligned rows using Polars structs.
    # .to_struct().to_list() returns plain Python dicts — no pyarrow needed.
    unique_groups = (
        df.select(group_cols)
        .unique()
        .to_struct(name='g')
        .to_list()
    )

    new_rows = []

    for group_dict in unique_groups:
        # Filter the fit window to this group using Polars expressions
        group_mask = pl.lit(True)
        for col, val in group_dict.items():
            group_mask = group_mask & (pl.col(col) == val)

        window_pl = fit_window_pl.filter(group_mask).sort(year_col)

        if len(window_pl) == 0:
            continue

        # Extract years and values as plain Python lists — no pyarrow needed
        win_years  = window_pl.get_column(year_col).cast(pl.Int64).to_list()
        win_values = window_pl.get_column(value_col).cast(pl.Float64).to_list()

        # Find anchor value at fit_start_year
        try:
            anchor_idx = win_years.index(fit_start_year)
            anchor = float(win_values[anchor_idx])
        except ValueError:
            continue  # fit_start_year not in this group's window

        slope = (
            stats.linregress(win_years, win_values).slope
            if len(win_years) >= 2 else 0.0
        )

        for y in early_years:
            row = dict(group_dict)
            row[year_col] = y
            row[value_col] = max(anchor + slope * (y - fit_start_year), 0.0)
            new_rows.append(row)

    if not new_rows:
        return df

    early_df = pl.DataFrame(new_rows).cast(
        {year_col: pl.Int64, value_col: pl.Float64}
    )
    return pl.concat([early_df, df], how="diagonal_relaxed").sort(
        group_cols + [year_col]
    )
