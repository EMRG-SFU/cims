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
    if decrease_periods is None:
        decrease_periods = [(2031, 2051, 0.05), (2051, 2101, 0.10)]

    # Determine the full year range needed and pre-allocate
    last_year = max(end for _, end, _ in decrease_periods)
    full_index = range(int(series.index.min()), last_year)
    out = pd.Series(index=full_index, dtype=float)
    # Copy historical values in
    for yr, v in series.items():
        if yr in out.index:
            out[int(yr)] = v

    if base_year not in out.index or pd.isna(out[base_year]):
        return series.copy().astype(float)

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
        prev_year = d_start - 1
        val_start = float(out[prev_year]) if pd.notna(out[prev_year]) else val
        n_years = d_end - d_start
        if n_years <= 0:
            continue
        annual_decrease = (val_start * pct) / n_years
        val = val_start
        for year in range(d_start, d_end):
            val = val - annual_decrease
            val = max(val, 0.0)
            out[year] = val

    return out.sort_index()

def extend_series_trend_dampener(series: pd.Series, base_year: int = 2022,
                                  trend_start: int = 2000, trend_end: int = 2022,
                                  trend_period: tuple = (2023, 2031),
                                  decline_periods: list = None) -> pd.Series:
    """
    Extend a series by:
      1. Continuing the historical linear trend through trend_period
      2. Applying a growth-rate dampener in each decline period

    The dampener formula mirrors the Excel formula:
        next = current * (1 + (current/previous - 1) * (1 + rate))

    Where rate is negative (e.g. -0.05) to progressively slow growth toward zero.
    A rate of -1.0 would freeze growth entirely; a rate of 0.0 leaves it unchanged.

    Parameters
    ----------
    series : pd.Series
        Year-indexed historical series.
    base_year : int
        Last historical year (default 2022).
    trend_start : int
        First year used to fit the historical trend (default 2000).
    trend_end : int
        Last year used to fit the historical trend (default 2022).
    trend_period : tuple of (int, int)
        (start_year, end_year) for trend continuation. End is exclusive.
        e.g. (2023, 2031) projects 2023–2030 using the historical trend.
    decline_periods : list of (start_year, end_year, rate)
        Each tuple defines a dampener phase. End is exclusive.
        rate should be negative to slow growth (e.g. -0.05 = dampen by 5%).
        Default: [(2031, 2051, -0.05), (2051, 2101, -0.10)]

    Returns
    -------
    pd.Series
        Extended series through the last decline period's end year.
    """
    if decline_periods is None:
        decline_periods = [(2031, 2051, -0.05), (2051, 2101, -0.10)]

    # Pre-allocate full index covering all years needed
    last_year = max(end for _, end, _ in decline_periods)
    full_index = range(int(series.index.min()), last_year)
    out = pd.Series(index=full_index, dtype=float)
    for yr, v in series.items():
        if int(yr) in out.index:
            out[int(yr)] = v

    if base_year not in out.index or pd.isna(out[base_year]):
        return series.copy().astype(float)

    # -- Fit historical linear trend ------------------------------------------
    hist_years, hist_vals = [], []
    for y in range(trend_start, trend_end + 1):
        if y in out.index and pd.notna(out[y]):
            hist_years.append(y)
            hist_vals.append(float(out[y]))

    if len(hist_years) >= 2:
        slope = stats.linregress(hist_years, hist_vals).slope
    else:
        slope = 0.0

    # -- Trend continuation phase ---------------------------------------------
    t_start, t_end = trend_period
    val = float(out[base_year])
    for year in range(t_start, t_end):
        val = val + slope
        val = max(val, 0.0)
        out[year] = val

    # -- Dampener decline phases ----------------------------------------------
    # Formula: next = current * (1 + (current/previous - 1) * (1 + rate))
    # This progressively slows the growth rate each year.
    for d_start, d_end, rate in decline_periods:
        prev_year = d_start - 1
        if prev_year not in out.index or pd.isna(out[prev_year]):
            continue
        prev = float(out[prev_year])

        # We need two consecutive values to compute the growth rate.
        # For the first year of the phase, prev_year-1 is the year before that.
        pre_prev_year = prev_year - 1
        if pre_prev_year not in out.index or pd.isna(out[pre_prev_year]):
            pre_prev = prev  # no growth if we can't find the prior year
        else:
            pre_prev = float(out[pre_prev_year])

        curr = prev
        prev_val = pre_prev

        for year in range(d_start, d_end):
            growth_rate = (curr / prev_val - 1) if prev_val != 0 else 0.0
            next_val = curr * (1 + growth_rate * (1 + rate))
            next_val = max(next_val, 0.0)
            out[year] = next_val
            prev_val = curr
            curr = next_val

    return out.sort_index()


# ==============================================================================
# POLARS DATAFRAME — gap interpolation
# ==============================================================================

def interpolate_gaps(
    df: "pl.DataFrame",
    group_cols: "list[str]",
    year_col: str,
    value_col: str,
) -> "pl.DataFrame":
    """
    Linearly interpolate mid-series NaN gaps within each group.

    Fills NaN values that are bounded on both sides by valid observations
    using linear interpolation. Leading or trailing NaNs (before the first
    or after the last valid value) are left untouched — use backfill_constant
    or extend_constant for those.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with at least group_cols, year_col, and value_col.
        Must contain one row per group per year (no duplicate year/group pairs).
    group_cols : list of str
        Columns that identify each group (e.g. ['Province']).
    year_col : str
        Name of the integer year column. Used as the x-axis for interpolation
        so uneven gaps are handled correctly.
    value_col : str
        Name of the numeric column to interpolate.

    Returns
    -------
    pl.DataFrame
        Same shape as input, with mid-series NaNs replaced by linearly
        interpolated values. Row order is preserved.

    Examples
    --------
    >>> interpolate_gaps(df, group_cols=['Province'], year_col='Year',
    ...                  value_col='Processed')
    # NWT 2015-2019 (suppressed) filled by linear interp between 2014 and 2020.
    """
    import numpy as np

    result_frames = []

    unique_groups = (
        df.select(group_cols)
        .unique()
        .to_struct(name="g")
        .to_list()
    )

    for group_dict in unique_groups:
        group_mask = pl.lit(True)
        for col, val in group_dict.items():
            group_mask = group_mask & (pl.col(col) == val)

        grp = df.filter(group_mask).sort(year_col)

        years  = grp.get_column(year_col).to_list()
        values = grp.get_column(value_col).to_list()

        # Identify indices with valid (non-null, non-NaN) values
        valid_idx = [
            i for i, v in enumerate(values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]

        if len(valid_idx) >= 2:
            first_valid = valid_idx[0]
            last_valid  = valid_idx[-1]

            for i in range(first_valid, last_valid + 1):
                if values[i] is None or (isinstance(values[i], float) and np.isnan(values[i])):
                    # Find the nearest valid neighbours on each side
                    lo = max(j for j in valid_idx if j < i)
                    hi = min(j for j in valid_idx if j > i)
                    x0, y0 = years[lo], values[lo]
                    x1, y1 = years[hi], values[hi]
                    values[i] = y0 + (y1 - y0) * (years[i] - x0) / (x1 - x0)

        grp = grp.with_columns(
            pl.Series(name=value_col, values=values, dtype=pl.Float64)
        )
        result_frames.append(grp)

    if not result_frames:
        return df

    return pl.concat(result_frames).sort(group_cols + [year_col])

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
