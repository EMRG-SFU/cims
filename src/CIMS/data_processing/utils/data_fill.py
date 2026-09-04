"""
Data Fill Functions

Utility functions for repairing and filling gaps in historical data:
  - backfill_constant: repeat first valid value backward to a start year
  - interpolate_5year_to_annual: linearly interpolate 5-year data to annual
  - interpolate_gaps: fill mid-series NaN gaps within groups in a Polars DataFrame
  - trend_backwards: back-extrapolate values using a fitted linear trend

All functions operate on pandas Series (indexed by year) or Polars DataFrames.
"""

import pandas as pd
import polars as pl
import numpy as np
from scipy import stats


# ==============================================================================
# PANDAS SERIES — historical gap filling
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
    full_range = pd.Series(index=range(start_year, int(series.index.max()) + 1), dtype=float)
    full_range.update(series)

    first_idx = series.first_valid_index()
    if first_idx is not None:
        first_valid = series.loc[first_idx]
        full_range.loc[:first_idx] = full_range.loc[:first_idx].fillna(first_valid)

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
    """
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

        valid_idx = [
            i for i, v in enumerate(values)
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]

        if len(valid_idx) >= 2:
            first_valid = valid_idx[0]
            last_valid  = valid_idx[-1]

            for i in range(first_valid, last_valid + 1):
                if values[i] is None or (isinstance(values[i], float) and np.isnan(values[i])):
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
    """
    early_years = list(range(start_year, fit_start_year))
    if not early_years:
        return df

    fit_window_pl = df.filter(pl.col(year_col).is_between(fit_start_year, fit_end_year))

    unique_groups = (
        df.select(group_cols)
        .unique()
        .to_struct(name='g')
        .to_list()
    )

    new_rows = []

    for group_dict in unique_groups:
        group_mask = pl.lit(True)
        for col, val in group_dict.items():
            group_mask = group_mask & (pl.col(col) == val)

        window_pl = fit_window_pl.filter(group_mask).sort(year_col)

        if len(window_pl) == 0:
            continue

        win_years  = window_pl.get_column(year_col).cast(pl.Int64).to_list()
        win_values = window_pl.get_column(value_col).cast(pl.Float64).to_list()

        try:
            anchor_idx = win_years.index(fit_start_year)
            anchor = float(win_values[anchor_idx])
        except ValueError:
            continue

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
