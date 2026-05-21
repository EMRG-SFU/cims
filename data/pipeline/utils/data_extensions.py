"""
Data Extension Functions

Utility functions for projecting data forward from a base year through future years:
  - extend_constant: hold last valid value constant to an end year
  - extend_series_constant: hold a specific base-year value constant
  - extend_series_linear: apply CAGR-style growth over one or more periods
  - extend_series_trend_decline: fit historical trend then apply decline phases
  - extend_series_trend_dampener: fit historical trend then progressively dampen growth rate
  - load_cagr_assumptions: parse CAGR periods and dampeners from a CSV
  - compute_cagr: compute compound annual growth rate between two years
  - extend_cagr_periods: project forward through multiple periods with dampened CAGR

All functions operate on pandas Series indexed by year.
"""

import pandas as pd
import numpy as np
from scipy import stats


# ==============================================================================
# PANDAS SERIES — forward fill
# ==============================================================================

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


# ==============================================================================
# PANDAS SERIES — projection extensions
# ==============================================================================

def extend_series_constant(series: pd.Series, base_year: int = 2022,
                            end_year: int = 2101) -> pd.Series:
    """
    Hold the base-year value constant through all future years.

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

    last_year = max(end for _, end, _ in decrease_periods)
    full_index = range(int(series.index.min()), last_year)
    out = pd.Series(index=full_index, dtype=float)
    for yr, v in series.items():
        if yr in out.index:
            out[int(yr)] = v

    if base_year not in out.index or pd.isna(out[base_year]):
        return series.copy().astype(float)

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

    t_start, t_end = trend_period
    for year in range(base_year + 1, t_end):
        val = val + slope
        out[year] = val

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

    last_year = max(end for _, end, _ in decline_periods)
    full_index = range(int(series.index.min()), last_year)
    out = pd.Series(index=full_index, dtype=float)
    for yr, v in series.items():
        if int(yr) in out.index:
            out[int(yr)] = v

    if base_year not in out.index or pd.isna(out[base_year]):
        return series.copy().astype(float)

    hist_years, hist_vals = [], []
    for y in range(trend_start, trend_end + 1):
        if y in out.index and pd.notna(out[y]):
            hist_years.append(y)
            hist_vals.append(float(out[y]))

    if len(hist_years) >= 2:
        slope = stats.linregress(hist_years, hist_vals).slope
    else:
        slope = 0.0

    t_start, t_end = trend_period
    val = float(out[base_year])
    for year in range(base_year + 1, t_end):
        val = val + slope
        val = max(val, 0.0)
        out[year] = val

    # Formula: next = current * (1 + (current/previous - 1) * (1 + rate))
    for d_start, d_end, rate in decline_periods:
        prev_year = d_start - 1
        if prev_year not in out.index or pd.isna(out[prev_year]):
            continue
        prev = float(out[prev_year])

        pre_prev_year = prev_year - 1
        if pre_prev_year not in out.index or pd.isna(out[pre_prev_year]):
            pre_prev = prev
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
# PANDAS SERIES — CAGR projection with per-period dampeners
# ==============================================================================

def load_cagr_assumptions(
    sector: str,
    assumptions_file,
) -> tuple[int, int, list[tuple[int, int, float]]]:
    """
    Parse CAGR periods and dampeners for one sector from the assumptions CSV.

    Expected CSV columns:
        Sector, CAGR Period,
        1st period, 1st period dampener,
        2nd period, 2nd period dampener,
        3d period,  3rd period dampener

    Parameters
    ----------
    sector : str
        Value to match in the 'Sector' column.
    assumptions_file : str or Path
        Path to the CSV file.

    Returns
    -------
    (cagr_start, cagr_end, periods)
        cagr_start, cagr_end : int — historical window for CAGR calculation.
        periods : list of (start_year, end_year, dampener) — projection periods.
    """
    def _parse_period(s) -> tuple[int, int] | None:
        try:
            a, b = str(s).strip().split('-')
            return int(a), int(b)
        except (ValueError, AttributeError):
            return None

    def _parse_pct(s) -> float | None:
        if s is None:
            return None
        try:
            if pd.isna(s):
                return None
        except (TypeError, ValueError):
            pass
        text = str(s).strip()
        if not text:
            return None
        return float(text.replace('%', '')) / 100.0

    df = pd.read_csv(assumptions_file, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['Sector'] = df['Sector'].str.strip()
    row = df[df['Sector'] == sector].iloc[0]

    parsed = _parse_period(row['CAGR Period'])
    if parsed is None:
        raise ValueError(f"Could not parse 'CAGR Period' for sector '{sector}'")
    cagr_start, cagr_end = parsed

    period_col_pairs = [
        ('1st period', '1st period dampener'),
        ('2nd period', '2nd period dampener'),
        ('3d period',  '3rd period dampener'),
    ]
    periods: list[tuple[int, int, float]] = []
    for p_col, d_col in period_col_pairs:
        p_val = row.get(p_col)
        d_val = row.get(d_col)
        p = _parse_period(p_val)
        d = _parse_pct(d_val)
        if p is None or d is None:
            break
        periods.append((*p, d))

    if not periods:
        raise ValueError(f"No valid projection periods found for sector '{sector}'")

    return cagr_start, cagr_end, periods


def compute_cagr(series: pd.Series, start_year: int, end_year: int) -> float:
    """
    Compute a robust CAGR as the median of consecutive year-over-year growth
    rates within [start_year, end_year].

    Using the median rather than the point-to-point formula avoids sensitivity
    to outlier values at either endpoint (e.g. revised or preliminary data).
    Falls back to point-to-point if fewer than two consecutive year pairs exist
    in the window.

    Parameters
    ----------
    series : pd.Series
        Year-indexed numeric series.
    start_year : int
        First year of the CAGR window.
    end_year : int
        Last year of the CAGR window.

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.03 = 3 % per year).
    """
    window = series.dropna()
    window = window[(window.index >= start_year) & (window.index <= end_year)]
    idx = sorted(window.index)
    yoy = [
        float(window.loc[idx[i]] / window.loc[idx[i - 1]] - 1)
        for i in range(1, len(idx))
        if idx[i] == idx[i - 1] + 1 and window.loc[idx[i - 1]] > 0
    ]
    if yoy:
        return float(np.median(yoy))
    # Fallback: point-to-point
    if start_year not in series.index or end_year not in series.index:
        return 0.0
    val_start = series.loc[start_year]
    val_end   = series.loc[end_year]
    if pd.isna(val_start) or pd.isna(val_end) or val_start <= 0:
        return 0.0
    return float((val_end / val_start) ** (1.0 / (end_year - start_year)) - 1.0)


def extend_cagr_periods(
    base_val: float,
    raw_cagr: float,
    periods: list[tuple[int, int, float]],
    override: tuple[float, ...] | None = None,
) -> pd.Series:
    """
    Extend a value forward through multiple periods using a dampened CAGR.

    Parameters
    ----------
    base_val : float
        Starting value at the last historical year.
    raw_cagr : float
        Undampened CAGR from the historical anchor window (e.g. from
        compute_cagr).
    periods : list of (start_year, end_year, dampener)
        Each tuple defines one projection period. End year is inclusive.
        The dampener fraction (0–1) is multiplied by raw_cagr to get the
        effective annual rate for that period.
    override : tuple of float, optional
        Explicit per-period annual rates, one entry per period. When
        provided, bypasses raw_cagr × dampener entirely.

    Returns
    -------
    pd.Series
        Year-indexed series from periods[0][0] through periods[-1][1],
        floored at 0.
    """
    val = base_val
    years: list[int]    = []
    values: list[float] = []
    for i, (p_start, p_end, dampener) in enumerate(periods):
        period_cagr = override[i] if override is not None else raw_cagr * dampener
        for yr in range(p_start, p_end + 1):
            val = max(val * (1 + period_cagr), 0.0)
            years.append(yr)
            values.append(val)
    return pd.Series(values, index=years, dtype=float)
