"""Utilities for building output dataframes"""
import numpy as np
import pandas as pd
import polars as pl

YEARS = list(range(2000, 2101))
META_COLS = ["Branch", "Type", "Region", "Sector", "Service", "Technology", "Parameter",
             "Context", "Sub_Context", "Target", "Source", "Unit"]


def make_row(meta: dict, series: dict = None, scale: float = 1.0, extend_func=None):
    """Build a row for the output dataframe
    
    Args:
        meta: Dictionary of metadata columns
        series: Dictionary of {year: value}
        scale: Multiplier to apply to all values
        extend_func: Optional function to extend the series (e.g., extend_households)
    
    Returns:
        Dictionary representing one row
    """
    row = {k: meta.get(k, "") for k in META_COLS}
    
    # Apply extension function if provided
    if extend_func is not None and series is not None:
        series = extend_func(series)
    
    for y in YEARS:
        v = None
        if series is not None and y in series:
            vv = series[y]
            if vv is not None and not (isinstance(vv, float) and np.isnan(vv)):
                v = float(vv) * scale
        row[str(y)] = v
    return row


def pl_to_series(df: pl.DataFrame) -> pd.Series:
    """Extract year→value from a long-format Polars DataFrame as a pd.Series."""
    years  = df.get_column('year').cast(pl.Int64).to_list()
    values = df.get_column('value').cast(pl.Float64).to_list()
    return pd.Series(values, index=years, dtype=float)


def pl_get_scalar(df: pl.DataFrame, col: str) -> object:
    """Return the first value of a column from a one-row Polars DataFrame."""
    return df.get_column(col).to_list()[0]
