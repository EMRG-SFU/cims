"""Utilities for building output dataframes"""
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

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


def log_output(
    df: pl.DataFrame,
    path,
    *,
    region_col: str = "Region",
    variable_col: str = "Variable",
    year_col: str = "Year",
) -> None:
    """Print a consistent save summary to the terminal.

    Prints the output path, row count, and — where the named columns exist —
    unique region count, variable list, and year range.

    Args:
        df:           The DataFrame that was written.
        path:         The file path it was written to.
        region_col:   Column name for regions (default "Region").
        variable_col: Column name for variables (default "Variable").
        year_col:     Column name for years (default "Year").
    """
    cols = df.columns
    print(f"\n✅  Saved → {Path(path)}")
    print(f"    Rows:      {len(df):,}")
    if region_col in cols:
        print(f"    Regions:   {df[region_col].n_unique()} unique")
    if variable_col in cols:
        variables = sorted(df[variable_col].unique().to_list())
        preview = ", ".join(str(v) for v in variables[:5])
        suffix = f" … (+{len(variables) - 5} more)" if len(variables) > 5 else ""
        print(f"    Variables: {preview}{suffix}")
    if year_col in cols:
        print(f"    Years:     {df[year_col].min()} – {df[year_col].max()}")
