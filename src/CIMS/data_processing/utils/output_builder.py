"""Utilities for building output dataframes"""
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from CIMS.data_processing.utils.collapse_constant_years import collapse_constant_years

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


def write_per_region_csvs(
    output: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    skip_if_all_zero: bool = False,
    collapse_years: bool = False,
    report_written: bool = False,
    region_col: str = "Region",
    value_col: str = "Value",
) -> list[str]:
    """Split a long-format output DataFrame by region and write one CSV per region.

    Args:
        output:           The assembled output DataFrame, containing region_col.
        output_dir:        Directory to write into (created if missing).
        prefix:           Sector name used as the output filename prefix,
                          e.g. 'agriculture' -> 'agriculture_ab.csv'.
        skip_if_all_zero: Skip a region entirely if every numeric value_col
                          entry is zero or null (calibration outputs).
        collapse_years:   Apply collapse_constant_years to each region's rows
                          before writing (model-input outputs).
        report_written:   Return only the regions actually written (i.e.
                          excluding those skipped by skip_if_all_zero) instead
                          of every candidate region.
        region_col:       Column to split on (default "Region").
        value_col:        Column checked by skip_if_all_zero (default "Value").

    Returns:
        By default, the sorted list of every region found in output[region_col]
        (matching most callers' pre-existing "Files: N" summary count, even
        where a region ends up skipped). With report_written=True, only the
        regions that were actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = output[region_col].drop_nulls().unique().sort().to_list()
    written = []
    for region in regions:
        region_df = output.filter(pl.col(region_col) == region)
        if skip_if_all_zero and not (
            region_df[value_col].cast(pl.Float64, strict=False).fill_null(0) != 0
        ).any():
            continue
        if collapse_years:
            region_df = collapse_constant_years(region_df)
        out_path = output_dir / f'{prefix}_{region.lower()}.csv'
        region_df.write_csv(out_path)
        print(f'  Wrote {len(region_df):,} rows → {out_path.name}')
        written.append(region)
    return written if report_written else regions
