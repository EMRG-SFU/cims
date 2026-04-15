"""
Helper functions for NRCan/CEUD Excel format

This format is used by multiple sectors:
- Residential (res_bc_e.xls)
- Commercial (commercial_bc_e.xls)
- Industrial (various files)

Only create extractors for formats used by MULTIPLE sectors!
"""
import polars as pl
import numpy as np


def extract_year_cols(df: pl.DataFrame, header_row: int = 4, start_col: int = 2):
    """Extract year columns from CEUD format
    
    Args:
        df: Polars DataFrame
        header_row: Row index containing years
        start_col: First column containing data
        
    Returns:
        List of (year, column_index) tuples
    """
    arr = df.to_numpy()
    years = []
    for c in range(start_col, arr.shape[1]):
        v = arr[header_row, c]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            break
        try:
            year = int(v)
            years.append((year, c))
        except Exception:
            break
    return years


def find_row_indices(df: pl.DataFrame, label: str, label_col: int = 1):
    """Find all row indices that match a label in CEUD format
    
    Args:
        df: Polars DataFrame
        label: Row label to search for
        label_col: Column containing labels
        
    Returns:
        List of row indices
    """
    return (
        df.select(pl.col(df.columns[label_col]).cast(pl.Utf8).str.strip_chars().alias("lab"))
          .with_row_index("row")
          .filter(pl.col("lab") == label)
          .select("row")["row"].to_list()
    )


def get_row_series(df: pl.DataFrame, label: str, match_n: int = 0, 
                   label_col: int = 1, header_row: int = 4, start_col: int = 2):
    """Get time series from a specific row in CEUD format
    
    Args:
        df: Polars DataFrame
        label: Row label to search for
        match_n: Which match to use if label appears multiple times
        label_col: Column containing labels
        header_row: Row containing years
        start_col: First data column
        
    Returns:
        Dictionary mapping years to values
    """
    idxs = find_row_indices(df, label, label_col)
    if len(idxs) <= match_n:
        raise KeyError(f"Label '{label}' not found (need #{match_n+1}, got {len(idxs)})")
    
    arr = df.to_numpy()
    years = extract_year_cols(df, header_row, start_col)
    return {y: arr[idxs[match_n], c] for y, c in years}
