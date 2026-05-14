"""
Statistics Canada Extractor Functions

Shared functions for loading Statistics Canada CSVs consistently across the pipeline.

  read_statscan_csv      — base reader: all columns as strings, suppression codes loggable
  load_resd              — RESD family (25100029, 2510002901-residential/commercial):
                           standardised column names + year/value cast
  build_population_shares — load quarterly population CSV, compute annual regional shares,
                            project constant through a given end year
"""

import pandas as pd
import polars as pl
from pathlib import Path

# Known Stats Can suppression / unavailability codes
SUPPRESSION_CODES = ["x", "..", "F", "E"]


def read_statscan_csv(
    path: Path,
    columns: list[str] = None,
    log_suppressed: bool = False,
) -> pl.DataFrame:
    """
    Load any Statistics Canada CSV with all columns read as strings.

    Reading as strings (infer_schema_length=0) preserves suppression codes
    ('x', '..', 'F', 'E') so they can be detected and handled explicitly by
    the caller rather than silently coerced to nulls by schema inference.
    Callers cast VALUE to Float64 with strict=False as needed.

    Parameters
    ----------
    path : Path
        CSV file to load.
    columns : list of str, optional
        Subset of columns to load. If None, all columns are loaded.
    log_suppressed : bool
        If True, print a count and list of suppression codes found in the
        VALUE column. Useful for auditing data quality at load time.

    Returns
    -------
    pl.DataFrame
        All columns as Utf8. VALUE is NOT cast — callers do their own casting.
    """
    df = pl.read_csv(path, infer_schema_length=0, columns=columns)

    if log_suppressed and "VALUE" in df.columns:
        suppressed = df.filter(pl.col("VALUE").is_in(SUPPRESSION_CODES))
        n = len(suppressed)
        if n > 0:
            codes = suppressed["VALUE"].unique().to_list()
            print(f"  {Path(path).name}: {n} suppressed cells found {codes}")

    return df


def load_resd(path: Path) -> pl.DataFrame:
    """
    Load a Statistics Canada RESD CSV with standardised column names and types.

    Covers the 25100029 / 2510002901 table family (residential, commercial,
    and general energy use by sector). Suppression codes in VALUE become null.

    Parameters
    ----------
    path : Path
        Path to the RESD CSV file.

    Returns
    -------
    pl.DataFrame with columns:
        year           (Int32)  — calendar year parsed from REF_DATE
        geo            (Utf8)   — province / territory name as published
        fuel           (Utf8)   — Fuel type column
        characteristic (Utf8)   — Supply and demand characteristics column
        value          (Float64) — demand in TJ; null for suppressed cells
    """
    return (
        read_statscan_csv(path)
        .rename({
            "REF_DATE":                          "year",
            "GEO":                               "geo",
            "Fuel type":                         "fuel",
            "Supply and demand characteristics": "characteristic",
            "VALUE":                             "value",
        })
        .with_columns([
            pl.col("year").cast(pl.Int32),
            pl.col("value").cast(pl.Float64, strict=False),
        ])
    )


def build_population_shares(
    pop_csv: Path,
    regions: list[str] = None,
    projection_end: int = 2100,
    date_format: str = '%Y-%m',
) -> pd.DataFrame:
    """
    Load a Statistics Canada quarterly population CSV, average to annual,
    compute each region's share of the filtered group total, then hold
    the last observed share constant through projection_end.

    Parameters
    ----------
    pop_csv : Path
        Path to the Stats Can population CSV (e.g. table 17-10-0009-01).
        Expected columns: REF_DATE, GEO, VALUE.
    regions : list of str, optional
        GEO values to include. If None, all regions are included.
        Shares are computed relative to the total of included regions only.
    projection_end : int
        Last year to extend shares to (default 2100).
    date_format : str
        strftime format for parsing REF_DATE (default '%Y-%m' for '2000-01').

    Returns
    -------
    pd.DataFrame with columns: year (int), territory (str), pop_share (float)
        pop_share sums to 1.0 across all included regions for each year.
        Covers all years from first observed year through projection_end.
    """
    pop = pd.read_csv(pop_csv, usecols=['REF_DATE', 'GEO', 'VALUE'])
    pop.columns = ['date', 'territory', 'population']
    pop['year'] = pd.to_datetime(pop['date'], format=date_format).dt.year

    if regions is not None:
        pop = pop[pop['territory'].isin(regions)].copy()

    annual = (
        pop.groupby(['year', 'territory'])['population']
        .mean()
        .reset_index()
    )
    total = (
        annual.groupby('year')['population']
        .sum()
        .rename('total_pop')
        .reset_index()
    )
    annual = annual.merge(total, on='year')
    annual['pop_share'] = annual['population'] / annual['total_pop']
    historical = annual[['year', 'territory', 'pop_share']].copy()

    last_year = int(historical['year'].max())
    last_shares = historical[historical['year'] == last_year][['territory', 'pop_share']]
    projected_rows = pd.DataFrame([
        {'year': yr, 'territory': row['territory'], 'pop_share': row['pop_share']}
        for yr in range(last_year + 1, projection_end + 1)
        for _, row in last_shares.iterrows()
    ])

    return pd.concat([historical, projected_rows], ignore_index=True)
