"""
Statistics Canada Data Utilities

Shared functions for loading and processing Statistics Canada data files
used across multiple pipeline modules.
"""

from pathlib import Path
import pandas as pd


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
        REF_DATE format controlled by date_format parameter.
    regions : list of str, optional
        GEO values to include. If None, all regions are included.
        Shares are computed relative to the total of included regions only.
    projection_end : int
        Last year to extend shares to (default 2100).
    date_format : str
        strftime format for parsing REF_DATE (default '%Y-%m' for '2000-01').
        Use '%b-%y' for old-style 'Jan-00' format.

    Returns
    -------
    pd.DataFrame with columns: year (int), territory, pop_share
        pop_share sums to 1.0 across all included regions for each year.
        Covers all years from first observed year through projection_end.

    Examples
    --------
    # All territories (shares sum to 1 across YT/NT/NU)
    terr_shares = build_population_shares(
        pop_csv,
        regions=['Yukon', 'Northwest Territories', 'Nunavut']
    )

    # All Atlantic provinces (shares sum to 1 across NL/PE/NS/NB)
    at_shares = build_population_shares(
        pop_csv,
        regions=['Newfoundland and Labrador', 'Prince Edward Island',
                 'Nova Scotia', 'New Brunswick']
    )

    # All provinces and territories
    all_shares = build_population_shares(pop_csv)
    """
    pop = pd.read_csv(pop_csv, usecols=['REF_DATE', 'GEO', 'VALUE'])
    pop.columns = ['date', 'territory', 'population']
    pop['year'] = pd.to_datetime(pop['date'], format=date_format).dt.year

    # Filter to specified regions if provided
    if regions is not None:
        pop = pop[pop['territory'].isin(regions)].copy()

    # Annual average population per territory
    annual = (
        pop.groupby(['year', 'territory'])['population']
        .mean()
        .reset_index()
    )

    # Each territory's share of the group total for that year
    total = (
        annual.groupby('year')['population']
        .sum()
        .rename('total_pop')
        .reset_index()
    )
    annual = annual.merge(total, on='year')
    annual['pop_share'] = annual['population'] / annual['total_pop']
    historical = annual[['year', 'territory', 'pop_share']].copy()

    # Hold last observed share constant through projection_end
    last_year = int(historical['year'].max())
    last_shares = (
        historical[historical['year'] == last_year]
        [['territory', 'pop_share']]
    )

    projected_rows = pd.DataFrame([
        {'year': yr, 'territory': row['territory'], 'pop_share': row['pop_share']}
        for yr in range(last_year + 1, projection_end + 1)
        for _, row in last_shares.iterrows()
    ])

    return pd.concat([historical, projected_rows], ignore_index=True)
