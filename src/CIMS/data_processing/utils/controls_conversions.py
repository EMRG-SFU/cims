"""
Utility functions for energy price processing.
Uses Polars for data processing with pandas for Excel reading.
"""


import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Paths: every layer root is in common/paths.py (plan Phase 2). These names are
# kept as aliases, so existing imports keep working.
from CIMS.data_processing.common import paths

_ENV_DATA_ROOT = paths.ENV_DATA_ROOT


def find_repo_root() -> Path:
    """Return the repository root (the directory containing src/ and data/)."""
    return paths.REPO_ROOT


def find_project_root() -> Path:
    """Return the data root: <repo root>/data, unless CIMS_DATA_ROOT is set (see common.paths)."""
    return paths.DATA_ROOT


REPO_ROOT        = paths.REPO_ROOT
BASE_PATH        = paths.DATA_ROOT
PIPELINE_ROOT    = paths.PIPELINE_ROOT
MAPPINGS_PATH    = paths.CONFIG / 'mapping'
CONTROL_FILE     = paths.CONFIG / 'control.py'
ENERGY_MAP_FILE  = MAPPINGS_PATH / 'energy.csv'
REGION_MAP_FILE  = MAPPINGS_PATH / 'region.csv'
SECTOR_MAP_FILE  = MAPPINGS_PATH / 'sector.csv'
CONVERSIONS_FILE = paths.CONFIG / 'conversion/energy-conversion.csv'
SECTOR_REGION_MAP_FILE = MAPPINGS_PATH / 'sector-region.csv'
EXCLUDED_BRANCHES_FILE = paths.CONFIG / 'excluded-branch.csv'

# calibration.py Parameter values considered "demand" or "emissions" data.
# Used to scope config/excluded-branch.csv so it can't accidentally drop
# technology/market-share rows for a branch that's only bad for demand
# or emissions purposes.
DEMAND_EMISSIONS_PARAMS = {
    'calibration_quantity_requested',
    'calibration_emissions_total',
    'calibration_emissions_by_type',
}

def load_control_config() -> Dict:
    """
    Load control settings from the CONTROLS dict in control.py.

    Returns
    -------
    dict
        Control key/value pairs (e.g. currency years, scenario name).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location('control', CONTROL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONTROLS


# Pipeline-wide constants — importable directly so scripts don't need to
# dig into the nested dict each time.
_controls      = load_control_config()
_pipeline      = _controls["pipeline"]
DATA_START     = _pipeline["data_start"]
PROJECTION_END = _pipeline["projection_end"]
LAST_DATA_YEAR = _controls["last_data_year"]
ARCHIVED_DATA  = _controls["archived_data"]


def _parse_numeric(value_str: str) -> Optional[float]:
    """
    Parse a numeric value from the 'Equivalent to' column.

    Handles plain numbers like "0.159" and scientific notation written
    as "1.0551 x 10-3" (the format used in the CER conversions CSV).

    Parameters
    ----------
    value_str : str
        Raw string from the CSV, e.g. "0.159" or "1.7111 x 10-3".

    Returns
    -------
    float or None
        Parsed number, or None if the string could not be parsed.
    """
    # Strip commas (used in large numbers like "1,000")
    cleaned = value_str.replace(',', '').strip()

    # Handle "A x 10B" scientific notation (e.g. "1.7111 x 10-3")
    if ' x 10' in cleaned:
        parts = cleaned.split(' x 10')
        try:
            mantissa = float(parts[0])
            exponent = int(parts[1])
            return mantissa * (10 ** exponent)
        except (ValueError, IndexError):
            return None

    # Plain number
    try:
        return float(cleaned.split()[0])
    except (ValueError, IndexError):
        return None


def parse_cell(value) -> float:
    """Parse a CSV cell to float, mapping dashes and blanks to 0.0.

    Handles em-dashes (–, —), plain dashes, spaces, and common null strings.
    """
    s = str(value).strip().replace('–', '-').replace('—', '-').replace(' ', '')
    if s in ('-', 'nan', '', 'None'):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_all_conversions() -> pd.DataFrame:
    """
    Load the full energy conversions table from config/conversion/energy-conversion.csv
    as a clean, typed DataFrame.

    The CSV has four columns: Group, Energy, Unit, Equivalent to, Source.
    This function returns all rows with a parsed numeric 'Value' column
    added, so downstream code can filter by Group/Energy/Unit without
    needing to re-parse strings.

    Returns
    -------
    pd.DataFrame
        Columns: Group, Energy, Unit, Equivalent_to, Source, Value (float).
        Rows where the numeric value could not be parsed are dropped.

    Examples
    --------
    >>> df = load_all_conversions()
    >>> # Get all GJ-per-m³ energy content rows
    >>> mask = (
    ...     (df['Group'] == 'approximate energy content') &
    ...     (df['Unit'].str.contains('Cubic metres')) &
    ...     (df['Equivalent_to'].str.contains('Gigajoules'))
    ... )
    >>> df[mask][['Energy', 'Value']]
    """
    df = pd.read_csv(CONVERSIONS_FILE, encoding='utf-8-sig')

    # Normalise column names (strip whitespace, replace spaces with _)
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    # The fifth column is named "Equivalent_to" after normalisation;
    # rename to something shorter for convenience.
    df = df.rename(columns={'Equivalent_to': 'Equivalent_to'})

    # Lowercase the Group column so callers can filter case-insensitively
    df['Group'] = df['Group'].str.strip().str.lower()
    df['Energy'] = df['Energy'].fillna('').str.strip()
    df['Unit'] = df['Unit'].fillna('').str.strip()
    df['Equivalent_to'] = df['Equivalent_to'].fillna('').str.strip()
    df['Source'] = df['Source'].fillna('').str.strip()

    # Parse the leading number from 'Equivalent_to' into a float column
    df['Value'] = df['Equivalent_to'].apply(_parse_numeric)

    # Drop rows where we couldn't get a number (e.g. condition/note rows)
    df = df.dropna(subset=['Value']).reset_index(drop=True)

    return df


def load_energy_conversions() -> Dict:
    """
    Load energy conversion factors from config/conversion/energy-conversion.csv.

    Builds a flat dictionary of conversion factors keyed by the names used
    throughout the pipeline. All GJ-per-m³ values come from the
    'approximate energy content' group; volume and energy unit conversions
    come from the 'volume' and 'energy terms' groups.

    Returns
    -------
    dict
        Flat dictionary of conversion factors, e.g.:
        {
            'mmbtu_to_gj': 1.0551,
            'bbl_to_m3': 0.159,
            'diesel_gj_per_m3': 38.68,
            ...
        }
    """
    df = load_all_conversions()

    def _unit_conversion(unit_contains: str, equiv_contains: str) -> float:
        """Return conversion factor where Unit and Equivalent_to match the given strings."""
        mask = (
            df['Unit'].str.contains(unit_contains, case=False, na=False) &
            df['Equivalent_to'].str.contains(equiv_contains, case=False, na=False)
        )
        rows = df[mask]
        if rows.empty:
            raise KeyError(
                f"No conversion row found: Unit~{unit_contains!r}, "
                f"Equivalent_to~{equiv_contains!r}"
            )
        return float(rows.iloc[0]['Value'])

    def _gj_per_m3(energy_name: str) -> float:
        """Return GJ/m³ for a named fuel from the approximate energy content rows."""
        row = df[
            (df['Group'] == 'approximate energy content') &
            (df['Energy'].str.lower() == energy_name.lower()) &
            (df['Unit'].str.contains('Cubic metres', na=False)) &
            (df['Equivalent_to'].str.contains('Gigajoules', na=False))
        ]
        if row.empty:
            raise KeyError(f"No approximate energy content row found for: {energy_name!r}")
        return float(row.iloc[0]['Value'])

    def _gj_per_t(energy_name: str) -> float:
        """Return GJ/tonne for a named fuel from the approximate energy content rows."""
        row = df[
            (df['Group'] == 'approximate energy content') &
            (df['Energy'].str.lower() == energy_name.lower()) &
            (df['Unit'].str.contains('Tonnes', na=False)) &
            (df['Equivalent_to'].str.contains('Gigajoules', na=False))
        ]
        if row.empty:
            raise KeyError(f"No approximate energy content row found for: {energy_name!r}")
        return float(row.iloc[0]['Value'])

    # --- Scalar unit conversions ---
    result = {
        'mmbtu_to_gj': _unit_conversion('Million British thermal units', 'Gigajoules'),
        'gj_to_mmbtu': _unit_conversion('Gigajoules', 'Million British thermal units'),
        'bbl_to_m3':   _unit_conversion('Barrels', 'Cubic metres'),
        'm3_to_bbl':   _unit_conversion('Cubic metres', 'Barrels'),
    }

    # --- All approximate energy content rows keyed by lowercase fuel name ---
    # Automatically picks up every fuel in the CSV so no manual listing is needed.
    # emission_factors.py looks up _EC[csv_name.lower()] directly.
    ec = df[df['Group'] == 'approximate energy content'].copy()
    for _, row in ec.iterrows():
        key = row['Energy'].strip().lower()
        if key and row['Value'] is not None:
            result[key] = float(row['Value'])

    return result


def load_energy_mapping() -> pd.DataFrame:
    """
    Load the CIMS↔JCIMS↔CER energy mapping from config/mapping/energy.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(ENERGY_MAP_FILE)

def load_sector_mapping() -> pd.DataFrame:
    """
    Load the sector map.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(SECTOR_MAP_FILE)

def load_region_mapping() -> pd.DataFrame:
    """
    Load the region map.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: CIMS, JCIMS, CER Prices.
    """
    return pd.read_csv(REGION_MAP_FILE)

def load_sector_regions() -> Dict[str, set]:
    """
    Load allowed regions per sector from config/mapping/sector-region.csv.

    Sectors not present in the map are unrestricted (assumed present in
    every region). Used by sector calibration.py scripts to drop
    demand/emissions rows for region/sector combinations that don't
    actually exist (e.g. Coal Mining in PE).

    Returns
    -------
    dict
        {sector_name: {region_abbr, ...}}.
    """
    df = pd.read_csv(SECTOR_REGION_MAP_FILE)
    return {
        row['Sector']: set(row['Regions'].split(';'))
        for _, row in df.iterrows()
        if pd.notna(row.get('Regions'))
    }


def load_excluded_branches() -> set:
    """
    Load CIMS branches to exclude from demand/emissions calibration output.

    Rows are flagged Active TRUE/FALSE in config/excluded-branch.csv rather than
    deleted, so re-including a branch later is a one-cell edit that keeps
    the row's Notes (why it was excluded) instead of relying on git history.

    Returns
    -------
    set
        Full CIMS branch strings (e.g. 'CIMS.CAN.AB.Petroleum Crude.Transmission')
        currently flagged Active=TRUE.
    """
    df = pd.read_csv(EXCLUDED_BRANCHES_FILE)
    active = df[df['Active'].astype(str).str.strip().str.upper() == 'TRUE']
    return set(active['Branch'].str.strip())


def filter_excluded_branches(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop demand/emissions calibration rows for branches flagged Active in
    config/excluded-branch.csv. Rows for other Parameter types (e.g. technology
    market shares) are left untouched.

    Parameters
    ----------
    df : pl.DataFrame
        A calibration.py `output` DataFrame with 'Branch' and 'Parameter' columns.

    Returns
    -------
    pl.DataFrame
        Same DataFrame with excluded branch/parameter rows removed.
    """
    excluded = load_excluded_branches()
    if not excluded:
        return df
    drop_mask = (
        pl.col('Branch').is_in(list(excluded))
        & pl.col('Parameter').is_in(list(DEMAND_EMISSIONS_PARAMS))
    )
    return df.filter(~drop_mask)

def force_single_technology_market_share(df: pl.DataFrame) -> pl.DataFrame:
    """
    Force market_share_total to 1 for any Branch whose competition has
    exactly one technology but was recorded as 0 in fixed_data.

    A competition with exactly one technology option must award it 100% of
    that service's demand. fixed_data is consistent about this everywhere
    except a handful of region/branch combinations (e.g. NT/NU/YT Electricity
    Storage/CCS/Transmission) that were left at 0 -- compare AB's
    CIMS.CAN.AB.Electricity.Storage.Battery (market_share_total=1) against
    NT's identical node (market_share_total=0), both with the same
    available=2025.

    Parameters
    ----------
    df : pl.DataFrame
        A flattened fixed_data DataFrame (or model `fixed`/`output`)
        with 'Branch', 'Parameter', 'Technology', 'Value' columns.

    Returns
    -------
    pl.DataFrame
        Same DataFrame with single-technology branches' market_share_total
        corrected to '1' where they were '0'.
    """
    tech_counts = (
        df.filter(
            (pl.col('Parameter') == 'technology')
            & pl.col('Technology').is_not_null()
            & (pl.col('Technology') != '')
        )
        .group_by('Branch')
        .agg(pl.col('Technology').n_unique().alias('n_tech'))
    )
    single_tech_branches = set(
        tech_counts.filter(pl.col('n_tech') == 1)['Branch'].to_list()
    )
    if not single_tech_branches:
        return df

    is_target = (
        pl.col('Branch').is_in(list(single_tech_branches))
        & (pl.col('Parameter') == 'market_share_total')
        & (pl.col('Value').cast(pl.Float64, strict=False).fill_null(0.0) == 0.0)
    )
    return df.with_columns(
        pl.when(is_target).then(pl.lit('1')).otherwise(pl.col('Value')).alias('Value')
    )


def load_macro_indicators(filepath: str, scenario: str) -> pl.DataFrame:
    """Load macro indicators (deflators and exchange rates)."""
    # Read CSV with Polars
    df = pl.read_csv(filepath)
    df = df.filter(pl.col('Scenario') == scenario)
    
    # Pivot to wide format for easier access
    pivot = df.pivot(
        values='Value',
        index='Year',
        on='Variable'
    )
    
    return pivot


def convert_currency(
    prices: pd.Series,
    from_year: int,
    to_year: int,
    from_currency: str,
    to_currency: str,
    macro_df: pl.DataFrame = None,
    constant_dollars: bool = False
) -> pd.Series:
    """
    Convert prices from one currency/year to another.

    Uses the same deflator and exchange tables, and the same method, as
    CIMS.Model (``CIMS.unit_conversion.CurrencyConverter``): deflate within the
    source currency, then exchange at the target year. See
    ``CIMS.data_processing.utils.currency``.

    Args:
        prices: Price series indexed by year
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Ignored. Kept so existing calls still work; the CER macro
                  deflator and exchange rate are no longer used for currency.
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    from CIMS.data_processing.utils.currency import convert_series

    return convert_series(
        prices,
        from_currency=from_currency,
        to_year=to_year,
        to_currency=to_currency,
        from_year=from_year if constant_dollars else None,
    )
