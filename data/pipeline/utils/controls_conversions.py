"""
Utility functions for energy price processing.
Uses Polars for data processing with pandas for Excel reading.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Paths — derived from this file's location so nothing is hardcoded.
# controls_conversions.py lives at pipeline/utils/, so:
#   parent       → pipeline/utils/
#   parent.parent → pipeline/
#   parent.parent.parent → data/  (the project root)
def find_project_root() -> Path:
    """Return the data root directory (the parent of pipeline/)."""
    return Path(__file__).resolve().parent.parent.parent

BASE_PATH        = find_project_root()
PIPELINE_ROOT    = BASE_PATH / 'pipeline'
MAPPINGS_PATH    = BASE_PATH / 'mappings_conversions'
CONTROL_FILE     = MAPPINGS_PATH / 'control.py'
ENERGY_MAP_FILE  = MAPPINGS_PATH / 'energy_map.csv'
REGION_MAP_FILE  = MAPPINGS_PATH / 'region_map.csv'
SECTOR_MAP_FILE  = MAPPINGS_PATH / 'sector_map.csv'
CONVERSIONS_FILE = MAPPINGS_PATH / 'energy_conversions.csv'

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
    Load the full energy conversions table from energy_conversions.csv
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
    Load energy conversion factors from energy_conversions.csv.

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
    Load the CIMS↔JCIMS↔CER energy mapping from energy_map.csv.

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

def load_macro_indicators(filepath: str, scenario: str) -> pl.DataFrame:
    """Load macro indicators (deflators and exchange rates)."""
    # Read CSV with Polars
    df = pl.read_csv(filepath)
    df = df.filter(pl.col('Scenario') == scenario)
    
    # Pivot to wide format for easier access
    pivot = df.pivot(
        values='Value',
        index='Year',
        columns='Variable'
    )
    
    return pivot


def get_deflator(macro_df: pl.DataFrame, from_year: int, to_year: int) -> pl.Series:
    """Calculate deflator to convert prices from one year to another."""
    deflator_col = 'Gross Domestic Product Deflator (2017=100)'
    
    if deflator_col not in macro_df.columns:
        raise ValueError(f"GDP Deflator column not found in macro data")
    
    # Base deflator value for target year
    base_deflator = macro_df.filter(pl.col('Year') == to_year)[deflator_col][0]
    
    # Calculate deflator for each year
    deflator = base_deflator / macro_df[deflator_col]
    
    return deflator


def get_exchange_rate(macro_df: pl.DataFrame, from_currency: str) -> pl.Series:
    """Get exchange rate from foreign currency to CAD."""
    if from_currency == 'CAD':
        # CAD to CAD, no conversion needed
        return pl.Series('exchange_rate', [1.0] * len(macro_df))
    
    # For USD, use the exchange rate column
    if from_currency == 'USD':
        rate_col = 'Canada-US Exchange Rate (C$/US$)'
        if rate_col not in macro_df.columns:
            raise ValueError(f"Exchange rate column for USD not found")
        
        return macro_df[rate_col]
    
    # For other currencies, assume 1.0
    return pl.Series('exchange_rate', [1.0] * len(macro_df))


def convert_currency_polars(
    prices: pl.Series,
    from_year: int,
    to_year: int,
    from_currency: str,
    to_currency: str,
    macro_df: pl.DataFrame,
    constant_dollars: bool = False
) -> pl.Series:
    """
    Convert prices from one currency/year to another (Polars version).
    
    Args:
        prices: Price series with 'Year' column
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Polars DataFrame with macro indicators
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    if constant_dollars:
        # All prices are in constant from_year dollars
        # Apply single conversion factor to all values
        deflator_col = 'Gross Domestic Product Deflator (2017=100)'
        deflator_from = macro_df.filter(pl.col('Year') == from_year)[deflator_col][0]
        deflator_to = macro_df.filter(pl.col('Year') == to_year)[deflator_col][0]
        conversion_factor = deflator_to / deflator_from
        
        # Exchange rate (use from_year rate for constant dollars)
        if from_currency != to_currency:
            exchange_rate = get_exchange_rate(macro_df, from_currency)
            year_mask = macro_df['Year'] == from_year
            if year_mask.sum() > 0:
                er_value = exchange_rate.filter(year_mask)[0]
            else:
                er_value = 1.0
        else:
            er_value = 1.0
        
        converted = prices * conversion_factor * er_value
        
    else:
        # Nominal prices: each year's price is in that year's dollars
        # This needs to work with pandas Series for compatibility
        # Convert to dict, process, convert back
        raise NotImplementedError("Nominal currency conversion not yet implemented for Polars. Use pandas version.")
    
    return converted


def convert_currency(
    prices: pd.Series,
    from_year: int,
    to_year: int,
    from_currency: str,
    to_currency: str,
    macro_df: pl.DataFrame,
    constant_dollars: bool = False
) -> pd.Series:
    """
    Convert prices from one currency/year to another.

    Queries macro_df directly with Polars — no to_pandas() call, so no
    pyarrow dependency regardless of which columns are present.

    Args:
        prices: Price series indexed by year
        from_year: Base year of input prices (only used if constant_dollars=True)
        to_year: Target year for output prices
        from_currency: Source currency code
        to_currency: Target currency code
        macro_df: Polars DataFrame with macro indicators (pivoted, numeric cols)
        constant_dollars: If True, all prices are in constant from_year dollars.
                         If False, each year's price is in that year's dollars (nominal).
    """
    DEFLATOR_COL  = 'Gross Domestic Product Deflator (2017=100)'
    EXCHANGE_COL  = 'Canada-US Exchange Rate (C$/US$)'

    def _scalar(col: str, year: int) -> float:
        """Return a single float from macro_df for (col, year). No pyarrow needed."""
        vals = macro_df.filter(pl.col('Year') == year).get_column(col).to_list()
        if not vals:
            raise KeyError(f"No macro data for year {year}, column '{col}'")
        return float(vals[0])

    if constant_dollars:
        # Single conversion factor applied to all years
        deflator_from = _scalar(DEFLATOR_COL, from_year)
        deflator_to   = _scalar(DEFLATOR_COL, to_year)
        conversion_factor = deflator_to / deflator_from

        er_value = _scalar(EXCHANGE_COL, from_year) if from_currency != to_currency else 1.0

        converted = prices * conversion_factor * er_value

    else:
        # Nominal: convert year-by-year
        deflator_to = _scalar(DEFLATOR_COL, to_year)

        # Pre-build a lookup dict for the years we need — one Polars filter per
        # unique year is much faster than converting the whole DataFrame.
        needed_years = [int(y) for y in prices.index]

        deflator_lookup: dict[int, float] = {}
        exchange_lookup: dict[int, float] = {}
        for year in needed_years:
            row = macro_df.filter(pl.col('Year') == year)
            if len(row) == 0:
                continue
            deflator_lookup[year] = float(row.get_column(DEFLATOR_COL).to_list()[0])
            if from_currency != to_currency:
                exchange_lookup[year] = float(row.get_column(EXCHANGE_COL).to_list()[0])

        converted_dict: dict[int, float] = {}
        for year, price in prices.items():
            year = int(year)
            if year not in deflator_lookup:
                continue
            price_deflated = price * (deflator_to / deflator_lookup[year])
            er = exchange_lookup.get(year, 1.0)
            converted_dict[year] = price_deflated * er

        converted = pd.Series(converted_dict)

    return converted
