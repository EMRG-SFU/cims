"""
Unit conversion utilities for normalizing parameter values to a common currency and dollar-year
at model read-time. Designed for future extension to physical unit conversion (e.g. energy units).
"""
import re
import pandas as pd
from pathlib import Path

# Matches: 4-digit year, optional separator (_, space, or none), 3-letter currency code,
# optional physical unit after / or _ (e.g. '2010_CAD', '2010_USD/GJ', '2010 CAD')
_MONETARY_PREFIX_RE = re.compile(r'^(\d{4})\s*_?\s*([A-Za-z]{3})(?:[/_](.+))?$')


def parse_unit_string(unit: str | None) -> tuple[int | None, str | None, str | None]:
    """
    Parse a unit string into its monetary prefix and physical unit components.

    Accepts the canonical format (e.g. '2010_CAD', '2010_USD/GJ') and common
    variations (e.g. '2010 CAD', '2010CAD').

    Parameters
    ----------
    unit : str | None
        Unit string from the model description.

    Returns
    -------
    tuple[int | None, str | None, str | None]
        (dollar_year, currency, physical_unit). dollar_year and currency are None
        if no monetary prefix is present.
    """
    if not unit:
        return None, None, unit

    match = _MONETARY_PREFIX_RE.match(unit.strip())
    if match:
        return int(match.group(1)), match.group(2).upper(), match.group(3)

    return None, None, unit


class CurrencyConverter:
    """Converts values between currencies and dollar-years using CPI and exchange rate tables."""

    def __init__(self, deflator_path: str | Path, exchange_path: str | Path):
        self._deflator = _load_table(deflator_path, key_col='Context')
        self._exchange = _load_table(exchange_path, key_col='Context')

    def conversion_factor(self, source_currency: str, source_year: int,
                          target_currency: str, target_year: int) -> float:
        """
        Compute the factor to convert a value from source_currency/source_year
        to target_currency/target_year.

        Applies inflation in the source currency first, then exchanges to the target currency.

        Parameters
        ----------
        source_currency : str
            ISO currency code of the input value (e.g. 'USD').
        source_year : int
            Dollar-year of the input value.
        target_currency : str
            ISO currency code to convert to.
        target_year : int
            Dollar-year to convert to.

        Returns
        -------
        float
            Multiplicative conversion factor.
        """
        if source_currency == target_currency and source_year == target_year:
            return 1.0
        return (_inflate_factor(self._deflator, source_currency, source_year, target_year) *
                _exchange_factor(self._exchange, source_currency, target_currency, target_year))

    def convert(self, value: float, source_currency: str | None, source_year: int | None,
                target_currency: str, target_year: int) -> float:
        """Apply currency conversion to a value. Returns value unchanged if source_currency is None."""
        if source_currency is None:
            return value
        return value * self.conversion_factor(source_currency, source_year, target_currency, target_year)


def normalize_currency_in_df(
    df: pd.DataFrame,
    converter: CurrencyConverter,
    target_currency: str,
    target_dollar_year: int,
) -> pd.DataFrame:
    """
    Apply currency/dollar-year conversion to monetary values in a DataFrame.

    Rows whose Unit column has no monetary prefix are passed through unchanged.
    Rows with a monetary prefix have their Value converted and their Unit updated
    to reflect the target currency and dollar-year (e.g. '2010_USD/GJ' → '2020_CAD/GJ').

    Parameters
    ----------
    df : pd.DataFrame
        Model description DataFrame with 'Unit' and 'Value' columns.
    converter : CurrencyConverter
    target_currency : str
    target_dollar_year : int

    Returns
    -------
    pd.DataFrame
        A copy of df with converted values and updated unit strings.
    """
    df = df.copy()

    if df.empty:
        return df

    extracted = df['Unit'].fillna('').astype(str).str.extract(_MONETARY_PREFIX_RE)
    extracted.columns = ['dollar_year', 'currency', 'physical_unit']
    monetary_mask = extracted['dollar_year'].notna()

    if not monetary_mask.any():
        return df

    numeric_values = pd.to_numeric(df['Value'], errors='coerce')
    # Only rows that are both monetary-prefixed AND have a convertible Value are
    # actually converted. A row whose Value can't be converted must not have its
    # Unit rewritten either — otherwise the label would claim a conversion that
    # never happened.
    convertible_mask = monetary_mask & numeric_values.notna()

    extracted['currency'] = extracted['currency'].str.upper()
    extracted['dollar_year'] = extracted['dollar_year'].astype('Int64')

    for (currency, year), group in extracted[convertible_mask].groupby(['currency', 'dollar_year']):
        factor = converter.conversion_factor(str(currency), int(year), target_currency, target_dollar_year)
        df.loc[group.index, 'Value'] = numeric_values.loc[group.index] * factor

    new_prefix = f"{target_dollar_year}_{target_currency}"
    new_units = extracted.loc[convertible_mask, 'physical_unit'].apply(
        lambda pu: f"{new_prefix}/{pu}" if pd.notna(pu) and pu else new_prefix
    )
    df.loc[convertible_mask, 'Unit'] = new_units

    return df


def apply_currency_conversion(
    node_dfs: dict,
    tech_dfs: dict,
    converter: CurrencyConverter,
    target_currency: str,
    target_dollar_year: int,
) -> tuple[dict, dict]:
    """
    Apply currency/dollar-year conversion to all node and technology DataFrames.

    Parameters
    ----------
    node_dfs : dict[str, pd.DataFrame]
    tech_dfs : dict[str, dict[str, pd.DataFrame]]
    converter : CurrencyConverter
    target_currency : str
    target_dollar_year : int

    Returns
    -------
    tuple[dict, dict]
        Converted (node_dfs, tech_dfs).
    """
    converted_node_dfs = {
        node: normalize_currency_in_df(df, converter, target_currency, target_dollar_year)
        for node, df in node_dfs.items()
    }
    converted_tech_dfs = {
        node: {
            tech: normalize_currency_in_df(df, converter, target_currency, target_dollar_year)
            for tech, df in tech_dict.items()
        }
        for node, tech_dict in tech_dfs.items()
    }
    return converted_node_dfs, converted_tech_dfs


def _load_table(path: str | Path, key_col: str) -> dict[tuple[str, int], float]:
    df = pd.read_csv(path)
    return {(row[key_col], int(row['Year'])): float(row['Value']) for _, row in df.iterrows()}


def _inflate_factor(deflator: dict, currency: str, source_year: int, target_year: int) -> float:
    if source_year == target_year:
        return 1.0
    source_key = (currency, source_year)
    target_key = (currency, target_year)
    _check_deflator_key(deflator, source_key, currency, source_year)
    _check_deflator_key(deflator, target_key, currency, target_year)
    return deflator[target_key] / deflator[source_key]


def _exchange_factor(exchange: dict, source_currency: str, target_currency: str, year: int) -> float:
    if source_currency == target_currency:
        return 1.0
    key = (f"{target_currency}_per_{source_currency}", year)
    if key not in exchange:
        if year not in {k[1] for k in exchange}:
            raise ValueError(f"Year {year} not found in exchange rate table.")
        raise ValueError(f"{target_currency}_per_{source_currency} not found in exchange rate table.")
    return exchange[key]


def _check_deflator_key(table: dict, key: tuple, currency: str, year: int) -> None:
    if key not in table:
        if year not in {k[1] for k in table}:
            raise ValueError(f"Year {year} not found in deflator table.")
        raise ValueError(f"Currency {currency} not found in deflator table.")
