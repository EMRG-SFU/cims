"""
Unit conversion utilities for normalizing parameter values to a common currency and dollar-year
at model read-time. Designed for future extension to physical unit conversion (e.g. energy units).

Conversion order
----------------
A value is rebased within its own (source) currency first, and only then exchanged
into the target currency. This ordering is deliberate — see
``CurrencyConverter.conversion_factor`` for the full rationale and its consequences.
"""
import re
import pandas as pd
from pathlib import Path

# Matches: 4-digit year, optional separator (_, space, or none), 3-letter currency code,
# optional physical unit after / or _ (e.g. '2010_CAD', '2010_USD/GJ', '2010 CAD')
MONETARY_PREFIX_RE = re.compile(r'^(\d{4})\s*_?\s*([A-Za-z]{3})(?:[/_](.+))?$')


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

    match = MONETARY_PREFIX_RE.match(unit.strip())
    if match:
        return int(match.group(1)), match.group(2).upper(), match.group(3)

    return None, None, unit


class CurrencyConverter:
    """
    Converts values between currencies and dollar-years using price deflator and
    exchange rate tables.

    The deflator table is expected to hold a GDP deflator at market prices, indexed
    per currency; see ``conversion_factor`` for why that series, and why deflation
    is applied before the exchange step.
    """

    def __init__(self, deflator_path: str | Path, exchange_path: str | Path):
        self._deflator = _load_table(deflator_path, key_col='Context')
        self._exchange = _load_table(exchange_path, key_col='Context')

    def conversion_factor(self, source_currency: str, source_year: int,
                          target_currency: str, target_year: int) -> float:
        """
        Compute the factor to convert a value from source_currency/source_year
        to target_currency/target_year.

        Methodology
        -----------
        Deflate first (within the source currency), then exchange:

            factor = deflator[source_currency][target_year]
                     / deflator[source_currency][source_year]
                     * exchange[target_currency per source_currency][target_year]

        Rebasing within the source (foreign) currency using the GDP deflator at
        market prices ensures that the value is adjusted consistently for changes in
        the domestic price level of output, preserving the real economic meaning of
        the original estimate. Performing this step first avoids mixing inflation
        dynamics across countries. The subsequent conversion to the target currency
        using the exchange rate then translates this real value into the target
        currency, separating time (inflation) effects from space (currency) effects
        and maintaining consistency with national accounts principles.

        Consequence: the factor is *not* symmetric, because the two directions
        deflate in different currencies and exchange in different years. Converting
        USD 2010 -> CAD 2020 and back to USD 2010 does not return the original value.
        Convert once, from the source the estimate was published in, rather than
        chaining or reversing conversions.

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

    extracted = df['Unit'].fillna('').astype(str).str.extract(MONETARY_PREFIX_RE)
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
    missing = [c for c in (key_col, 'Year', 'Value') if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {key_col}, Year, Value."
        )
    return dict(zip(
        zip(df[key_col].astype(str), df['Year'].astype(int)),
        df['Value'].astype(float),
    ))


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
    pair = f"{target_currency}_per_{source_currency}"
    if (pair, year) in exchange:
        return exchange[(pair, year)]

    # Distinguish "this currency pair is absent entirely" from "the pair exists
    # but not for this year", so the message points at the right fix.
    years_for_pair = {k[1] for k in exchange if k[0] == pair}
    if not years_for_pair:
        raise ValueError(
            f"{pair} not found in exchange rate table. "
            "Please review the exchange rate table for missing currencies."
        )
    raise ValueError(
        f"Year {year} not found in exchange rate table for {pair} "
        f"(available: {min(years_for_pair)}-{max(years_for_pair)}). "
        "Please review the exchange rate table for missing or out-of-range years."
    )


def _check_deflator_key(table: dict, key: tuple, currency: str, year: int) -> None:
    if key in table:
        return

    # As above: a missing currency and a missing year need different messages.
    years_for_currency = {k[1] for k in table if k[0] == currency}
    if not years_for_currency:
        raise ValueError(
            f"Currency {currency} not found in deflator table. "
            "Please review the deflator table for missing currencies."
        )
    raise ValueError(
        f"Year {year} not found in deflator table for {currency} "
        f"(available: {min(years_for_currency)}-{max(years_for_currency)}). "
        "Please review the deflator table for missing or out-of-range years."
    )
