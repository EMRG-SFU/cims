"""
Currency handling shared by the data pipeline and the CIMS model.

The pipeline and ``CIMS.Model`` convert money with the same
``CIMS.unit_conversion.CurrencyConverter`` and the same two tables, built by
``source/deflator_exchange/deflator_exchange.py``:

    data/processed/currency/currency-deflator.csv
    data/processed/currency/currency-exchange.csv

Unit labels
-----------
CIMS converts a row only when its ``Unit`` starts with a ``YYYY_CCC`` prefix
(``2020_CAD``, ``2025_CAD/GJ``, ``2017_CAD_millions``); anything else passes
through unconverted. ``canonical_unit`` rewrites the pipeline's legacy labels
into that form:

    '$'             -> '2020_CAD'           (untagged: CONTROLS currency_*_untagged)
    '$/tCO2e'       -> '2020_CAD/tCO2e'
    '2025 C$/GJ'    -> '2025_CAD/GJ'
    '2025$ millions'-> '2025_CAD_millions'
    'US$/GJ'        -> '2020_USD/GJ'
    blank, on a fic or dic_min row -> '2020_CAD'   (BLANK_UNIT_MONETARY; untagged)

The fixed data and older calibration outputs leave fic and dic_min without a
unit, although both are money. ``finalise_currency`` gives those rows the
untagged dollar-year and currency, so CIMS converts them like any other cost.
New FIC fits carry the run's own label (calibration/.../write_fics.py).

Units that name GDP as an activity are quantities of output (real GDP in
chained 2017 dollars), not prices, so they must not convert. They are written as
GDP_ACTIVITY_UNIT, 'million 2017 CAD GDP': no '$' and no leading YYYY_CCC, so
CIMS neither converts them nor flags them as missing a currency prefix. The
legacy labels 'M$ GDP', '$M GDP' and '$M 2017 GDP' are renamed to it.

Output basis
------------
By default every monetary row keeps the dollar-year and currency it was tagged
with, and CIMS converts at read time to the scenario's ``target_units``. Set
``currency_output_mode`` to ``"convert"`` in ``control.py`` to have the pipeline
convert every tagged row to ``currency_output_year`` / ``currency_output_country``
before writing instead. Either way the conversion uses the tables above.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

from CIMS.unit_conversion import MONETARY_PREFIX_RE, CurrencyConverter
from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.controls_conversions import load_control_config

DEFLATOR_TABLE = paths.PROCESSED / 'currency/currency-deflator.csv'
EXCHANGE_TABLE = paths.PROCESSED / 'currency/currency-exchange.csv'

_CONTROLS = load_control_config()
UNTAGGED_YEAR = int(_CONTROLS.get('currency_year_untagged', 2020))
UNTAGGED_CURRENCY = str(_CONTROLS.get('currency_country_untagged', 'CAD')).upper()
OUTPUT_MODE = str(_CONTROLS.get('currency_output_mode', 'tagged')).lower()

# Monetary parameters that the fixed data and older calibration outputs leave
# without a unit; finalise_currency labels them with the untagged currency.
BLANK_UNIT_MONETARY = frozenset({'fic', 'dic_min'})
if OUTPUT_MODE not in ('tagged', 'convert'):
    raise ValueError(f"control.py currency_output_mode must be 'tagged' or 'convert', not {OUTPUT_MODE!r}")
OUTPUT_CONVERT = OUTPUT_MODE == 'convert'
OUTPUT_YEAR = int(_CONTROLS.get('currency_output_year', 2025))
OUTPUT_CURRENCY = str(_CONTROLS.get('currency_output_country', 'CAD')).upper()

# '$' symbols with a currency attached. A bare '$' means the untagged currency.
_SYMBOLS = {'': None, 'C': 'CAD', 'CA': 'CAD', 'CAD': 'CAD', 'US': 'USD', 'USD': 'USD'}
_DATED_RE = re.compile(r'^(\d{4})\s*(C|CA|CAD|US|USD)?\$\s*(.*)$')   # '2025 C$/GJ', '2025$ millions'
_UNDATED_RE = re.compile(r'^(C|CA|CAD|US|USD)?\$\s*(.*)$')          # '$', '$/GJ', 'US$/GJ'
_PREFIX_RE = re.compile(r'^\d{4}\s*_?\s*[A-Za-z]{3}')

GDP_ACTIVITY_UNIT = 'million 2017 CAD GDP'
UNIT_RENAMES = {'M$ GDP': GDP_ACTIVITY_UNIT, '$M GDP': GDP_ACTIVITY_UNIT,
                '$M 2017 GDP': GDP_ACTIVITY_UNIT}


@lru_cache(maxsize=1)
def get_converter() -> CurrencyConverter:
    """The converter CIMS uses, on the same tables. Built once per process."""
    missing = [p for p in (DEFLATOR_TABLE, EXCHANGE_TABLE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Currency tables not found: {', '.join(p.name for p in missing)}. Run "
            'python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange first.'
        )
    return CurrencyConverter(DEFLATOR_TABLE, EXCHANGE_TABLE)


def conversion_factor(from_currency: str, from_year: int,
                      to_currency: str, to_year: int) -> float:
    """Factor from from_year from_currency to to_year to_currency (CIMS method)."""
    return get_converter().conversion_factor(
        from_currency.upper(), int(from_year), to_currency.upper(), int(to_year))


def convert_series(prices: pd.Series, from_currency: str, to_year: int,
                   to_currency: str, from_year: int | None = None) -> pd.Series:
    """
    Convert a year-indexed price series.

    from_year given: constant from_year dollars, one factor for every year.
    from_year None: nominal, each value is in its own index year's dollars.
    """
    if from_year is not None:
        return prices * conversion_factor(from_currency, from_year, to_currency, to_year)
    factors = {y: conversion_factor(from_currency, int(y), to_currency, to_year)
               for y in prices.index}
    return prices * pd.Series(factors).reindex(prices.index)


def _rest(rest: str) -> str:
    rest = rest.strip()
    if not rest:
        return ''
    return rest if rest[0] in '/_' else '_' + rest.replace(' ', '_')


def canonical_unit(unit):
    """Rewrite a legacy monetary unit label to CIMS's YYYY_CCC form; others unchanged."""
    if unit is None or not isinstance(unit, str):
        return unit
    u = unit.strip()
    if u in UNIT_RENAMES:
        return UNIT_RENAMES[u]
    if not u or MONETARY_PREFIX_RE.match(u) or '$' not in u or 'GDP' in u:
        return unit
    m = _DATED_RE.match(u)
    if m:
        year, sym, rest = m.groups()
        cur = _SYMBOLS[(sym or '').upper()] or UNTAGGED_CURRENCY
        return f'{year}_{cur}{_rest(rest)}'
    m = _UNDATED_RE.match(u)
    if m:
        sym, rest = m.groups()
        cur = _SYMBOLS[(sym or '').upper()] or UNTAGGED_CURRENCY
        return f'{UNTAGGED_YEAR}_{cur}{_rest(rest)}'
    return unit


def _retarget(unit: str, year: int, currency: str) -> str:
    return _PREFIX_RE.sub(f'{year}_{currency}', unit.strip(), count=1)


def _target_factor(unit: str, year: int, currency: str):
    m = MONETARY_PREFIX_RE.match(unit.strip()) if isinstance(unit, str) else None
    if not m:
        return None
    return conversion_factor(m.group(2), int(m.group(1)), currency, year)


def finalise_currency(df: pl.DataFrame, convert: bool | None = None,
                      year: int | None = None, currency: str | None = None) -> pl.DataFrame:
    """
    Canonicalise the Unit column and, if configured, convert monetary rows.

    convert/year/currency default to the control.py currency_output_* settings
    (convert is True when currency_output_mode is 'convert').
    """
    if 'Unit' not in df.columns or df.is_empty():
        return df
    convert = OUTPUT_CONVERT if convert is None else convert
    year = OUTPUT_YEAR if year is None else int(year)
    currency = (OUTPUT_CURRENCY if currency is None else currency).upper()

    if 'Parameter' in df.columns:
        blank = pl.col('Unit').cast(pl.Utf8).fill_null('').str.strip_chars() == ''
        money = pl.col('Parameter').cast(pl.Utf8).is_in(list(BLANK_UNIT_MONETARY))
        df = df.with_columns(
            pl.when(blank & money).then(pl.lit(f'{UNTAGGED_YEAR}_{UNTAGGED_CURRENCY}'))
              .otherwise(pl.col('Unit').cast(pl.Utf8)).alias('Unit'))

    units = df.get_column('Unit').cast(pl.Utf8)
    mapping = {u: canonical_unit(u) for u in units.unique().to_list() if u is not None}
    df = df.with_columns(pl.col('Unit').cast(pl.Utf8).replace(mapping).alias('Unit'))

    if not convert or 'Value' not in df.columns:
        return df

    factors = {u: _target_factor(u, year, currency) for u in set(mapping.values())}
    factors = {u: f for u, f in factors.items() if f is not None}
    if not factors:
        return df
    retarget = {u: _retarget(u, year, currency) for u in factors}

    value_dtype = df.schema['Value']
    num = pl.col('Value').cast(pl.Float64, strict=False)
    factor = pl.col('Unit').replace_strict(factors, default=None, return_dtype=pl.Float64)
    convertible = factor.is_not_null() & num.is_not_null()
    new_value = num * factor
    if value_dtype == pl.Utf8 or value_dtype == pl.String:
        new_value = new_value.map_elements(repr, return_dtype=pl.Utf8)
    return df.with_columns(
        pl.when(convertible).then(new_value).otherwise(pl.col('Value')).alias('Value'),
        pl.when(convertible).then(pl.col('Unit').replace(retarget))
          .otherwise(pl.col('Unit')).alias('Unit'),
    )


def write_model_csv(df: pl.DataFrame, path) -> pl.DataFrame:
    """Write a model-input CSV with currency units finalised. Returns what was written."""
    out = finalise_currency(df)
    out.write_csv(str(path))
    return out


def finalise_currency_csv(path) -> None:
    """Finalise currency units in an existing CSV in place (for writers outside polars)."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows or 'Unit' not in rows[0]:
        return
    header = [h.strip() for h in rows[0]]
    df = pl.DataFrame({h: [r[i] if i < len(r) else '' for r in rows[1:]]
                       for i, h in enumerate(header)}, schema={h: pl.Utf8 for h in header})
    out = finalise_currency(df)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(out.rows())
