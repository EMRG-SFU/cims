"""
Currency deflator and exchange rate table builder.

Converts the raw GDP deflator and exchange rate downloads in
``data/raw_data/deflator_exchange/`` into the two reference tables that back
``CIMS.unit_conversion.CurrencyConverter``:

    data/processed_data/deflator_exchange/currency_deflator.csv   Context = ISO currency code
    data/processed_data/deflator_exchange/currency_exchange.csv   Context = <TARGET>_per_<SOURCE>

Both use the ``Context, Year, Value`` contract that ``unit_conversion._load_table``
expects, plus a ``Source`` column the loader ignores.

Sources
-------
world-bank_2026_gdp-deflator.csv
    World Bank WDI series NY.GDP.DEFL.ZS, "GDP deflator (base year varies by
    country)" -- the GDP deflator at market prices called for by the CIMS
    conversion methodology (NOT CPI). Wide format, one column per year.
ecb_2026_gdp-deflator.csv
    ECB series MNA.A.N.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.IX.D.N, euro area gross
    domestic product at market prices, deflator. Annual, already long format.
bank-of-canada_2026_fx-rates-pre-2017.csv
    Bank of Canada "Legacy Monthly Average Rates" (Valet export), monthly, 1999-01
    to 2017-04. Wide format: a SERIES block mapping series ids to ISO codes, then
    an OBSERVATIONS block with one column per series. Discontinued 2017-04-28.
bank-of-canada_2026_fx-rates-post-2017.csv
    Statistics Canada / Bank of Canada monthly average exchange rates, 2017-01
    onward. Long format, one row per currency-month.

Both exchange files quote Canadian dollars per one unit of the foreign currency,
i.e. CAD_per_<X>. They overlap in early 2017; the current series takes precedence
wherever both carry a month, so the legacy series only fills the years before the
current one begins.

Notes
-----
Deflator base years differ by currency. That is fine and deliberate: the deflator
enters ``conversion_factor`` only as a ratio of two years *within the same
currency*, so each currency may carry its own base. Values are therefore passed
through from source rather than rebased.

Only calendar years with a full twelve monthly observations are averaged, so a
partial year is never emitted as an annual rate. This drops the legacy file's
partial 2017 (Jan-Apr) and the current file's partial final year. The year range
of each output is whatever the raw files support at or after ``START_YEAR`` --
re-export a longer raw series and rerun; no code change is needed.

Exchange rates are published against CAD, so cross rates are derived through CAD:

    A_per_B = CAD_per_B / CAD_per_A

All directed pairs among the currencies that have deflator coverage are written,
because ``_exchange_factor`` looks up the exact ``<TARGET>_per_<SOURCE>`` key and
does not invert or chain rates itself.

Usage
-----
    python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This file lives at src/CIMS/data_processing/source/deflator_exchange/, so
# parents[5] is the repository root.
DATA_ROOT = Path(__file__).resolve().parents[5] / 'data'
RAW_DIR = DATA_ROOT / 'raw_data' / 'deflator_exchange'
OUT_DIR = DATA_ROOT / 'processed_data' / 'deflator_exchange'

WORLD_BANK_FILE = RAW_DIR / 'world-bank_2026_gdp-deflator.csv'
ECB_FILE = RAW_DIR / 'ecb_2026_gdp-deflator.csv'
FX_LEGACY_FILE = RAW_DIR / 'bank-of-canada_2026_fx-rates-pre-2017.csv'
FX_CURRENT_FILE = RAW_DIR / 'bank-of-canada_2026_fx-rates-post-2017.csv'

DEFLATOR_OUT = OUT_DIR / 'currency_deflator.csv'
EXCHANGE_OUT = OUT_DIR / 'currency_exchange.csv'

START_YEAR = 2000

# World Bank "Country Name" -> ISO currency code
WORLD_BANK_CURRENCIES = {
    'Canada': 'CAD',
    'United States': 'USD',
    'United Kingdom': 'GBP',
    'Japan': 'JPY',
    'India': 'INR',
    'China': 'CNY',
}

# The euro area deflator comes from the ECB rather than the World Bank, because
# the World Bank series is per country and the euro has no single country.
ECB_CURRENCY = 'EUR'

# StatCan / Bank of Canada "Type of currency" -> ISO currency code. Only
# currencies that also have deflator coverage are carried through; every other
# series in the raw file (AUD, BRL, MXN, ...) is dropped, since a currency with
# no deflator cannot be converted anyway.
FX_CURRENT_CURRENCIES = {
    'U.S. dollar, monthly average': 'USD',
    'European euro, monthly average': 'EUR',
    'U.K. pound sterling, monthly average': 'GBP',
    'Japanese yen, monthly average': 'JPY',
    'Indian rupee, monthly average': 'INR',
    'Chinese renminbi, monthly average': 'CNY',
}

# Legacy Valet series id -> ISO currency code. Ids rather than the file's own
# `label` column because several series share a label: USD alone has six
# (noon, close, high, low, and two 90-day series). IEXM0101 is the Bank of
# Canada noon rate, the official published rate for that era.
FX_LEGACY_SERIES = {
    'IEXM0101': 'USD',
    'EUROCAM01': 'EUR',
    'IEXM1201': 'GBP',
    'IEXM0701': 'JPY',
    'IEXM3001': 'INR',
    'IEXM2201': 'CNY',
}

BASE_CURRENCY = 'CAD'  # the currency the raw rates are quoted against

# Deflator values sit near 100, so six decimals leaves ample significant digits.
# Exchange rates span four orders of magnitude within one table (CAD_per_JPY is
# ~0.0092 while JPY_per_CAD is ~108), so a fixed number of decimals would strip
# CAD_per_JPY down to three significant figures and break reciprocal consistency.
# Rates are therefore written to a fixed number of *significant* digits instead.
DEFLATOR_DECIMALS = 6
EXCHANGE_SIG_DIGITS = 12

MONTHS_IN_YEAR = 12


# ---------------------------------------------------------------------------
# Deflator
# ---------------------------------------------------------------------------

def load_world_bank_deflator(path: Path = WORLD_BANK_FILE) -> pd.DataFrame:
    """Read the wide World Bank WDI export into long Context/Year/Value rows."""
    df = pd.read_csv(path)

    # The export carries blank spacer rows and a "Data from database: ..." /
    # "Last Updated: ..." footer. Keeping only the rows we recognise drops both.
    df = df[df['Country Name'].isin(WORLD_BANK_CURRENCIES)].copy()

    missing = set(WORLD_BANK_CURRENCIES) - set(df['Country Name'])
    if missing:
        raise ValueError(
            f"{path.name} is missing expected countries: {', '.join(sorted(missing))}"
        )

    # Year columns are labelled "2020 [YR2020]".
    year_cols = {c: int(c.split(' ')[0]) for c in df.columns if c.endswith(']')}

    long = df.melt(
        id_vars=['Country Name'],
        value_vars=list(year_cols),
        var_name='year_col',
        value_name='Value',
    )
    long['Year'] = long['year_col'].map(year_cols)
    long['Context'] = long['Country Name'].map(WORLD_BANK_CURRENCIES)

    # World Bank writes ".." for unavailable observations.
    long['Value'] = pd.to_numeric(long['Value'], errors='coerce')

    long['Source'] = (
        'World Bank WDI, GDP deflator at market prices (NY.GDP.DEFL.ZS); '
        f'base year varies by country; from {path.name}'
    )

    return long[['Context', 'Year', 'Value', 'Source']].dropna(subset=['Value'])


def load_ecb_deflator(path: Path = ECB_FILE) -> pd.DataFrame:
    """Read the ECB euro area GDP deflator into long Context/Year/Value rows."""
    df = pd.read_csv(path)

    value_cols = [c for c in df.columns if c not in ('DATE', 'TIME PERIOD')]
    if len(value_cols) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one value column, found {value_cols}"
        )

    out = pd.DataFrame({
        'Context': ECB_CURRENCY,
        'Year': df['TIME PERIOD'].astype(int),
        'Value': pd.to_numeric(df[value_cols[0]], errors='coerce'),
        'Source': (
            'ECB, euro area gross domestic product at market prices, deflator '
            '(MNA.A.N.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.IX.D.N); '
            f'from {path.name}'
        ),
    })

    return out.dropna(subset=['Value'])


def build_deflator_table(start_year: int = START_YEAR) -> pd.DataFrame:
    """Assemble the full deflator table, one row per (currency, year)."""
    table = pd.concat(
        [load_world_bank_deflator(), load_ecb_deflator()],
        ignore_index=True,
    )
    table = table[table['Year'] >= start_year]
    return table.sort_values(['Context', 'Year']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exchange -- monthly rates
# ---------------------------------------------------------------------------

def load_current_monthly(path: Path = FX_CURRENT_FILE) -> pd.DataFrame:
    """
    Read the StatCan long-format monthly rates.

    Returns Currency/Month/Value rows, where Value is CAD per one foreign unit.
    """
    # The StatCan export is UTF-8 with a BOM.
    df = pd.read_csv(path, encoding='utf-8-sig')

    df = df[df['Type of currency'].isin(FX_CURRENT_CURRENCIES)].copy()

    # Defensive: the CERI index rows are excluded by the currency filter above,
    # but the unit check makes the "Canadian dollars per one foreign unit"
    # assumption explicit and would catch an export that changes units.
    unexpected_uom = set(df['UOM']) - {'Dollars'}
    if unexpected_uom:
        raise ValueError(
            f"{path.name}: unexpected unit(s) on bilateral rate rows: "
            f"{', '.join(sorted(unexpected_uom))}. Rates must be quoted in "
            'Canadian dollars per one unit of the foreign currency.'
        )

    out = pd.DataFrame({
        'Currency': df['Type of currency'].map(FX_CURRENT_CURRENCIES),
        'Month': df['REF_DATE'].str.slice(0, 7),
        'Value': pd.to_numeric(df['VALUE'], errors='coerce'),
    })
    out['File'] = path.name
    return out.dropna(subset=['Value'])


def load_legacy_monthly(path: Path = FX_LEGACY_FILE) -> pd.DataFrame:
    """
    Read the Bank of Canada legacy Valet export into Currency/Month/Value rows.

    The file is not a plain CSV: metadata blocks (TERMS AND CONDITIONS, NAME,
    DESCRIPTION, LINK, SERIES) precede an OBSERVATIONS block that holds the
    actual wide table, so the observation header has to be located by name.
    """
    lines = path.read_text(encoding='utf-8-sig').splitlines()

    try:
        start = lines.index('"OBSERVATIONS"')
    except ValueError as exc:
        raise ValueError(
            f'{path.name}: no OBSERVATIONS block found. Expected a Bank of Canada '
            'Valet export.'
        ) from exc

    rows = list(csv.reader(lines[start + 1:]))
    header = rows[0]

    missing = set(FX_LEGACY_SERIES) - set(header)
    if missing:
        raise ValueError(
            f"{path.name} is missing expected series: {', '.join(sorted(missing))}"
        )

    columns = {series: header.index(series) for series in FX_LEGACY_SERIES}

    records = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        month = row[0][:7]
        for series, currency in FX_LEGACY_SERIES.items():
            raw = row[columns[series]].strip()
            if raw:
                records.append((currency, month, float(raw), path.name))

    return pd.DataFrame(records, columns=['Currency', 'Month', 'Value', 'File'])


def load_annual_cad_rates(start_year: int = START_YEAR) -> pd.DataFrame:
    """
    Combine both exchange rate files and average them into annual rates.

    The current series takes precedence over the legacy series wherever both
    carry the same currency-month, so the legacy noon rates only fill the period
    before the current series begins. Only complete calendar years (twelve
    monthly observations) are kept, so a partial year never leaves this function.
    """
    monthly = pd.concat(
        [load_current_monthly(), load_legacy_monthly()],
        ignore_index=True,
    )
    # concat order puts the current file first, so keep='first' prefers it.
    monthly = monthly.drop_duplicates(subset=['Currency', 'Month'], keep='first')

    monthly['Year'] = monthly['Month'].str.slice(0, 4).astype(int)

    grouped = monthly.groupby(['Currency', 'Year']).agg(
        cad_per_unit=('Value', 'mean'),
        months=('Value', 'size'),
        files=('File', lambda s: tuple(sorted(set(s)))),
    ).reset_index()

    complete = grouped[grouped['months'] == MONTHS_IN_YEAR]
    annual = complete[complete['Year'] >= start_year]
    return annual[['Currency', 'Year', 'cad_per_unit', 'files']].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exchange -- pair table
# ---------------------------------------------------------------------------

_FILE_LABELS = {
    FX_LEGACY_FILE.name: f'Bank of Canada legacy monthly average noon rates ({FX_LEGACY_FILE.name})',
    FX_CURRENT_FILE.name: (
        'Statistics Canada / Bank of Canada monthly average rates '
        f'({FX_CURRENT_FILE.name})'
    ),
}


def _source_note(files: tuple[str, ...], is_cross: bool) -> str:
    labels = ' + '.join(_FILE_LABELS.get(f, f) for f in sorted(files))
    note = f'Annual average of monthly rates; from {labels}'
    if is_cross:
        note += f'; cross rate derived via {BASE_CURRENCY}'
    return note


def build_exchange_table(start_year: int = START_YEAR) -> pd.DataFrame:
    """
    Build every directed <TARGET>_per_<SOURCE> pair among the covered currencies.

    Rates are published against CAD, so cross rates go through CAD:
    A_per_B = CAD_per_B / CAD_per_A. CAD itself is added as a rate of 1.0 so the
    base currency falls out of the same arithmetic as every other pair.
    """
    annual = load_annual_cad_rates(start_year=start_year)

    # CAD per one CAD is 1 by definition; including it lets the cross-rate loop
    # below produce the CAD_per_X and X_per_CAD pairs without a special case.
    # It contributes no source file of its own.
    years = sorted(annual['Year'].unique())
    base = pd.DataFrame({
        'Currency': BASE_CURRENCY,
        'Year': years,
        'cad_per_unit': 1.0,
        'files': [()] * len(years),
    })
    annual = pd.concat([annual, base], ignore_index=True)

    rates = annual.pivot(index='Year', columns='Currency', values='cad_per_unit')
    files = annual.pivot(index='Year', columns='Currency', values='files')

    rows = []
    currencies = sorted(rates.columns)
    for target in currencies:
        for source in currencies:
            if target == source:
                continue  # conversion_factor short-circuits same-currency pairs
            is_cross = BASE_CURRENCY not in (target, source)
            pair_files = [
                tuple(sorted(set(t) | set(s)))
                for t, s in zip(files[target], files[source])
            ]
            pair = pd.DataFrame({
                'Context': f'{target}_per_{source}',
                'Year': rates.index,
                # CAD per source unit / CAD per target unit = target per source
                'Value': (rates[source] / rates[target]).values,
                'Source': [_source_note(f, is_cross) for f in pair_files],
            })
            rows.append(pair.dropna(subset=['Value']))

    table = pd.concat(rows, ignore_index=True)
    return table.sort_values(['Context', 'Year']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write(table: pd.DataFrame, path: Path, fmt: str) -> None:
    out = table.copy()
    out['Value'] = out['Value'].map(lambda v: format(v, fmt))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator='\n')


def _summarise(name: str, table: pd.DataFrame) -> None:
    contexts = table['Context'].nunique()
    print(f'{name}: {len(table)} rows, {contexts} contexts, '
          f'years {table["Year"].min()}-{table["Year"].max()}')
    coverage = table.groupby('Context')['Year'].agg(['min', 'max', 'size'])
    gaps = coverage[coverage['size'] != coverage['max'] - coverage['min'] + 1]
    if len(gaps):
        print(f'  WARNING: {len(gaps)} context(s) have gaps in their year range')
    spans = coverage.groupby(['min', 'max', 'size']).size()
    for (lo, hi, n), count in spans.items():
        print(f'  {count} context(s): {lo}-{hi} ({n} years)')


def main() -> None:
    deflator = build_deflator_table()
    exchange = build_exchange_table()

    _write(deflator, DEFLATOR_OUT, f'.{DEFLATOR_DECIMALS}f')
    _write(exchange, EXCHANGE_OUT, f'.{EXCHANGE_SIG_DIGITS}g')

    _summarise(DEFLATOR_OUT.name, deflator)
    print()
    _summarise(EXCHANGE_OUT.name, exchange)

    # A currency whose exchange coverage is shorter than its deflator coverage
    # cannot be converted for the missing years; currency_table_coverage will
    # report it against real model data, but flag it here too.
    deflator_start = deflator['Year'].min()
    exchange_start = exchange['Year'].min()
    if exchange_start > deflator_start:
        print(
            f'\nNote: exchange rates start in {exchange_start}, but deflators go back '
            f'to {deflator_start}. Cross-currency conversion for dollar-years before '
            f'{exchange_start} will fail; same-currency rebasing is unaffected.'
        )


if __name__ == '__main__':
    main()
