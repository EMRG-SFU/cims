"""
Currency deflator and exchange rate table builder.

Converts the raw GDP deflator and exchange rate downloads in
``data/extracted/currency/`` into the two reference tables that back
``CIMS.unit_conversion.CurrencyConverter``:

    data/processed/currency/currency-deflator.csv   Context = ISO currency code
    data/processed/currency/currency-exchange.csv   Context = <TARGET>_per_<SOURCE>

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
can-statcan_2026_fx-rates-post-2017.csv
    Statistics Canada / Bank of Canada monthly average exchange rates, 2017-01
    onward. Long format, one row per currency-month.
ecb_2026_euro-reference-rates.csv
    ECB euro foreign exchange reference rates, daily, 1999-01-04 onward. Wide
    format: one row per business day, one column per currency, quoted as units
    of that currency per one euro; "N/A" where a currency is not quoted. Used
    only for currencies the Bank of Canada does not publish against CAD
    (ECB_VIA_EUR_CURRENCIES, currently DKK).
oecd_2026_gdp-deflator-forecast.csv
    OECD Economic Outlook 117 long-term scenarios (dataflow
    OECD.ECO.MAD:DSD_EO_LTB@DF_EO_LTB), annual to 2100, filtered at download to
    the PGDP measure (GDP at market prices, deflator) and the BAU1 scenario.
    It extends the deflators past the last historical year; see "Forecast
    years" below. The OECD's PPP measure was left out of the download: a
    purchasing power parity is not a market exchange rate.

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
partial 2017 (Jan-Apr) and the current file's partial final year. Every year the
raw files support is written (``START_YEAR = None``): the World Bank deflators
from 1976, the ECB euro-area deflator from 1995, and exchange rates from 1999,
the first complete year of the Bank of Canada legacy series. Re-export a longer
raw series and rerun; no code change is needed. Set ``START_YEAR`` to a year to
cut the tables off again.

A deflator year with no exchange rate is still useful: the CIMS method deflates
inside the source currency first and exchanges only at the target year, so a
1985_USD value needs the 1985 USD deflator and the target-year USD rate, not a
1985 rate.

Currencies with no Bank of Canada series are brought onto the CAD basis through
the euro: the ECB daily rates are averaged to an annual X_per_EUR (complete years
only: observations in all twelve months), and

    CAD_per_X = CAD_per_EUR (Bank of Canada annual average) / X_per_EUR (ECB annual average)

so every currency rests on the same Bank of Canada / StatCan CAD rates.

Exchange rates are published against CAD, so cross rates are derived through CAD:

    A_per_B = CAD_per_B / CAD_per_A

All directed pairs among the currencies that have deflator coverage are written,
because ``_exchange_factor`` looks up the exact ``<TARGET>_per_<SOURCE>`` key and
does not invert or chain rates itself.

Forecast years
--------------
Values quoted in future dollar-years (e.g. 2030_USD) need a deflator for that
future year, and nothing else: conversion_factor exchanges only at the target
year, which is historical. Each currency's deflator is therefore extended past
its last historical year H with the OECD PGDP growth path, spliced at H:

    deflator[X][y] = deflator[X][H] * oecd[X][y] / oecd[X][H]    for y > H

Historical values are never replaced, so no existing conversion changes; the
OECD series contributes only its year-on-year growth, not its level or base.
NLG and DEM follow the Dutch and German national deflators, as their history
does. The exchange table is not extended, so a TARGET year past the last
complete year of exchange rates still fails, by design.

Usage
-----
    python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from CIMS.data_processing.common import paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = paths.EXTRACTED / 'currency'
OUT_DIR = paths.PROCESSED / 'currency'

WORLD_BANK_FILE = RAW_DIR / 'world-bank_2026_gdp-deflator.csv'
ECB_FILE = RAW_DIR / 'ecb_2026_gdp-deflator.csv'
ECB_FX_FILE = RAW_DIR / 'ecb_2026_euro-reference-rates.csv'
FX_LEGACY_FILE = RAW_DIR / 'bank-of-canada_2026_fx-rates-pre-2017.csv'
FX_CURRENT_FILE = RAW_DIR / 'can-statcan_2026_fx-rates-post-2017.csv'

OECD_FORECAST_FILE = RAW_DIR / 'oecd_2026_gdp-deflator-forecast.csv'

DEFLATOR_OUT = OUT_DIR / 'currency-deflator.csv'
EXCHANGE_OUT = OUT_DIR / 'currency-exchange.csv'

START_YEAR = None   # None: every year the raw files carry (2026-10-02; was 2000)

# World Bank "Country Name" -> ISO currency code
WORLD_BANK_CURRENCIES = {
    'Canada': 'CAD',
    'United States': 'USD',
    'United Kingdom': 'GBP',
    'Japan': 'JPY',
    'India': 'INR',
    'China': 'CNY',
    'Denmark': 'DKK',
    'Australia': 'AUD',
    'Sweden': 'SEK',
    'Switzerland': 'CHF',
    # pre-euro legacy currencies: national deflator; CAD rate through the fixed euro rate
    'Netherlands': 'NLG',
    'Germany': 'DEM',
}

# OECD EO117 long-term scenario used for forecast deflators: BAU1 =
# business-as-usual energy transition, median climate damage curve, no carbon
# mitigation costs. The raw file holds only this scenario and measure; the loader
# still filters on both, so a re-download with more series cannot slip through.
# The six EO117 scenarios agree to within about 0.5% on deflators through 2050.
OECD_SCENARIO = 'BAU1'
OECD_DEFLATOR_MEASURE = 'PGDP'

# ISO currency code -> OECD REF_AREA whose deflator forecast extends it.
OECD_AREAS = {
    'CAD': 'CAN',
    'USD': 'USA',
    'EUR': 'EA17',
    'GBP': 'GBR',
    'JPY': 'JPN',
    'INR': 'IND',
    'CNY': 'CHN',
    'DKK': 'DNK',
    'AUD': 'AUS',
    'SEK': 'SWE',
    'CHF': 'CHE',
    'NLG': 'NLD',
    'DEM': 'DEU',
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
    'Australian dollar, monthly average': 'AUD',
    'Swedish krona, monthly average': 'SEK',
    'Swiss franc, monthly average': 'CHF',
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
    'IEXM1601': 'AUD',
    'IEXM1001': 'SEK',
    'IEXM1101': 'CHF',
}

# Currencies the Bank of Canada does not publish against CAD; their CAD rate is
# derived through the euro from the ECB daily reference rates (column = ISO code).
ECB_VIA_EUR_CURRENCIES = ['DKK']

# Pre-euro legacy currencies, converted to EUR at the irrevocable rates fixed on
# 1 January 1999 (Council Regulation (EC) No 2866/98): units per one euro. Their CAD
# rate in any year is CAD_per_EUR / rate. Added 2026-10-03 for 1990s Dutch and
# German cost data in the tech_data database.
EURO_LEGACY_RATES = {'NLG': 2.20371, 'DEM': 1.95583}
EURO_LEGACY_LABEL = 'fixed euro conversion rate (Council Regulation (EC) No 2866/98)'

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


def load_oecd_deflator_forecast(path: Path = OECD_FORECAST_FILE,
                                scenario: str = OECD_SCENARIO) -> pd.DataFrame:
    """Read the OECD long-term PGDP index into Context/Year/Value rows (Value = index)."""
    df = pd.read_csv(path)
    df = df[(df['MEASURE'] == OECD_DEFLATOR_MEASURE) & (df['SCENARIO'] == scenario)
            & (df['FREQ'] == 'A')]
    if df.empty:
        raise ValueError(
            f'{path.name}: no annual {OECD_DEFLATOR_MEASURE} rows for scenario {scenario}. '
            f"Available scenarios: {', '.join(sorted(pd.read_csv(path)['SCENARIO'].unique()))}"
        )
    area_to_cur = {a: c for c, a in OECD_AREAS.items()}
    missing = set(OECD_AREAS.values()) - set(df['REF_AREA'])
    if missing:
        raise ValueError(f"{path.name} is missing expected areas: {', '.join(sorted(missing))}")
    df = df[df['REF_AREA'].isin(area_to_cur)]
    if df.duplicated(['REF_AREA', 'TIME_PERIOD']).any():
        raise ValueError(f'{path.name}: duplicate (area, year) rows for {scenario}')
    out = pd.DataFrame({
        'Context': df['REF_AREA'].map(area_to_cur),
        'Year': df['TIME_PERIOD'].astype(int),
        'Value': pd.to_numeric(df['OBS_VALUE'], errors='coerce'),
    })
    return out.dropna(subset=['Value']).reset_index(drop=True)


def extend_with_forecast(history: pd.DataFrame, forecast: pd.DataFrame,
                         source_name: str, scenario: str = OECD_SCENARIO) -> pd.DataFrame:
    """
    Append forecast years to each currency's historical deflator.

    Splices at the currency's last historical year H: for y > H,
    value = history[H] * forecast[y] / forecast[H]. History is left untouched.
    """
    fc = forecast.set_index(['Context', 'Year'])['Value']
    parts = []
    for cur, hist in history.groupby('Context'):
        if cur not in OECD_AREAS:
            continue
        last = hist.loc[hist['Year'].idxmax()]
        h_year, h_value = int(last['Year']), float(last['Value'])
        if (cur, h_year) not in fc.index:
            raise ValueError(
                f'{source_name}: no {OECD_AREAS[cur]} value for {h_year}, the last '
                f'historical {cur} deflator year, so the forecast cannot be spliced.'
            )
        f = fc.loc[cur]
        f = f[f.index > h_year]
        if f.empty:
            continue
        parts.append(pd.DataFrame({
            'Context': cur,
            'Year': f.index.astype(int),
            'Value': h_value * f.values / fc[(cur, h_year)],
            'Source': (
                f'Forecast: {last["Source"].split(";")[0]} {h_year} value, grown at the OECD '
                f'Economic Outlook 117 long-term scenario {scenario} GDP deflator '
                f'({OECD_DEFLATOR_MEASURE}, {OECD_AREAS[cur]}) path; from {source_name}'
            ),
        }))
    return pd.concat([history] + parts, ignore_index=True)


def _raw(default: Path, raw_dir: Path | None) -> Path:
    """A raw file: the default under RAW_DIR, or the same name under raw_dir."""
    return default if raw_dir is None else Path(raw_dir) / default.name


def build_deflator_table(start_year: int | None = START_YEAR,
                         raw_dir: Path | None = None,
                         forecast: bool | None = None) -> pd.DataFrame:
    """Assemble the full deflator table, one row per (currency, year).

    raw_dir: read the raw files from there instead of RAW_DIR (the technology
    database keeps them in its own deposit, data/extracted/technology-database/currency/).
    forecast: extend past the last historical year with the OECD deflator path.
    True requires the OECD file, False skips it, None (default) extends only if
    the file is present, so a deposit without it still builds history alone.
    """
    table = pd.concat(
        [load_world_bank_deflator(_raw(WORLD_BANK_FILE, raw_dir)),
         load_ecb_deflator(_raw(ECB_FILE, raw_dir))],
        ignore_index=True,
    )
    oecd_path = _raw(OECD_FORECAST_FILE, raw_dir)
    if forecast or (forecast is None and oecd_path.exists()):
        table = extend_with_forecast(table, load_oecd_deflator_forecast(oecd_path),
                                     oecd_path.name)
    if start_year is not None:
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


def load_annual_cad_rates(start_year: int | None = START_YEAR,
                          raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Combine both exchange rate files and average them into annual rates.

    The current series takes precedence over the legacy series wherever both
    carry the same currency-month, so the legacy noon rates only fill the period
    before the current series begins. Only complete calendar years (twelve
    monthly observations) are kept, so a partial year never leaves this function.
    """
    monthly = pd.concat(
        [load_current_monthly(_raw(FX_CURRENT_FILE, raw_dir)),
         load_legacy_monthly(_raw(FX_LEGACY_FILE, raw_dir))],
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
    annual = complete if start_year is None else complete[complete['Year'] >= start_year]
    return annual[['Currency', 'Year', 'cad_per_unit', 'files']].reset_index(drop=True)


def load_ecb_annual_per_eur(path: Path = ECB_FX_FILE,
                            currencies: list[str] = ECB_VIA_EUR_CURRENCIES,
                            start_year: int | None = START_YEAR) -> pd.DataFrame:
    """
    Average the ECB daily reference rates (units of X per one euro) to annual
    X_per_EUR. A year is kept only if it has observations in all twelve months,
    so the current partial year is never emitted.
    """
    df = pd.read_csv(path, na_values=['N/A'])
    missing = set(currencies) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected currencies: {', '.join(sorted(missing))}")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    out = []
    for cur in currencies:
        d = df[['Year', 'Month', cur]].dropna()
        g = d.groupby('Year').agg(per_eur=(cur, 'mean'), months=('Month', 'nunique'),
                                  days=(cur, 'size')).reset_index()
        g = g[(g['months'] == MONTHS_IN_YEAR)
              & ((g['Year'] >= start_year) if start_year is not None else True)]
        g['Currency'] = cur
        out.append(g[['Currency', 'Year', 'per_eur', 'days']])
    return pd.concat(out, ignore_index=True)


def add_euro_legacy(annual: pd.DataFrame) -> pd.DataFrame:
    """Append CAD_per_X for EURO_LEGACY_RATES: CAD_per_EUR / fixed X_per_EUR."""
    if not EURO_LEGACY_RATES:
        return annual
    eur = annual[annual['Currency'] == 'EUR'][['Year', 'cad_per_unit', 'files']]
    parts = []
    for cur, rate in EURO_LEGACY_RATES.items():
        m = eur.copy()
        m['Currency'] = cur
        m['cad_per_unit'] = m['cad_per_unit'] / rate
        m['files'] = m['files'].map(lambda f: tuple(sorted(set(f) | {EURO_LEGACY_LABEL})))
        parts.append(m[['Currency', 'Year', 'cad_per_unit', 'files']])
    return pd.concat([annual] + parts, ignore_index=True)


def add_via_eur(annual: pd.DataFrame, path: Path = ECB_FX_FILE) -> pd.DataFrame:
    """Append CAD_per_X for ECB_VIA_EUR_CURRENCIES: CAD_per_EUR / X_per_EUR."""
    if not ECB_VIA_EUR_CURRENCIES:
        return annual
    eur = annual[annual['Currency'] == 'EUR'][['Year', 'cad_per_unit', 'files']]
    ecb = load_ecb_annual_per_eur(path)
    m = ecb.merge(eur, on='Year', how='inner')
    m['cad_per_unit'] = m['cad_per_unit'] / m['per_eur']
    m['files'] = m['files'].map(lambda f: tuple(sorted(set(f) | {path.name})))
    return pd.concat([annual, m[['Currency', 'Year', 'cad_per_unit', 'files']]],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Exchange -- pair table
# ---------------------------------------------------------------------------

_FILE_LABELS = {
    FX_LEGACY_FILE.name: f'Bank of Canada legacy monthly average noon rates ({FX_LEGACY_FILE.name})',
    FX_CURRENT_FILE.name: (
        'Statistics Canada / Bank of Canada monthly average rates '
        f'({FX_CURRENT_FILE.name})'
    ),
    ECB_FX_FILE.name: (
        f'ECB daily euro reference rates averaged to annual ({ECB_FX_FILE.name}), '
        'CAD rate derived via EUR'
    ),
}


def _source_note(files: tuple[str, ...], is_cross: bool) -> str:
    labels = ' + '.join(_FILE_LABELS.get(f, f) for f in sorted(files))
    note = f'Annual average of monthly rates; from {labels}'
    if is_cross:
        note += f'; cross rate derived via {BASE_CURRENCY}'
    return note


def build_exchange_table(start_year: int | None = START_YEAR,
                         raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Build every directed <TARGET>_per_<SOURCE> pair among the covered currencies.

    Rates are published against CAD, so cross rates go through CAD:
    A_per_B = CAD_per_B / CAD_per_A. CAD itself is added as a rate of 1.0 so the
    base currency falls out of the same arithmetic as every other pair.
    """
    annual = load_annual_cad_rates(start_year=start_year, raw_dir=raw_dir)
    annual = add_via_eur(annual, _raw(ECB_FX_FILE, raw_dir))
    annual = add_euro_legacy(annual)

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
    deflator = build_deflator_table(forecast=True)
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
            f'\nNote: exchange rates start in {exchange_start}, deflators in '
            f'{deflator_start}. Values dated before {exchange_start} still convert: '
            'conversion_factor deflates inside the source currency and exchanges only '
            f'at the target year. Only a TARGET year before {exchange_start} would fail.'
        )
    is_fc = deflator['Source'].str.startswith('Forecast:')
    if is_fc.any():
        print(
            f'Note: deflators for {deflator.loc[is_fc, "Year"].min()}-'
            f'{deflator.loc[is_fc, "Year"].max()} are OECD {OECD_SCENARIO} forecasts. '
            f'Exchange rates end in {exchange["Year"].max()}, so a TARGET year after '
            'that still fails.'
        )


if __name__ == '__main__':
    main()
