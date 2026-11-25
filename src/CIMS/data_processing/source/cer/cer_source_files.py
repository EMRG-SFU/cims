"""
CER source files: carbon-free end-use prices and history back to 2000.

Reads the Canada's Energy Future (CEF) open-data files in raw_data/cer/ and
writes the versions the rest of the pipeline reads to processed_data/cer/,
under the same file names (for example end-use-prices-2026.csv):

  end-use-prices   Carbon costs removed, 2000-2004 back-cast, and reshaped to
                   the columns the energy-price scripts use (Scenario, Year,
                   Area, Sector, Fuel, Units, Sum of Price).
  end-use-demand   2000-2004 added, held at 2005 values (used only as weights).
  benchmark-prices 2000-2004 added from EIA spot prices (2025 US$).
  macro-indicators 2000-2004 added from Statistics Canada and the deflator and
                   exchange tables.

It also writes end-use-prices_carbon-<edition>.csv, an audit table with the
price before and after carbon removal and each carbon component.

Why remove carbon: CER's end-use prices include carbon pricing (consumer
charges to 2024, industrial pricing, and in Canada Net-zero the aggregate
cost of carbon). CIMS adds its own carbon prices through policy files, so the
input prices must exclude carbon or it is counted twice.

The method and every assumption are documented in README.md in this folder.
The assumptions themselves sit in data/mappings_conversions/:
  config/interim-source/cer_carbon-price.csv                carbon price by component, region, year
  config/interim-source/cer_emission-factor.csv             tCO2e per GJ by fuel and sector
  config/interim-source/cer_benchmark-price_2000-2005.csv   EIA nominal spot prices

Run after deflator_exchange (it reads the deflator and exchange tables).

Usage:
    python -m CIMS.data_processing.source.cer.cer_source_files
"""

from pathlib import Path

import pandas as pd

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.extractors.cer import find_cer_file

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = paths.RAW / 'cer'
OUT_DIR       = paths.PROCESSED / 'cer'
MAPPINGS      = paths.CONFIG / 'mapping'
CARBON_FILE   = paths.CONFIG / 'interim-source/cer_carbon-price.csv'
EF_FILE       = paths.CONFIG / 'interim-source/cer_emission-factor.csv'
BENCH_HISTORY = paths.CONFIG / 'interim-source/cer_benchmark-price_2000-2005.csv'
CPI_FILE      = paths.RAW / 'stats_can/cpi/18100005.xlsx'
POP_FILE      = paths.RAW / 'stats_can/population/1710000901.csv'
GDP_FILE      = paths.RAW / 'stats_can/gdp/36100222.csv'
DEFLATOR_FILE = paths.PROCESSED / 'deflator_exchange/currency_deflator.csv'
EXCHANGE_FILE = paths.PROCESSED / 'deflator_exchange/currency_exchange.csv'

FIRST_YEAR  = 2000          # extend every file back to this year
CER_START   = 2005          # first year in the CER open data
PRICE_YEAR  = 2025          # CEF2026 dollar basis (C$2025, US$2025)
PRICE_UNITS = f'{PRICE_YEAR} C$/GJ'
DEFLATOR_VAR = 'Gross Domestic Product Deflator (2017=100)'
# CER restates nominal carbon prices in C$2025 with the CPI, not the GDP deflator:
# its implied factors for 2030 (1.0964) and 2050 (1.687) match the CPI ratios.
CPI_VAR = 'Consumer Price Index (2002=100)'

# CPI component used to back-cast each CER fuel. CPI tracks consumer prices;
# it is also used for industrial prices, which have no provincial index.
CPI_COMPONENT = {
    'Natural Gas': 'Natural gas',
    'Oil':         'Fuel oil and other fuels',
    'Diesel':      'Fuel oil and other fuels',
    'Gasoline':    'Gasoline',
    'Electricity': 'Electricity',
}
# CPI geography for each CER region. Missing components fall back to Canada.
CPI_GEO = {
    'Yukon': 'Whitehorse, Yukon',
    'Northwest Territories': 'Yellowknife, Northwest Territories',
    'Nunavut': 'Canada',
}
# Benchmarks without an EIA series: hold their 2005 ratio to a marker price.
BENCH_MARKER = {
    'Canadian Light Sweet (CLS) - US$/bbl':        'West Texas Intermediate (WTI) - US$/bbl',
    'Western Canadian Select (WCS) - US$/bbl':     'West Texas Intermediate (WTI) - US$/bbl',
    'Nova Inventory Transfer (NIT) - US$/MMBtu':   'Henry Hub - US$/MMBtu',
}


# ── reference tables ──────────────────────────────────────────────────────────

def _read_cer(stem: str) -> tuple[pd.DataFrame, Path]:
    path = find_cer_file(RAW_DIR, stem)
    return pd.read_csv(path, encoding='utf-8-sig'), path


def _deflator_tables() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return CAD and USD GDP deflators and CAD per USD, indexed by year."""
    d = pd.read_csv(DEFLATOR_FILE, encoding='utf-8-sig')
    x = pd.read_csv(EXCHANGE_FILE, encoding='utf-8-sig')
    cad = d[d['Context'] == 'CAD'].set_index('Year')['Value']
    usd = d[d['Context'] == 'USD'].set_index('Year')['Value']
    fx = x[x['Context'] == 'CAD_per_USD'].set_index('Year')['Value']
    return cad, usd, fx


def _cpi() -> pd.DataFrame:
    """Statistics Canada 18-10-0005, annual CPI (2002=100), long format."""
    df = pd.read_excel(CPI_FILE)
    df = df.rename(columns={'REF_DATE': 'Year', 'GEO': 'Geo',
                            'Products and product groups': 'Product', 'VALUE': 'Value'})
    df['Year'] = df['Year'].astype(int)
    return df[['Year', 'Geo', 'Product', 'Value']].dropna(subset=['Value'])


def _cpi_lookup(cpi: pd.DataFrame, region: str, fuel: str) -> pd.Series:
    """CPI series for a region and fuel, falling back to Canada."""
    product = CPI_COMPONENT[fuel]
    for geo in (CPI_GEO.get(region, region), 'Canada'):
        s = cpi[(cpi['Geo'] == geo) & (cpi['Product'] == product)].set_index('Year')['Value']
        if all(y in s.index for y in range(FIRST_YEAR, CER_START + 1)):
            return s
    raise ValueError(f'No CPI series for {region} {fuel}')


# ── macro indicators ──────────────────────────────────────────────────────────

def _statcan_canada(path: Path, filters: dict, month: str | None = None) -> pd.Series:
    """Canada total; sums the provinces and territories when no Canada row exists."""
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    if (df['GEO'] == 'Canada').any():
        df = df[df['GEO'] == 'Canada']
    for col, val in filters.items():
        df = df[df[col] == val]
    if month:
        df = df[df['REF_DATE'].astype(str).str.endswith(month)]
    years = df['REF_DATE'].astype(str).str[:4].astype(int)
    return df['VALUE'].astype(float).groupby(years.values).sum()


def extend_macro(macro: pd.DataFrame, cad: pd.Series, fx: pd.Series, cpi: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2000-2004 by chaining each series to CER's 2005 value with the
    year-on-year movement of a matching Statistics Canada or deflator series.
    """
    pop = _statcan_canada(POP_FILE, {}, month='-07') / 1000        # July 1, thousands
    gdp = _statcan_canada(GDP_FILE, {'Prices': 'Chained (2017) dollars',
                                     'Estimates': 'Gross domestic product at market prices'})
    cpi_all = cpi[(cpi['Geo'] == 'Canada') & (cpi['Product'] == 'All-items')].set_index('Year')['Value']
    movers = {
        'Canada-US Exchange Rate (C$/US$)': fx,
        'Consumer Price Index (2002=100)': cpi_all,
        DEFLATOR_VAR: cad,
        'Population (thousands)': pop,
        'Real Gross Domestic Product ($2017 Millions)': gdp,
    }
    new = []
    for (scen, reg, var), g in macro.groupby(['Scenario', 'Region', 'Variable']):
        mover = next((s for k, s in movers.items() if k.lower() == var.lower()), None)
        if mover is None:
            raise ValueError(f'No history series for macro variable {var!r}')
        base = g.loc[g['Year'] == CER_START, 'Value'].iloc[0]
        for y in range(FIRST_YEAR, CER_START):
            new.append({'Scenario': scen, 'Region': reg, 'Variable': var, 'Year': y,
                        'Value': base * mover[y] / mover[CER_START]})
    return pd.concat([pd.DataFrame(new), macro]).sort_values(['Scenario', 'Variable', 'Year'])


def deflator_index(macro: pd.DataFrame) -> dict[str, pd.Series]:
    """Per scenario: CPI factor that turns nominal dollars of year y into C$2025."""
    out = {}
    for scen, g in macro[macro['Variable'] == CPI_VAR].groupby('Scenario'):
        d = g.set_index('Year')['Value']
        out[scen] = d[PRICE_YEAR] / d
    return out


# ── benchmark prices ──────────────────────────────────────────────────────────

def extend_benchmark(bench: pd.DataFrame, usd: pd.Series) -> pd.DataFrame:
    """
    Add 2000-2004. EIA series: CER's 2005 value times the change in the real
    (US GDP deflator) EIA price. Others: hold their 2005 ratio to a marker.
    """
    hist = pd.read_csv(BENCH_HISTORY)
    real = {}
    for var, g in hist.groupby('Variable'):
        nominal = g.set_index('Year')['Value']
        real[var] = nominal * usd[PRICE_YEAR] / usd.loc[nominal.index]
    new = []
    for (scen, var), g in bench.groupby(['Scenario', 'Variable']):
        base = g.loc[g['Year'] == CER_START, 'Value'].iloc[0]
        marker = var if var in real else BENCH_MARKER[var]
        r = real[marker]
        for y in range(FIRST_YEAR, CER_START):
            new.append({'Scenario': scen, 'Variable': var, 'Year': y,
                        'Value': base * r[y] / r[CER_START]})
    return pd.concat([pd.DataFrame(new), bench]).sort_values(['Scenario', 'Variable', 'Year'])


# ── end-use demand ────────────────────────────────────────────────────────────

def extend_demand(demand: pd.DataFrame) -> pd.DataFrame:
    """Add 2000-2004 at 2005 values; the pipeline uses demand only as price weights."""
    base = demand[demand['Year'] == CER_START]
    early = [base.assign(Year=y) for y in range(FIRST_YEAR, CER_START)]
    return pd.concat(early + [demand]).sort_values(['Scenario', 'Region', 'Sector', 'Variable', 'Year'])


# ── end-use prices ────────────────────────────────────────────────────────────

def _expand_sectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(Sector=df['Sectors'].str.split(';')).explode('Sector')
    return df.drop(columns='Sectors')


def carbon_costs(prices: pd.DataFrame, defl: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Add one column per carbon component (C$2025/t), the emission factor, and
    the carbon cost (C$2025/GJ) to the CER end-use prices.
    """
    carbon = _expand_sectors(pd.read_csv(CARBON_FILE))
    unknown = set(carbon['Scenario']) - set(prices['Scenario']) - {'All'}
    if unknown:
        raise ValueError(f'Scenario in config/interim-source/cer_carbon-price.csv not in the price data: {sorted(unknown)}')
    ef = _expand_sectors(pd.read_csv(EF_FILE)).rename(
        columns={'Emission_factor_tCO2e_per_GJ': 'Emission factor (tCO2e/GJ)'})

    out = prices.merge(ef[['Fuel', 'Sector', 'Emission factor (tCO2e/GJ)']],
                       on=['Fuel', 'Sector'], how='left')
    missing = out['Emission factor (tCO2e/GJ)'].isna()
    if missing.any():
        pairs = out.loc[missing, ['Fuel', 'Sector']].drop_duplicates().values.tolist()
        raise ValueError(f'No emission factor for {pairs}')

    total = pd.Series(0.0, index=out.index)
    for comp, g in carbon.groupby('Component'):
        col = f'{comp} price (C$2025/t)'
        rows = []
        for scen in out['Scenario'].unique():
            s = g[g['Scenario'].isin(['All', scen])].copy()
            real = s['Price'].astype(float)
            nominal = s['Basis'] == 'nominal'
            real[nominal] = real[nominal] * s.loc[nominal, 'Year'].map(defl[scen]).values
            rows.append(s.assign(Scenario=scen, **{col: real})[['Scenario', 'Region', 'Sector', 'Year', col]])
        comp_df = pd.concat(rows).rename(columns={'Region': 'Area'})
        out = out.merge(comp_df, on=['Scenario', 'Area', 'Sector', 'Year'], how='left')
        for key in ('Area', 'Sector'):
            unknown = set(comp_df[key]) - set(out[key])
            if unknown:
                raise ValueError(f'{key} in config/interim-source/cer_carbon-price.csv not in the price data: {sorted(unknown)}')
        out[col] = out[col].fillna(0.0)
        total += out[col]
    out['Carbon price (C$2025/t)'] = total
    out['Carbon cost (C$2025/GJ)'] = total * out['Emission factor (tCO2e/GJ)']
    return out


def build_end_use_prices(raw: pd.DataFrame, macro: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    prices = raw.rename(columns={'Region': 'Area', 'Variable': 'Fuel',
                                 'Value': 'Price incl. carbon (C$2025/GJ)'})
    out = carbon_costs(prices, deflator_index(macro))
    out['Sum of Price'] = out['Price incl. carbon (C$2025/GJ)'] - out['Carbon cost (C$2025/GJ)']
    out['Method'] = 'CER'
    out.loc[out['Carbon cost (C$2025/GJ)'] > 0, 'Method'] = 'CER less carbon'

    negative = out['Sum of Price'] < 0
    if negative.any():
        print(f'  WARNING: {negative.sum()} prices fall below zero after carbon removal; set to zero')
        out.loc[negative, 'Sum of Price'] = 0.0

    # Back-cast 2000-2004: 2005 price times the real change in the matching CPI.
    defl_hist = deflator_index(macro)
    base = out[out['Year'] == CER_START]
    early = []
    for (area, fuel), g in base.groupby(['Area', 'Fuel']):
        cpi_s = _cpi_lookup(cpi, area, fuel)
        for y in range(FIRST_YEAR, CER_START):
            for _, row in g.iterrows():
                factor = (cpi_s[y] / cpi_s[CER_START]) * (defl_hist[row['Scenario']][y] / defl_hist[row['Scenario']][CER_START])
                r = row.copy()
                r['Year'] = y
                r['Price incl. carbon (C$2025/GJ)'] = row['Price incl. carbon (C$2025/GJ)'] * factor
                r['Sum of Price'] = row['Sum of Price'] * factor
                r['Method'] = f'CPI back-cast ({CPI_COMPONENT[fuel]})'
                early.append(r)
    out = pd.concat([pd.DataFrame(early), out]).sort_values(['Scenario', 'Area', 'Sector', 'Fuel', 'Year'])
    out['Units'] = PRICE_UNITS
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=' * 60)
    print('CER SOURCE FILES')
    print('=' * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cad, usd, fx = _deflator_tables()
    cpi = _cpi()

    macro_raw, macro_path = _read_cer('macro-indicators')
    macro = extend_macro(macro_raw, cad, fx, cpi)
    macro.to_csv(OUT_DIR / macro_path.name, index=False)
    print(f'  {macro_path.name}: {len(macro_raw)} → {len(macro)} rows')

    bench_raw, bench_path = _read_cer('benchmark-prices')
    bench = extend_benchmark(bench_raw, usd)
    bench.to_csv(OUT_DIR / bench_path.name, index=False)
    print(f'  {bench_path.name}: {len(bench_raw)} → {len(bench)} rows')

    demand_raw, demand_path = _read_cer('end-use-demand')
    demand = extend_demand(demand_raw)
    demand.to_csv(OUT_DIR / demand_path.name, index=False)
    print(f'  {demand_path.name}: {len(demand_raw)} → {len(demand)} rows')

    prices_raw, prices_path = _read_cer('end-use-prices')
    prices = build_end_use_prices(prices_raw, macro, cpi)
    prices[['Scenario', 'Year', 'Area', 'Sector', 'Fuel', 'Units', 'Sum of Price']].to_csv(
        OUT_DIR / prices_path.name, index=False)
    audit_name = prices_path.name.replace('end-use-prices-', 'end-use-prices_carbon-')
    prices.rename(columns={'Sum of Price': 'Price excl. carbon (C$2025/GJ)'}).drop(columns='Units').to_csv(
        OUT_DIR / audit_name, index=False)
    print(f'  {prices_path.name}: {len(prices_raw)} → {len(prices)} rows (carbon removed)')
    print(f'  {audit_name}: audit table')
    print('Done.')


if __name__ == '__main__':
    main()
