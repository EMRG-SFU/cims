# Currency deflator and exchange rate tables

`currency_deflator.csv` and `currency_exchange.csv` back `CIMS.Model`'s optional
currency conversion (`target_units` / `deflator_path` / `exchange_path`).

[`deflator_exchange.py`](deflator_exchange.py) in this directory builds them into
**`data/processed_data/deflator_exchange/`**, alongside every other generated
output. They are therefore **not committed** — `data/.gitignore` excludes
`processed_data/**/*.csv` — and neither are the raw downloads they derive from.
A fresh clone has neither, so anything that reads these tables (see
`scenarios/Reference.py`) needs the raw files staged and this script rerun first.

Both files use the same three-column contract, plus a `Source` column that the
loader ignores:

| Column | `currency_deflator.csv` | `currency_exchange.csv` |
| --- | --- | --- |
| `Context` | ISO currency code (e.g. `CAD`, `USD`) | rate name, `<TARGET>_per_<SOURCE>` (e.g. `CAD_per_USD`) |
| `Year` | dollar-year | year the rate applies to |
| `Value` | price index for that currency/year | units of target currency per one source currency |

### Shared by the pipeline and the model

The data pipeline converts currency with the same tables and the same
`CurrencyConverter`, through `CIMS.data_processing.utils.currency`, so a value
converted in the pipeline and one converted by `CIMS.Model` at read time agree
exactly. Every model input the pipeline writes carries a `YYYY_CCC` unit
(`2020_CAD`, `2025_CAD/GJ`), which is what CIMS needs to convert a row. The
`currency_*` settings in `data/mappings_conversions/control.py` choose the year and
currency assumed for bare `$` template units, and whether the pipeline writes
values as tagged (CIMS converts) or converts them itself to a set output year and
currency. Run this builder before the rest of the pipeline.

### Conversion order and why it matters

A value is **deflated first, within its own (source) currency, and only then
exchanged** into the target currency:

```
factor = deflator[source_currency][target_year] / deflator[source_currency][source_year]
         x exchange[target_per_source][target_year]
```

Rebasing within the source (foreign) currency using the GDP deflator at market prices
ensures that the value is adjusted consistently for changes in the domestic price
level of output, preserving the real economic meaning of the original estimate.
Performing this step first avoids mixing inflation dynamics across countries. The
subsequent conversion to CAD using the exchange rate then translates this real value
into the target currency, separating time (inflation) effects from space (currency)
effects and maintaining consistency with national accounts principles.

Two consequences follow:

- **Use the GDP deflator at market prices**, not CPI. CPI tracks the price level of a
  consumption basket; the GDP deflator tracks the price level of domestic output,
  which is the right reference for the cost and price estimates CIMS carries.
- **The conversion is not symmetric.** The two directions deflate in different
  currencies and exchange in different years, so USD 2010 -> CAD 2020 -> USD 2010 does
  not return the original value. Convert once, from the currency and dollar-year the
  estimate was published in; do not chain or reverse conversions.

Deflator values only ever enter the calculation as a ratio of two years *within the
same currency*, so each currency may use its own index base, and the tables carry
each source's own base year rather than a rebased one.

Exchange rates must be present in the direction being requested — `_exchange_factor`
looks up the exact `<TARGET>_per_<SOURCE>` key and neither inverts nor chains rates —
so all 42 directed pairs among the covered currencies are supplied.

Coverage is checked at validation time by the `currency_table_coverage` check,
which reports any (currency, dollar-year) pair in the model data that these tables
cannot convert to the configured target.

## Coverage and sources

The deflator table covers **1960-2100** (EUR from 1995) and the exchange table
**1999-2025**, for **CAD, USD, EUR, GBP, JPY, INR, CNY, DKK, AUD, SEK and CHF**, plus
the pre-euro **NLG** and **DEM**. Deflators after 2025 are forecasts (see below).

| Table | Series | Source |
| --- | --- | --- |
| Deflator | GDP deflator at market prices, `NY.GDP.DEFL.ZS` | World Bank WDI (CAD, USD, GBP, JPY, INR, CNY) |
| Deflator | Euro area GDP at market prices, deflator | ECB, `MNA.A.N.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.IX.D.N` (EUR) |
| Deflator | GDP at market prices, deflator (`PGDP`), forecast years only | OECD Economic Outlook 117 long-term scenarios, scenario `BAU1` |
| Exchange | Monthly average rates, 2017 onward | Statistics Canada / Bank of Canada |
| Exchange | Legacy monthly average noon rates, through 2016 | Bank of Canada (discontinued 2017-04-28) |

Both tables are built from the raw downloads in `data/raw_data/deflator_exchange/`.
Rerun the script to refresh or extend them; the year range follows whatever the raw
files support, so a longer export needs no code change. Per-row provenance is
recorded in the `Source` column.

Run it with:

```
python -m CIMS.data_processing.source.deflator_exchange.deflator_exchange
```

Three details are worth knowing when reading these tables:

- **The exchange table is spliced at 2016/2017.** The Bank of Canada's current
  bilateral series begins 2017-01; everything earlier comes from the discontinued
  legacy series. The builder prefers the current series wherever both carry a month.
  The two overlap Jan-Apr 2017 and agree to within 0.0004 CAD/USD, but genuine
  market moves across that seam can be large (GBP fell ~7% between the 2016 and 2017
  annual averages), so it is the first place to look if a conversion looks wrong.
- **Only complete 12-month calendar years are averaged**, so the current partial
  year is never emitted as an annual rate.
- **Exchange values carry 12 significant digits, not fixed decimals.** One table
  spans `CAD_per_JPY` (~0.0092) to `JPY_per_CAD` (~108); fixed decimals would strip
  the small rates to a few significant figures and break reciprocal consistency.

### Forecast deflators (future dollar-years)

A value quoted in a future dollar-year, such as `2030_USD`, needs only that
currency's deflator for 2030: the exchange step happens at the target year, which
is historical. The builder extends each currency's deflator past its last
historical year H (2025) with the OECD long-term GDP deflator path, spliced at H:

```
deflator[X][y] = deflator[X][H] * oecd[X][y] / oecd[X][H]    for y > H
```

So the OECD series supplies growth only; historical values and base years are
untouched, and no existing conversion changes. The `Source` column of forecast rows
starts with `Forecast:`. When the next historical year is published, rerun the
builder and the splice point moves forward on its own.

The raw file, `oecd_2026_gdp-deflator-forecast.csv`, is the OECD download filtered
to the `PGDP` measure and the `BAU1` scenario. The six EO117 scenarios agree to
within about 0.5% on deflators through 2050. To use another, download that scenario
and set `OECD_SCENARIO` in `deflator_exchange.py` to match.

The download leaves out the OECD's purchasing power parity (`PPP`) measure: a PPP
is not a market exchange rate, and conversions never need a future exchange rate
unless the *target* year is in the future. The exchange table therefore still ends
at the last complete year of market rates.

Cross rates are derived through CAD (`A_per_B = CAD_per_B / CAD_per_A`), since the
published rates are all quoted against the Canadian dollar.
