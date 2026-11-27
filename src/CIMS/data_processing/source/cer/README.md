# CER source files

`cer_source_files.py` turns the Canada's Energy Future (CEF) open-data files in `data/extracted/gov-can/cer/` into the versions the pipeline reads, in `data/processed/gov-can/cer/`. It keeps CER's file names, so `find_cer_file()` finds them the same way. Run it after `deflator_exchange` and before `energy_prices`; `process_source_data.py` does this.

| File | What changes |
|---|---|
| `end-use-prices-<edition>.csv` | Carbon costs removed, 2000 to 2004 back-cast, and columns renamed to the layout `energy_prices.py` and `energy_price_multipliers.py` read |
| `end-use-prices_carbon-<edition>.csv` | New audit table: price before and after removal, each carbon component, the emission factor, and the method used for each row |
| `end-use-demand-<edition>.csv` | 2000 to 2004 added |
| `benchmark-prices-<edition>.csv` | 2000 to 2004 added |
| `macro-indicators-<edition>.csv` | 2000 to 2004 added |

The crude oil and natural gas production files are not changed. `oil_production.py` and `gas_production.py` already back-cast 2000 to 2004 themselves.

## Why remove carbon from end-use prices

CER's end-use prices include carbon pricing. CIMS adds its own carbon prices through policy files (`carbon-tax_ref`, `obps_federal`, and others). If the input prices kept CER's carbon costs, CIMS would count carbon twice. The script subtracts, for each row:

```
carbon cost (C$2025/GJ) = sum over components of carbon price (C$2025/t) × emission factor (tCO2e/GJ)
price excl. carbon      = CER price − carbon cost
```

Electricity prices are left alone. Their carbon costs sit inside generation costs, and CIMS models the power sector itself.

## Carbon price components

The prices are in `data/config/interim-source/cer_carbon-price.csv`, one row per component, region, sector, and year.

| Component | Applies to | Years | Price | Source |
|---|---|---|---|---|
| Consumer | All sectors except electricity, every region | 2000 to 2024 | The `carbon-tax_ref` policy files: BC carbon tax from 2008, Quebec cap-and-trade from 2013, and the federal fuel charge from 2019 everywhere else, rising to $80/t in 2024 (nominal) | Policy files, so the price removed equals the price CIMS adds back |
| Consumer, Quebec | All sectors except electricity, Quebec | 2025 to 2050 | $95/t in 2025, rising $15/yr to $170/t in 2030, then flat (nominal) | CEF2026 Table A1.2 (cap-and-trade to 2030, then the federal systems) |
| Industrial | Industrial, every region except Quebec | 2025 to 2050 | Same path: $95/t (2025) to $170/t (2030), then flat (nominal) | CEF2026 Table A1.1 |
| Aggregate cost of carbon | Industrial, every region, Canada Net-zero only | 2031 to 2050 | Rises $20/yr to $200/t in 2040, then $25/yr to $450/t in 2050 (C$2025) | CEF2026 Table A1.3 for the end points; the path is inferred (below) |

The consumer charge ended in April 2025, so CER carries no consumer carbon price from 2025, except in Quebec.

### Why history uses the policy files

CER's historical prices are observed prices, so they carry whatever carbon charge applied at the time. Removing the same schedule that CIMS adds back means that, in every historical year, the carbon-free price plus the CIMS carbon tax equals CER's price. Calibration therefore sees the observed price. Where a historical charge is missing from the policy files (for example Alberta's 2017 to 2019 levy), it stays inside the fuel price, which still counts it once.

### How the projection prices were confirmed

Each assumption was checked against the CER data before use:

- **Industrial price.** In Alberta, industrial and commercial gas prices are identical through 2024. From 2025, the industrial price is higher by exactly $95/t × 0.0509 tCO2e/GJ, and by the C$2025 value of $170/t in 2030 and later.
- **Quebec.** Quebec's residential gas price rises against Ontario's by the benchmark path from 2025 to 2030, then falls slowly as a flat nominal price loses value to inflation. Its industrial gap with Ontario stays constant, so the same price applies to both sectors.
- **Aggregate cost of carbon.** Canada Net-zero less Current Measures, industrial less commercial, divided by the emission factor, gives $40/t in 2032, $100 in 2035, $200 in 2040, $325 in 2045, and $450 in 2050. That is the stepped path above, not a straight line from $0 to $450.
- **Dollar basis.** CER restates nominal carbon prices in C$2025 with the CPI (`Consumer Price Index (2002=100)` in the macro file), not the GDP deflator. Its implied factors for 2030 (1.0964) and 2050 (1.687) match the CPI ratios (1.0963 and 1.6878). The script uses the CPI.

## Emission factors

`data/config/interim-source/cer_emission-factor.csv` holds CER's implied emission factors, backed out of its own data rather than taken from a published table:

| Fuel | Sectors | tCO2e/GJ | Derived from |
|---|---|---|---|
| Natural gas | Residential, commercial, industrial | 0.0509 | Aggregate cost of carbon gap (22.91 $/GJ at $450/t, 2050); matches the 2024 consumer gap (4.140 $/GJ at $80/t) |
| Oil | Residential, commercial | 0.0721 | 2024 consumer gap (5.864 $/GJ); light fuel oil |
| Oil | Industrial | 0.0786 | Aggregate cost of carbon gap (35.37 $/GJ, 2050); matches the 2024 gap (6.393 $/GJ); heavy fuel oil |
| Diesel | Transportation | 0.0699 | 2024 consumer gap (5.684 $/GJ) |
| Gasoline | Transportation | 0.0690 | 2024 consumer gap (5.611 $/GJ) |

The 2024 gaps compare the open-data file with a carbon-free Current Measures export that CER supplied earlier. That export is no longer in the repository.

## Validation

| Check | Result |
|---|---|
| Against the earlier carbon-free Current Measures export (Alberta and Ontario, 2024 to 2050) | Within 0.017 $/GJ in every year checked |
| Alberta industrial less commercial gas, after removal | 0.000 to 0.003 $/GJ in every year, matching CER history |
| Canada Net-zero less Current Measures, industrial less commercial, after removal | Under 0.06 $/GJ, except Ontario gas (0.1 to 0.2 $/GJ in 2035 to 2050) |
| Quebec less Ontario residential gas, 2023 to 2033, after removal | Flat at 3.5 to 3.7 $/GJ through the 2025 and 2030 policy changes |
| Prices below zero after removal | None (lowest 1.44 $/GJ) |

The audit table records every component, so any row can be traced.

## Extending to 2000

CER's open data starts in 2005. Each series is extended back to 2000 by holding CER's 2005 value and applying the year-on-year movement of a matching series:

| File | Series | Movement used for 2000 to 2004 | Source |
|---|---|---|---|
| End-use prices | Every fuel, region, and sector | Matching CPI component (natural gas; fuel oil and other fuels for oil and diesel; gasoline; electricity), deflated by the all-items CPI. Falls back to Canada where a province lacks the component; Yukon and the Northwest Territories use Whitehorse and Yellowknife | Statistics Canada 18-10-0005 (`extracted/gov-can/statcan/cpi/18100005.xlsx`) |
| Benchmark prices | Brent, WTI, Henry Hub | EIA annual spot price, restated in 2025 US$ with the US GDP deflator | `config/interim-source/cer_benchmark-price_2000-2005.csv` (FRED: ACOILBRENTEU, ACOILWTICO, AHHNGSP) |
| Benchmark prices | WCS, CLS, NIT | Held at their 2005 ratio to WTI (WCS, CLS) or Henry Hub (NIT) | No public annual series in the repository |
| Macro indicators | Exchange rate; GDP deflator | `processed/currency/currency-exchange.csv` and `processed/currency/currency-deflator.csv` (CAD) | `processed/currency/` |
| Macro indicators | CPI | All-items CPI, Canada | Statistics Canada 18-10-0005 |
| Macro indicators | Population; real GDP | July 1 population (sum of provinces and territories); GDP at market prices, chained 2017 dollars | Statistics Canada 17-10-0009 and 36-10-0222 |
| End-use demand | Every series | Held at 2005 values | The pipeline uses demand only to weight prices across provinces |

No carbon price applied before 2008, so the back-cast years need no removal.

## Limits

- The CPI tracks consumer prices. It also drives the industrial back-cast, since no provincial industrial index exists for these fuels.
- The historical removal follows the CIMS policy files, not each province's exact legislated rates. Nova Scotia (2019 to 2022) and the Atlantic provinces' 2023 transition are approximate, but the removed and added prices stay equal.
- The emission factors are inferred from CER's data. Each holds across every year and province checked, but CER has not published them.
- Ontario's industrial gas in Canada Net-zero keeps a small extra cost (up to 0.2 $/GJ in 2050) that the carbon components above do not explain.

## Updating for a new CEF edition

1. Put the new files in `extracted/gov-can/cer/`. Check that the end-use price columns are still `Scenario, Region, Variable, Year, Value, Sector`.
2. Recheck the components against CEF Appendix 1, and the implied emission factors and aggregate cost of carbon path against the data (the checks above).
3. Update `config/interim-source/cer_carbon-price.csv` and `config/interim-source/cer_emission-factor.csv`, rerun, and review the audit table.
