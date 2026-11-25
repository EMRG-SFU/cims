# CIMS Data Processing Control

Houses conversions, mappings, and control scenarios/years

## CER source file assumptions

Read by `src/CIMS/data_processing/source/cer/cer_source_files.py`; see the README in that folder for the method and checks.

| File | Contents |
|---|---|
| `cer_carbon-prices.csv` | Carbon prices embedded in CER end-use prices, by component, region, sector, and year |
| `cer_emission-factors.csv` | Emission factors implied by CER prices, by fuel and sector |
| `cer_benchmark-prices_2000-2005.csv` | EIA spot prices used to extend the benchmark prices to 2000 |
