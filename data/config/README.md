# config/

Every hand-edited record the data processing pipeline reads (conventions §2). Everything here
is committed. A value someone chose, such as a proxy region, a growth rate, or a fuel
assumption, belongs here with its rationale, not in code or in a generated file.

| Path | Contents | Read by |
| - | - | - |
| `control.py` | Pipeline-wide settings: data years per source, the CER scenario, currency years | `utils/controls_conversions.py` (`CONTROLS`) |
| `mapping/` | Source-to-CIMS maps: `cer-to-cims.csv`, `energy.csv`, `region.csv`, `sector.csv`, `sector-region.csv`, `nir-to-cims.csv`, `nir-crosswalk.csv`, `cims-node.csv` | most source and sector modules |
| `conversion/energy-conversion.csv` | Unit conversions (GJ ↔ TJ, L ↔ tonnes, …) | `utils/controls_conversions.py` |
| `excluded-branch.csv` | Branches left out of processing | `utils/controls_conversions.py` |
| `assumption/` | Growth rates and shares that extend the data through the projection period | the activity and NRCan CEUD source modules, `utils/steam_chp_shares.py` |
| `calibration/` | `fic-proxy-map.csv` and `fic-decline-techs.csv` | `calibration/Calibration/Utility/extend_fics.py` |
| `model-default/` | `default-parameter.csv` and `default-list.csv`: the model's parameter defaults and its column list | `scenarios/Reference.py` (the model) |
| `interim-source/` | Hand-entered CER tables (see below), until the primary sources replace CER | `source/cer/cer_source_files.py` |
| `technology-database/` | The technology database's records: archetypes, source-example mapping, fill rules, reference and vocabulary tables | the database build |
| `results-general.txt`, `results-tech.txt` | Which results the model writes | `scenarios/Reference.py` |

## interim-source/: the CER tables

Read by `src/CIMS/data_processing/source/cer/cer_source_files.py`; see the README in that
folder for the method and checks.

| File | Contents |
|---|---|
| `cer_carbon-price.csv` | Carbon prices embedded in CER end-use prices, by component, region, sector, and year |
| `cer_emission-factor.csv` | Emission factors implied by CER prices, by fuel and sector |
| `cer_benchmark-price_2000-2005.csv` | EIA spot prices used to extend the benchmark prices to 2000 |
