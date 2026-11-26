# model/

Everything CIMS reads (conventions §2). The build writes it, and nothing in it is committed
apart from this README. `scenarios/Reference.py` lists which folders a run loads, in order: the
base (`root/`), then the scenario files (`region/`, `sector/`, `cost/`, `exogenous/`, policies, and
calibration outputs), where a later file wins.

File names give the region in lowercase, with RoW as `rest-of-world`
(`common/paths.region_file_code`); the values inside the files keep `RoW` until phase 14.

| Path | Contents | Written by |
| - | - | - |
| `root/` | `root_cims.csv`: the root node and model-wide settings (currency year, GWPs, tax foresight, retrofit parameters). The model's base (`base_model`). | `sector/cims_base_model_inputs.py` |
| `region/` | `region_<region>.csv`: the region nodes (CAN, rest-of-world, and each province and territory), with GDP and population, and each parent's request to its region. Loaded as scenario files, before the sectors. | `sector/cims_base_model_inputs.py` |
| `sector/<sector>/` | `<sector>_<region>.csv` | `sector/<sector>_model_inputs.py` |
| `cost/` | `declining-capital-cost/`, `declining-intangible-cost/`, `fixed-intangible-cost/` | `sector/dcc_`, `dic_`, and `fic_model_inputs.py` |
| `exogenous/` | `exogenous-price/`, `exogenous-demand/`, `market-share-limit/` | the matching `sector/*_model_inputs.py` |
| `policy/<policy>/` | `<policy>_<region>.csv` | **made by hand** until the policy stage exists: keep a copy |
| `calibration/target/<sector>/` | Calibration targets | `sector/<sector>_calibration.py` |
| `calibration/fitted/<sector>/` | `fitted_fics/`, `fitted_lifetimes/`, `extended_fics/`, and `logs/fit_summary.csv` | the calibration notebooks (`src/CIMS/calibration/`) |

The optimizer logs go to `results/logs/calibration/`, outside the data folder.
