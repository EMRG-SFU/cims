# model/

Everything CIMS reads (conventions §2). The build writes it, and nothing in it is committed
apart from this README. `scenarios/Reference.py` lists which folders a run loads.

| Path | Contents | Written by |
| - | - | - |
| `base/` | `cims-base_<region>.csv` | `sector/cims_base_model_inputs.py` |
| `sector/<sector>/` | `<sector>_<region>.csv` | `sector/<sector>_model_inputs.py` |
| `cost/` | `declining-capital-cost/`, `declining-intangible-cost/`, `fixed-intangible-cost/` | `sector/dcc_`, `dic_`, and `fic_model_inputs.py` |
| `exogenous/` | `exogenous-price/`, `exogenous-demand/`, `market-share-limit/` | the matching `sector/*_model_inputs.py` |
| `policy/<policy>/` | `<policy>_<region>.csv` | **made by hand** until the policy stage exists: keep a copy |
| `calibration/target/<sector>/` | Calibration targets | `sector/<sector>_calibration.py` |
| `calibration/fitted/<sector>/` | `fitted_fics/`, `fitted_lifetimes/`, `extended_fics/`, and `logs/fit_summary.csv` | the calibration notebooks (`src/CIMS/calibration/`) |

The optimizer logs go to `results/logs/calibration/`, outside the data folder.
