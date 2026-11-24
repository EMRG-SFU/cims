# Natural Gas Market as an optional sector

Branch `natural-gas-market`, 2026-11-18; rebased onto `emh-port` and split on 2026-11-19.

**Status:** the market is **off by default** in `scenarios/Reference.py`. With it off, Natural Gas (and Petroleum Crude, which needs it) runs for any set of regions. Turning it on relies on the #512 sector-check exemption (#631), which is merged and on this branch.

## How the switch works

- **Market off** (`'Natural Gas'` in `sector_list`, `'Natural Gas Market'` not): each region's `CIMS.CAN.<r>.Natural Gas.Production` requests its own `Supply`, `Transmission`, and `Distribution` at 1.0. Any subset of regions can run.
- **Market on** (both sectors in `sector_list`; needs the validator exemption): `natural_gas_market/natural_gas_market_can.csv` loads after the `natural_gas` files and overrides each region's `Production` to request `CIMS.CAN.Natural Gas Market` at 1.0 and its own three stages at 0. Every region must run.
- **Supply shares:** the market's provincial `Supply` shares are each region's share of national production requests (`CIMS.CAN.<r>` → `CIMS.CAN.<r>.Natural Gas`), year by year. `Transmission` and `Distribution` keep their fixed-data shares.
- **Prices:** consumers pay the `CIMS.Generic Fuels.Natural Gas` price either way. The market never set it.

The generator is `src/CIMS/data_processing/sector/natural_gas_model_inputs.py`; its module docstring has the details.

## To explore: declare overlay sectors instead of inferring them

The market's switch rows are `service_request` rows on regional `Production` nodes, tagged `Sector = Natural Gas Market`. The #512 check `branch_path_missing_sector` (an error, so it blocks `construct_graph`) flags them, because the Sector isn't part of the Branch.

The exemption (#631) accepts a flagged `service_request` row when either:

1. its Sector is part of its Target path, or
2. its Sector is a national node (`CIMS.<country>.<sector>`).

That works with today's files but weakens the check:

- Rule 1 hides a copied row whose Sector was taken from its target by mistake.
- Rule 2 accepts a `Natural Gas Market` tag on any branch.
- The list of national nodes is worked out from branch paths, so it also includes region codes (harmless, but indirect).
- There's no test file for the exemption.

**Option to explore:** declare overlay sectors explicitly, for example with an `overlay` flag in the sector list (`config/sector.csv` in Phase 4 of the data reorganization plan). The check would then accept only `service_request` rows tagged with a declared overlay sector, and drop both rules above.

- **Owner:** Jillian (author of #512). The exemption is a separate pull request into `staging`, so she can take it, replace it with the declared approach, or fix the check her own way.
- **Also:** add a test that a wrong-sector request inside a region is still flagged, and that a declared overlay sector's rows pass.
