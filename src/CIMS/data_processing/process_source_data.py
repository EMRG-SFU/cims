"""
Run all source pipeline scripts in the correct order.

Usage:
    python -m CIMS.data_processing.process_source_data
    python -m CIMS.data_processing.process_source_data --stop-on-error
    python -m CIMS.data_processing.process_source_data --use-existing-technology-database
    python -m CIMS.data_processing.process_source_data --strict

--use-existing-technology-database skips the technology database build and
uses the tables already in processed_data/technology_database/ (for example
one built from the tech_data repository's authoring copy). The run stops if
those tables aren't there.
"""

import os
import subprocess
import sys
import time

from CIMS.data_processing.common import paths
from CIMS.data_processing.utils.controls_conversions import BASE_PATH

PACKAGE = "CIMS.data_processing.source"
TECHDB_BUILD = "CIMS.data_processing.technology_database.build"
# Tables the steam code (techdb_steam) reads from the technology database build
TECHDB_TABLES = ["archetype-parameter.csv", "archetype-alternative-carrier.csv", "archetype-flow.csv"]
# Explicit order where dependencies matter (energy_prices before multipliers).
# All other scripts are independent and run after.
MODULES = [
    # Currency reference tables (GDP deflators and exchange rates)
    f"{PACKAGE}.deflator_exchange.deflator_exchange",
    # CER source files: carbon-free end-use prices and history back to 2000.
    # Needs the deflator tables; energy_prices, multipliers and pop_gdp read its output.
    f"{PACKAGE}.cer.cer_source_files",
    # ECCC GHG inventory
    f"{PACKAGE}.eccc.nir.nir_to_cims",
    f"{PACKAGE}.eccc.nir.nir_crosswalk_tables_cims",
    # Activity drivers
    f"{PACKAGE}.activity.emissions_drivers",
    f"{PACKAGE}.activity.electricity",
    f"{PACKAGE}.activity.light_industrial",
    f"{PACKAGE}.activity.petroleum_refining",
    f"{PACKAGE}.activity.coal_mining",
    f"{PACKAGE}.activity.oil_production",
    f"{PACKAGE}.activity.gas_production",
    f"{PACKAGE}.activity.heavy_industry",
    # Emission factors
    f"{PACKAGE}.emission_factors.emission_factors",
    # Energy prices — multipliers imports prices directly, so run prices
    # first to also produce the intermediate processed_data output.
    f"{PACKAGE}.energy_prices.energy_prices",
    f"{PACKAGE}.energy_prices.energy_price_multipliers",
    # NRCan CEUD
    f"{PACKAGE}.nrcan.ceud.residential.residential",
    f"{PACKAGE}.nrcan.ceud.residential.residential_heating_intensity",
    f"{PACKAGE}.nrcan.ceud.commercial.commercial",
    f"{PACKAGE}.nrcan.ceud.transportation_passenger.transportation_passenger",
    f"{PACKAGE}.nrcan.ceud.transportation_freight.transportation_freight",
    # CER RESD demand data
    f"{PACKAGE}.cer.cer_resd_demand",
    # EIA MECS steam shares (used by sector calibration.py via mecs_steam_calibration)
    f"{PACKAGE}.eia.mecs_steam_shares",
    # Statistics Canada macro drivers
    f"{PACKAGE}.stats_can.pop_gdp",
    # StatCan biomass (SPL and wood waste) for pulp and paper / light industrial
    f"{PACKAGE}.stats_can.biomass_demand",
    # StatCan refinery fuel (own + purchased) for petroleum refining
    f"{PACKAGE}.stats_can.refinery_fuel",
    # Technology database build (processed_data/technology_database/). The
    # steam code (techdb_steam) reads its archetype tables in the calibration
    # and model-input stages. It has its own currency tables, so it can run
    # anywhere in this list; it runs here, next to the steam stage it serves.
    TECHDB_BUILD,
    # Steam fuel targets for the six steam sectors (CER + MECS + the two
    # StatCan tables above). Read by steam_chp_shares and techdb_steam in the
    # calibration and model-input stages, so it must run before both.
    f"{PACKAGE}.steam.steam_fuel_targets",
]

def main() -> int:
    STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]
    USE_EXISTING_TECHDB = "--use-existing-technology-database" in sys.argv[1:]
    paths.announce(strict=True if "--strict" in sys.argv[1:] else None)

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    results = []

    modules = MODULES
    if USE_EXISTING_TECHDB:
        techdb_dir = BASE_PATH / "processed_data" / "technology_database"
        missing = [t for t in TECHDB_TABLES if not (techdb_dir / t).exists()]
        if missing:
            print(f"{RED}--use-existing-technology-database: {techdb_dir} lacks "
                  f"{', '.join(missing)}.{RESET} Build it, or run without the option.")
            return 1
        modules = [m for m in MODULES if m != TECHDB_BUILD]
        print(f"Using the existing technology database build in {techdb_dir}\n")

    print(f"{BOLD}Running {len(modules)} source scripts{RESET}\n")

    for module in modules:
        label = module.removeprefix(f"{PACKAGE}.").removeprefix("CIMS.data_processing.")
        print(f"  {label} ... ", end="", flush=True)
        t0 = time.monotonic()

        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env={**os.environ, 'PYTHONUTF8': '1'},
        )

        elapsed = time.monotonic() - t0
        ok = result.returncode == 0

        if ok:
            print(f"{GREEN}ok{RESET} ({elapsed:.1f}s)")
        else:
            print(f"{RED}FAILED{RESET} ({elapsed:.1f}s)")
            # Show only the last 30 lines of stdout to surface the traceback
            # without flooding the terminal with normal script output.
            if result.stdout.strip():
                tail = result.stdout.rstrip().splitlines()[-30:]
                print("\n".join(tail))
            if result.stderr.strip():
                print(result.stderr.rstrip())

        results.append((label, ok, elapsed))

        if not ok and STOP_ON_ERROR:
            print(f"\n{RED}Stopped after first failure.{RESET}")
            break

    # Summary
    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n{BOLD}Results: {GREEN}{len(passed)} passed{RESET}{BOLD}, "
          f"{RED if failed else ''}{len(failed)} failed{RESET}{BOLD} "
          f"({sum(r[2] for r in results):.1f}s total){RESET}")

    if failed:
        print(f"\n{RED}Failed scripts:{RESET}")
        for label, _, _ in failed:
            print(f"  {label}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
