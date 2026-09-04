"""
Run all source pipeline scripts in the correct order.

Usage:
    python -m CIMS.data_processing.run_all_source
    python -m CIMS.data_processing.run_all_source --stop-on-error
"""

import os
import subprocess
import sys
import time

PACKAGE = "CIMS.data_processing.source"
# Explicit order where dependencies matter (energy_prices before multipliers).
# All other scripts are independent and run after.
MODULES = [
    # Currency reference tables (GDP deflators and exchange rates)
    f"{PACKAGE}.deflator_exchange.deflator_exchange",
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
    f"{PACKAGE}.nrcan.ceud.commercial.commercial",
    f"{PACKAGE}.nrcan.ceud.transportation_passenger.transportation_passenger",
    f"{PACKAGE}.nrcan.ceud.transportation_freight.transportation_freight",
    # CER RESD demand data
    f"{PACKAGE}.cer.cer_resd_demand",
    # Statistics Canada macro drivers
    f"{PACKAGE}.stats_can.pop_gdp",
]

def main() -> int:
    STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    results = []

    print(f"{BOLD}Running {len(MODULES)} source scripts{RESET}\n")

    for module in MODULES:
        label = module[len(PACKAGE) + 1:]
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
