"""
Run all sector pipeline scripts in the correct order.

Usage:
    python -m CIMS.data_processing.run_all_model_inputs
    python -m CIMS.data_processing.run_all_model_inputs --stop-on-error
"""

import os
import subprocess
import sys
import time

PACKAGE = "CIMS.data_processing.sector"
MODULES = [
    f"{PACKAGE}.agriculture_model_inputs",
    f"{PACKAGE}.biodiesel_model_inputs",
    f"{PACKAGE}.chemical_products_model_inputs",
    f"{PACKAGE}.cims_base_model_inputs",
    f"{PACKAGE}.coal_mining_model_inputs",
    f"{PACKAGE}.commercial_model_inputs",
    f"{PACKAGE}.construction_model_inputs",
    f"{PACKAGE}.dcc_model_inputs",
    f"{PACKAGE}.dic_model_inputs",
    f"{PACKAGE}.electricity_model_inputs",
    f"{PACKAGE}.ethanol_model_inputs",
    f"{PACKAGE}.exogenous_demand_model_inputs",
    f"{PACKAGE}.exogenous_prices_model_inputs",
    f"{PACKAGE}.fic_model_inputs",
    f"{PACKAGE}.forestry_model_inputs",
    f"{PACKAGE}.fuels_model_inputs",
    f"{PACKAGE}.hydrogen_model_inputs",
    f"{PACKAGE}.industrial_minerals_model_inputs",
    f"{PACKAGE}.iron_and_steel_model_inputs",
    f"{PACKAGE}.light_industrial_model_inputs",
    f"{PACKAGE}.market_share_limits_model_inputs",
    f"{PACKAGE}.metal_smelting_model_inputs",
    f"{PACKAGE}.mining_model_inputs",
    f"{PACKAGE}.natural_gas_model_inputs",
    f"{PACKAGE}.petroleum_crude_model_inputs",
    f"{PACKAGE}.petroleum_refining_model_inputs",
    f"{PACKAGE}.pulp_and_paper_model_inputs",
    f"{PACKAGE}.residential_model_inputs",
    f"{PACKAGE}.transmission_model_inputs",
    f"{PACKAGE}.transportation_passenger_model_inputs",
    f"{PACKAGE}.transportation_freight_model_inputs",
    f"{PACKAGE}.waste_model_inputs",
]

def main() -> int:
    STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    results = []

    print(f"{BOLD}Running {len(MODULES)} sector scripts{RESET}\n")

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
