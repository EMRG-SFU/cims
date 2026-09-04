"""
Run all sector calibration scripts.

Usage:
    python -m CIMS.data_processing.run_all_calibration
    python -m CIMS.data_processing.run_all_calibration --stop-on-error
"""

import os
import subprocess
import sys
import time

PACKAGE = "CIMS.data_processing.sector"
MODULES = [
    f"{PACKAGE}.agriculture_calibration",
    f"{PACKAGE}.biodiesel_calibration",
    f"{PACKAGE}.chemical_products_calibration",
    f"{PACKAGE}.coal_mining_calibration",
    f"{PACKAGE}.commercial_calibration",
    f"{PACKAGE}.construction_calibration",
    f"{PACKAGE}.electricity_calibration",
    f"{PACKAGE}.ethanol_calibration",
    f"{PACKAGE}.forestry_calibration",
    f"{PACKAGE}.hydrogen_calibration",
    f"{PACKAGE}.industrial_minerals_calibration",
    f"{PACKAGE}.iron_and_steel_calibration",
    f"{PACKAGE}.light_industrial_calibration",
    f"{PACKAGE}.metal_smelting_calibration",
    f"{PACKAGE}.mining_calibration",
    f"{PACKAGE}.natural_gas_calibration",
    f"{PACKAGE}.petroleum_crude_calibration",
    f"{PACKAGE}.petroleum_refining_calibration",
    f"{PACKAGE}.pulp_and_paper_calibration",
    f"{PACKAGE}.residential_calibration",
    f"{PACKAGE}.transportation_passenger_calibration",
    f"{PACKAGE}.transportation_freight_calibration",
    f"{PACKAGE}.waste_calibration",
]

def main() -> int:
    STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    results = []

    print(f"{BOLD}Running {len(MODULES)} calibration scripts{RESET}\n")

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
