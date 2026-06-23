"""
Run all source pipeline scripts in the correct order.

Usage:
    python pipeline/source/run_all.py
    python pipeline/source/run_all.py --stop-on-error
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SOURCE_DIR = Path(__file__).parent

# Explicit order where dependencies matter (energy_prices before multipliers).
# All other scripts are independent and run after.
SCRIPTS = [
    # ECCC GHG inventory
    SOURCE_DIR / "eccc/nir/nir_to_cims.py",
    SOURCE_DIR / "eccc/nir/nir_crosswalk_tables_cims.py",
    # Activity drivers
    SOURCE_DIR / "activity/emissions_drivers.py",
    SOURCE_DIR / "activity/electricity.py",
    SOURCE_DIR / "activity/light_industrial.py",
    SOURCE_DIR / "activity/petroleum refining.py",
    SOURCE_DIR / "activity/coal_mining.py",
    SOURCE_DIR / "activity/oil_production.py",
    SOURCE_DIR / "activity/gas_production.py",
    SOURCE_DIR / "activity/heavy_industry.py",
    # Emission factors
    SOURCE_DIR / "emission_factors/emission_factors.py",
    # Energy prices — multipliers imports prices directly, so run prices first
    # to also produce the intermediate processed_data output.
    SOURCE_DIR / "energy_prices/energy_prices.py",
    SOURCE_DIR / "energy_prices/energy_price_multipliers.py",
    # NRCan CEUD
    SOURCE_DIR / "nrcan/ceud/residential/residential.py",
    SOURCE_DIR / "nrcan/ceud/commercial/commercial.py",
    SOURCE_DIR / "nrcan/ceud/transportation_passenger/transportation_passenger.py",
    SOURCE_DIR / "nrcan/ceud/transportation_freight/transportation_freight.py",
    # CER RESD demand data
    SOURCE_DIR / "cer/cer_resd_demand.py",
    # Statistics Canada macro drivers
    SOURCE_DIR / "stats_can/pop_gdp.py",
]

STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

results = []

print(f"{BOLD}Running {len(SCRIPTS)} source scripts{RESET}\n")

for script in SCRIPTS:
    label = script.relative_to(SOURCE_DIR)
    print(f"  {label} ... ", end="", flush=True)
    t0 = time.monotonic()

    result = subprocess.run(
        [sys.executable, str(script)],
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

sys.exit(0 if not failed else 1)
