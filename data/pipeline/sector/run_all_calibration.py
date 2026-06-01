"""
Run all sector calibration scripts.

Usage:
    python pipeline/sector/run_all_calibration.py
    python pipeline/sector/run_all_calibration.py --stop-on-error
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SECTOR_DIR = Path(__file__).parent

SCRIPTS = [
    SECTOR_DIR / "agriculture/calibration.py",
    SECTOR_DIR / "biodiesel/calibration.py",
    SECTOR_DIR / "chemical products/calibration.py",
    SECTOR_DIR / "coal mining/calibration.py",
    SECTOR_DIR / "commercial/calibration.py",
    SECTOR_DIR / "construction/calibration.py",
    SECTOR_DIR / "electricity/calibration.py",
    SECTOR_DIR / "ethanol/calibration.py",
    SECTOR_DIR / "forestry/calibration.py",
    SECTOR_DIR / "hydrogen/calibration.py",
    SECTOR_DIR / "industrial minerals/calibration.py",
    SECTOR_DIR / "iron and steel/calibration.py",
    SECTOR_DIR / "light industrial/calibration.py",
    SECTOR_DIR / "metal smelting/calibration.py",
    SECTOR_DIR / "mining/calibration.py",
    SECTOR_DIR / "natural gas/calibration.py",
    SECTOR_DIR / "petroleum crude/calibration.py",
    SECTOR_DIR / "petroleum refining/calibration.py",
    SECTOR_DIR / "pulp and paper/calibration.py",
    SECTOR_DIR / "residential/calibration.py",
    SECTOR_DIR / "waste/calibration.py",
]

STOP_ON_ERROR = "--stop-on-error" in sys.argv[1:]

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

results = []

print(f"{BOLD}Running {len(SCRIPTS)} calibration scripts{RESET}\n")

for script in SCRIPTS:
    label = script.relative_to(SECTOR_DIR)
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
