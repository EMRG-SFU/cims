"""
Run all sector pipeline scripts in the correct order.

Run process_source_data first: the six steam sectors read
processed_data/steam/steam_fuel_targets.csv and the technology database
build output (processed_data/technology_database/), which it writes last.

Usage:
    python -m CIMS.data_processing.run_all_model_inputs
    python -m CIMS.data_processing.run_all_model_inputs --stop-on-error
    python -m CIMS.data_processing.run_all_model_inputs --from natural_gas
    python -m CIMS.data_processing.run_all_model_inputs --only natural_gas residential
    python -m CIMS.data_processing.run_all_model_inputs --strict

--from NAME    Start at the step whose name starts with NAME (for example after a
               failure), and run every step after it.
--only NAME..  Run only the named steps (names may be shortened the same way).

Step names are the module names without "_model_inputs" (natural_gas, fic, ...).
"""

import argparse
import os
import subprocess
import sys
import time

from CIMS.data_processing.common import paths

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

def _step_name(module: str) -> str:
    return module.rsplit(".", 1)[-1].removesuffix("_model_inputs")


def _match(name: str, steps: list[str]) -> str:
    """Return the one module whose step name equals, or else starts with, name."""
    exact = [m for m in steps if _step_name(m) == name]
    if exact:
        return exact[0]
    hits = [m for m in steps if _step_name(m).startswith(name)]
    if len(hits) != 1:
        options = ", ".join(_step_name(m) for m in (hits or steps))
        raise SystemExit(f"Step {name!r} matches {len(hits)} steps. Choose from: {options}")
    return hits[0]


def select_modules(argv: list[str]) -> tuple[list[str], bool]:
    parser = argparse.ArgumentParser(description="Run the sector model-input scripts.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="stop if the data root is not inside the current folder")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", metavar="NAME",
                       help="start at this step and run every step after it")
    group.add_argument("--only", nargs="+", metavar="NAME",
                       help="run only these steps")
    args = parser.parse_args(argv)

    if args.start:
        chosen = MODULES[MODULES.index(_match(args.start, MODULES)):]
    elif args.only:
        picked = {_match(n, MODULES) for n in args.only}
        chosen = [m for m in MODULES if m in picked]
    else:
        chosen = MODULES
    return chosen, args.stop_on_error


def main() -> int:
    modules, STOP_ON_ERROR = select_modules(sys.argv[1:])
    paths.announce(strict=True if "--strict" in sys.argv[1:] else None)

    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    results = []

    print(f"{BOLD}Running {len(modules)} of {len(MODULES)} sector scripts{RESET}\n")

    for module in modules:
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
            print(f"\n{RED}Stopped after first failure. Resume with: "
                  f"--from {_step_name(module)}{RESET}")
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
        names = " ".join(label.removesuffix("_model_inputs") for label, _, _ in failed)
        print(f"\nRerun just these with: --only {names}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
