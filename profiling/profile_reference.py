"""cProfile harness for the CIMS Reference scenario.

Mirrors scenarios/Reference.py cell-for-cell, but as a plain script so it can
run headless under cProfile.
"""
import cProfile
import pstats
import sys
import time
import os

REPO = "/home/claude/cims"
OUT = "/home/claude/prof"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import CIMS  # noqa: E402

PHASES = []


def phase(name, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    PHASES.append((name, dt))
    print(f"[phase] {name}: {dt:.2f}s", flush=True)
    return result


# ---------------------------------------------------------------- file paths
model_path = 'data/model_inputs/model'
base_model = 'CIMS_base'
default_path = 'data/model_inputs/defaults/defaults_Parameters.csv'
list_path = 'data/model_inputs/defaults/defaults_Lists.csv'
deflator_path = 'data/processed_data/deflator_exchange/currency_deflator.csv'
exchange_path = 'data/processed_data/deflator_exchange/currency_exchange.csv'

update_files = {}

sector_req = [
    'fuels', 'transmission', 'coal_mining', 'natural_gas', 'petroleum_crude',
    'petroleum_refining', 'electricity', 'biodiesel', 'ethanol', 'hydrogen',
    'mining', 'industrial_minerals', 'iron_and_steel', 'metal_smelting',
    'chemical_products', 'pulp_and_paper', 'light_industrial', 'construction',
    'residential', 'commercial', 'transportation_passenger',
    'transportation_freight', 'waste', 'agriculture', 'forestry',
]
update_files.setdefault(model_path, []).extend(sector_req)

model_req = ['DCC', 'DIC', 'FIC', 'market_share_limits']
update_files.setdefault(model_path, []).extend(model_req)

# ------------------------------------------------------------ scenario setup
target_units = {"currency": "CAD", "dollar_year": 2020}

region_list = ['CIMS', 'RoW', 'CAN', 'BC']

year_list = [2000, 2005, 2010, 2015, 2020,
             2025, 2030, 2035, 2040, 2045, 2050]

sector_list = [
    'Coal Mining', 'Petroleum Refining', 'Electricity', 'Biodiesel', 'Ethanol',
    'Hydrogen', 'Mining', 'Industrial Minerals', 'Iron and Steel',
    'Metal Smelting', 'Chemical Products', 'Pulp and Paper',
    'Light Industrial', 'Construction', 'Residential', 'Commercial',
    'Transportation Passenger', 'Transportation Freight', 'Waste',
    'Agriculture', 'Forestry',
]

model_optional = ['exogenous_prices']
update_files.setdefault(model_path, []).extend(model_optional)

ref_path = 'data/model_inputs/policies/reference'
ref_policies = [
    'ref_carbon_tax', 'ref_obps_fed',
    'ref_coal_phase_out', 'ref_nuclear_ban', 'ref_nuclear_decommission',
    'ref_clean_electricity', 'ref_cer',
    'ref_incandescent_phase_out',
    'ref_ldv_zev_federal', 'ref_ldv_zev_prov',
    'ref_renewable_fuel_content_fed', 'ref_renewable_fuel_content_prov',
    'ref_waste_methane_large_sites',
]
update_files.setdefault(ref_path, []).extend(ref_policies)

scenario_name = 'Reference'
results_path = f'{OUT}/results/{scenario_name}'


def build_model():
    return CIMS.Model(
        model_path=model_path,
        base_model=base_model,
        region_list=region_list,
        update_files=update_files,
        year_list=year_list,
        sector_list=sector_list,
        default_values_csv_path=default_path,
        list_csv_path=list_path,
        target_units=target_units,
        deflator_path=deflator_path,
        exchange_path=exchange_path,
    )


def main():
    model = phase("Model.__init__ (read inputs)", build_model)
    phase("validate_files", model.validate_files)
    phase("construct_graph", model.construct_graph)
    phase("validate_graph", model.validate_graph)
    phase("model.run", lambda: model.run(
        equilibrium_threshold=0.05, max_iterations=10,
        show_warnings=False, print_eq=True))
    os.makedirs(results_path, exist_ok=True)
    phase("log_model (general)", lambda: CIMS.log_model(
        model=model,
        output_file=f"{results_path}/results_general.csv",
        parameter_file=f"{OUT}/results/results_general.txt",
        ensure_dir=True))
    phase("log_model (tech)", lambda: CIMS.log_model(
        model=model,
        output_file=f"{results_path}/results_tech.csv",
        parameter_file=f"{OUT}/results/results_tech.txt",
        ensure_dir=True))
    return model


if __name__ == "__main__":
    prof = cProfile.Profile()
    wall0 = time.perf_counter()
    prof.enable()
    try:
        main()
    finally:
        prof.disable()
        wall = time.perf_counter() - wall0
        prof.dump_stats(f"{OUT}/reference.prof")
        print("\n===== PHASE SUMMARY =====", flush=True)
        for name, dt in PHASES:
            print(f"{dt:8.2f}s  {100*dt/wall:5.1f}%  {name}")
        print(f"{wall:8.2f}s  100.0%  TOTAL (wall, under profiler)")
