import CIMS
from pprint import pprint
import numpy as np

import config # A config.py file in the tutorials directory


CIMS.download_models(
    out_path='cims/models/',
    token=config.TOKEN,
    model_name="all_models",
    overwrite=False,
    unzip=True
)

file = 'cims/models/all_models-v0.0.0-alpha/CIMS_base model.xlsb'

# --- Validate Model ---
my_validator = CIMS.ModelValidator(
    infile=model_file,
    sheet_map={
        'model': ['CIMS', 'CAN', 'BC'],
        'incompatible': 'Incompatible',
        'default_param': 'Default values'},
    node_col='Branch',
    root_node='CIMS'
)
# my_validator.validate(raise_warnings=False)
# pprint(my_validator.warnings)

# --- Build Model ---
col_list1 = ['Branch', 'Sector', 'Technology', 'Parameter', 'Context', 'Sub_Context',
             'Target', 'Source', 'Unit']
year_columns = list(np.arange(2000, 2051, 5))
col_list = col_list1 + year_columns
my_reader = CIMS.ModelReader(
    infile=model_file,
    sheet_map={
        'model': ['CIMS', 'CAN', 'BC'],
        'incompatible': 'Incompatible',
        'default_param': 'Default values'},
    node_col='Branch',
    year_list=year_columns,
    col_list=col_list,
    sector_list=[],
    root_node='CIMS'
)

base_model = CIMS.Model(my_reader)
model = base_model

# --- Run Model ---
model.run(show_warnings=False, max_iterations=10, print_eq=False)

# --- Visualize Results ---
model.visualize_prices_change_over_time(
    out_file='./fuel_prices_over_years.png')

model.visualize_price_comparison_with_benchmark(
    benchmark_file='./benchmark/prices.csv',
    out_file='./price_comparison_to_benchmark.png')

# --- Export Model to CSV ---
df = CIMS.log_model(model=model, output_file="log.csv")
