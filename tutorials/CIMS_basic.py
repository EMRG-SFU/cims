import CIMS
import numpy as np
from pathlib import Path
import pandas as pd
from os import path
from datetime import datetime

import config # A config.py file in the tutorials directory

#---------------------------------------------------
# Download Compatible Model
#---------------------------------------------------
CIMS.download_models(
    out_path='cims/models/',
    token=config.TOKEN,
    model_name="all_models",
    overwrite=False,
    unzip=True
)


#---------------------------------------------------
# Set Up Scenario
#---------------------------------------------------
region_list = [
    'CIMS',
    'CAN',
    'BC',
]

year_list = [
    ### Historical ###
    2000,
    2005,
    2010,
    2015,
    2020,
    ### Forecast ###
    # 2025,
    # 2030,
    # 2035,
    # 2040,
    # 2045,
    # 2050,
]

sector_list = [
    'Coal Mining',
    'Natural Gas Production',
    'Petroleum Crude',
    'Mining',
    'Electricity',
    'Biodiesel',
    'Ethanol',
    'Hydrogen',
    'Petroleum Refining',
    'Industrial Minerals',
    'Iron and Steel',
    'Metal Smelting',
    'Chemical Products',
    'Pulp and Paper',
    'Light Industrial',
    'Residential',
    'Commercial',
    'Transportation Personal',
    'Transportation Freight',
    'Waste',
    'Agriculture',
]

col_list1 = [
    'Branch',
    'Region',
    'Sector',
    'Technology',
    'Parameter',
    'Context',
    'Sub_Context',
    'Target',
    'Source',
    'Unit',
]
col_list2 = pd.Series(np.arange(2000,2051,5), dtype='string').tolist()
col_list = col_list1 + col_list2

### Base model and standard files below are required for the model to run
model_path = 'cims-models/model'    # Path to location of model file directories

### Default values file
default_values_path = 'cims-models/defaults/CIMS_defaults_Parameters.csv'

### Model Files
model_req = [
    'CIMS_DCC',
    'CIMS_calibration_FIC_DIC',
    'CIMS_calibration_MS',    
    ]
model_optional = [
    'CIMS_exogenous prices',  # needed for correct calibration of historical years
    ]
update_files = {model_path: model_req + model_optional}

# Base Files
base_model = 'CIMS_base'
print(f'Loading base model: {base_model}')
load_paths = []
for reg in region_list:
    load_path = f'{model_path}/{base_model}/{base_model}_{reg}.csv'
    if path.exists(load_path):
        print(f'\t{reg} - loaded')
        load_paths.append(load_path)
    else:
        print(f'\t{reg} - file does not exist')


# Update Files
print('Loading model updates files:')
if bool(update_files):
    update_paths = []
    for dir, file in update_files.items():
        for file in update_files[dir]:
            print(file)
            for reg in region_list:
                update_path = f'{dir}/{file}/{file}_{reg}.csv'
                if path.exists(update_path):
                    print(f'\t{reg} - loaded')
                    update_paths.append(update_path)
                else:
                    print(f'\t{reg} - file does not exist')
else:
    print('None')


#---------------------------------------------------
# Validate Model
#---------------------------------------------------
model_validator = CIMS.ModelValidator(
            csv_file_paths = load_paths,
            col_list = col_list,
            year_list = year_list,
            sector_list = sector_list,
            scenario_files = update_paths,
            default_values_csv_path = default_values_path,
            )
print("Validating the model ...")
model_validator.validate(verbose=False)


#---------------------------------------------------
# Build Base Model
#---------------------------------------------------
print("Building the model ...")
model_reader = CIMS.ModelReader(
            csv_file_paths = load_paths,
            col_list = col_list,
            year_list = year_list,
            sector_list = sector_list,
            default_values_csv_path = default_values_path,
            )
model = CIMS.Model(model_reader)


#---------------------------------------------------
# Update Model
#---------------------------------------------------
model_reader = CIMS.ScenarioReader(
            csv_file_paths = update_paths,
            col_list = col_list,
            year_list = year_list,
            sector_list = sector_list,
            )
model = model.update(model_reader)


#---------------------------------------------------
# Run Model
#---------------------------------------------------
print("Running the model ...")
model.run(equilibrium_threshold=0.05, max_iterations=10, show_warnings=False, print_eq=True)


#---------------------------------------------------
# Export Model Results
#---------------------------------------------------
print("Exporting model results ...")
def get_active_branch_name():
    head_dir = Path("cims") / ".git" / "HEAD"
    with head_dir.open("r") as f: content = f.read().splitlines()
    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]

output_file_name = f"{get_active_branch_name()}-{datetime.today().strftime('%Y-%m-%d')}.csv"
df = CIMS.log_model(model=model, output_file=f"results/{output_file_name}")
