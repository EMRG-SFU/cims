# Multiple Models
import CIMS

reference_model_file = 'pycims_prototype/models/multiple_models/CIMS_model_reference.xlsb'
scenario_model_file = 'pycims_prototype/models/multiple_models/CIMS_model_scenario1.xlsb'

# ***************************************************
# Design Option 1
# ***************************************************
# Reference
# =========
# Create reference model reader
reference_reader = CIMS.ModelReader(infile=reference_model_file,
                                      sheet_map={'model': 'Lists',
                                                 'incompatible': 'Incompatible',
                                                 'default_param': 'Default values'},
                                      node_col='Node')
# create reference model
reference_model = CIMS.Model(reference_reader)

# Scenario 1
# ==========
# create scenario reader
scenario_1_reader = CIMS.ModelReader(infile=scenario_model_file,
                                       sheet_map={'model': 'Lists'},
                                       node_col='Node')


# Update reference model with scenario reader
scenario_1_model = reference_model.update(scenario_1_reader)

scenario_1_model.run(show_warnings=False, max_iterations=5)