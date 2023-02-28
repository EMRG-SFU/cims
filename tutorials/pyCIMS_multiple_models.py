# Multiple Models
import pyCIMS

reference_model = 'pycims_prototype/models/multiple_models/pyCIMS_model_reference.xlsb'
scenario_model = 'pycims_prototype/models/multiple_models/pyCIMS_model_scenario1.xlsb'

# ***************************************************
# Design Option 1
# ***************************************************
# Reference
# =========
# Create reference model reader
reference_reader = pyCIMS.ModelReader(infile=reference_model,
                                      sheet_map={'model': 'Lists',
                                                 'incompatible': 'Incompatible',
                                                 'default_param': 'Default values'},
                                      node_col='Node')
# create reference model
reference_model = pyCIMS.Model(reference_reader)

# Scenario 1
# ==========
# create scenario reader
scenario_1_reader = pyCIMS.ModelReader(infile=scenario_model,
                                       sheet_map={'model': 'Lists'},
                                       node_col='Node')


# Update reference model with scenario reader
scenario_1_model = reference_model.update(scenario_1_reader)