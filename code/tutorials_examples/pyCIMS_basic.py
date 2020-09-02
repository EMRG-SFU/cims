# Run this from the the /pycims_prototype/ directory
import pyCIMS
from pprint import pprint

file = 'pycims_prototype/model_descriptions/pyCIMS_model_description_Alberta_Validated-Jillian-July 7.xlsm'

my_validator = pyCIMS.ModelValidator(file)
my_validator.validate(raise_warnings=False)
pprint(my_validator.warnings)

# Create a model description reader
my_reader = pyCIMS.ModelReader(infile=file,
                               sheet_map={'model': 'Model',
                                          'incompatible': 'Incompatible',
                                          'default_tech': 'Technologies'},
                               node_col='Node')

# Create a model from the reader
my_model = pyCIMS.Model(my_reader)

# Run the Model
my_model.run(show_warnings=False, max_iterations=5)

