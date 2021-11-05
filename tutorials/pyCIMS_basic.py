# Run this from the the /pycims_prototype/ directory
import pyCIMS
from pprint import pprint

file = 'pycims_prototype/model_descriptions/pyCIMS_model_description_ALL_value.xlsb'

my_validator = pyCIMS.ModelValidator(infile=file,
                                     sheet_map={'model': 'Lists',
                                                'default_tech': 'Technology_Node templates'},
                                     node_col='Node')
my_validator.validate(raise_warnings=False)
pprint(my_validator.warnings)

# Create a model description reader
my_reader = pyCIMS.ModelReader(infile=file,
                               sheet_map={'model': 'Lists',
                                          'incompatible': 'Incompatible',
                                          'default_tech': 'Technology_Node templates'},
                               node_col='Node')

# Create a model from the reader
my_model = pyCIMS.Model(my_reader)

# Run the Model
my_model.run(show_warnings=False, max_iterations=5)

# Log the result
pyCIMS.log_model(model=my_model, output_file="ALL_values_log.csv")