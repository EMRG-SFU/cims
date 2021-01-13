# Run this from the the /pycims_prototype/ directory
import pyCIMS
from pprint import pprint

file = 'pycims_prototype/model_descriptions/pyCIMS_model_description_Alberta_Test.xlsb'

my_validator = pyCIMS.ModelValidator(file)
my_validator.validate(raise_warnings=False)
pprint(my_validator.warnings)

# Create a model description reader
my_reader = pyCIMS.ModelReader(infile=file,
                               sheet_map={'model': 'Model',
                                          'incompatible': 'Incompatible',
                                          'default_tech': 'Technology_Node templates'},
                               node_col='Node')

# Create a model from the reader
my_model = pyCIMS.Model(my_reader)

# Run the Model
my_model.run(show_warnings=False, max_iterations=5)

# Log the result
pyCIMS.log_model(model=my_model, output_file="Alberta_Test_Log.csv")