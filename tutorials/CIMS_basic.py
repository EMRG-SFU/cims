# Run this from the /CIMS/ directory
import CIMS
from pprint import pprint

file = 'pycims_prototype/models/CIMS_base model.xlsb'

my_validator = CIMS.ModelValidator(
    infile=file,
    sheet_map={
        'model': 'RunSheets',
        'default_param': 'Default values'},
    node_col='Node')
my_validator.validate(raise_warnings=False)
pprint(my_validator.warnings)

# Create a model description reader
my_reader = CIMS.ModelReader(
    infile=file,
    sheet_map={
        'model': 'RunSheets',
        'incompatible': 'Incompatible',
        'default_param': 'Default values'},
    node_col='Node')

# Create a model from the reader
my_model = CIMS.Model(my_reader)

# Run the Model
my_model.run(show_warnings=False, max_iterations=10, print_eq=True)

# Log the result
df = CIMS.log_model(model=my_model, output_file="log.csv")
