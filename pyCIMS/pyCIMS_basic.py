# Run this from the the /pycims_prototype/ directory

import pyCIMS

file = 'pycims_prototype/pyCIMS_model_description.xlsm'

mv = pyCIMS.ModelValidator(file)
mv.validate(verbose=True)
pycims_reader = pyCIMS.Reader(infile=file,
                              sheet_map={'model': 'Model', 'incompatible': 'Incompatible'})
my_model = pyCIMS.Model(pycims_reader)