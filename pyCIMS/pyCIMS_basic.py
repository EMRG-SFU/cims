# Run this from the the /pycims_prototype/ directory

import pyCIMS

file = 'pycims_prototype/pyCIMS_model_description.xlsm'

# Validate the Model Description
mv = pyCIMS.ModelValidator(file)
mv.validate(verbose=True)

# Create a model description reader
pycims_reader = pyCIMS.Reader(infile=file,
                              sheet_map={'model': 'Model', 'incompatible': 'Incompatible'})

# From the reader, create a model
my_model = pyCIMS.Model(pycims_reader)

# Access the networkx graph of the model
print(my_model.graph)
len(my_model.graph.nodes)
len(my_model.graph.edges)

# Search for a node within the graph
print(my_model.search_nodes("Alberta"))