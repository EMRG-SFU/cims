
# Columns needed in output file:
# Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit, Year, Value

from Calibration.SubGraphs.graphFunctions import getDescendants 

def write_fics(model, nodeName, include_subtree = False):

    if include_subtree:
        nodes_to_process = [nodeName] + getDescendants(model, nodeName)

    pass


