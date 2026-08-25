
# Columns needed in output file:
# Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit, Year, Value

import re

from Calibration.SubGraphs.graphFunctions import getDescendants 
from Calibration.Data.node_info import list_years, list_techs

def extractRegion(node):
    pattern = r"^CIMS\.CAN\.([A-Za-z]{2})(\.|$)"
    try:
        res = re.match(pattern, node)
        return res.groups()[0]
    except AttributeError as ee:
        return ""

def extractSector(node):
    pattern = r'^CIMS\.CAN\.([A-Za-z]{2})\.(.*?)(\.|$)'
    try:
        res = re.match(pattern, node)
        return res.groups()[1]
    except AttributeError as ee:
        return ""

def extractService(node):
    pattern = r'^CIMS\.CAN\.([A-Za-z]{2})\.(.*?)\.(.*?\.)*(.*?)$'
    try:
        res = re.match(pattern, node)
        return res.groups()[3]
    except AttributeError as ee:
        return ''


def get_fic_file_strs(model, nodeName):
    all_years = list_years(model.graph, nodeName)
    all_techs = list_techs(model.graph, nodeName)

    fic_str_list = []
    for tt in all_techs:
        for yy in all_years:
            out_value = model.get_param('fic', nodeName, year=yy, tech=tt)
            
            str_to_write = [
                nodeName,                    # Branch
                "",                          # Type
                extractRegion(nodeName),     # Region
                extractSector(nodeName),     # Sector
                extractService(nodeName),    # Service
                tt,                          # Technology
                'fic',                       # Parameter
                "",                          # Context
                "",                          # Sub_Context
                "",                          # Target
                "calibration_fic_export",    # Source
                "",                          # Unit
                yy,                          # Year
                out_value                    # Value
            ]
            fic_str_list.append(",".join([str(a) for a in str_to_write]))

    return fic_str_list


def write_fics(model, nodeName, outputFile, include_subtree = False):

    if include_subtree:
        nodes_to_process = sorted( [nodeName] + list(getDescendants(model, nodeName)) )
    else:
        nodes_to_process = [nodeName]

    print(nodes_to_process)

    with open(outputFile, 'w') as _f:
        _f.write("Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit, Year, Value\n")
        for node in nodes_to_process:
            
            outputStrs = get_fic_file_strs(model, node)
            for s in outputStrs:
                _f.write(s+"\n")



