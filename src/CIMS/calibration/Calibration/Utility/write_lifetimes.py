
# Columns needed in output file:
# Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit, Year, Value

from Calibration.SubGraphs.graphFunctions import getDescendants
from Calibration.Data.node_info import list_years, list_techs
from Calibration.Utility.write_fics import extractRegion, extractSector, extractService, write_rows_by_region


def get_lifetime_file_rows(model, nodeName):
    all_years = list_years(model.graph, nodeName)
    all_techs = list_techs(model.graph, nodeName)

    rows = []
    for tt in all_techs:
        # lifetime is a single value per technology, applied to every year
        # identically, so only one row is written, with the Year left blank.
        out_value = model.get_param('lifetime', nodeName, year=all_years[0], tech=tt)

        rows.append({
            "Branch": nodeName,
            "Type": "",
            "Region": extractRegion(nodeName),
            "Sector": extractSector(nodeName),
            "Service": extractService(nodeName),
            "Technology": tt,
            "Parameter": 'lifetime',
            "Context": "",
            "Sub_Context": "",
            "Target": "",
            "Source": "calibration_lifetime_export",
            "Unit": "",
            "Year": "",
            "Value": out_value,
        })

    return rows


def write_lifetimes(model, nodeName, outputDir, name='fitted_lifetimes', include_subtree=False):

    if include_subtree:
        nodes_to_process = sorted( [nodeName] + list(getDescendants(model, nodeName)) )
    else:
        nodes_to_process = [nodeName]

    rows = []
    for node in nodes_to_process:
        rows.extend(get_lifetime_file_rows(model, node))

    write_rows_by_region(rows, outputDir, name, nodes_to_process)
