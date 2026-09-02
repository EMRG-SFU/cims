
# Columns needed in output file:
# Branch, Type, Region, Sector, Service, Technology, Parameter, Context, Sub_Context, Target, Source, Unit, Year, Value

import os
import re

import pandas as pd

from Calibration.SubGraphs.graphFunctions import getDescendants
from Calibration.Data.node_info import list_years, list_techs

COLUMNS = ["Branch", "Type", "Region", "Sector", "Service", "Technology", "Parameter",
           "Context", "Sub_Context", "Target", "Source", "Unit", "Year", "Value"]


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


def get_fic_file_rows(model, nodeName):
    all_years = list_years(model.graph, nodeName)
    all_techs = list_techs(model.graph, nodeName)

    rows = []
    for tt in all_techs:
        for yy in all_years:
            out_value = model.get_param('fic', nodeName, year=yy, tech=tt)

            rows.append({
                "Branch": nodeName,
                "Type": "",
                "Region": extractRegion(nodeName),
                "Sector": extractSector(nodeName),
                "Service": extractService(nodeName),
                "Technology": tt,
                "Parameter": 'fic',
                "Context": "",
                "Sub_Context": "",
                "Target": "",
                "Source": "calibration_fic_export",
                "Unit": "",
                "Year": yy,
                "Value": out_value,
            })

    return rows


def write_rows_by_region(rows, outputDir, name, nodes_written):
    """
    Write `rows` to `<outputDir>/<name>/<name>_<region>.csv`, one file per region,
    matching the layout `update_files` expects for calibration input files.

    Rows for `nodes_written` replace any existing rows for those same nodes in
    each region file, so re-running calibration on a node updates it in place
    rather than duplicating rows; rows for other nodes already in the file are
    preserved.
    """
    new_df = pd.DataFrame(rows, columns=COLUMNS)

    for region, region_df in new_df.groupby("Region"):
        region_dir = os.path.join(outputDir, name)
        os.makedirs(region_dir, exist_ok=True)
        region_file = os.path.join(region_dir, f"{name}_{region.lower()}.csv")

        if os.path.exists(region_file):
            existing_df = pd.read_csv(region_file)
            existing_df = existing_df[~existing_df["Branch"].isin(nodes_written)]
            combined_df = pd.concat([existing_df, region_df], ignore_index=True)
        else:
            combined_df = region_df

        combined_df.to_csv(region_file, index=False)


def write_fics(model, nodeName, outputDir, name='fitted_fics', include_subtree=False):

    if include_subtree:
        nodes_to_process = sorted( [nodeName] + list(getDescendants(model, nodeName)) )
    else:
        nodes_to_process = [nodeName]

    rows = []
    for node in nodes_to_process:
        rows.extend(get_fic_file_rows(model, node))

    write_rows_by_region(rows, outputDir, name, nodes_to_process)
