"""
Where calibration outputs go, and a record of how well each fit did.

Reference.py reads fitted values from one subfolder per sector:

    data/model_inputs/calibration_outputs/<sector>/fitted_fics/
    data/model_inputs/calibration_outputs/<sector>/fitted_lifetimes/

for every sector listed in its `calibration_output_sectors`. `sector_folder`
maps a node to that subfolder name from its branch (`CIMS.CAN.<region>.<Sector>...`),
so the notebooks only need the root folder and every node lands where
Reference.py looks for it.

`fit_summary.csv` (one per sector, under `<sector>/logs/`) records the L1
market-share error of every fit, so a later stage can tell whether a full
Reference.py run kept the fit or whether the node needs refitting.
"""

import os
import re
import time
import warnings

import pandas as pd

from Calibration.Data.market_share import get_marketShare_both_dict
from Calibration.Utility.write_fics import write_fics
from Calibration.Utility.write_lifetimes import write_lifetimes

# Sector names in the model whose output folder is not just the snake-cased name.
SECTOR_FOLDER_OVERRIDES = {
    "Fuel Blends": "fuels",
}

SUMMARY_COLUMNS = ["Branch", "Stage", "Target", "L1_before", "L1_after", "Time"]


def sector_folder(nodeName):
    """`CIMS.CAN.AB.Transportation Passenger.Car...` -> `transportation_passenger`."""
    match = re.match(r"^CIMS\.CAN\.[A-Za-z]{2}\.([^.]+)", nodeName)
    if not match:
        raise ValueError(f"cannot read a sector from node name {nodeName!r}")
    sector = match.group(1)
    if sector in SECTOR_FOLDER_OVERRIDES:
        return SECTOR_FOLDER_OVERRIDES[sector]
    return re.sub(r"[^a-z0-9]+", "_", sector.lower()).strip("_")


def sector_output_dir(output_root, nodeName):
    """`<output_root>/<sector>` for this node."""
    return os.path.join(output_root, sector_folder(nodeName))


def group_by_sector(nodeNames):
    """{sector folder: [nodes]}, keeping the order nodes were given in."""
    groups = {}
    for node in nodeNames:
        groups.setdefault(sector_folder(node), []).append(node)
    return groups


def log_file_for(output_root, nodeName, stage):
    """Per-node solver log path, `<output_root>/<sector>/logs/<node>_<stage>.log`."""
    log_dir = os.path.join(sector_output_dir(output_root, nodeName), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, re.sub(r"[^A-Za-z0-9]+", "_", nodeName) + f"_{stage}.log")


def write_calibration_outputs(model, nodeName, output_root, lifetimes=True,
                              fic_source="calibration_fic_export",
                              lifetime_source="calibration_lifetime_export"):
    """
    Write `nodeName`'s FICs, and its lifetimes when `lifetimes` is True, into the
    node's sector folder under `output_root`. Returns that folder.

    Only write lifetimes after a fit that chose them. A FIC-only fit leaves
    whatever lifetimes the model was loaded with, and if that Reference.py run
    did not load the fitted lifetimes, writing them would overwrite the fitted
    values with the originals.
    """
    out_dir = sector_output_dir(output_root, nodeName)
    write_fics(model, nodeName, out_dir, include_subtree=False, source=fic_source)
    if lifetimes:
        write_lifetimes(model, nodeName, out_dir, include_subtree=False, source=lifetime_source)
    return out_dir


def market_share_l1(model, nodeName, target_key="calibration_market_share_total",
                    estimate_key="market_share_total"):
    """
    Total L1 error between modelled and target market shares, summed over every
    year except the base year: the same quantity the fitters report as
    'final'. Reads the shares stored on the model, so it measures the run that
    produced the pickle without fitting or recomputing anything. A missing
    target counts as 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        both = get_marketShare_both_dict(model, nodeName, key_cims=estimate_key, key_cal=target_key)
    return sum(abs(v["cims"] - v["cal"])
               for year, techs in both.items() if int(year) != model.base_year
               for v in techs.values())


def record_fit(output_root, nodeName, stage, target_key, l1_before, l1_after):
    """Append one fit's before/after L1 to `<sector>/logs/fit_summary.csv`."""
    log_dir = os.path.join(sector_output_dir(output_root, nodeName), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "fit_summary.csv")
    row = pd.DataFrame([[nodeName, stage, target_key, l1_before, l1_after,
                         time.strftime("%Y-%m-%d %H:%M:%S")]], columns=SUMMARY_COLUMNS)
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def last_recorded_fits(output_root, nodeNames):
    """{node: latest summary row as a dict} for nodes that have one."""
    latest = {}
    for sector in group_by_sector(nodeNames):
        path = os.path.join(output_root, sector, "logs", "fit_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for node, rows in df.groupby("Branch"):
            latest[node] = rows.iloc[-1].to_dict()
    return {node: latest[node] for node in nodeNames if node in latest}
