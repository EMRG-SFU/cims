"""
custom_model.py

This generalizes the subgraph extraction process in
`single_sector_all_region.py` and enables the selection of regions using a
list, and the selection of sectors using a list. Then you get a compound graph,
showing just those sectors, in those regions, but all the information needed to
do calibration and everything else properly will be in the resulting graph.
"""


import networkx as nx
import numpy as np
import json
import re
import pickle
import gzip

from Calibration.SubGraphs.graphFunctions import getSubgraph

def get_custom_model():
    pass
