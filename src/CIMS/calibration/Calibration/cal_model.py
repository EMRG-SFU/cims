
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
import sys
import os, os.path
import re
from collections.abc import Iterable
import types

import Calibration.utility_functions as UF
import Calibration.Plotting as plotting

def find_calibration_nodes(g, searchStr = r'calibration', retAll=False):
    ret = UF.findTechsWithParam_anyYears(g, searchStr)
    if retAll:
        return(ret)
    else:
        return(sorted([b for b in set([a['node'] for a in ret])]))


class Cal_Model:

    def __init__(self, model, initialNodeSet):

        self.model = model
        self.initialNodeSet = initialNodeSet




