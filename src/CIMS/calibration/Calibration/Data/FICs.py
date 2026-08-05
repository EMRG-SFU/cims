import pandas as pd
import polars as pl
import re

from collections.abc import Iterable
from functools import reduce
import types

import Calibration.Data.node_info as node_info

def get_FICs(model, nodeName, key="fic"):

    allYears = node_info.list_years(model.graph, nodeName)
    allTechs = node_info.list_techs(model.graph, nodeName)
    retDict = {"tech": allTechs}
    retDict.update(
            {yv:[model.get_param(key, nodeName, year=yv, tech=tv) for tv in allTechs] for yv in allYears}
    )
    return pl.DataFrame(retDict)

def tweak_FICs():
    pass

def fic_set():
    pass
