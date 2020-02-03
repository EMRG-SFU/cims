import networkx as nx
import copy as copy
import re
import random
import numpy as np
from pprint import pprint
import traceback
import sys
import logging

from utils import get_name, split_unit
from graph import get_parent


def get_provided(g, node, year, parent_provide):
    '''
    '''
    node_name = get_name(node)
    provided = copy.deepcopy(g.nodes[node][year]["Service provided"])
    service = list(provided.keys())[0]
    if service == node_name:
        parent = get_parent(g, node, year)
        parent_request = parent["Service requested"][service]
        request_units = split_unit(parent_request['unit'])
        service_unit = provided[service]['unit']

        provide_val = parent_request['year_value'] * parent_provide[year][request_units[-1]]
        g.nodes[node][year]["Service provided"][service]["year_value"] = provide_val
        return service_unit, provide_val


def lastyear_tech(g, node, year, tech, param, step=5):
    """
    Check if there's a value filled out within a tech, if not, take last years value
    """
    value = g.nodes[node][year]["technologies"][tech][param]["year_value"]
    if value == None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, tech {tech}, parameter {param}")
        else:
            last_year = str( int(year) - step )
            value = g.nodes[node][last_year]["technologies"][tech][param]["year_value"]
    return value


def lastyear_fuel(g, node, year, fuel, param, step=5):
    """
    In an operation related to fuel, check if there's a value filled out, if not, take last years value
    """
    value = g.nodes[node][year][param][fuel]["year_value"]

    if value == None:
        if year == "2000":
            raise ValueError(f"No initial value for node {node}, fuel {fuel}, parameter {param}")
        else:
            last_year = str( int(year) - step )
            value = g.nodes[node][last_year][param][fuel]["year_value"]

    return value
