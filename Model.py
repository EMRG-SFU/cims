from __future__ import print_function

import networkx as nx
from copy import deepcopy
import copy
import re
import random
import numpy as np
from pprint import pprint
import traceback
import sys
import logging

from graph import *# find_value, get_parent, parent_name, make_nodes, make_edges
# from graph import traverse_graph, depth_first_post, get_subgraph
# from graph import  get_fuels
from utils import *# get_name, split_unit, check_type, is_year, aggregate
from econ import *
### NEXT STEP: deal with iter = 1 to see when it's appropriate to keep a value constant throughout iteration


__true_print = print  # for when using cmd line
def print(*args, **kwargs):
    __true_print(*args, **kwargs)
    sys.stdout.flush()


# Configure logging and set flag to raise exceptions
logging.raiseExceptions = True
logger = logging.getLogger(__name__)
logger.info('Start')





class Model:
    """
    Relevant dataframes and associated information taken from the model description provided in `reader`. Also includes
    methods needed for building and running the Model.

    Parameters
    ----------
    reader : pyCIMS.Reader
        The Reader set up to ingest the description (excel file) for our model.

    Attributes
    ----------
    graph : networkx.DiGraph
        Model Graph populated using the `build_graph` method. Model services are nodes in `graph`, with data contained
        within an associated dictionary. Structural and Request/Provide relationships are edges in the `graph`.

    node_dfs : dict {str: pandas.DataFrame}
        Node names (branch form) are the keys in the dictionary. Associated DataFrames (specified in the excel model
        description) are the values. DataFrames do not include 'Technology' or 'Service' information for a node.

    tech_dfs : dict {str: dict {str: pandas.DataFrame}}
        Technology & service information from the excel model description. Node names (branch form) are keys in
        `tech_dfs` to sub-dictionaries. These sub-dictionaries have technology/service names as keys and pandas
         DataFrames as values. These DataFrames contain information from the excel model description.

    fuels : list [str]
        List of supply-side sector nodes (fuels, etc) requested by the demand side of the Model Graph.  Populated using
        the `build_graph` method.

    years : list [str or int]
        List of the years for which the model will be run.



    """

    def __init__(self, reader):
        self.graph = nx.DiGraph()
        self.node_dfs, self.tech_dfs = reader.get_model_description()
        self.fuels = []
        self.years = reader.get_years()

        self.prices = {}
        self.quantity = {}
        self.results = {}

        # Special container for estimated parameters that need to be held for later aggregation with nodes from other branches
        self.tech_results = {}


        self.build_graph()




    def build_graph(self):
        """

        Populates self.graph with nodes & edges and sets self.fuels to a list of fuel nodes.

        Returns
        -------
        None

        """
        graph = deepcopy(self.graph)
        node_dfs = self.node_dfs
        tech_dfs = self.tech_dfs
        years = self.years

        graph = make_nodes(graph, node_dfs, tech_dfs)
        graph = make_edges(graph, node_dfs, tech_dfs)
        self.fuels = get_fuels(graph, years)

        self.graph = graph



    def run(self, equilibrium_threshold=0.05):
        """
        Runs the entire model, progressing year-by-year until an equilibrium has been reached for each year.

        # TODO: add max_iterations parameter

        Parameters
        ----------
        equilibrium_threshold : float, optional
            The largest relative difference between prices allowed for an equilibrium to be reached. Must be between
            [0, 1]. Relative difference is calculated as the absolute difference between two prices, divided by the
            first price.

        Returns
        -------
        None
            Nothing is returned, but `self.graph` will be updated with the resulting prices, quantities, etc calculated
            for each year.

        """

        g_demand = get_subgraph(self.graph, ['demand', 'standard'])

        for year in self.years:
            # ML! TEMPORARY SETTING EQUIL TO True then False here: just for tests
            equilibrium = True
            while equilibrium:
                print(f"year: {year}")
                self.iter = 0
                self.results[year] = {}
                self.prices[year] = {get_name(f): {"year_value": None, "to_estimate": False} for f in self.fuels}
                self.quantity[year] = {get_name(q): {"year_value": None, "to_estimate": False} for q in self.fuels}


                # # DEMAND
                # get initial prices

                # get prices inside `region` node
                traverse_graph(g_demand, self.init_prices, year)

                depth_first_post(g_demand, self.get_service_cost, year)
                traverse_graph(g_demand, self.passes, year)


                # RUN SUPPLY HERE AND DO EQUILIBRIUM CHECK

                # temporary
                equilibrium = False
                self.iter += 1
                # TODO: Make sure (estimated) prices are changed between years IN THE GRAPH





    def get_service_cost(self, sub_graph, node, year):

        results = self.results
        fuels = self.fuels
        prices = self.prices

        total_lcc_v = 0.0

        child = child_name(sub_graph, node, return_empty=True)

        if sub_graph.nodes[node]["competition type"] == "tech compete":

            v = self.get_heterogeneity(sub_graph, node, year)
            results[year][get_name(node)] = {}

            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                results[year][get_name(node)][tech] = {}

                # insert values that will be needed inside of nodes TEMP
                tech_params = ["Service cost", "CRF", "LCC", "Full capital cost"]
                param_vals = [0.0, 0.0, 0.0, 0.0]

                for param_name, param_val in zip(tech_params, param_vals):
                    # go through all tech parameters and make sure we don't overwrite something that exists
                    try:
                        print(sub_graph.nodes[node][year]["technologies"][tech][param_name])

                    except KeyError:
                        self.add_tech_element(sub_graph, node, year, tech, param_name, value=param_val)

                    except:
                        raise Exception

                # TODO Make default values `automatic` (based on default value sheet)
                if sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] == None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] = 0.0

                    # Get Full Cap Cost (will have more terms later)

                if sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"] == None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"] = 1.0

                # go through available techs (range is [lower year, upper year + 1] to work with range function
                low, up = range_available(sub_graph, node, tech)

                if int(year) in range(low, up):

                    # get service cost
                    service_cost = get_service_cost(sub_graph, node, year, tech, fuels, prices)
                    sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost


                    # get CRF:
                    crf = get_crf(sub_graph, node, year, tech, finance_base=0.1, life_base=10.0)
                    sub_graph.nodes[node][year]["technologies"][tech]["CRF"]["year_value"] = crf

                    # get Capital Cost
                    full_cc = get_capcost(sub_graph, node, year, tech, crf)
                    sub_graph.nodes[node][year]["technologies"][tech]["Full capital cost"]["year_value"] = full_cc

                    # Get LCC
                    operating_cost = sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"]
                    lcc = service_cost + operating_cost + full_cc
                    sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"] = lcc

                    # get marketshare (calculate instead of using given ones (base year 2000 only)(?))
                    # TODO: catch min and max marketshares
                    total_lcc_v += lcc**(-1.0*v)   # will need to catch other competing lccs in other branches?

            weighted_lccs = 0
            # get marketshares based on each competing service/tech
            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                # go through available techs
                avail = sub_graph.nodes[node][year]['technologies'][tech]['Available']['year_value']
                unavail = sub_graph.nodes[node][year]['technologies'][tech]['Unavailable']['year_value']
                marketshare = 0.0

                if int(year) <= unavail and int(year) >= avail:
                    curr_lcc = sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"]
                    # print(f"tech: {tech}, lcc: {curr_lcc}")
                    # print(f"curr_lcc: {curr_lcc}")
                    if curr_lcc > 0.0:
                        marketshare = curr_lcc**(-1.0*v)/total_lcc_v

                    results[year][get_name(node)][tech] = {"marketshare": marketshare}
                    # print(f"marketshare: {marketshare} for tech {tech}")
                    weighted_lccs += marketshare * curr_lcc

                sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value'] = marketshare

            sub_graph.nodes[node][year]["total lcc"] = weighted_lccs
            # print(f"node: {get_name(node)}, weighted lccs: {weighted_lccs}")

        self.results = results
        self.prices = prices



    def add_tech_element(self, g, node, year, tech, param, value=0.0, source=None, unit=None):
        '''
        Include a new set of parameter: {values} as a nested dict
        '''
        g.nodes[node][year]["technologies"][tech].update({str(param): {"year_value": value,
                                                                         "branch": str(node),
                                                                         "source": source,
                                                                         "unit": unit}})
    def get_heterogeneity(self, g, node, year):
        try:
            v = g.nodes[node][year]["Heterogeneity"]["v"]["year_value"]
        except:
            v = 10 # default val
        return v



    def init_prices(self, sub_graph, node, year):

        blueprint = find_value(sub_graph, node, "blueprint", year)

        if blueprint == "region":
            self.region_node(sub_graph, node, year)

        if blueprint == "sector":
            self.sector_node(sub_graph, node, year)







    def passes(self, sub_graph, node, year):
        '''
        '''

        blueprint = find_value(sub_graph, node, "blueprint", year)


        if blueprint == "stacked":
            self.stacked_node(sub_graph, node, year)


        if blueprint == "compete":
            self.compete_node(sub_graph, node, year)




    def region_node(self, sub_graph, node, year):
        # check if "active" or not
        try:
            prices = sub_graph.nodes[node][year]["Price"]
        except:
            # if non active (region without price (or attributes))
            return

        # if active:
        self.results[year][get_name(node)] = {}
        results = self.results

        for fuel, price in prices.items():
            if fuel in self.prices[year].keys():
                if price["year_value"] == None:
                    # ML: check if we should check ahead when on base year instead
                    # to flag estimate right away
                    # TODO remove 5 and put step
                    self.prices[year][fuel]["raw_year_value"] = self.prices[str(int(year) - 5)][fuel]["raw_year_value"]
                    self.prices[year][fuel]["year_value"] = self.prices[str(int(year) - 5)][fuel]["raw_year_value"]
                    self.prices[year][fuel]["to_estimate"] = True

                else:
                    # when there's a price at first iteration
                    self.prices[year][fuel]["raw_year_value"] = price["year_value"]
                    self.prices[year][fuel]["year_value"] = price["year_value"]
                    self.prices[year][fuel]["to_estimate"] = False


        attributes = sub_graph.nodes[node][year]["Attribute"]
        # keeping attribute values (macroeconomic indicators) in result dict
        for indicator, value in attributes.items():
            # indicator is name, for now naming by unit for ease in fetching from children
            results[year][get_name(node)][value["unit"]] = value["year_value"]
        print("done region")
        self.results = results
        return



    def sector_node(self, sub_graph, node, year):

        self.results[year][get_name(node)] = {}
        results = self.results

        service_unit, provided = get_provided(sub_graph, node, year, results)
        results[year][get_name(node)][service_unit] = provided
        requested = sub_graph.nodes[node][year]["Service requested"]

        # ML catch if there's no multiplier
        multiplier = sub_graph.nodes[node][year]["Price Multiplier"]
        prices = copy.copy(self.prices)

        for fuel, multi in multiplier.items():
            if fuel in self.prices[year].keys():
                if multi:
                    self.prices[year][fuel]["year_value"] = prices[year][fuel]["raw_year_value"] * multi["year_value"]
                else:
                    print(f"No multiplier in node {get_name(node)}, for year {year}")

        for req in requested.values():
            results[year][get_name(node)][service_unit] = provided * req["year_value"]

        self.results = results
        print('done sector')

        return


    def stacked_node(self, sub_graph, node, year):

        self.results[year][get_name(node)] = {}
        results = self.results

        children = child_name(sub_graph, node)
        temp_results = {c: 0.0 for c in children}
        # TODO change get_provided
        service_unit, provided = get_provided(sub_graph, node, year, results)


        results[year][get_name(node)][service_unit] = provided

        for tech, vals in sub_graph.nodes[node][year]["technologies"].items():
            results[year][get_name(node)][tech] = {}

            marketshare = vals["Market share"]
            requested = vals["Service requested"]

            results[year][get_name(node)][tech]["marketshare"] = marketshare["year_value"]

            for req in requested:

                results[year][get_name(node)][tech][get_name(req["branch"])] = {}
                downflow = {"value": req["year_value"],
                            "result_unit": split_unit(req["unit"])[0],
                            "location": req["branch"],
                            "result": req["year_value"] * results[year][get_name(node)][service_unit] * marketshare["year_value"]}

                results[year][get_name(node)][tech][get_name(req["branch"])] = downflow
                temp_results[req["branch"]] += downflow["result"]

        for child in children:
            prov_dict = sub_graph.nodes[child][year]["Service provided"]
            for object in prov_dict.keys():

                sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[child]

        self.results = results

        print("complete stacked (fixed ratio)")



    def compete_node(self, sub_graph, node, year):
        self.results[year][get_name(node)] = {}
        results = self.results

        children = child_name(sub_graph, node)

        if isinstance(children, list):
            temp_results = {c: 0.0 for c in children}

        elif isinstance(children, str):
            temp_results = {children: 0.0}



        prov_dict = sub_graph.nodes[node][year]["Service provided"]

        for obj, vals in prov_dict.items():
            provided = vals["year_value"]
            service_unit = vals["unit"]


        results[year][get_name(node)][service_unit] = provided

        for tech, vals in sub_graph.nodes[node][year]["technologies"].items():

            results[year][get_name(node)][tech] = {}
            marketshare = vals["Market share"]
            results[year][get_name(node)][tech]["marketshare"] = marketshare["year_value"]
            requested = vals["Service requested"]

            if isinstance(requested, list):

                for req in requested:
                    if req["branch"] not in self.fuels:

                        results[year][get_name(node)][tech][get_name(req["branch"])] = {}
                        downflow = {"value": req["year_value"],
                                    "result_unit": split_unit(req["unit"])[0],
                                    "location": req["branch"],
                                    "result": req["year_value"] * results[year][get_name(node)][service_unit] * marketshare["year_value"]}

                        results[year][get_name(node)][tech][get_name(req["branch"])] = downflow
                        temp_results[req["branch"]] += downflow["result"]

                    else:
                        # if req is a fuel
                        temp_results.update({req["branch"]: 0})



                        results[year][get_name(node)][tech][get_name(req["branch"])] = {}
                        downflow = {"value": req["year_value"],
                                    "result_unit": split_unit(req["unit"])[0],
                                    "location": req["branch"],
                                    "result": req["year_value"] * marketshare["year_value"]}

                        results[year][get_name(node)][tech][get_name(req["branch"])] = downflow
                        temp_results[req["branch"]] += downflow["result"]



            elif isinstance(requested, dict):
                if requested["branch"] not in self.fuels:

                    results[year][get_name(node)][tech][get_name(requested["branch"])] = {}
                    downflow = {"value": requested["year_value"],
                                "result_unit": split_unit(requested["unit"])[0],
                                "location": requested["branch"],
                                "result": requested["year_value"] * results[year][get_name(node)][service_unit] * marketshare["year_value"]}

                    results[year][get_name(node)][tech][get_name(requested["branch"])] = downflow
                    temp_results[requested["branch"]] += downflow["result"]



        if isinstance(children, list):
            for child in children:
                if child not in self.fuels:
                    prov_dict = sub_graph.nodes[child][year]["Service provided"]
                    for object in prov_dict.keys():
                        sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[child]

        elif isinstance(children, str):
            child = children
            if child not in self.fuels:
                prov_dict = sub_graph.nodes[child][year]["Service provided"]
                for object in prov_dict.keys():
                    sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[child]

        self.quantity[year] = temp_results

        self.results = results



        print("complete compete")
