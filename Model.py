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
                self.tech_results[year] = {}
                self.prices[year] = {get_name(f): {"year_value": None, "to_estimate": False} for f in self.fuels}
                self.quantity[year] = {get_name(q): {"year_value": None, "to_estimate": False} for q in self.fuels}


                # # DEMAND
                # get initial prices
                if self.iter == 0:
                    # get prices inside `region` node
                    traverse_graph(g_demand, self.init_prices, year)
                depth_first_post(g_demand, self.get_service_cost, year)
                traverse_graph(g_demand, self.passes, year)

                # temp
                equilibrium = False
                self.iter += 1

                # TODO: Make sure (estimated) prices are changed between years IN THE GRAPH

        pprint(self.results)



    def get_service_cost(self, sub_graph, node, year):

        results = self.results

        total_lcc_v = 0.0

        child = child_name(sub_graph, node, return_empty=True)

        # print(f"node: {get_name(node)}")
        # print(f"child name: {child}")

        if sub_graph.nodes[node]["competition type"] == "tech compete":
            v = self.get_heterogeneity(sub_graph, node, year)
            # print(f"node, v: {get_name(node)}, {v}")
            try:
                results[year][get_name(node)] = {}
            except:
                pass

            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                results[year][get_name(node)][tech] = {}

                # insert values that will be needed inside of nodes
                tech_params = ["Service cost", "Output", "CRF", "LCC", "Full capital cost"]
                param_vals = [0.0, 1.0, 0.0, 0.0, 0.0]
                # output is set to 1 to work with cap cost formula (cc/output)*crf
                for param_name, param_val in zip(tech_params, param_vals):
                    # go through all tech parameters and make sure we don't overwrite something that exists
                    try:
                        param_exists = sub_graph.nodes[node][year]["technologies"][tech][param_name]

                    except KeyError:
                        self.add_tech_element(sub_graph, node, year, tech, param_name, value=param_val)
                        param_exists = sub_graph.nodes[node][year]["technologies"][tech][param_name]

                    except:
                        raise Exception
                    # else:
                    #     pprint(f"Parameter already filled out as: {param_name}: {param_exists}")


                if sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] == None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] = 0.0

                service_req = sub_graph.nodes[node][year]["technologies"][tech]["Service requested"]

                # go through available techs
                avail = sub_graph.nodes[node][year]['technologies'][tech]['Available']['year_value']
                unavail = sub_graph.nodes[node][year]['technologies'][tech]['Unavailable']['year_value']

                if int(year) <= unavail and int(year) >= avail:

                    # sometimes more than one thing requested, that would make results into a list
                    # (eg [{year_value:... , branch:...electricity, ...},
                    #      {year_value:... , branch:...furnace, ...}}])
                    if isinstance(service_req, dict):
                        if service_req['branch'] in self.fuels:
                            service_req_val = service_req["year_value"]
                            price_tech = self.prices[year][get_name(service_req['branch'])]["year_value"]
                            service_cost = price_tech * service_req_val
                            sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost
                        else:
                            service_req_val = service_req["year_value"]

                            for i, c in enumerate(child):
                                # print(f"node within: {get_name(node)}")
                                # print(f"child within: {c}")
                                child_lccs = sub_graph.nodes[c][year]["total lcc"]
                                # print(f"child_lccs: {child_lccs}")
                                service_cost = child_lccs * service_req_val
                                # if i > 0:
                                    # print(f"(dict)Check service cost at node {get_name(node)}, for tech {tech}: Multiple children not leading to leaf")
                                sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost


                    elif isinstance(service_req, list):
                        for reqs in service_req:
                            if reqs['branch'] in self.fuels:
                                service_req_val = reqs["year_value"]
                                service_cost = self.prices[year][get_name(reqs['branch'])]["year_value"] * service_req_val
                                sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost
                            else:
                                service_req_val = reqs["year_value"]

                                for i, c in enumerate(child):

                                    child_lccs = sub_graph.nodes[c][year]["total lcc"]
                                    service_cost = child_lccs * service_req_val
                                    if i > 0:
                                        print(f"Check service cost at node {get_name(node)}, for tech {tech}: Multiple children not leading to leaf")
                                    sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost

                    else:
                        print(f"type for service requested? {type(service_req)}")


                    # CRF:
                    finance_discount = sub_graph.nodes[node][year]["technologies"][tech]["Discount rate_Financial"]["year_value"]
                    lifespan = sub_graph.nodes[node][year]["technologies"][tech]["Lifetime"]["year_value"]
                    if lifespan != None:
                        crf = finance_discount/(1 - (1 + finance_discount)**(-1.0*lifespan))
                    else:
                        # TODO: Check that this is correct - (lifespan is None at tech: Furnace, for example)
                        crf = 1.0

                    sub_graph.nodes[node][year]["technologies"][tech]["CRF"]["year_value"] = crf

                    # Get Full Cap Cost (will have more terms later)
                    output = sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"]
                    if output == None:
                        self.add_tech_element(sub_graph, node, year, tech, "Output", value=1.0)
                        output = sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"]


                    cap_cost = sub_graph.nodes[node][year]["technologies"][tech]["Capital cost"]["year_value"]
                    if cap_cost == None:
                        self.add_tech_element(sub_graph, node, year, tech, "Capital cost", value=0.0)
                        cap_cost = sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"]

                    full_cap_cost = (cap_cost/output)*crf
                    sub_graph.nodes[node][year]["technologies"][tech]["Full capital cost"]["year_value"] = full_cap_cost

                    # print(f"full capital cost:{full_cap_cost}")

                    # Get LCC
                    operating_cost = sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"]
                    service_cost = sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"]

                    lcc = service_cost + operating_cost + full_cap_cost
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
                    # print("\n")
                    # print(get_name(node))
                    # print(f"tech: {tech}, lcc: {curr_lcc}")
                    # print(f"curr_lcc: {curr_lcc}")
                    if curr_lcc > 0.0:
                        marketshare = curr_lcc**(-1.0*v)/total_lcc_v

                    results[year][get_name(node)][tech] = {"marketshare": marketshare}
                    # print(f"marketshare: {marketshare} for tech {tech}")
                    weighted_lccs += marketshare * curr_lcc

                sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value'] = marketshare

            sub_graph.nodes[node][year]["total lcc"] = weighted_lccs


        self.results = results

            # print(f"node: {get_name(node)}, weighted lccs: {weighted_lccs}")




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
        Getting values from non-compete nodes, nodes outside of iteration process
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
            requested = vals["Service requested"]
            print(f"ms: {marketshare}")

            results[year][get_name(node)][tech]["marketshare"] = marketshare["year_value"]

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

        self.results = results


        print("complete compete")













    # def stacked_node(self, sub_graph, node, year):
    #     self.results[year][get_name(node)] = {}
    #     results = self.results
    #     service_unit, provided = get_provided(sub_graph, node, year, results)
    #     # # pprint(f"provided: {provided}") -- number
    #     # # print(f"unnit: {service_unit}") -- buildings
    #     results[year][get_name(node)][service_unit] = provided
    #
    #     self.results = results
    #
    #
    #
    #
    # def compete_node(self, sub_graph, node, year):
    #     "Runs tech compete nodes"
    #     self.results[year][get_name(node)]["quantity"] = 0
    #     results = self.results
    #     tech_results = {}
    #     tech_results[get_name(node)] = {}
    #
    #     parent_node = parent_name(node)
    #     parent_compete = find_value(sub_graph, parent_node, "competition type", year)
    #
    #     # if parent_compete == "fixed market shares":
    #     parent_tech = find_value(sub_graph, parent_node, "technologies", year)
    #
    #     for tech, params in parent_tech.items():
    #         req_from_par = sub_graph.nodes[parent_node][year]["technologies"][tech]["Service requested"]
    #
    #         # note: results[year][tech]['Service requested'] is a list of dicts
    #         for idx, service_req in enumerate(req_from_par):
    #         # ML!!! TODO change this because we don't want to operate local node (say, shell) over parent node
    #         #            (say, building). In this case, all would just be shell and other requests from techs are overlooked
    #             if service_req["branch"] == node:
    #
    #                 parent_tech_unit = sub_graph.nodes[parent_node][year]["technologies"][tech]["Service requested"][idx]["unit"]
    #
    #                 # weird way to get market share since we are fetching the parent node info in service_req
    #                 parent_marketshare = sub_graph.nodes[parent_node][year]["technologies"][tech]["Market share"]["year_value"]
    #
    #
    #                 par = split_unit(parent_tech_unit)
    #
    #                 # left hand side of mesure of service provided (i.e. `m2 floorspace` for `m2 floorspace/building`)
    #                 requested = par[0]
    #                 # right hand side of mesure of service provided (i.e. `building` for `m2 floorspace/building`)
    #                 parent_provides = par[1]
    #                 print(f"parent_provedes: {parent_provides}")
    #                 print(results[year][get_name(node)])
    #                 print("\n")
    #
    #                 value_provided = results[year][get_name(node)][parent_provides]
    #                 parent_val_required = service_req["year_value"]
    #                 # gather info without marketshare, in case it is needed when aggregating with various nodes outside of this branch
    #                 tech_results[get_name(node)][tech] = (parent_val_required * value_provided)
    #                 # aggregate all the things requested by this node from the parent node
    #                 # ML! TODO: Arrange things to aggregate properly when item to aggregate comes from various nodes (see Furnace example)
    #                 #           One way to do this might be to check the "is_leaf" field and aggregate likes from other leaves, then 'remove'
    #                 results[year][get_name(node)]["quantity"] += tech_results[get_name(node)][tech] * parent_marketshare
    #         self.tech_results = tech_results
    #         self.results = results
    #
    #
    #
    #
    #     # else:
    #     #     print(f"Node {get_name(node)}: fill compete when parent not fixed")
    #
    #     # print(f"Tech compete at node {get_name(node)} initialized")
