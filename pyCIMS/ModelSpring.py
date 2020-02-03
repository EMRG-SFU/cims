import networkx as nx
import copy as copy
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

### TO MICAH: Check music authorize (??) 02/02/20

__true_print = print  # noqa
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
        self.tech_results = None

        self.build_graph()




    def build_graph(self):
        """

        Populates self.graph with nodes & edges and sets self.fuels to a list of fuel nodes.

        Returns
        -------
        None

        """
        graph = self.graph
        node_dfs = self.node_dfs
        tech_dfs = self.tech_dfs
        graph = make_nodes(graph, node_dfs, tech_dfs)
        graph = make_edges(graph, node_dfs, tech_dfs)
        self.fuels = get_fuels(graph, self.years)

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
            # TEMPORARY SETTING EQUIL TO True then False here: just for tests
            equilibrium = True
            while equilibrium:
                print(f"year: {year}")
                self.iter = 0
                self.results[year] = {}
                self.prices[year] = {get_name(f): {"year_value": None, "to_estimate": False} for f in self.fuels}
                self.quantity[year] = {get_name(q): {"year_value": None, "to_estimate": False} for q in self.fuels}
                depth_first_post(g_demand, self.get_service_cost, year)
                traverse_graph(g_demand, self.first_pass, year)

                # temp
                equilibrium = False
        print('yo')
        print(sub_graph.nodes[node][year]["technologies"].keys())









    def first_pass(self, sub_graph, node, year):
        '''
        Getting values from non-compete nodes, nodes outside of iteration process
        '''

        blueprint = find_value(sub_graph, node, "blueprint", year)

        if blueprint == "root":
            pass

        if blueprint == "region":
            self.region_node(sub_graph, node, year)


        if blueprint == "sector":
            self.sector_node(sub_graph, node, year)


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
        for fuel, price in prices.items():
            if fuel in self.prices[year].keys():
                if price["year_value"] == None:
                    # ML: check if we should check ahead when on base year instead
                    # to flag estimate right away

                    self.prices[year][fuel]["raw_year_value"] = self.prices[str(int(year) - 5)][fuel]["raw_year_value"]
                    self.prices[year][fuel]["year_value"] = self.prices[str(int(year) - 5)][fuel]["raw_year_value"]
                    self.prices[year][fuel]["to_estimate"] = True

                else:
                    # when there's a price at first iteration
                    self.prices[year][fuel]["raw_year_value"] = price["year_value"]
                    self.prices[year][fuel]["year_value"] = price["year_value"]
                    self.prices[year][fuel]["to_estimate"] = False


        attributes = sub_graph.nodes[node][year]["Attribute"]

        for indicator, value in attributes.items():
            # indicator is name, for now naming by unit for ease in fetching from children
            self.results[year][value["unit"]] = value["year_value"]
        print("done region")
        return



    def sector_node(self, sub_graph, node, year):
        results = self.results

        service_unit, provided = get_provided(sub_graph, node, year, results)


        self.results[year][service_unit] = provided
        # ML catch if there's no multiplier
        multiplier = sub_graph.nodes[node][year]["Price Multiplier"]

        prices = copy.copy(self.prices)

        for fuel, multi in multiplier.items():
            if fuel in self.prices[year].keys():
                if multi:
                    self.prices[year][fuel]["year_value"] = prices[year][fuel]["raw_year_value"] * multi["year_value"]
                else:
                    print(f"No multiplier in node {get_name(node)}, for year {year}")
        print('done sector')
        return


    def stacked_node(self, sub_graph, node, year):
        results = self.results
        service_unit, provided = get_provided(sub_graph, node, year, results)
        # # pprint(f"provided: {provided}") -- number
        # # print(f"unnit: {service_unit}") -- buildings
        results[year][service_unit] = provided
        self.results = results




    def compete_node(self, sub_graph, node, year):
        "Runs tech compete nodes"
        results = self.results
        tech_results = {}
        tech_results[get_name(node)] = {}
        results[year][get_name(node)] = 0
        parent_node = parent_name(node)
        parent_compete = find_value(sub_graph, parent_node, "competition_type", year)

        if parent_compete == "fixed market shares":
            parent_tech = find_value(sub_graph, parent_node, "technologies", year)

            for tech, params in parent_tech.items():
                req_from_par = sub_graph.nodes[parent_node][year]["technologies"][tech]["Service requested"]

                # note: results[year][tech]['Service requested'] is a list of dicts
                for idx, service_req in enumerate(req_from_par):
                # ML!!! TODO change this because we don't want to operate local node (say, shell) over parent node
                #            (say, building). In this case, all would just be shell and other requests from techs are overlooked
                    if service_req["branch"] == node:

                        parent_tech_unit = sub_graph.nodes[parent_node][year]["technologies"][tech]["Service requested"][idx]["unit"]

                        # weird way to get market share since we are fetching the parent node info in service_req
                        parent_marketshare = sub_graph.nodes[parent_node][year]["technologies"][tech]["Market share"]["year_value"]


                        par = split_unit(parent_tech_unit)

                        # left hand side of mesure of service provided (i.e. `m2 floorspace` for `m2 floorspace/building`)
                        requested = par[0]
                        # right hand side of mesure of service provided (i.e. `building` for `m2 floorspace/building`)
                        parent_provides = par[1]
                        value_provided = results[year][parent_provides]
                        parent_val_required = service_req["year_value"]
                        # gather info without marketshare, in case it is needed when aggregating with various nodes outside of this branch
                        tech_results[get_name(node)][tech] = (parent_val_required * value_provided)
                        # aggregate all the things requested by this node from the parent node
                        # ML! TODO: Arrange things to aggregate properly when item to aggregate comes from various nodes (see Furnace example)
                        #           One way to do this might be to check the "is_leaf" field and aggregate likes from other leaves, then 'remove'
                        results[year][get_name(node)] += tech_results[get_name(node)][tech] * parent_marketshare
            self.tech_results = tech_results
            self.results = results


        else:
            print(f"Node {get_name(node)}: fill compete when parent not fixed")

        print(f"Tech compete at node {get_name(node)} initialized")


    def get_service_cost(self, sub_graph, node, year):
        results = self.results
        # ML: to change once we fix the node types
        if get_name(node) not in ["pyCIMS", "Canada", "Alberta", "Residential", "Buildings"]:

            for tech in sub_graph.nodes[node][year]["technologies"].keys():

                # insert values that will be needed inside of nodes
                sub_graph.nodes[node][year]["technologies"][tech].update({"Service cost": {"year_value": 0.0}})
                sub_graph.nodes[node][year]["technologies"][tech].update({"Output": {"year_value": 0.0}})
                sub_graph.nodes[node][year]["technologies"][tech].update({"CRF": {"year_value": 0.0}})
                sub_graph.nodes[node][year]["technologies"][tech].update({"LCC": {"year_value": 0.0}})
                sub_graph.nodes[node][year]["technologies"][tech].update({"Full capital cost": {"year_value": 0.0}})

                if sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] == None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] = 0.0

            try:
                v = sub_graph.nodes[node][year]["Heterogeneity"]["v"]["year_value"]
            except:
                v = 10 # default val
        # techdata = find_value(sub_graph, node, 'technologies', year)
        # techs = copy.deepcopy(techdata)
        # check_type(techs, dict, node=node, passing=True)
        # for key, val in techs.items():
        #     try:
        #         branch_req = val['Service requested']['branch']
        #         fuel_req = get_name(branch_req)
        #         req_val = val['Service requested']['year_value']
        #     except TypeError as err:
        #         # ML: Sometimes this is a list (??)
        #         if isinstance(val['Service requested'], list):
        #             # ML! I think this only happens when trying to get child of a leaf
        #             break
        #         else:
        #             # raise
        #             break
        #
        #     if sub_graph.nodes[node]["is_leaf"]:
        #         fuel_req = get_name(branch_req)
        #         if fuel_req in self.prices.keys():
        #             # ML! This will be summed when tech in more than 1 node
        #             fuel_price = self.prices[fuel_req]['year_value']
        #             # raise Exception
        #             sc = fuel_price * req_val
        #             s_cost = {"Service cost": {"year_value": sc}}
        #
        #             sub_graph.nodes[node][year]["technologies"][key].update(s_cost)
        #
        #             crf = {"CRF": {"year_value": get_crf(sub_graph, node, key, year)}}
        #             sub_graph.nodes[node][year]["technologies"][key].update(crf)
        #
        #             output = {"Output": {"year_value": get_output()}}
        #             sub_graph.nodes[node][year]["technologies"][key].update(output)
        #
        #             cap_cost = {"Full capital cost": {"year_value": get_capcost(sub_graph, node, key, year)}}
        #             sub_graph.nodes[node][year]["technologies"][key].update(cap_cost)
        #
        #             lcc = {"LCC": {"year_value": get_lcc(sub_graph, node, key, year)}}
        #             sub_graph.nodes[node][year]["technologies"][key].update(lcc)
        #
        #             market_share = {"Market share": {"year_value": get_marketshare(sub_graph, node, key, year, v)}}
        #             sub_graph.nodes[node][year]["technologies"][key].update(market_share)
        #
        #     else:
        #         pass
