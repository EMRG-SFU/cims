from __future__ import print_function

import copy
import networkx as nx
import graph_utils
import econ
import utils
import random
import logging

### NEXT STEP: deal with iter = 1 to see when it's appropriate to keep a value constant throughout iteration


# __true_print = print  # for when using cmd line
# def print(*args, **kwargs):
#     __true_print(*args, **kwargs)
#     sys.stdout.flush()


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
        self.step = 5  # Make this an input later

        self.fuels = []
        self.years = reader.get_years()
        self.base_year = int(self.years[0])

        self.prices = {}
        self.quantity = {}
        self.results = {}

        # Special container for estimated parameters that need to be held for later aggregation with nodes from
        # other branches
        self.tech_results = {}

        self.build_graph()

    def build_graph(self):
        """

        Populates self.graph with nodes & edges and sets self.fuels to a list of fuel nodes.

        Returns
        -------
        None

        """
        graph = copy.deepcopy(self.graph)
        node_dfs = self.node_dfs
        tech_dfs = self.tech_dfs
        years = self.years

        graph = graph_utils.make_nodes(graph, node_dfs, tech_dfs)
        graph = graph_utils.make_edges(graph, node_dfs, tech_dfs)
        self.fuels = graph_utils.get_fuels(graph, years)

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
            Nothing is returned, but `self.graph` will be updated with the resulting prices, quantities, etc calculated
            for each year.

        """

        g_demand = graph_utils.get_subgraph(self.graph, ['demand', 'standard'])
        g_supply = graph_utils.get_subgraph(self.graph, ['supply', 'standard'])

        for year in self.years:
            equilibrium = False
            # prev_prices = {}
            cur_prices = copy.deepcopy(self.prices)
            print(f"***** ***** year: {year} ***** *****")
            self.iter = 0
            self.results[year] = {}
            self.prices[year] = {utils.get_name(f): {"year_value": None, "to_estimate": False} for f in self.fuels}
            self.quantity[year] = {utils.get_name(q): {"year_value": None, "to_estimate": False} for q in self.fuels}

            graph_utils.traverse_graph(g_demand, self.init_prices, year)

            while not equilibrium:
                print('----- ----- ITER {} ----- ----- '.format(self.iter))
                print('Prices\n-----')
                print([(k, v['raw_year_value']) for k, v in self.prices[year].items()])

                print('Results\n-----')

                # print(f"year: {year}")
                # self.iter = 0
                # self.results[year] = {}
                # self.prices[year] = {get_name(f): {"year_value": None, "to_estimate": False} for f in self.fuels}
                # self.quantity[year] = {get_name(q): {"year_value": None, "to_estimate": False} for q in self.fuels}

                # DEMAND
                # ******************
                # get initial prices
                # print("***** ***** DEMAND ***** *****")

                # get prices inside `region` node
                # traverse_graph(g_demand, self.init_prices, year)
                graph_utils.breadth_first_post(g_demand, self.get_service_cost, year)
                graph_utils.traverse_graph(g_demand, self.passes, year)                 # HERE

                # Supply
                # ******************
                # print("***** ***** SUPPLY ***** *****")
                graph_utils.breadth_first_post(g_supply, self.get_service_cost, year)

                # Update Prices
                prev_prices = cur_prices
                cur_prices = self.update_prices(year, g_supply)
                self.prices = cur_prices

                # Equilibrium
                # ******************
                # Check if first iteration
                def eq_check(prev, new):
                    """
                    Return False unless an equilibrium has been reached.
                    1. Check if prev is empty or year not in previous (first year or first iteration)
                    2. For every fuel, check if the relative difference exceeds the threshold
                        (A) If it does, return False
                        (B) Otherwise, keep checking
                    3. If all fuels are checked and no relative difference exceeds the threshold, return True
                    """
                    print('----- Checking Equilibrium')
                    if (len(prev) == 0) or (year not in prev.keys()):
                        return False

                    for fuel in new[year]:
                        prev_fuel_price = prev[year][fuel]['raw_year_value']
                        new_fuel_price = new[year][fuel]['raw_year_value']

                        abs_diff = abs(new_fuel_price - prev_fuel_price)
                        rel_diff = abs_diff/prev_fuel_price

                        print("\tFuel:", fuel, prev_fuel_price, new_fuel_price, rel_diff)
                        # print("Diff:{}".format(rel_diff))
                        if rel_diff > equilibrium_threshold:
                            return False

                    return True

                equilibrium = eq_check(prev_prices, cur_prices)
                self.iter += 1

                x = 1
                # TODO: Make sure (estimated) prices are changed between years IN THE GRAPH

    def get_service_cost(self, sub_graph, node, year, rand=False):
        """
        1. Initialize
            * results, fuels, and prices to what is in the model object.
            * total_lcc_v = 0
            * child to be all the children of the node
        2. Check if the node is a tech compete node. If its not, do nothing.
        3. For every tech in the node
            (A) Initialize the parameters we have
            (B) Calculate service cost, CRF, capital cost, LCC, marketshare
        4. Calculate the weighted lcc for the node
        5. Return results

        """
        results = self.results
        fuels = self.fuels
        prices = self.prices
        print(node)  # HERE

        total_lcc_v = 0.0

        child = graph_utils.child_name(sub_graph, node, return_empty=True)

        if sub_graph.nodes[node]["competition type"] == "tech compete":

            v = self.get_heterogeneity(sub_graph, node, year)
            results[year][utils.get_name(node)] = {}

            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                results[year][utils.get_name(node)][tech] = {}

                # insert values that will be needed inside of nodes TEMP
                tech_params = ["Service cost", "CRF", "LCC", "Full capital cost"]
                param_vals = [0.0, 0.0, 0.0, 0.0]

                for param_name, param_val in zip(tech_params, param_vals):
                    # go through all tech parameters and make sure we don't overwrite something that exists
                    try:
                        sub_graph.nodes[node][year]["technologies"][tech][param_name]

                    except KeyError:
                        self.add_tech_element(sub_graph, node, year, tech, param_name, value=param_val)

                    except:
                        raise Exception

                # TODO Make default values `automatic` (based on default value sheet)
                if sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] is None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"] = 0.0

                if sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"] is None:
                    sub_graph.nodes[node][year]["technologies"][tech]["Output"]["year_value"] = 1.0

                # go through available techs (range is [lower year, upper year + 1] to work with range function
                low, up = utils.range_available(sub_graph, node, tech)

                if int(year) in range(low, up):

                    # get service cost
                    service_cost = econ.get_service_cost(sub_graph, node, year, tech, fuels, prices)
                    sub_graph.nodes[node][year]["technologies"][tech]["Service cost"]["year_value"] = service_cost
                    # get CRF:
                    crf = econ.get_crf(sub_graph, node, year, tech, finance_base=0.1, life_base=10.0)
                    sub_graph.nodes[node][year]["technologies"][tech]["CRF"]["year_value"] = crf

                    # get Full Cap Cost (will have more terms later)

                    full_cc = econ.get_capcost(sub_graph, node, year, tech, crf)
                    sub_graph.nodes[node][year]["technologies"][tech]["Full capital cost"]["year_value"] = full_cc

                    # Get LCC
                    operating_cost = sub_graph.nodes[node][year]["technologies"][tech]["Operating cost"]["year_value"]
                    lcc = service_cost + operating_cost + full_cc
                    sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"] = lcc

                    # get marketshare (calculate instead of using given ones (base year 2000 only)(?))
                    # TODO: catch min and max marketshares
                    total_lcc_v += lcc ** (-1.0 * v)  # will need to catch other competing lccs in other branches?

            weighted_lccs = 0
            # get marketshares based on each competing service/tech
            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                # go through available techs
                marketshare = 0.0

                # go through available techs (range is [lower year, upper year + 1] to work with range function
                low, up = utils.range_available(sub_graph, node, tech)

                if int(year) in range(low, up):

                    curr_lcc = sub_graph.nodes[node][year]["technologies"][tech]["LCC"]["year_value"]
                    # print(f"tech: {tech}, lcc: {curr_lcc}")
                    # print(f"curr_lcc: {curr_lcc}")
                    if curr_lcc > 0.0:
                        if rand:
                            marketshare = random.random()
                        else:
                            marketshare = curr_lcc ** (-1.0 * v) / total_lcc_v

                    results[year][utils.get_name(node)][tech] = {"marketshare": marketshare}
                    # print(f"marketshare: {marketshare} for tech {tech}")
                    weighted_lccs += marketshare * curr_lcc
                print('\t', tech)
                print('\t\tservice cost:', service_cost)
                print('\t\tmarketshare:', marketshare)
                sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value'] = marketshare

            sub_graph.nodes[node][year]["total lcc"] = weighted_lccs
            # print(f"node: {get_name(node)}, weighted lccs: {weighted_lccs}")

        # if sub_graph.nodes[node]["competition type"] in ['fixed ratio', 'sector']:
        #     # We need to go to the children of the node and do a weighted average of their lccs.
        #     # Find Requested Services
        #     total_lcc = 0
        #     total_req = 0
        #     for service in sub_graph.nodes[node][year]['Service requested']:
        #         service_node = service['branch']
        #         service_req = service['year_value']
        #         total_req += service_req
        #         # total_lcc += sub_graph.
        #
        #     pass

        self.results = results
        # self.prices = prices

    def add_tech_element(self, g, node, year, tech, param, value=0.0, source=None, unit=None):
        """
        Include a new set of parameter: {values} as a nested dict

        """
        g.nodes[node][year]["technologies"][tech].update({str(param): {"year_value": value,
                                                                       "branch": str(node),
                                                                       "source": source,
                                                                       "unit": unit}})

    def get_heterogeneity(self, g, node, year):
        try:
            v = g.nodes[node][year]["Heterogeneity"]["v"]["year_value"]
        except KeyError:
            v = 10  # default val
        if v is None:
            v = 10
        return v

    def init_prices(self, sub_graph, node, year):

        blueprint = graph_utils.find_value(sub_graph, node, "blueprint", year)

        if blueprint == "region":
            self.region_node(sub_graph, node, year)

        if blueprint == "sector":
            self.sector_node(sub_graph, node, year)

    def passes(self, sub_graph, node, year):

        blueprint = graph_utils.find_value(sub_graph, node, "blueprint", year)

        if blueprint == "stacked":
            self.stacked_node(sub_graph, node, year)

        if blueprint == "compete":
            self.compete_node(sub_graph, node, year)

    def region_node(self, sub_graph, node, year):
        """
        JA Notes:
        1. Check if the region node is active. Active nodes are those which have Prices.
        2. If it is active, clear the results for the node.
        3. for every fuel whose price is specified at the node, check if the fuel is in our prices dict. If so, check
           if price was set exogenously.
            (A) NO -> initialize the current year prices (raw, year_value) for that fuel to be the prices from the
                      previous year. Set that fuel as an estimateable fuel.
            (B) YES -> Turn off estimation for the fuel. Set prices based on exogenous value.
        4. Add attributes to results.
        5. Return results
        """
        # check if "active" or not
        try:
            prices = sub_graph.nodes[node][year]["Price"]
        except KeyError:
            # if non active (region without price (or attributes))
            return

        # if active:
        self.results[year][utils.get_name(node)] = {}
        results = self.results

        for fuel, price in prices.items():
            price_short_names = [p.split('.')[-1] for p in self.prices[year].keys()]
            # if fuel in self.prices[year].keys():
            if fuel in price_short_names:
                fuel_key = [k for k in self.prices[year].keys() if k.split('.')[-1] == fuel][0]  # TODO: REMOVE!
                if price["year_value"] is None:
                    # ML: check if we should check ahead when on base year instead
                    # to flag estimate right away
                    # TODO remove 5 and put step
                    self.prices[year][fuel_key]["raw_year_value"] = self.prices[str(int(year) - self.step)][fuel_key]["raw_year_value"]
                    self.prices[year][fuel_key]["year_value"] = self.prices[str(int(year) - self.step)][fuel_key]["raw_year_value"]
                    self.prices[year][fuel_key]["to_estimate"] = True

                else:
                    # when there's a price at first iteration
                    self.prices[year][fuel_key]["raw_year_value"] = price["year_value"]
                    self.prices[year][fuel_key]["year_value"] = price["year_value"]
                    self.prices[year][fuel_key]["to_estimate"] = False

        attributes = sub_graph.nodes[node][year]["Attribute"]
        # keeping attribute values (macroeconomic indicators) in result dict
        for indicator, value in attributes.items():
            # indicator is name, for now naming by unit for ease in fetching from children
            results[year][utils.get_name(node)][value["unit"]] = value["year_value"]
        # print("done region")
        self.results = results
        return

    def sector_node(self, sub_graph, node, year):
        """
        Fixed Ratio
        """
        """
        JA Notes: 
        1. Check if the sector contains price multipliers. 
        2. For each multiplier: 
            (A) Check if the corresponding fuel is in our price list. If yes: 
                i) If multiplier has been defined -> update price's year_value with multiplier * raw_year_value
        3. For each service requested: 
            (A) multiply requested by the provided
        4. Save results. Return nothing. 
        """
        self.results[year][utils.get_name(node)] = {}
        results = self.results

        service_unit, provided = econ.get_provided(sub_graph, node, year, results)
        results[year][utils.get_name(node)][service_unit] = provided
        requested = sub_graph.nodes[node][year]["Service requested"]

        # ML catch if there's no multiplier
        try:
            multiplier = sub_graph.nodes[node][year]["Price Multiplier"]
        except KeyError:
            ("Sector has no multiplier field")
            multiplier = None

        prices = copy.copy(self.prices)

        for fuel, multi in multiplier.items():
            if fuel in [f.split('.')[-1] for f in self.prices[year].keys()]:
                if multi:
                    fuel_key = [k for k in self.prices[year].keys() if k.split('.')[-1] == fuel][0]  # TODO: REMOVE!
                    self.prices[year][fuel_key]["year_value"] = prices[year][fuel_key]["raw_year_value"] * multi["year_value"]
                else:
                    print(f"No multiplier in node {utils.get_name(node)}, for year {year}")

        for req in requested.values():
            results[year][utils.get_name(node)][service_unit] = provided * req["year_value"]

        self.results = results
        # print('done sector')
        return

    def stacked_node(self, sub_graph, node, year):
        """
        Fixed Market Share
        """

        self.results[year][utils.get_name(node)] = {}
        results = self.results

        children = graph_utils.child_name(sub_graph, node)
        temp_results = {c: 0.0 for c in children}
        # TODO change get_provided
        service_unit, provided = econ.get_provided(sub_graph, node, year, results)

        results[year][utils.get_name(node)][service_unit] = provided

        for tech, vals in sub_graph.nodes[node][year]["technologies"].items():
            results[year][utils.get_name(node)][tech] = {}

            marketshare = vals["Market share"]
            requested = vals["Service requested"]

            results[year][utils.get_name(node)][tech]["marketshare"] = marketshare["year_value"]

            for req in requested:
                results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = {}
                downflow = {"value": req["year_value"],
                            "result_unit": utils.split_unit(req["unit"])[0],
                            "location": req["branch"],
                            "result": req["year_value"] * results[year][utils.get_name(node)][service_unit] * marketshare[
                                "year_value"]}

                results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = downflow
                temp_results[req["branch"]] += downflow["result"]

        for child in children:
            prov_dict = sub_graph.nodes[child][year]["Service provided"]
            for object in prov_dict.keys():
                sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[child]

        self.results = results

        # print("complete stacked (fixed ratio)")

    def compete_node(self, sub_graph, node, year):
        """
        1. Initialize the results entry for this node in this year to an empty dict
        2. Find the children of this node, initialize their results to 0.
        3. Find all of the services being provided by this node. Update results with this info for this node & year
        4. For each technology at this node:
            (A) update the tech in the results with marketshare and services requested
            (B) For each service being requested, calculate the quantity being requested.
        """
        self.results[year][utils.get_name(node)] = {}
        results = self.results

        # print(f"node: {get_name(node)}")
        # pprint(f"quantity{self.quantity}")

        children = graph_utils.child_name(sub_graph, node)
        temp_results = {utils.get_name(c): 0.0 for c in children}

        # vv OLD IMPLEMENTATION vv
        # if isinstance(children, list):
        #     temp_results = {c: 0.0 for c in children}
        #
        # elif isinstance(children, str):
        #     temp_results = {children: 0.0}
        #
        # prov_dict = sub_graph.nodes[node][year]["Service provided"]
        #
        # for obj, vals in prov_dict.items():
        #     provided = vals["year_value"]
        #     service_unit = vals["unit"]
        # ^^ OLD IMPLEMENTATION ^^

        # vv New IMPLEMENTATION vv
        provided_dict = sub_graph.nodes[node][year]["Service provided"]
        # NOTE: Is traversal order correct?? Printing seems to show depth first (shell-space h-furnace-dish-clothes)

        provided_vals = list(provided_dict.values())[0]
        provided = provided_vals["year_value"]
        service_unit = provided_vals["unit"]
        # ^^ NEW IMPLEMENTATION ^^

        results[year][utils.get_name(node)][service_unit] = provided

        for tech, vals in sub_graph.nodes[node][year]["technologies"].items():
            results[year][utils.get_name(node)][tech] = {}
            marketshare = vals["Market share"]
            results[year][utils.get_name(node)][tech]["marketshare"] = marketshare["year_value"]
            requested = vals["Service requested"]

            if isinstance(requested, list):
                for req in requested:
                    if req["branch"] not in self.fuels:
                        results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = {}
                        downflow = {"value": req["year_value"],
                                    "result_unit": utils.split_unit(req["unit"])[0],
                                    "location": req["branch"],
                                    "result": req["year_value"] * results[year][utils.get_name(node)][service_unit] *
                                              marketshare["year_value"]}

                        results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = downflow
                        temp_results[utils.get_name(req["branch"])] += downflow["result"]
                        # print(f"downflow list not fuel: {downflow['result']}")

                    else:
                        # if req is a fuel
                        # CHANGE THIS in case it overwrites in corner cases
                        temp_results.update({req["branch"]: 0})

                        results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = {}
                        downflow = {"value": req["year_value"],
                                    "result_unit": utils.split_unit(req["unit"])[0],
                                    "location": req["branch"],
                                    "result": req["year_value"] * marketshare["year_value"]}

                        results[year][utils.get_name(node)][tech][utils.get_name(req["branch"])] = downflow
                        temp_results[utils.get_name(req["branch"])] += downflow["result"]
                        # print(f"downflow list fuel: {downflow['result']}")


            elif isinstance(requested, dict):
                if requested["branch"] not in self.fuels:
                    results[year][utils.get_name(node)][tech][utils.get_name(requested["branch"])] = {}
                    downflow = {"value": requested["year_value"],
                                "result_unit": utils.split_unit(requested["unit"])[0],
                                "location": requested["branch"],
                                "result": requested["year_value"] * results[year][utils.get_name(node)][service_unit] *
                                          marketshare["year_value"]}

                    results[year][utils.get_name(node)][tech][utils.get_name(requested["branch"])] = downflow
                    temp_results[utils.get_name(requested["branch"])] += downflow["result"]
                    # print(f"downflow dict: {downflow['result']}")
                else:
                    # print(f"fuel dict: {requested}")
                    # print("get info here")
                    temp_results.update({utils.get_name(requested["branch"]): 0.0})


                    results[year][utils.get_name(node)][tech][utils.get_name(requested["branch"])] = {}
                    downflow = {"value": requested["year_value"],
                                "result_unit": utils.split_unit(requested["unit"])[0],
                                "location": requested["branch"],
                                "result": requested["year_value"] * results[year][utils.get_name(node)][service_unit] * marketshare["year_value"]}

                    # vv NEW IMPLEMENTATION vv
                    results[year][utils.get_name(node)][tech][utils.get_name(requested["branch"])] = downflow
                    temp_results[utils.get_name(requested["branch"])] += downflow["result"]
                    # print(f"downflow dict: {downflow['result']}")
                    # ^^ NEW IMPLEMENTATION

        if isinstance(children, list):
            for child in children:
                if child not in self.fuels:
                    prov_dict = sub_graph.nodes[child][year]["Service provided"]
                    for object in prov_dict.keys():
                        sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[utils.get_name(child)]

        elif isinstance(children, str):
            child = children
            if child not in self.fuels:
                prov_dict = sub_graph.nodes[child][year]["Service provided"]
                for object in prov_dict.keys():
                    sub_graph.nodes[child][year]["Service provided"][object]["year_value"] = temp_results[utils.get_name(child)]



        for tech, vals in sub_graph.nodes[node][year]["technologies"].items():
            requested = vals["Service requested"]

            if isinstance(requested, list):
                for req in requested:
                    if req["branch"] in self.fuels:
                        self.quantity[year][utils.get_name(req["branch"])]["year_value"] = temp_results[utils.get_name(req["branch"])]


            elif isinstance(requested, dict):
                if requested["branch"] in self.fuels:
                    self.quantity[year][utils.get_name(requested["branch"])]["year_value"] = temp_results[utils.get_name(requested["branch"])]

        # print(f"temp_results: {temp_results}")
        # pprint(f"quantity{self.quantity[year][get_name(node)]}")

        return

        # print("complete compete")

    def update_prices(self, year, supply_subgraph):
        """
        1. Determine what fuels need to be calculated
        2. For each fuel whose price need to be estimated. Calculate that price . To calculate:
            (A) If a tech compete node, just return the lcc. This is production cost.
            (B) If its a fixed ratio or sector node:
                  i) find all the services being requested
                 ii) for each service, multiply the price by the requested amount to get a weighted cost.
                iii) calculate the weighted average production cost of the requested services.
        3. Store the prices of the nodes in the new prices dictionary. Return dictionary.
        """
        # Figure out which fuels I need to calculate
        price_keys_to_estimate = [p for p, d in self.prices[year].items() if d['to_estimate']]
        new_prices = copy.deepcopy(self.prices)

        # For each price we need to estimate, go find the price
        def calc_price(node_name):
            # Base Case is that we hit a tech compete node
            if supply_subgraph.nodes[node_name]['competition type'] == 'tech compete':
                return supply_subgraph.nodes[node_name][year]['total lcc']
            elif supply_subgraph.nodes[node_name]['competition type'] in ['fixed ratio', 'sector']:
                # Find all the services being requested
                services_requested = [(d['branch'], d['year_value']) for s, d in supply_subgraph.nodes[node_name][year]['Service requested'].items()]

                # For each service multiply their price by their request amount
                total_req = 0
                total_lcc = 0
                for branch, req_amount in services_requested:
                    total_req += req_amount
                    total_lcc += req_amount * calc_price(branch)

                # Divide it by the total request amount
                # return random.randint(5, 200)
                return total_lcc / total_req

        for fuel in price_keys_to_estimate:
            fuel_price = calc_price(node_name=fuel)
            # print(fuel_price)
            new_prices[year][fuel]['raw_year_value'] = fuel_price

        return new_prices

        # node = 'pyCIMS.Canada.Alberta.Electricity'
        # price = calc_price(node)