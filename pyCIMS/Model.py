from __future__ import print_function
import copy
import networkx as nx
import random
import logging
import warnings

from . import graph_utils
from . import utils
from . import econ

# TODO: deal with iter = 1 to see when it's appropriate to keep a value constant throughout iteration
# TODO: Evaluate whether methods should avoid side effects.
# TODO: Implement automatic defaults
# TODO: Implement logic for determining when to read, calculate, inherit, or use default values
# TODO: Separate the get_service_cost code out into smaller functions & document.
# TODO: Should we be initializing quantities when we initialize prices?


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
        self.quantities = {}
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

    def run(self, equilibrium_threshold=0.005, max_iterations=10):
        """
        Runs the entire model, progressing year-by-year until an equilibrium has been reached for each year.

        Parameters
        ----------
        equilibrium_threshold : float, optional
            The largest relative difference between prices allowed for an equilibrium to be reached. Must be between
            [0, 1]. Relative difference is calculated as the absolute difference between two prices, divided by the
            first price. Defaults to 0.05.

        max_iterations : int, optional
            The maximum number of times to iterate between supply and demand in an attempt to reach an equilibrium. If
            max_iterations is reached, a warning will be raised, iteration for that year will stop, and iteration for
            the next year will begin.

        Returns
        -------
            Nothing is returned, but `self.graph` will be updated with the resulting prices, quantities, etc calculated
            for each year.

        """

        g_demand = graph_utils.get_subgraph(self.graph, ['demand', 'standard'])
        g_supply = graph_utils.get_subgraph(self.graph, ['supply', 'standard'])

        for year in self.years:
            print(f"***** ***** year: {year} ***** *****")

            # Initialize Basic Variables
            equilibrium = False
            iteration = 0

            # Initialize Results
            self.results[year] = {}
            self.quantities[year] = {}

            # Initialize Values
            graph_utils.traverse_graph(self.graph, self.init_prices, year)

            while not equilibrium:
                print('iter {}'.format(iteration))

                # Printing Prices
                prices = {f: list(self.graph.nodes[f][year]['Production Cost'].values())[0]['year_value'] for f
                          in self.fuels}
                for f, p in prices.items():
                    print('\tprice of {}: {}'.format(f, p))


                if iteration > max_iterations:
                    warnings.warn("Max iterations reached for year {}. Continuing on to next year.".format(year))
                    break

                # DEMAND
                # ******************
                # Calculate Service Costs on Demand Side
                graph_utils.breadth_first_post(g_demand, self.get_service_cost, year)

                # Calculate Quantities
                graph_utils.traverse_graph(g_demand, self.calc_quantity, year)

                # Supply
                # ******************
                # Calculate Service Costs on Supply Side
                graph_utils.breadth_first_post(g_supply, self.get_service_cost, year)

                # Calculate Fuel Quantities
                graph_utils.traverse_graph(g_supply, self.calc_quantity, year)

                # Update Prices
                # Go and get all the previous prices
                prev_prices = {f: list(self.graph.nodes[f][year]['Production Cost'].values())[0]['year_value']
                               for f in self.fuels}
                # Now actually update prices
                self.update_prices(year, g_supply)
                # Go get all the new prices
                new_prices = {f: list(self.graph.nodes[f][year]['Production Cost'].values())[0]['year_value']
                              for f in self.fuels}

                # Equilibrium
                # ******************
                def eq_check(prev, new):
                    """
                    Return False unless an equilibrium has been reached.
                        1. Check if prev is empty or year not in previous (first year or first iteration)
                        2. For every fuel, check if the relative difference exceeds the threshold
                            (A) If it does, return False
                            (B) Otherwise, keep checking
                        3. If all fuels are checked and no relative difference exceeds the threshold, return True

                    Parameters
                    ----------
                    prev : dict

                    new : dict


                    Returns
                    -------
                    True if every fuel has changed less than `equilibrium_threshold`. False otherwise.
                    """

                    # For every fuel, check if the relative difference exceeds the threshold
                    for fuel in new:
                        prev_fuel_price = prev[fuel]
                        new_fuel_price = new[fuel]
                        if (prev_fuel_price is None) or (new_fuel_price is None):
                            return False
                        abs_diff = abs(new_fuel_price - prev_fuel_price)
                        rel_diff = abs_diff/prev_fuel_price
                        # If any fuel's relative difference exceeds the threshold, an equilibrium has not been reached
                        if rel_diff > equilibrium_threshold:
                            return False

                    # Otherwise, an equilibrium has been reached
                    return True

                equilibrium = eq_check(prev_prices, new_prices)

                iteration += 1

                # Printing Quantities
                for node, quantity_data in self.quantities[year].items():
                    print('\tquantity of {}: {}'.format(node, quantity_data))

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

        total_lcc_v = 0.0

        # Check if the node is a tech compete node.
        if sub_graph.nodes[node]["competition type"] == "tech compete":
            # Initialize the results for the node at year
            results[year][node] = {}

            # Find the v parameter
            v = self.get_heterogeneity(sub_graph, node, year)

            # For every tech in the node, compute econ calculations
            for tech in sub_graph.nodes[node][year]["technologies"].keys():
                # Calculate Econ Measures
                def calculate_tech_econ():
                    # Initialize calculable value
                    tech_param_names = ["Service cost", "CRF", "LCC", "Full capital cost"]
                    tech_param_values = [0.0, 0.0, 0.0, 0.0]
                    # go through all tech parameters and make sure we don't overwrite something that exists
                    for param_name, param_val in zip(tech_param_names, tech_param_values):
                        if param_name not in sub_graph.nodes[node][year]['technologies'][tech].keys():
                            self.add_tech_element(sub_graph, node, year, tech, param_name, value=param_val)

                    # Initialize values with defaults
                    def_param_names = ['Operating cost', 'Output']
                    def_param_values = [0, 1]
                    for param_name, param_val in zip(def_param_names, def_param_values):
                        if sub_graph.nodes[node][year]["technologies"][tech][param_name]["year_value"] is None:
                            sub_graph.nodes[node][year]["technologies"][tech][param_name]["year_value"] = param_val

                calculate_tech_econ()

                # go through available techs (range is [lower year, upper year + 1] to work with range function
                low, up = utils.range_available(sub_graph, node, tech)

                if int(year) in range(low, up):
                    # get service cost
                    service_cost = econ.get_technology_service_cost(sub_graph, self.graph, node, year, tech, fuels)
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

                    if curr_lcc > 0.0:
                        if rand:
                            marketshare = random.random()
                        else:
                            marketshare = curr_lcc ** (-1.0 * v) / total_lcc_v

                    results[year][node][tech] = {"marketshare": marketshare}
                    weighted_lccs += marketshare * curr_lcc

                sub_graph.nodes[node][year]['technologies'][tech]['Market share']['year_value'] = marketshare

            sub_graph.nodes[node][year]["total lcc"] = weighted_lccs

        self.results = results

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

    def init_prices(self, graph, node, year, step=5):
        def init_node_price_multipliers():
            # Grab the price multipliers from the parents (if they exist)
            parents = list(graph.predecessors(node))
            parent_price_multipliers = {}
            if len(parents) > 0:
                parent = parents[0]
                if 'Price Multiplier' in graph.nodes[parent][year].keys():
                    parent_price_multipliers.update(graph.nodes[parent][year]['Price Multiplier'])

            # Grab the price multipliers from the current node
            node_price_multipliers = {}
            # Grab the price multiplier from the current node (if they exist)
            if 'Price Multiplier' in graph.nodes[node][year].keys():
                node_price_multipliers.update(graph.nodes[node][year]['Price Multiplier'])

            # Update the Parent's Multipliers by the Child's (through multiplication)
            for fuel, mult in node_price_multipliers.items():
                if fuel in parent_price_multipliers.keys():
                    parent_price_multipliers[fuel]['year_value'] *= mult['year_value']
                else:
                    parent_price_multipliers[fuel] = mult

            # Set Price Multiplier of node in the graph
            graph.nodes[node][year]['Price Multiplier'] = parent_price_multipliers

        def old_init():
            # --- OLD CODE BELOW HERE -- STILL IN USE
            self.results[year][node] = {}
            # Attributes
            if (graph.nodes[node]['competition type'] == 'region') & ('Attribute' in graph.nodes[node][year].keys()):
                attributes = graph.nodes[node][year]["Attribute"]
                # keeping attribute values (macroeconomic indicators) in result dict
                for indicator, value in attributes.items():
                    # TODO: CHANGE to be unit agnostic. Instead be something like node_name.attribute_name
                    # indicator is name, for now naming by unit for ease in fetching from children
                    self.results[year][node][value["unit"]] = value["year_value"]

            # Services Being Provided
            if (graph.nodes[node]['competition type'] == 'sector') & (graph.nodes[node]['type'] == 'demand'):
                service_unit, provided = econ.get_provided(graph, node, year, self.results)
                self.results[year][node][service_unit] = provided
                # Services Being Requested
                requested = graph.nodes[node][year]["Service requested"]
                for req in requested.values():
                    self.results[year][node][service_unit] = provided * req["year_value"]

        def init_prices_to_be_estimated():
            """
            Needs to estimate Production Costs for values to be estimated. Will start by using the value settled on
            in the previous year.
            """
            # Determine if a fuel
            if node in self.fuels:
                fuel_name = node.split('.')[-1]
                if graph.nodes[node][year]['Production Cost'][fuel_name]['year_value'] is None:
                    graph.nodes[node][year]['Production Cost'][fuel_name]['to_estimate'] = True
                    last_year = str(int(year) - step)
                    last_year_value = graph.nodes[node][last_year]['Production Cost'][fuel_name]['year_value']
                    graph.nodes[node][year]['Production Cost'][fuel_name]['year_value'] = last_year_value
                else:
                    graph.nodes[node][year]['Production Cost'][fuel_name]['to_estimate'] = False

        init_node_price_multipliers()
        old_init()
        init_prices_to_be_estimated()

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

        Parameters
        ----------
        year : str
        supply_subgraph: nx.Graph

        Returns
        -------
        A dictionary containing the updated prices. Keys are fuel nodes, values are prices.
        """
        prev_prices = {f: list(self.graph.nodes[f][year]['Production Cost'].values())[0] for f in
                       self.fuels}
        price_keys_to_estimate = [f for f, d in prev_prices.items() if d['to_estimate']]
        new_prices = copy.deepcopy(prev_prices)

        # For each price we need to estimate, go find the price
        def calc_price(node_name):
            # Base Case is that we hit a tech compete node
            if supply_subgraph.nodes[node_name]['competition type'] == 'tech compete':
                return supply_subgraph.nodes[node_name][year]['total lcc']

            elif supply_subgraph.nodes[node_name]['competition type'] in ['fixed ratio', 'sector']:
                # Find all the services being requested
                services_requested = [(d['branch'], d['year_value'])
                                      for s, d in supply_subgraph.nodes[node_name][year]['Service requested'].items()]

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
            new_prices[fuel]['year_value'] = fuel_price
            # Update the price @ the fuel node
            fuel_name = fuel.split('.')[-1]
            self.graph.nodes[fuel][year]['Production Cost'][fuel_name]['year_value'] = fuel_price

        return new_prices

    def calc_quantity(self, sub_graph, node, year):
        """
        Calculate how much of each provided service node is providing in year. Expectation is this will be used in the
        `traverse_graph` function, where we only visit a node once its parents have already been visited.

        Parameters
        ----------
        node : str
            The name of the node whose quantity is being calculated.

        sub_graph: nx.DiGraph
            The subgraph being traversed. This is not used in this function, but is required of all functions used in
            the `traverse_graph` function.

        year : str
            The year for which we are calculating quantity.

        Returns
        -------
        None
            self.quantities will be updated with the amount provided by the node's services in the given year.
        """

        node_year_data = self.graph.nodes[node][year]

        # What is being provided by the node?
        services_to_provide = node_year_data['Service provided']

        # How much needs to be provided, based on what was requested of it?
        totals_to_provide = {}
        for provided_service, provided_service_data in services_to_provide.items():
            service_unit = provided_service_data['unit'].strip().lower()
            if self.graph.nodes[node]['competition type'] == 'root':
                totals_to_provide[service_unit] = 1
            else:
                totals_to_provide[service_unit] = self.quantities[year][node][service_unit]

        # Based on what this node needs to provide, find out how much it must request from other services
        def calc_quantity_from_services(requested_services):
            if isinstance(requested_services, dict):
                requested_services = [requested_services]

            for service_data in requested_services:

                # Find the units
                result_unit = service_data['unit'].split('/')[0].strip().lower()
                per_unit = service_data['unit'].split('/')[-1].strip().lower()

                # Find the multiplier based on quantity to be provided
                service_amount_mult = 1
                for provided_service_unit, provided_service_amount in totals_to_provide.items():
                    if provided_service_unit.lower() == per_unit:
                        service_amount_mult = provided_service_amount

                # Find market share
                try:
                    market_share = service_data['Market share']['year_value']
                except KeyError:
                    market_share = 1

                # Find the ratio to provide
                year_value = service_data['year_value']

                # Calculate the total quantity requested
                quant_requested = market_share * year_value * service_amount_mult

                # Check if we need to initialize quantities
                year_quantities = self.quantities[year]
                if service_data['branch'] not in year_quantities.keys():
                    year_quantities[service_data['branch']] = {}
                    year_quantities[service_data['branch']][result_unit] = 0

                # Save in results
                self.quantities[year][service_data['branch']][result_unit] += quant_requested

        if 'technologies' in node_year_data:
            # For each technology, find the services being requested
            for tech, tech_data in node_year_data['technologies'].items():
                if 'Service requested' in tech_data.keys():
                    services_being_requested = tech_data['Service requested']
                    # Calculate the quantities being for each of the services
                    calc_quantity_from_services(services_being_requested)

        elif 'Service requested' in node_year_data:
            # Calculate the quantities being for each of the services
            services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
            calc_quantity_from_services(services_being_requested)
