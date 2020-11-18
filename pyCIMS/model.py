import copy
import math
import warnings
import pprint as pp
import networkx as nx

from . import graph_utils
from . import utils
from . import econ
from . import lcc_calculation

# TODO: Evaluate whether methods should avoid side effects.
# TODO: Implement automatic defaults
# TODO: Implement logic for determining when to read, calculate, inherit, or use default values
# TODO: Separate the get_service_cost code out into smaller functions & document.
# TODO: Should we be initializing quantities when we initialize prices?


class NodeQuantity:
    def __init__(self):
        self.quantities = {}

    def add_quantity_request(self, amount, requesting_node, requesting_technology=None):
        node_tech = '{}[{}]'.format(requesting_node, requesting_technology)
        self.quantities[node_tech] = amount

    def get_total_quantity(self):
        total = 0
        for amount in self.quantities.values():
            total += amount
        return total


class Model:
    """
    Relevant dataframes and associated information taken from the model description provided in
    `reader`. Also includes methods needed for building and running the Model.

    Parameters
    ----------
    reader : pyCIMS.reader
        The Reader set up to ingest the description (excel file) for our model.

    Attributes
    ----------
    graph : networkx.DiGraph
        Model Graph populated using the `build_graph` method. Model services are nodes in `graph`,
        with data contained within an associated dictionary. Structural and Request/Provide
        relationships are edges in the `graph`.

    node_dfs : dict {str: pandas.DataFrame}
        Node names (branch form) are the keys in the dictionary. Associated DataFrames (specified in
        the excel model description) are the values. DataFrames do not include 'Technology' or
        'Service' information for a node.

    tech_dfs : dict {str: dict {str: pandas.DataFrame}}
        Technology & service information from the excel model description. Node names (branch form)
        are keys in `tech_dfs` to sub-dictionaries. These sub-dictionaries have technology/service
        names as keys and pandas DataFrames as values. These DataFrames contain information from the
        excel model description.

    fuels : list [str]
        List of supply-side sector nodes (fuels, etc) requested by the demand side of the Model
        Graph.  Populated using the `build_graph` method.

    years : list [str or int]
        List of the years for which the model will be run.

    """

    def __init__(self, model_reader):
        self.graph = nx.DiGraph()
        self.node_dfs, self.tech_dfs = model_reader.get_model_description()
        self.step = 5  # Make this an input later

        self.fuels = []
        self.years = model_reader.get_years()
        self.base_year = int(self.years[0])

        self.prices = {}

        self.build_graph()

        self.show_run_warnings = True

    def build_graph(self):
        """

        Builds graph based on the model reader used in instantiation of the class. Stores this graph
        in `self.graph`. Additionally, initializes `self.fuels`.

        Returns
        -------
        None

        """
        graph = nx.DiGraph()
        node_dfs = self.node_dfs
        tech_dfs = self.tech_dfs

        graph = graph_utils.make_nodes(graph, node_dfs, tech_dfs)
        graph = graph_utils.make_edges(graph, node_dfs, tech_dfs)

        self.fuels = graph_utils.get_fuels(graph, self.years)
        self.graph = graph

    def run(self, equilibrium_threshold=0.005, max_iterations=10, show_warnings=True):
        """
        Runs the entire model, progressing year-by-year until an equilibrium has been reached for
        each year.

        Parameters
        ----------
        equilibrium_threshold : float, optional
            The largest relative difference between prices allowed for an equilibrium to be reached.
            Must be between [0, 1]. Relative difference is calculated as the absolute difference
            between two prices, divided by the first price. Defaults to 0.05.

        max_iterations : int, optional
            The maximum number of times to iterate between supply and demand in an attempt to reach
            an equilibrium. If max_iterations is reached, a warning will be raised, iteration for
            that year will stop, and iteration for the next year will begin.

        verbose : bool, optional
            Whether or not to have verbose printing during iterations. If true, fuel prices are
            printed at the end of each iteration.

        Returns
        -------
            Nothing is returned, but `self.graph` will be updated with the resulting prices,
            quantities, etc calculated for each year.

        """
        self.show_run_warnings = show_warnings

        # Find the demand subtree
        g_demand = graph_utils.get_subgraph(self.graph, ['demand', 'standard'])

        # Find the supply subtree
        g_supply = graph_utils.get_subgraph(self.graph, ['supply', 'standard'])

        for year in self.years:
            print(f"***** ***** year: {year} ***** *****")

            # Initialize Basic Variables
            equilibrium = False
            iteration = 0

            # Initialize Graph Values
            self.initialize_graph(self.graph, year)
            # graph_utils.top_down_traversal(self.graph, self.initialize_node, year)

            while not equilibrium:
                print('iter {}'.format(iteration))
                # Early exit if we reach the maximum number of iterations
                if iteration > max_iterations:
                    warnings.warn("Max iterations reached for year {}. "
                                  "Continuing to next year.".format(year))
                    break

                # Initialize Iteration Specific Values
                self.iteration_initialization(year)

                # DEMAND
                # ******************
                # Calculate LCC values on demand side
                graph_utils.bottom_up_traversal(g_demand,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self.step,
                                                self.graph,
                                                self.fuels)

                for ix in range(4):
                    # Calculate Quantities (Total Stock Needed)
                    graph_utils.top_down_traversal(g_demand,
                                                   self.stock_retirement_and_allocation,
                                                   year)

                    if int(year) == self.base_year:
                        break

                    # Calculate Service Costs on Demand Side
                    # graph_utils.bottom_up_traversal(g_demand, self.get_service_cost, year)
                    graph_utils.bottom_up_traversal(g_demand,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self.step,
                                                    self.graph,
                                                    self.fuels)

                # Supply
                # ******************
                # Calculate Service Costs on Supply Side
                # graph_utils.bottom_up_traversal(g_supply, self.initial_lcc_calculation, year)
                graph_utils.bottom_up_traversal(g_supply,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self.step,
                                                self.graph,
                                                self.fuels)
                for _ in range(4):
                    # Calculate Fuel Quantities
                    graph_utils.top_down_traversal(g_supply,
                                                   self.stock_retirement_and_allocation,
                                                   year)

                    if int(year) == self.base_year:
                        break

                    # Calculate Service Costs on Demand Side
                    # graph_utils.bottom_up_traversal(g_demand, self.get_service_cost, year)
                    graph_utils.bottom_up_traversal(g_supply,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self.step,
                                                    self.graph,
                                                    self.fuels)

                # Update Prices
                # *************
                # Go and get all the previous prices
                prev_prices = {f: list(self.graph.nodes[f][year]['Life Cycle Cost'].values())[0]['year_value']
                               for f in self.fuels}

                # Now actually update prices
                self.update_prices(year, g_supply)

                # Go get all the new prices
                new_prices = {f: list(self.graph.nodes[f][year]['Life Cycle Cost'].values())[0]['year_value']
                              for f in self.fuels}

                # Check for an Equilibrium
                # ************************
                equilibrium = (int(year) == self.base_year) or \
                              self.check_equillibrium(prev_prices,
                                                      new_prices,
                                                      equilibrium_threshold)

                # Print if Verbose
                # ****************
                # if verbose:
                #     self.print_prices(year)

                # Next Iteration
                # **************
                iteration += 1

    def check_equillibrium(self, prev, new, threshold):
        """
        Return False unless an equilibrium has been reached.
            1. Check if prev is empty or year not in previous (first year or first
               iteration)
            2. For every fuel, check if the relative difference exceeds the threshold
                (A) If it does, return False
                (B) Otherwise, keep checking
            3. If all fuels are checked and no relative difference exceeds the
               threshold, return True

        Parameters
        ----------
        prev : dict

        new : dict

        threshold : float

        Returns
        -------
        True if all fuels changed less than `threshold`. False otherwise.
        """

        # For every fuel, check if the relative difference exceeds the threshold
        for fuel in new:
            prev_fuel_price = prev[fuel]
            if prev_fuel_price == 0:
                if self.show_run_warnings:
                    warnings.warn("Previous fuel price is 0 for {}".format(fuel))
                prev_fuel_price = 1
            new_fuel_price = new[fuel]
            if (prev_fuel_price is None) or (new_fuel_price is None):
                return False
            abs_diff = abs(new_fuel_price - prev_fuel_price)
            rel_diff = abs_diff / prev_fuel_price
            # If any fuel's relative difference exceeds the threshold, an equilibrium
            # has not been reached
            if rel_diff > threshold:
                return False

        # Otherwise, an equilibrium has been reached
        return True

    def initialize_graph(self, graph, year):
        """
        Initializes the graph at the start of a simulation year.
        Specifically, initializes (1) price multiplier values and (2) fuel nodes' LCC value.

        Parameters
        ----------
        graph : NetworkX.DiGraph
            The graph object being initialized.

        year: str
            The string representing the current simulation year (e.g. "2005").

        Returns
        -------
        Nothing is returned, but `self.graph` will be updated with the initialized nodes.
        """
        def init_node_price_multipliers(graph, node, year):
            """
            Function for initializing the Price Multipler values for a given node in a graph. This
            function assumes all of node's parents have already had their price multipliers
            initialized.

            Parameters
            ----------
            graph : NetworkX.DiGraph
                A graph object containing the node of interest.
            node : str
                Name of the node to be initialized.

            year: str
                The string representing the current simulation year (e.g. "2005").

            Returns
            -------
            Nothing is returned, but `graph.nodes[node]` will be updated with the initialized price
            multiplier values.
            """
            # Retrieve price multipliers from the parents (if they exist)
            parents = list(graph.predecessors(node))
            parent_price_multipliers = {}
            if len(parents) > 0:
                parent = parents[0]
                if 'Price Multiplier' in graph.nodes[parent][year]:
                    parent_price_multipliers.update(graph.nodes[parent][year]['Price Multiplier'])

            # Grab the price multiplier from the current node (if they exist)
            node_price_multipliers = {}
            if 'Price Multiplier' in graph.nodes[node][year]:
                node_price_multipliers.update(graph.nodes[node][year]['Price Multiplier'])

            # Multiply the node's price multipliers by its parents' price multipliers
            for fuel, mult in node_price_multipliers.items():
                if fuel in parent_price_multipliers:
                    parent_price_multipliers[fuel]['year_value'] *= mult['year_value']
                else:
                    parent_price_multipliers[fuel] = mult

            # Set Price Multiplier of node in the graph
            graph.nodes[node][year]['Price Multiplier'] = parent_price_multipliers

        def init_fuel_lcc(graph, node, year, step=5):
            """
            Function for initializing LCC for a node in a graph, if that node is a fuel node. This
            function assumes all of node's children have already been processed by this function.

            Parameters
            ----------
            graph : NetworkX.DiGraph
                A graph object containing the node of interest.

            node : str
                Name of the node to be initialized.

            year: str
                The string representing the current simulation year (e.g. "2005").

            step: int, optional
                The number of years between simulation years. Default is 5.

            Returns
            -------
            Nothing is returned, but `graph.nodes[node]` will be updated with the initialized LCC if
            node is a fuel node.
            """

            def calc_lcc_from_children():
                """
                Helper function to calculate a node's LCC from its children.

                Returns
                -------
                Nothing is returned, but the node will be updated with a new LCC value.
                """
                # Find the subtree rooted at the fuel node
                descendants = [n for n in graph.nodes if node in n]
                descendant_tree = nx.subgraph(graph, descendants)

                # Calculate the LCCs for the sub-tree
                graph_utils.bottom_up_traversal(descendant_tree,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self.step,
                                                self.graph,
                                                self.fuels,
                                                root=node
                                                )

            if node in self.fuels:
                if "Life Cycle Cost" not in graph.nodes[node][year]:
                    # LCC needs to be calculated from children
                    calc_lcc_from_children()
                    lcc_dict = graph.nodes[node][year]['Life Cycle Cost']
                    fuel_name = list(lcc_dict.keys())[0]
                    lcc_dict[fuel_name]['to_estimate'] = True

                else:
                    lcc_dict = graph.nodes[node][year]['Life Cycle Cost']
                    fuel_name = list(lcc_dict.keys())[0]
                    if lcc_dict[fuel_name]['year_value'] is None:
                        lcc_dict[fuel_name]['to_estimate'] = True
                        last_year = str(int(year) - step)
                        last_year_value = graph.nodes[node][last_year]['Life Cycle Cost'][fuel_name]['year_value']
                        graph.nodes[node][year]['Life Cycle Cost'][fuel_name]['year_value'] = last_year_value
                    else:
                        graph.nodes[node][year]['Life Cycle Cost'][fuel_name]['to_estimate'] = False

        graph_utils.top_down_traversal(graph,
                                       init_node_price_multipliers,
                                       year)

        graph_utils.bottom_up_traversal(graph,
                                        init_fuel_lcc,
                                        year)

    def iteration_initialization(self, year):
        # Reset the quantities at each node
        for n in self.graph.nodes():
            self.graph.nodes[n][year]['quantities'] = NodeQuantity()

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
        prev_prices = {f: list(self.graph.nodes[f][year]['Life Cycle Cost'].values())[0] for f in
                       self.fuels}

        price_keys_to_estimate = [f for f, d in prev_prices.items() if d['to_estimate']]
        new_prices = copy.deepcopy(prev_prices)

        # For each price we need to estimate, go find the price
        def calc_price(node_name):
            # Base Case is that we hit a tech compete node
            if supply_subgraph.nodes[node_name]['competition type'] == 'tech compete':
                lcc_dict = supply_subgraph.nodes[node_name][year]['Life Cycle Cost']
                fuel_name = list(lcc_dict.keys())[0]
                return supply_subgraph.nodes[node_name][year]['Life Cycle Cost'][fuel_name]['year_value']

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
            fuel_name = list(self.graph.nodes[fuel][year]['Life Cycle Cost'].keys())[0]
            self.graph.nodes[fuel][year]['Life Cycle Cost'][fuel_name]['year_value'] = fuel_price

        return new_prices

    def stock_retirement_and_allocation(self, sub_graph, node, year):
        def helper_quantity_from_services(requested_services, assessed_demand, technology=None, technology_market_share=1):
            if isinstance(requested_services, dict):
                requested_services = [requested_services]

            for service_data in requested_services:
                # Find the multiplier based on quantity to be provided
                service_amount_mult = assessed_demand

                # Find the ratio to provide
                year_value = service_data['year_value']

                # Calculate the total quantity requested
                quant_requested = technology_market_share * year_value * service_amount_mult

                # Check if we need to initialize quantities
                year_node = self.graph.nodes[service_data['branch']][year]
                if 'quantities' not in year_node.keys():
                    year_node['quantities'] = NodeQuantity()

                # Save results
                year_node['quantities'].add_quantity_request(amount=quant_requested,
                                                             requesting_node=node,
                                                             requesting_technology=technology
                                                             )

        def general_allocation():
            node_year_data = self.graph.nodes[node][year]

            # What is being provided by the node?
            service_to_provide = node_year_data['Service provided']

            # How much needs to be provided, based on what was requested of it?
            if self.graph.nodes[node]['competition type'] == 'root':
                total_to_provide = 1
            else:
                total_to_provide = self.graph.nodes[node][year]['quantities'].get_total_quantity()

            # Based on what this node needs to provide, find out how much it must request from other
            # services
            if 'technologies' in node_year_data:
                # For each technology, find the services being requested
                for tech, tech_data in node_year_data['technologies'].items():
                    if 'Service requested' in tech_data.keys():
                        services_being_requested = tech_data['Service requested']
                        t_ms = tech_data['Market share']['year_value']
                        # Calculate the quantities being for each of the services
                        helper_quantity_from_services(services_being_requested,
                                                      total_to_provide,
                                                      technology=tech,
                                                      technology_market_share=t_ms)

            elif 'Service requested' in node_year_data:
                # Calculate the quantities being for each of the services
                services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
                helper_quantity_from_services(services_being_requested, total_to_provide)

        def tech_compete_allocation():
            # Demand Assessment
            # *****************
            # Count demanded quantities from above nodes or technologies (for the top node, derive
            # demand quantities external to the sector).
            node_year_data = self.graph.nodes[node][year]

            # How much needs to be provided, based on what was requested of it?
            assessed_demand = self.graph.nodes[node][year]['quantities'].get_total_quantity()

            # Existing Tech Specific Stocks
            # *****************************
            # Retrieve existing technology stocks quantities from ‘existing stock database’ for
            # simulation year once vintage-specific retirements are conducted for simulation year.
            existing_stock_per_tech = {}
            for t in node_year_data['technologies']:
                t_existing = self.small_calc_existing_stock(sub_graph, node, year, t)
                existing_stock_per_tech[t] = t_existing

            # Retrofit
            # TODO: Much later we will add this

            # Assessment of capital stock availability
            # ****************************************
            # Subtract total stock for each technology, after natural retirements, from demanded
            # quantities to determine how many new stock must be allocated through competition. If
            # remaining stock already meets quantity demanded, no competition is required and
            # un-needed surplus stock is force retired starting with the oldest vintage at a
            # proportion of each vintages's remaining total market shares.
            # TODO: Deal with surplus retirements
            new_stock_demanded = copy.copy(assessed_demand)
            for t, e_stocks in existing_stock_per_tech.items():
                new_stock_demanded -= e_stocks

            # Surplus Retirements (aka Early Retirement)
            if new_stock_demanded < 0:
                surplus = -1 * new_stock_demanded

                # Base Stock Retirement
                total_base_stock = 0
                for tech in existing_stock_per_tech:
                    t_base_stock = self.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining']
                    total_base_stock += t_base_stock

                if total_base_stock == 0:
                    pass
                else:
                    retirement_proportion = max(0, min(surplus / total_base_stock, 1))
                    for tech in existing_stock_per_tech:
                        t_base_stock = self.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining']
                        amount_tech_to_retire = t_base_stock * retirement_proportion
                        # Remove from existing stock
                        existing_stock_per_tech[tech] -= amount_tech_to_retire
                        # Remove from surplus & new stock demanded
                        surplus -= amount_tech_to_retire
                        new_stock_demanded += amount_tech_to_retire
                        # note early retirement in the model
                        self.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining'] -= amount_tech_to_retire

            # New Tech Competition
            # ********************
            # Calculate “New Market Share” percentages to allocate new technology stocks according
            # to demand.
            # TODO: Add other calculations based on whether tech compete, winner take all,
            #  fixed market share
            new_market_shares_per_tech = {}
            for t in node_year_data['technologies']:
                new_market_shares_per_tech[t] = {}
                market_share_exogenous = sub_graph.nodes[node][year]['technologies'][t]['Market share']['exogenous']

                if market_share_exogenous:
                    new_market_share = sub_graph.nodes[node][year]['technologies'][t]['new_market_share']
                else:
                    new_market_share = 0

                    # Find the years the technology is available
                    # TODO: Check that low, up is correct and not off by 1
                    # TODO: Check that marketshares are accurately being calculated
                    low, up = utils.range_available(sub_graph, node, t)
                    if low < int(year) < up:
                        v = econ.get_heterogeneity(sub_graph, node, year)
                        tech_lcc = sub_graph.nodes[node][year]["technologies"][t]["LCC"]["year_value"]
                        total_lcc_v = self.graph.nodes[node][year]["total_lcc_v"]
                        if tech_lcc == 0:
                            # TODO: Address the problem of a 0 LCC properly
                            if self.show_run_warnings:
                                warnings.warn("Technology {} @ node {} has an LCC=0".format(t, node))
                            tech_lcc = 0.0001
                        if tech_lcc < 0:
                            if self.show_run_warnings:
                                warnings.warn("Technology {} @ node {} has a negative LCC".format(t, node))
                            tech_lcc = 0.0001
                        try:
                            new_market_share = tech_lcc ** (-1 * v) / total_lcc_v
                        except OverflowError:
                            if self.show_run_warnings:
                                warnings.warn("Overflow Error when calculating new marketshare for "
                                              "tech {} @ node {}".format(t, node))

                self.graph.nodes[node][year]['technologies'][t]['base_stock'] = 0
                self.graph.nodes[node][year]['technologies'][t]['new_stock'] = 0

                new_market_shares_per_tech[t] = new_market_share
                if int(year) == self.base_year:
                    self.graph.nodes[node][year]['technologies'][t]['base_stock'] = new_stock_demanded * new_market_share
                else:
                    self.graph.nodes[node][year]['technologies'][t]['new_stock'] = new_stock_demanded * new_market_share

                self.graph.nodes[node][year]['technologies'][t]['new_market_share'] = new_market_share

            # Calculate Total Market shares -- remaining + new stock
            total_market_shares_by_tech = {}
            for t in node_year_data['technologies']:
                try:
                    existing_stock = existing_stock_per_tech[t]
                except KeyError:
                    existing_stock = 0

                tech_total_stock = existing_stock + new_market_shares_per_tech[t] * new_stock_demanded

                # TODO: Deal with assessed_demand == 0
                # TODO: WARN when assessed_demand is 0. Assign a total marketshare of 0. Might need
                #  to propogate this to nodes below/deal with the case of all marketshares being 0.
                if assessed_demand == 0:
                    if self.show_run_warnings:
                        warnings.warn("Assessed Demand is 0 for {}[{}]".format(node, t))
                    total_market_share = 0
                else:
                    total_market_share = tech_total_stock / assessed_demand

                total_market_shares_by_tech[t] = total_market_share
                self.graph.nodes[node][year]['technologies'][t]['total_market_share'] = total_market_share

            # Send demand quantities to services below
            # Based on what this node needs to provide, find out how much it must request from
            # other services
            if 'technologies' in node_year_data:
                # For each technology, find the services being requested
                for tech, tech_data in node_year_data['technologies'].items():
                    if 'Service requested' in tech_data.keys():
                        services_being_requested = tech_data['Service requested']
                        # Calculate the quantities being for each of the services
                        t_ms = total_market_shares_by_tech[tech]
                        helper_quantity_from_services(services_being_requested, assessed_demand,
                                                      technology=tech, technology_market_share=t_ms)

            elif 'Service requested' in node_year_data:
                # Calculate the quantities being for each of the services
                services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
                helper_quantity_from_services(services_being_requested, assessed_demand)

        # Move into the proper allocation function
        if self.graph.nodes[node]['competition type'] == 'tech compete':
            tech_compete_allocation()
        else:
            general_allocation()

    def small_calc_existing_stock(self, graph, node, year, tech):
        def base_stock_retirement(base_stock, initial_year, current_year, lifespan=10):
            # How much base stock remains if only natural retirements have occurred?
            naturally_unretired_base_stock = base_stock * (1 - (int(current_year) - int(initial_year)) / lifespan)

            # What is the remaining base stock from the previous year? This considers early
            # retirements.
            prev_year = str(int(year) - self.step)
            if int(prev_year) == self.base_year:
                prev_year_unretired_base_stock = graph.nodes[node][prev_year]['technologies'][tech]['base_stock']
            else:
                prev_year_unretired_base_stock = graph.nodes[node][prev_year]['technologies'][tech]['base_stock_remaining']

            base_stock_remaining = max(min(naturally_unretired_base_stock, prev_year_unretired_base_stock), 0.0)

            return base_stock_remaining

        def purchased_stock_retirement(purchased_stock, purchased_year, current_year, lifespan, intercept=11.513):
            exponent = (intercept / lifespan) * ((int(current_year) - int(purchased_year)) - lifespan)
            unretired_purchased_stock = purchased_stock / (1 + math.exp(exponent))
            retired_purchased_stock = purchased_stock - unretired_purchased_stock

            return max(unretired_purchased_stock, 0), retired_purchased_stock

        if graph.nodes[node]['competition type'] != 'tech compete':
            # we don't care about existing stock for non-tech compete nodes
            return 0

        earlier_years = [y for y in self.years if int(y) < int(year)]

        if len(earlier_years) == 0:
            return 0

        # Means we are not on the initial year & we need to calculate remaining base and new stock
        # (existing)
        existing_stock = 0
        remaining_base_stock = 0
        for y in earlier_years:
            # TODO: Allow default parameters
            tech_lifespan = self.graph.nodes[node][y]['technologies'][tech]['Lifetime']['year_value']
            tech_lifespan = 10 if tech_lifespan is None else tech_lifespan

            # Base Stock
            tech_base_stock_y = self.graph.nodes[node][y]['technologies'][tech]['base_stock']
            remain_base_stock_vintage_y = base_stock_retirement(tech_base_stock_y, y, year, tech_lifespan)
            remaining_base_stock += remain_base_stock_vintage_y
            existing_stock += remain_base_stock_vintage_y

            # New Stock
            tech_new_stock_y = self.graph.nodes[node][y]['technologies'][tech]['new_stock']
            remain_new_stock, retired_new_stock = purchased_stock_retirement(tech_new_stock_y, y,
                                                                             year, tech_lifespan)
            existing_stock += remain_new_stock

        graph.nodes[node][year]['technologies'][tech]['base_stock_remaining'] = remaining_base_stock

        return existing_stock