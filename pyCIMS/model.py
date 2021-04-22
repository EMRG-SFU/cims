import copy
import math
import warnings
import networkx as nx

from . import graph_utils
from . import utils
from . import lcc_calculation
from . import stock_allocation

from .utils import create_value_dict

# TODO: Separate the get_service_cost code out into smaller functions & document.


class ProvidedQuantity:
    def __init__(self):
        self.provided_quantities = {}

    def provide_quantity(self, amount, requesting_node, requesting_technology=None):
        node_tech = '{}[{}]'.format(requesting_node, requesting_technology)
        self.provided_quantities[node_tech] = amount

    def get_total_quantity(self):
        total = 0
        for amount in self.provided_quantities.values():
            total += amount
        return total

    def get_quantity_provided_to_node(self, node):
        """
        Find the quantity being provided to a specific node, across all it's technologies
        """
        total_provided_to_node = 0
        for pq in self.provided_quantities:
            pq_node, pq_tech = pq.split('[', 1)
            if pq_node == node:
                total_provided_to_node += self.provided_quantities[pq]
        return total_provided_to_node


class RequestedQuantity:
    def __init__(self):
        self.requested_quantities = {}

    def record_requested_quantity(self, providing_node, child, amount):
        if providing_node in self.requested_quantities:
            if child in self.requested_quantities[providing_node]:
                self.requested_quantities[providing_node][child] += amount
            else:
                self.requested_quantities[providing_node][child] = amount

        else:
            self.requested_quantities[providing_node] = {child: amount}

    def get_total_quantities_requested(self):
        total_quants = {}
        for service in self.requested_quantities:
            total_service = 0
            for child, quantity in self.requested_quantities[service].items():
                total_service += quantity
            total_quants[service] = total_service
        return total_quants

    def sum_requested_quantities(self):
        total_quantity = 0
        for fuel in self.requested_quantities:
            fuel_rq = self.requested_quantities[fuel]
            for source in fuel_rq:
                total_quantity += fuel_rq[source]

        return total_quantity


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
        self.technology_defaults, self.node_defaults = model_reader.get_default_tech_params()
        self.step = 5  # TODO: Make this an input or calculate
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

        self.fuels = graph_utils.get_fuels(graph)
        self.graph = graph

    def run(self, equilibrium_threshold=0.05, max_iterations=10, show_warnings=True):
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
            provided_quantities, etc calculated for each year.

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
                # Calculate Life Cycle Cost values on demand side
                graph_utils.bottom_up_traversal(g_demand,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self)

                for _ in range(4):
                    # Calculate Quantities (Total Stock Needed)
                    graph_utils.top_down_traversal(g_demand,
                                                   self.stock_retirement_and_allocation,
                                                   year)

                    if int(year) == self.base_year:
                        break

                    # Calculate Service Costs on Demand Side
                    graph_utils.bottom_up_traversal(g_demand,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self)

                # Supply
                # ******************
                # Calculate Service Costs on Supply Side
                graph_utils.bottom_up_traversal(g_supply,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self)
                for _ in range(4):
                    # Calculate Fuel Quantities
                    graph_utils.top_down_traversal(g_supply,
                                                   self.stock_retirement_and_allocation,
                                                   year)

                    if int(year) == self.base_year:
                        break

                    # Calculate Service Costs on Supply Side
                    graph_utils.bottom_up_traversal(g_supply,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self)

                # Check for an Equilibrium
                # ************************
                # Find the previous prices
                prev_prices = self.prices

                # Go get all the new prices
                new_prices = {fuel: self.get_param('Life Cycle Cost', fuel, year)
                              for fuel in self.fuels}


                equilibrium = (int(year) == self.base_year) or \
                              self.check_equilibrium(prev_prices,
                                                     new_prices,
                                                     equilibrium_threshold)
                self.prices = new_prices

                # Next Iteration
                # **************
                iteration += 1

            # Once we've reached an equilibrium, calculate the quantities requested by each node.
            graph_utils.bottom_up_traversal(self.graph,
                                            self.calc_requested_quantities,
                                            year)

    def check_equilibrium(self, prev, new, threshold):
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
                prev_fuel_price = self.get_node_parameter_default('Life Cycle Cost', 'sector')
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
        Specifically, initializes (1) price multiplier values and (2) fuel nodes' Life Cycle Cost
        value.

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
                    price_multipliers = self.get_param('Price Multiplier', parent, year)
                    parent_price_multipliers.update(price_multipliers)

            # Grab the price multiplier from the current node (if they exist)
            node_price_multipliers = {}
            if 'Price Multiplier' in graph.nodes[node][year]:
                price_multipliers = self.get_param('Price Multiplier', node, year)
                node_price_multipliers.update(price_multipliers)

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
            Function for initializing Life Cycle Cost for a node in a graph, if that node is a fuel
            node. This function assumes all of node's children have already been processed by this
            function.

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
            Nothing is returned, but `graph.nodes[node]` will be updated with the initialized Life
            Cycle Cost if node is a fuel node.
            """

            def calc_lcc_from_children():
                """
                Helper function to calculate a node's Life Cycle Cost from its children.

                Returns
                -------
                Nothing is returned, but the node will be updated with a new Life Cycle Cost value.
                """
                # Find the subtree rooted at the fuel node
                descendants = [n for n in graph.nodes if node in n]
                descendant_tree = nx.subgraph(graph, descendants)

                # Calculate the Life Cycle Costs for the sub-tree
                graph_utils.bottom_up_traversal(descendant_tree,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                root=node)

            if node in self.fuels:
                if "Life Cycle Cost" not in graph.nodes[node][year]:
                    # Life Cycle Cost needs to be calculated from children
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
                        last_year_value = self.get_param('Life Cycle Cost',
                                                         node, last_year)[fuel_name]['year_value']
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
        # Reset the provided_quantities at each node
        for n in self.graph.nodes():

            self.graph.nodes[n][year]['provided_quantities'] = create_value_dict(ProvidedQuantity(),
                                                                                 param_source='initialization')

    def stock_retirement_and_allocation(self, sub_graph, node, year):
        def helper_quantity_from_services(requested_services,
                                          assessed_demand,
                                          technology=None,
                                          technology_market_share=1):
            if isinstance(requested_services, dict):
                requested_services = [requested_services]

            for service_data in requested_services:
                # Find the multiplier based on quantity to be provided
                service_amount_mult = assessed_demand

                # Find the ratio to provide
                year_value = service_data['year_value']

                # Calculate the total quantity requested
                quant_requested = technology_market_share * year_value * service_amount_mult

                # Check if we need to initialize provided_quantities
                year_node = self.graph.nodes[service_data['branch']][year]
                if 'provided_quantities' not in year_node.keys():
                    year_node['provided_quantities'] = create_value_dict(ProvidedQuantity(),
                                                                         param_source='initialization')

                # Save results
                year_node['provided_quantities']['year_value'].provide_quantity(amount=quant_requested,
                                                                requesting_node=node,
                                                                requesting_technology=technology)

        def general_allocation():
            node_year_data = self.graph.nodes[node][year]

            # How much needs to be provided, based on what was requested of it?
            if self.get_param('competition type', node) == 'root':
                total_to_provide = 1
            else:
                total_to_provide = self.get_param('provided_quantities', node, year).get_total_quantity()
            # Based on what this node needs to provide, find out how much it must request from other
            # services
            if 'technologies' in node_year_data:
                # For each technology, find the services being requested
                for tech, tech_data in node_year_data['technologies'].items():
                    if 'Service requested' in tech_data.keys():
                        services_being_requested = tech_data['Service requested']
                        t_ms = tech_data['Market share']
                        # Calculate the provided_quantities being for each of the services
                        helper_quantity_from_services(services_being_requested,
                                                      total_to_provide,
                                                      technology=tech,
                                                      technology_market_share=t_ms)

            elif 'Service requested' in node_year_data:
                # Calculate the provided_quantities being requested for each of the services
                services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
                helper_quantity_from_services(services_being_requested, total_to_provide)

        def tech_compete_allocation():
            # Demand Assessment
            # *****************
            # Count demanded provided_quantities from above nodes or technologies (for the top node, derive
            # demand provided_quantities external to the sector).
            node_year_data = self.graph.nodes[node][year]

            # How much needs to be provided, based on what was requested of it?
            assessed_demand = self.get_param('provided_quantities', node, year).get_total_quantity()
            # Existing Tech Specific Stocks
            # *****************************
            # Retrieve existing technology stocks provided_quantities from ‘existing stock database’ for
            # simulation year once vintage-specific retirements are conducted for simulation year.
            existing_stock_per_tech = {}
            for t in node_year_data['technologies']:
                t_existing = self.calc_existing_stock(sub_graph, node, year, t)
                existing_stock_per_tech[t] = t_existing

            # Retrofit
            # TODO: Much later we will add this

            # Assessment of capital stock availability
            # ****************************************
            # Subtract total stock for each technology, after natural retirements, from demanded
            # provided_quantities to determine how many new stock must be allocated through competition. If
            # remaining stock already meets quantity demanded, no competition is required and
            # un-needed surplus stock is force retired starting with the oldest vintage at a
            # proportion of each vintages's remaining total market shares.
            new_stock_demanded = copy.copy(assessed_demand)
            for t, e_stocks in existing_stock_per_tech.items():
                new_stock_demanded -= e_stocks

            # Surplus Retirements (aka Early Retirement)
            if new_stock_demanded < 0:
                surplus = -1 * new_stock_demanded

                # Base Stock Retirement
                total_base_stock = 0
                for tech in existing_stock_per_tech:
                    t_base_stock = self.get_param('base_stock_remaining', node, year, tech)
                    total_base_stock += t_base_stock

                if total_base_stock == 0:
                    pass

                else:
                    retirement_proportion = max(0, min(surplus / total_base_stock, 1))
                    for tech in existing_stock_per_tech:
                        t_base_stock = self.get_param('base_stock_remaining', node, year, tech)

                        amount_tech_to_retire = t_base_stock * retirement_proportion
                        # Remove from existing stock
                        existing_stock_per_tech[tech] -= amount_tech_to_retire
                        # Remove from surplus & new stock demanded
                        surplus -= amount_tech_to_retire
                        new_stock_demanded += amount_tech_to_retire
                        # note early retirement in the model
                        self.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining']['year_value'] -= amount_tech_to_retire

                # New Stock Retirement
                possible_purchase_years = [y for y in self.years if (int(y) > self.base_year) &
                                                                    (int(y) < int(year))]
                for purchase_year in possible_purchase_years:
                    total_new_stock_remaining_pre_surplus = 0
                    if surplus > 0:
                        for tech in existing_stock_per_tech:
                            tech_data = self.graph.nodes[node][year]['technologies'][tech]
                            t_rem_new_stock_pre_surplus = self.get_param('new_stock_remaining_pre_surplus',
                                                                         node, year, tech,
                                                                    )[purchase_year]
                            total_new_stock_remaining_pre_surplus += t_rem_new_stock_pre_surplus

                    if total_new_stock_remaining_pre_surplus == 0:
                        retirement_proportion = 0
                    else:
                        retirement_proportion = max(0, min(surplus/total_new_stock_remaining_pre_surplus, 1))

                    for tech in existing_stock_per_tech:
                        tech_data = self.graph.nodes[node][year]['technologies'][tech]

                        t_rem_new_stock_pre_surplus = self.get_param('new_stock_remaining_pre_surplus',
                                                                     node, year, tech,)[purchase_year]

                        amount_tech_to_retire = t_rem_new_stock_pre_surplus * retirement_proportion
                        # Remove from existing stock
                        existing_stock_per_tech[tech] -= amount_tech_to_retire
                        # Remove from surplus & new stock demanded
                        surplus -= amount_tech_to_retire
                        new_stock_demanded += amount_tech_to_retire
                        # note new stock remaining (post surplus) in the model
                        tech_data['new_stock_remaining']['year_value'][purchase_year] -= amount_tech_to_retire

            # New Tech Competition
            # ********************
            # Calculate “New Market Share” percentages to allocate new technology stocks according
            # to demand.
            # TODO: Add other calculations based on whether tech compete, winner take all,
            #  fixed market share
            new_market_shares_per_tech = {}
            for t in node_year_data['technologies']:
                new_market_shares_per_tech[t] = {}
                ms, ms_source = self.get_param('Market share',
                                               node, year, t,
                                               return_source=True)
                ms_exogenous = ms_source == 'model'
                if ms_exogenous:
                    new_market_share = ms

                else:
                    new_market_share = 0

                    # Find the years the technology is available
                    first_year_available = self.get_param('Available', node, str(self.base_year), t)
                    first_year_unavailable = self.get_param('Unavailable', node, str(self.base_year), t)
                    if first_year_available <= int(year) < first_year_unavailable:
                        v = self.get_param('Heterogeneity', node, year)
                        tech_lcc = self.get_param('Life Cycle Cost', node, year, t)
                        total_lcc_v = self.get_param('total_lcc_v', node, year)

                        # TODO: Instead of calculating this in 2 places, set this value in
                        #  lcc_calculation.py. Or here. Not both.
                        if tech_lcc < 0.01:
                            # When lcc < 0.01, we will approximate it's weight using a TREND line
                            w1 = 0.1 ** (-1 * v)
                            w2 = 0.01 ** (-1 * v)
                            slope = (w2 - w1) / (0.01 - 0.1)
                            weight = slope * tech_lcc + (w1 - slope * 0.1)
                        else:
                            weight = tech_lcc ** (-1 * v)

                        new_market_share = weight / total_lcc_v

                self.graph.nodes[node][year]['technologies'][t]['base_stock'] = create_value_dict(0, param_source='initialization')
                self.graph.nodes[node][year]['technologies'][t]['new_stock'] = create_value_dict(0, param_source='initialization')

                new_market_shares_per_tech[t] = new_market_share

            # Min/Max New Marketshare Limit
            # *****************************
            # Apply Min/Max limits to calculated New Market Share percentages and adjust final
            # percentages to comply with limits.
            adjusted_new_ms = stock_allocation.apply_min_max_limits(self,
                                                                    node,
                                                                    year,
                                                                    new_market_shares_per_tech)

            # Calculate Total Market shares -- remaining + new stock
            # *****************************
            total_market_shares_by_tech = {}
            for t in node_year_data['technologies']:
                try:
                    existing_stock = existing_stock_per_tech[t]
                except KeyError:
                    existing_stock = 0

                tech_total_stock = existing_stock + adjusted_new_ms[t] * new_stock_demanded

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



            # Record Values in Model -- market shares & stocks
            # **********************
            for tech in node_year_data['technologies']:
                # New Market Share
                nms = adjusted_new_ms[tech]
                self.graph.nodes[node][year]['technologies'][tech]['new_market_share'] = create_value_dict(nms, param_source='calculation')

                # Total Market Share -- THIS WORKS (i.e. matches previous iterations #s)
                tms = total_market_shares_by_tech[tech]
                self.graph.nodes[node][year]['technologies'][tech]['total_market_share'] = create_value_dict(tms, param_source='calculation')

                # New Stock & Base Stock
                if int(year) == self.base_year:
                    self.graph.nodes[node][year]['technologies'][tech]['base_stock'] = create_value_dict(new_stock_demanded * nms, param_source ='calculation')
                else:
                    self.graph.nodes[node][year]['technologies'][tech]['new_stock'] = create_value_dict(new_stock_demanded * nms, param_source='calculation')

            # Send demand provided_quantities to services below
            # Based on what this node needs to provide, find out how much it must request from
            # other services
            if 'technologies' in node_year_data:
                # For each technology, find the services being requested
                for tech, tech_data in node_year_data['technologies'].items():
                    if 'Service requested' in tech_data.keys():
                        services_being_requested = tech_data['Service requested']
                        # Calculate the provided_quantities being for each of the services
                        t_ms = total_market_shares_by_tech[tech]
                        helper_quantity_from_services(services_being_requested, assessed_demand,
                                                      technology=tech, technology_market_share=t_ms)

            elif 'Service requested' in node_year_data:
                # Calculate the provided_quantities being for each of the services
                services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
                helper_quantity_from_services(services_being_requested, assessed_demand)

        # Move into the proper allocation function
        if self.get_param('competition type', node) == 'tech compete':
            tech_compete_allocation()
        else:
            general_allocation()

    def calc_existing_stock(self, graph, node, year, tech):
        def base_stock_retirement(base_stock, initial_year, current_year, lifespan=10):
            # How much base stock remains if only natural retirements have occurred?
            naturally_unretired_base_stock = base_stock * (1 - (int(current_year) - int(initial_year)) / lifespan)

            # What is the remaining base stock from the previous year? This considers early
            # retirements.
            prev_year = str(int(year) - self.step)
            if int(prev_year) == self.base_year:
                prev_year_unretired_base_stock = self.get_param('base_stock', node, prev_year, tech)
            else:
                prev_year_unretired_base_stock = self.get_param('base_stock_remaining', node, prev_year, tech)

            base_stock_remaining = max(min(naturally_unretired_base_stock, prev_year_unretired_base_stock), 0.0)

            return base_stock_remaining

        def purchased_stock_retirement(purchased_stock, purchased_year, current_year, lifespan, intercept=-11.513):
            prev_year = str(int(year) - self.step)
            prev_y_tech_data = graph.nodes[node][prev_year]['technologies'][tech]

            # Calculate the remaining purchased stock with only natural retirements
            prev_y_exponent = intercept * (1 - (int(prev_year) - int(purchased_year)) / lifespan)
            prev_y_fictional_purchased_stock_remain = purchased_stock / (1 + math.exp(prev_y_exponent))

            # Calculate Adjustment Multiplier
            adj_multiplier = 1

            if int(prev_year) > int(purchased_year):
                prev_y_unretired_new_stock = self.get_param('new_stock_remaining', node, prev_year, tech)[purchased_year]

                if prev_y_fictional_purchased_stock_remain > 0:
                    adj_multiplier = prev_y_unretired_new_stock / prev_y_fictional_purchased_stock_remain

            # Update the tech data
            tech_data = graph.nodes[node][current_year]['technologies'][tech]
            if 'adjustment_multiplier' not in tech_data:
                tech_data['adjustment_multiplier'] = {}
            tech_data['adjustment_multiplier'][purchased_year] = adj_multiplier

            # Calculate the remaining purchased stock
            exponent = intercept * (1 - (int(current_year) - int(purchased_year)) / lifespan)
            purchased_stock_remaining = purchased_stock / (1 + math.exp(exponent)) * adj_multiplier

            return purchased_stock_remaining

        if self.get_param('competition type', node) != 'tech compete':
            # we don't care about existing stock for non-tech compete nodes
            return 0

        earlier_years = [y for y in self.years if int(y) < int(year)]

        if len(earlier_years) == 0:
            return 0

        # Means we are not on the initial year & we need to calculate remaining base and new stock
        # (existing)
        existing_stock = 0
        remaining_base_stock = 0
        remaining_new_stock_pre_surplus = {}
        for y in earlier_years:
            tech_lifespan = self.get_param('Lifetime', node, y, tech)

            # Base Stock
            tech_base_stock_y = self.get_param('base_stock', node, y, tech)
            remain_base_stock_vintage_y = base_stock_retirement(tech_base_stock_y, y, year, tech_lifespan)
            remaining_base_stock += remain_base_stock_vintage_y
            existing_stock += remain_base_stock_vintage_y

            # New Stock
            tech_new_stock_y = self.get_param('new_stock', node, y, tech)
            remain_new_stock = purchased_stock_retirement(tech_new_stock_y, y, year, tech_lifespan)
            remaining_new_stock_pre_surplus[y] = remain_new_stock
            existing_stock += remain_new_stock

        graph.nodes[node][year]['technologies'][tech]['base_stock_remaining'] = create_value_dict(remaining_base_stock,
                                                                                                  param_source='calculation')
        graph.nodes[node][year]['technologies'][tech]['new_stock_remaining_pre_surplus'] = create_value_dict(remaining_new_stock_pre_surplus,
                                                                                                             param_source='calculation')
        # Note: retired stock will be removed later from ['new_stock_remaining']
        graph.nodes[node][year]['technologies'][tech]['new_stock_remaining'] = create_value_dict(remaining_new_stock_pre_surplus,
                                                                                                 param_source='calculation')
        return existing_stock

    def get_tech_parameter_default(self, parameter):
        return self.technology_defaults[parameter]

    def get_node_parameter_default(self, parameter, competition_type):
        return self.node_defaults[competition_type][parameter]

    def get_param(self, param, node, year=None, tech=None, sub_param=None, return_source=False):
        """
        Gets a parameter's value from the model, given a specific context (node, year, technology,
        and sub-parameter).

        This will not re-calculate the parameter's value, but will only retrieve
        values which are already stored in the model or obtained via inheritance, default values,
        or estimation using the previous year's value. If return_source is True, this function will
        also, return how this value was originally obtained (e.g. via calculation)

        Parameters
        ----------
        param : str
            The name of the parameter whose value is being retrieved.
        node : str
            The name of the node (branch format) whose parameter you are interested in retrieving.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            "competition type" can be retreived without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        sub_param : str, optional
            This is a rarely used parameter for specifying a nested key. Most commonly used when
            `get_param()` would otherwise return a dictionary where a nested value contains the
            parameter value of interest. In this case, the key corresponding to that value can be
            provided as a `sub_param`
        return_source : bool, default=False
            Whether or not to return the method by which this value was originally obtained.

        Returns
        -------
        any :
            The value of the specified `param` at `node`, given the context provided by `year` and
            `tech`.
        str :
            If return_source is `True`, will return a string indicating how the parameter's value
            was originally obtained. Can be one of {model, initialization, inheritance, calculation,
            default, or previous_year}.

        See Also
        --------
        Model.get_or_calc_param :  Gets a parameter's value from the model, given a specific context
        (node, year, technology, and sub-parameter), calculating the parameter's value if needed.
        """
        if tech:
            param_val = utils.get_tech_param(param, self, node, year, tech, sub_param,
                                             return_source=return_source,
                                             retrieve_only=True)

        else:
            param_val = utils.get_node_param(param, self, node, year,
                                             return_source=return_source,
                                             retrieve_only=True)

        return param_val

    def get_or_calc_param(self, param, node, year=None, tech=None, sub_param=None,
                          return_source=False):
        """
        Gets a parameter's value from the model, given a specific context (node, year, technology,
        and sub-parameter), calculating the parameter's value if needed.

        If return_source is True,
        this function will also, return how this value was originally obtained (e.g. via
        calculation)

        Parameters
        ----------
        param : str
            The name of the parameter whose value is being retrieved.
        node : str
            The name of the node (branch format) whose parameter you are interested in retrieving.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            "competition type" can be retreived without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        sub_param : str, optional
            This is a rarely used parameter for specifying a nested key. Most commonly used when
            `get_param()` would otherwise return a dictionary where a nested value contains the
            parameter value of interest. In this case, the key corresponding to that value can be
            provided as a `sub_param`
        return_source : bool, default=False
            Whether or not to return the method by which this value was originally obtained.

        Returns
        -------
        any :
            The value of the specified `param` at `node`, given the context provided by `year` and
            `tech` and potentially using a calculator function if the parameter's value was not
             exogenously defined.
        str :
            If return_source is `True`, will return a string indicating how the parameter's value
            was originally obtained. Can be one of {model, initialization, inheritance, calculation,
            default, or previous_year}.

        See Also
        --------
        Model.get_param : Gets a parameter's value from the model, given a specific context (node,
        year, technology, and sub-parameter).
        """

        if tech:
            param_val = utils.get_tech_param(param, self, node, year, tech, sub_param,
                                             return_source,
                                             retrieve_only=False)

        else:
            param_val = utils.get_node_param(param, self, node, year,
                                             return_source=return_source,
                                             retrieve_only=False)

        return param_val

    def calc_requested_quantities(self, graph, node, year):
        """
        Calculates the quantities which have been requested by a node in the specified year and
        records this in the Model. This calculates all quantities that can be traced back to this
        node. In other words, this will not only include the services that the node requests, but
        also any quantities requested by it's successors (children, grandchildren, etc).

        This method was built to be used with the bottom up traversal method
        (pyCIMS.graph_utils.bottom_up_traversal()), which ensures that a node is only visited once
        all its children have been visited (except when it needs to break a loop).

        Important things to note:
           * Fuel nodes pass along quantities requested by their successors via their structural
             parent rather than through request/provide parents.

        Parameters
        ----------
        graph : networkX.Graph
            The graph containing node & it's children.
        node : str
            The name of the node (in branch/path notation) for which the total requested quantities
            will be calculated.
        year : str
            The year of interest.

        Returns
        -------
        Nothing. Does set the requested_quantities parameter in the Model according to the
        quantities requested of node & all it's successors.
        """
        # Create an empty RequestedQuantity object to fill
        requested_quantity = RequestedQuantity()

        if self.get_param("competition type", node) in ['root', 'region']:
            # Find the node's children, who they have a structural relationship with
            children = graph.successors(node)
            structural_children = [c for c in children if 'structure' in
                                   graph.get_edge_data(node, c)['type']]

            # For each structural child, add it's provided quantities to the region/root
            for child in structural_children:
                child_requested_quant = self.get_param("requested_quantities",
                                                       child, year).get_total_quantities_requested()
                for child_rq_node, child_rq_amount in child_requested_quant.items():
                    requested_quantity.record_requested_quantity(child_rq_node,
                                                                 child,
                                                                 child_rq_amount)

        else:
            # Find the node's children, who they have a request/provide relationship with
            children = graph.successors(node)
            req_prov_children = [c for c in children if 'request_provide' in
                                 graph.get_edge_data(node, c)['type']]

            # For each child, calculate how much of each service has been requested
            for child in req_prov_children:
                # *********
                # Add the quantity requested of the child by node (if child is a fuel)
                # *********
                child_provided_quant = self.get_param("provided_quantities", child, year)
                child_quantity_provided_to_node = child_provided_quant.get_quantity_provided_to_node(node)
                if child in self.fuels:
                    requested_quantity.record_requested_quantity(child, child, child_quantity_provided_to_node)

                # *********
                # Calculate proportion of child's requested quantities that come from node. Record these
                # as well.
                # *********
                else:
                    try:
                        child_total_quantity_provided = child_provided_quant.get_total_quantity()
                        if child_total_quantity_provided == 0:
                            # If the child doesn't provide any quantities, move onto the next child without
                            # updating the node's quantity requested.
                            continue
                        else:
                            # Otherwise, find out what proportion of the child's requested can be traced
                            # back to node
                            proportion = child_quantity_provided_to_node / child_total_quantity_provided

                            child_requested_quant = self.get_param("requested_quantities", child, year)
                            for child_rq_node, child_rq_amount in child_requested_quant.get_total_quantities_requested().items():
                                requested_quantity.record_requested_quantity(child_rq_node,
                                                                             child,
                                                                             proportion * child_rq_amount)
                    except KeyError:
                        # Occurs when a requested quantity value doesn't exist yet b/c a loop has been
                        # broken for the base year.
                        continue

        # Save the requested quantities to the node's data
        self.graph.nodes[node][year]["requested_quantities"] = utils.create_value_dict(requested_quantity,
                                                                                       param_source='calculation')