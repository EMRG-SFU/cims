import copy
import warnings
import networkx as nx
import pandas as pd
import re
import time
import pickle

from . import graph_utils
from . import utils
from . import lcc_calculation
from . import stock_allocation
from . import loop_resolution

from .quantities import ProvidedQuantity, RequestedQuantity, DistributedSupply
from .emissions import Emissions, EmissionsCost, calc_cumul_emissions_rate

from .utils import create_value_dict, inheritable_params, inherit_parameter
from .quantity_aggregation import find_children, get_quantities_to_record, \
    get_direct_distributed_supply


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
        the excel model description) are the values. DataFrames do not include 'technology' or
        'service' information for a node.

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
        self.node_tech_defaults = model_reader.get_default_params()
        self.step = 5  # TODO: Make this an input or calculate
        self.fuels = []
        self.equilibrium_fuels = []
        self.GHGs = []
        self.emission_types = []
        self.gwp = {}
        self.years = model_reader.get_years()
        self.base_year = int(self.years[0])

        self.prices = {}

        self.build_graph()
        self.dcc_classes = self._dcc_classes()
        self._inherit_parameter_values()
        self._initialize_model()

        self.show_run_warnings = True

        self.model_description_file = model_reader.infile
        self.change_history = pd.DataFrame(
            columns=['base_model_description', 'parameter', 'node', 'year', 'technology',
                     'context', 'sub_context', 'old_value', 'new_value'])

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

        self.fuels, self.equilibrium_fuels = graph_utils.get_fuels(graph)
        self.GHGs, self.emission_types, self.gwp = graph_utils.get_GHG_and_Emissions(graph, str(self.base_year))
        self.graph = graph

    def _initialize_model(self):
        for year in self.years:
            # Pass tax to all children for carbon cost
            graph_utils.top_down_traversal(self.graph,
                                           self._init_tax_emissions,
                                           year)

            # Pass foresight methods to all children nodes
            sec_list = [node for node, data in self.graph.nodes(data=True)
                        if 'sector' in data['competition type'].lower()]

            foresight_context = self.get_param('tax_foresight', 'pyCIMS', year, dict_expected=True)
            if foresight_context is not None:
                for ghg, sectors in foresight_context.items():
                    for node in sec_list:
                        sector = node.split('.')[-1]
                        if sector in sectors:
                            # Initialize foresight method
                            if 'tax_foresight' not in self.graph.nodes[node][year]:
                                self.graph.nodes[node][year]['tax_foresight'] = {}

                            self.graph.nodes[node][year]['tax_foresight'][ghg] = sectors[sector]

            graph_utils.top_down_traversal(self.graph,
                                           self._init_foresight,
                                           year)

    def _dcc_classes(self):
        """
        Iterate through each node and technology in self to create a dictionary mapping Declining
        Capital Cost (DCC) Classes to a list of nodes that belong to that class.

        Returns
        -------
        dict {str: [str]}:
            Dictionary where keys are declining capital cost classes (str) and values are lists of
            nodes (str) belonging to that class.
        """
        dcc_classes = {}

        nodes = self.graph.nodes
        base_year = str(self.base_year)
        for node in nodes:
            if 'technologies' in nodes[node][base_year]:
                for tech in nodes[node][base_year]['technologies']:
                    try:
                        dccc = self.graph.nodes[node][base_year]['technologies'][tech]['dcc_class'][
                        'context']
                    except:
                        dccc = None
                    if dccc is not None:
                        if dccc in dcc_classes:
                            dcc_classes[dccc].append((node, tech))
                        else:
                            dcc_classes[dccc] = [(node, tech)]

        return dcc_classes

    def run(self, equilibrium_threshold=0.05, min_iterations=1, max_iterations=10,
            show_warnings=True, print_eq=False):
        """
        Runs the entire model, progressing year-by-year until an equilibrium has been reached for
        each year.

        Parameters
        ----------
        equilibrium_threshold : float, optional
            The largest relative difference between prices allowed for an equilibrium to be reached.
            Must be between [0, 1]. Relative difference is calculated as the absolute difference
            between two prices, divided by the first price. Defaults to 0.05.

        min_iterations : int, optional
            The minimum number of times to iterate between supply and demand in an attempt to reach
            an equilibrium.

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

        # Make a subgraph based on the type of node
        demand_nodes = ['demand', 'standard']
        supply_nodes = ['supply', 'standard']

        for year in self.years:
            print(f"***** ***** year: {year} ***** *****")

            # Initialize Basic Variables
            equilibrium = False
            iteration = 1

            # Initialize Graph Values
            self.initialize_graph(self.graph, year)
            while not equilibrium:
                print('iter {}'.format(iteration))
                # Early exit if we reach the maximum number of iterations
                if iteration >= max_iterations:
                    warnings.warn("Max iterations reached for year {}. "
                                  "Continuing to next year.".format(year))
                    break
                # Initialize Iteration Specific Values
                self.iteration_initialization(year)

                # DEMAND
                # ******************
                # Calculate Life Cycle Cost values on demand side
                graph_utils.bottom_up_traversal(self.graph,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                node_types=demand_nodes)

                # Calculate Quantities (Total Stock Needed)
                graph_utils.top_down_traversal(self.graph,
                                               self.stock_allocation_and_retirement,
                                               year,
                                               node_types=demand_nodes)

                if int(year) != self.base_year:
                    # Calculate Service Costs on Demand Side
                    graph_utils.bottom_up_traversal(self.graph,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self,
                                                    node_types=demand_nodes)

                # Supply
                # ******************
                # Calculate Service Costs on Supply Side
                graph_utils.bottom_up_traversal(self.graph,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                node_types=supply_nodes)
                # Calculate Fuel Quantities
                graph_utils.top_down_traversal(self.graph,
                                               self.stock_allocation_and_retirement,
                                               year,
                                               node_types=supply_nodes)

                if int(year) != self.base_year:
                    # Calculate Service Costs on Supply Side
                    graph_utils.bottom_up_traversal(self.graph,
                                                    lcc_calculation.lcc_calculation,
                                                    year,
                                                    self,
                                                    node_types=supply_nodes)

                # Check for an Equilibrium -- Across all nodes, not just fuels
                # ************************
                # Find the previous prices
                prev_prices = self.prices
                # Go get all the new prices
                new_prices = {node: self.get_param('price', node, year, do_calc=True) for node in self.graph.nodes()}

                # Check for an equilibrium in prices
                equilibrium = min_iterations <= iteration and \
                              (int(year) == self.base_year or
                               self.check_equilibrium(prev_prices,
                                                      new_prices,
                                                      iteration,
                                                      equilibrium_threshold,
                                                      print_eq))

                self.prices = new_prices

                # Next Iteration
                # **************
                iteration += 1

            # Once we've reached an equilibrium, calculate the quantities requested by each node.
            graph_utils.bottom_up_traversal(self.graph,
                                            self.calc_requested_quantities,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            fuels=self.fuels)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_direct_emissions,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            fuels=self.fuels)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_direct_emissions_cost,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            fuels=self.fuels)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_cumul_emissions,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            fuels=self.fuels)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_cumul_emissions_cost,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            fuels=self.fuels)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_distributed_supplies,
                                            year)
    def check_equilibrium(self, prev, new, iteration, threshold, print_eq):
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
        equil_check = 0

        for fuel in new:
            prev_fuel_price = prev[fuel]
            new_fuel_price = new[fuel]
            if (prev_fuel_price is None) or (new_fuel_price is None):
                print(f"   Not at equilibrium: {fuel} does not have an LCC calculated")
                return False
            abs_diff = abs(new_fuel_price - prev_fuel_price)

            if prev_fuel_price == 0:
                if self.show_run_warnings:
                    warnings.warn("Previous fuel price is 0 for {}".format(fuel))
                rel_diff = 0
            else:
                rel_diff = abs_diff / prev_fuel_price

            # If any fuel's relative difference exceeds the threshold, an equilibrium
            # has not been reached
            if rel_diff > threshold:
                equil_check += 1
                if iteration > 1 and print_eq == True:
                    rel_diff_formatted = "{:.1f}".format(rel_diff * 100)
                    print(f"   Not at equilibrium: {fuel} has {rel_diff_formatted}% difference between iterations")

        if equil_check > 0:
            return False

        # Otherwise, an equilibrium has been reached
        return True

    def _init_tax_emissions(self, graph, node, year):
        """
        Function for initializing the tax values (to multiply against emissions) for a given node in a graph. This
        function assumes all of node's parents have already had their tax emissions initialized.

        Parameters
        ----------
        self :
            A graph object containing the node of interest.
        node : str
            Name of the node to be initialized.

        year: str
            The string representing the current simulation year (e.g. "2005").

        Returns
        -------
        Nothing is returned, but `graph.nodes[node]` will be updated with the initialized tax emission values.
        """

        # Retrieve tax from the parents (if they exist)
        parents = list(graph.predecessors(node))
        parent_dict = {}
        if len(parents) > 0:
            parent = parents[0]
            if 'tax' in graph.nodes[parent][year]:
                parent_dict = graph.nodes[parent][year]['tax']

        # Store away tax at current node to overwrite parent tax later
        node_dict = {}
        if 'tax' in graph.nodes[node][year]:
            node_dict = graph.nodes[node][year]['tax']

        # Make final dict where we prioritize keeping node_dict and only unique parent taxes
        final_tax = copy.deepcopy(node_dict)
        for ghg in parent_dict:
            if ghg not in final_tax:
                final_tax[ghg] = {}
            for emission_type in parent_dict[ghg]:
                if emission_type not in final_tax[ghg]:
                    final_tax[ghg][emission_type] = parent_dict[ghg][emission_type]

        if final_tax:
            graph.nodes[node][year]['tax'] = final_tax

    def _init_foresight(self, graph, node, year):
        parents = list(graph.predecessors(node))
        parent_dict = {}
        if len(parents) > 0:
            parent = parents[0]
            if 'tax_foresight' in graph.nodes[parent][year] and parent != 'pyCIMS':
                parent_dict = graph.nodes[parent][year]['tax_foresight']
        if parent_dict:
            graph.nodes[node][year]['tax_foresight'] = parent_dict

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
                if 'price multiplier' in graph.nodes[parent][year]:
                    price_multipliers = copy.deepcopy(self.graph.nodes[parent][year]['price multiplier'])
                    parent_price_multipliers.update(price_multipliers)

            # Grab the price multipliers from the current node (if they exist) and replace the parent price multipliers
            node_price_multipliers = copy.deepcopy(parent_price_multipliers)
            if 'price multiplier' in graph.nodes[node][year]:
                price_multipliers = self.get_param('price multiplier', node, year, dict_expected=True)
                node_price_multipliers.update(price_multipliers)

            # Set Price Multiplier of node in the graph
            graph.nodes[node][year]['price multiplier'] = node_price_multipliers

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
                if 'financial life cycle cost' in graph.nodes[node][year]:
                    lcc_dict = graph.nodes[node][year]['financial life cycle cost']

                    fuel_name = list(lcc_dict.keys())[0]
                    if lcc_dict[fuel_name]['year_value'] is None:
                        lcc_dict[fuel_name]['to_estimate'] = True
                        last_year = str(int(year) - step)
                        last_year_value = self.get_param('financial life cycle cost',
                                                         node, last_year)[fuel_name]['year_value']
                        graph.nodes[node][year]['financial life cycle cost'][fuel_name]['year_value'] = last_year_value

                    else:
                        graph.nodes[node][year]['financial life cycle cost'][fuel_name]['to_estimate'] = False
                elif 'cost_curve_function' in graph.nodes[node]:
                    if int(year) == self.base_year:
                        lcc = lcc_calculation.calc_cost_curve_lcc(self, node, year)
                        service_name = node.split('.')[-1]
                        graph.nodes[node][year]['financial life cycle cost'] = {
                            service_name: utils.create_value_dict(lcc,
                                                                  param_source='initialization')}
                    else:
                        last_year = str(int(year) - self.step)
                        service_name = node.split('.')[-1]
                        last_year_value = self.get_param('financial life cycle cost', node, last_year)
                        graph.nodes[node][year]['financial life cycle cost'] = {
                            service_name: utils.create_value_dict(last_year_value,
                                                                  param_source='cost curve function')}

                else:
                    # Life Cycle Cost needs to be calculated from children
                    calc_lcc_from_children()
                    lcc_dict = graph.nodes[node][year]['financial life cycle cost']
                    fuel_name = list(lcc_dict.keys())[0]
                    lcc_dict[fuel_name]['to_estimate'] = True

        def init_convert_to_CO2e(graph, node, year, gwp):
            """
            Function for initializing all Emissions as tCO2e (instead of tCH4, tN2O, etc).
            This function assumes all of node's children have already been processed by this function.

            Parameters
            ----------
            graph : NetworkX.DiGraph
                A graph object containing the node of interest.

            node : str
                Name of the node to be initialized.

            year: str
                The string representing the current simulation year (e.g. "2005").

            gwp: dict
                The dictionary containing the GHGs (keys) and GWPs (values).

            Returns
            -------
            Nothing is returned, but `graph.nodes[node]` will be updated with the initialized Emissions.
            """

            # Emissions from a node with technologies
            if 'technologies' in graph.nodes[node][year]:
                techs = graph.nodes[node][year]['technologies']
                for tech in techs:
                    tech_data = techs[tech]
                    if 'emissions' in tech_data:
                        emission_data = tech_data['emissions']
                        for ghg in emission_data:
                            for emission_type in emission_data[ghg]:
                                try:
                                    emission_data[ghg][emission_type]['year_value'] *= gwp[ghg]
                                except KeyError:
                                    continue

            # Emissions from a node
            elif 'emissions' in graph.nodes[node][year]:
                emission_data = graph.nodes[node][year]['emissions']
                for ghg in emission_data:
                    for emission_type in emission_data[ghg]:
                        try:
                            emission_data[ghg][emission_type]['year_value'] *= gwp[ghg]
                        except KeyError:
                            continue

        def init_load_factor(graph, node, year):
            """
            Initialize the load factor for nodes & technologies using inheritence from either the
            node's parent or the technology's node.

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
            Nothing. Will update graph.nodes[node][year] with the initialized value of `Load Factor`
            (if there is one).
            """
            if 'load factor' not in graph.nodes[node][year]:
                # Check if a load factor was defined at the node's structural parent (its first
                # parent). If so, use this load factor for the node.
                parents = list(graph.predecessors(node))
                if len(parents) > 0:
                    parent = parents[0]
                    if 'load factor' in graph.nodes[parent][year]:
                        val = graph.nodes[parent][year]['load factor']['year_value']
                        units = graph.nodes[parent][year]['load factor']['unit']
                        graph.nodes[node][year]['load factor'] = utils.create_value_dict(val,
                                                                                         unit=units,
                                                                                         param_source='inheritance')

            if 'load factor' in graph.nodes[node][year]:
                # Ensure this load factor is recorded at each of the technologies within the node.
                if 'technologies' in graph.nodes[node][year]:
                    tech_data = graph.nodes[node][year]['technologies']
                    for tech in tech_data:
                        if 'load factor' not in tech_data[tech]:
                            val = graph.nodes[node][year]['load factor']['year_value']
                            units = graph.nodes[node][year]['load factor']['unit']
                            tech_data[tech]['load factor'] = utils.create_value_dict(val,
                                                                                     unit=units,
                                                                                     param_source='inheritance')

        def init_tax_emissions(graph, node, year):
            """
            Function for initializing the tax values (to multiply against emissions) for a given node in a graph. This
            function assumes all of node's parents have already had their tax emissions initialized.

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
            Nothing is returned, but `graph.nodes[node]` will be updated with the initialized tax emission values.
            """

            # Retrieve tax from the parents (if they exist)
            parents = list(graph.predecessors(node))
            parent_dict = {}
            if len(parents) > 0:
                parent = parents[0]
                if 'tax' in graph.nodes[parent][year]:
                    parent_dict = graph.nodes[parent][year]['tax']

            # Store away tax at current node to overwrite parent tax later
            node_dict = {}
            if 'tax' in graph.nodes[node][year]:
                node_dict = graph.nodes[node][year]['tax']

            # Make final dict where we prioritize keeping node_dict and only unique parent taxes
            final_tax = copy.deepcopy(node_dict)
            for ghg in parent_dict:
                if ghg not in final_tax:
                    final_tax[ghg] = {}
                for emission_type in parent_dict[ghg]:
                    if emission_type not in final_tax[ghg]:
                        final_tax[ghg][emission_type] = parent_dict[ghg][emission_type]

            if final_tax:
                graph.nodes[node][year]['tax'] = final_tax

        def init_agg_emissions_cost(graph):
            # Reset the aggregate_emissions_cost at each node
            for n in self.graph.nodes():
                self.graph.nodes[n][year]['aggregate_emissions_cost_rate'] = \
                    create_value_dict({}, param_source='initialization')
                self.graph.nodes[n][year]['cumul_emissions_cost_rate'] = \
                    create_value_dict(EmissionsCost(), param_source='initialization')

        init_agg_emissions_cost(graph)

        graph_utils.top_down_traversal(graph,
                                       init_convert_to_CO2e,
                                       year,
                                       self.gwp)
        graph_utils.top_down_traversal(graph,
                                       init_load_factor,
                                       year)
        graph_utils.bottom_up_traversal(graph,
                                        init_fuel_lcc,
                                        year)

    def iteration_initialization(self, year):
        # Reset the provided_quantities at each node
        for n in self.graph.nodes():
            self.graph.nodes[n][year]['provided_quantities'] = create_value_dict(ProvidedQuantity(),
                                                                                 param_source='initialization')

    def _inherit_parameter_values(self):
        def inherit_function(graph, node, year):
            for param in inheritable_params:
                inherit_parameter(graph, node, year, param)

        for year in self.years:
            graph_utils.top_down_traversal(self.graph, inherit_function, year)

    def stock_allocation_and_retirement(self, sub_graph, node, year):
        """
        Provided to `graph_utils.top_down_traversal()` by `Model.run()` as the processing function
        for stock allocation and retirement.
        Parameters
        ----------
        sub_graph: any
            This is not used in this function, but is a required parameter for any function used by
            `graph_utils.top_down_traversal()`.

        node: str
            The name of the node (in branch form) where stock stock retirement and allocation will
            be performed.

        year: str
            The year to perform stock retirement and allocation.

        Returns
        -------
            Nothing is returned. `self` will be updated to reflect the results of stock retirement
            and new stock competitions.
        """
        comp_type = self.get_param('competition type', node).lower()

        # Market acts the same as tech compete
        if comp_type == 'market':
            comp_type = 'tech compete'

        if comp_type in ['tech compete', 'node tech compete']:
            stock_allocation.all_tech_compete_allocation(self, node, year)
        else:
            stock_allocation.general_allocation(self, node, year)

    def calc_requested_quantities(self, graph, node, year, **kwargs):
        """
        Calculates and records fuel quantities requested by a node in the specified year.

        This includes fuel quantities directly requested of the node (e.g. a Lighting requests
        services directly from Electricity) and fuel quantities that are indirectly requested, but
        can be attributed to the node (e.g. Alberta indirectly requests Electricity via its
        children). In other words, this not only includes quantities a node requests, but also
        quantities requested by it's successors (children, grandchildren, etc).

        This method was built to be used with the bottom up traversal method
        (pyCIMS.graph_utils.bottom_up_traversal()), which ensures that a node is only visited once
        all its children have been visited (except when it needs to break a loop).

        Important things to note:
           * Fuel nodes pass along quantities requested by their successors via their structural
             parent rather than through request/provide parents.

        Parameters
        ----------
        graph : networkX.Graph
            The graph containing node & its children.
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
        requested_quantity = RequestedQuantity()

        if self.get_param('competition type', node) in ['root', 'region']:
            structural_children = find_children(graph, node, structural=True)
            for child in structural_children:
                # Find quantities provided to the node via its structural children
                child_requested_quant = self.get_param('requested_quantities',
                                                       child, year).get_total_quantities_requested()
                for child_rq_node, child_rq_amount in child_requested_quant.items():
                    # Record requested quantities
                    requested_quantity.record_requested_quantity(child_rq_node,
                                                                 child,
                                                                 child_rq_amount)

        elif 'technologies' in graph.nodes[node][year]:
            for tech in graph.nodes[node][year]['technologies']:
                tech_requested_quantity = RequestedQuantity()

                # Aggregate Fuel Quantities
                req_prov_children = find_children(graph, node, year, tech, request_provide=True)
                for child in req_prov_children:
                    quantities_to_record = get_quantities_to_record(self, child, node, year, tech)
                    # Record requested quantities
                    for providing_node, child, attributable_amount in quantities_to_record:
                        tech_requested_quantity.record_requested_quantity(providing_node,
                                                                          child,
                                                                          attributable_amount)
                        requested_quantity.record_requested_quantity(providing_node,
                                                                     child,
                                                                     attributable_amount)

                # Save the tech requested quantities
                self.graph.nodes[node][year]['technologies'][tech]['requested_quantities'] = \
                    tech_requested_quantity

        else:
            req_prov_children = find_children(graph, node, year, request_provide=True)
            for child in req_prov_children:
                # Record requested quantities
                quantities_to_record = get_quantities_to_record(self, child, node, year)
                for providing_node, child, attributable_amount in quantities_to_record:
                    requested_quantity.record_requested_quantity(providing_node,
                                                                 child,
                                                                 attributable_amount)

        self.graph.nodes[node][year]['requested_quantities'] = \
            utils.create_value_dict(requested_quantity, param_source='calculation')

    def _aggregate_distributed_supplies(self, graph, node, year, **kwargs):
        """
        We want to aggregate up the structural relationships in the tree. This means there are two
        different locations within the tree we need to think about:

        (1) @ a Node without techs — Find any distributed supply that has been generated at the
            node. Add any distributed supplies from structural children.
        (2) @ a Node with tech — For each tech, find the distributed supply generated at that node
           (See Q below). Sum up the distributed supplies across all techs.

        When doing sums, there is no need to worry about multiply by weights or service request
        ratios, since each node only has a single structural parent, everything will flow through
        that path.

        Question: I'm guessing that we want to store a "distributed_supply" value at the tech-level,
        even though technologies will only include distributed supply if they are directly
        producing it (since techs are never structural parents). Is this logic correct?
        """
        node_distributed_supply = DistributedSupply()
        if 'technologies' in self.graph.nodes[node][year]:
            # @ a Node with techs
            # Find distributed supply generated at the tech
            for tech in self.graph.nodes[node][year]['technologies']:
                tech_distributed_supply = DistributedSupply()
                distributed_supplies = get_direct_distributed_supply(self, node, year, tech)
                for service, amount in distributed_supplies:
                    tech_distributed_supply.record_distributed_supply(service, node, amount)
                    node_distributed_supply.record_distributed_supply(service, node, amount)
                self.graph.nodes[node][year]['technologies'][tech]['distributed_supply'] = \
                    tech_distributed_supply
        else:
            # @ a Node without techs
            node_distributed_supply = DistributedSupply()
            # Find distributed supply generated at the node
            distributed_supplies = get_direct_distributed_supply(self, node, year)
            for service, amount in distributed_supplies:
                node_distributed_supply.record_distributed_supply(service, node, amount)
            # Find distributed supply from structural children
            structural_children = find_children(graph, node, structural=True)
            for child in structural_children:
                node_distributed_supply += self.get_param('distributed_supply', child, year)

        self.graph.nodes[node][year]['distributed_supply'] = \
            utils.create_value_dict(node_distributed_supply, param_source='calculation')

    def _aggregate_cumul_emissions_cost(self, graph, node, year, **kwargs):
        """
        Calculates and records the total emissions cost for a particular node. This is done by
        taking the per-unit emissions cost and multiplying it by the number of units provided by a
        particular node or technology.

        To be used as part of a bottom_up_traversal once an equilibrium has been reached.

        Parameters
        ----------
        graph : NetworkX.Digraph
            The graph being traversed.

        node : str
            The node whose total_cumul_emissions_cost is being calculated.

        year : str
            Tthe year of interest.

        Returns
        -------
        None
            Returns nothing but does record total_cumul_emissions_cost for the node (and any
            technologies).
        """
        total_cumul_emissions_cost = EmissionsCost()

        comp_type = self.get_param('competition type', node)
        if comp_type in ['root', 'region']:
            structural_children = find_children(graph, node, structural=True)
            for child in structural_children:
                # Find quantities provided to the node via its structural children
                total_cumul_emissions_cost += self.get_param('total_cumul_emissions_cost', child, year)
        else:
            pq = self.get_param('provided_quantities', node, year).get_total_quantity()
            emissions_cost = self.get_param('cumul_emissions_cost_rate', node, year)
            total_cumul_emissions_cost = emissions_cost * pq
            total_cumul_emissions_cost.num_units = pq

            if 'technologies' in graph.nodes[node][year]:
                for tech in graph.nodes[node][year]['technologies']:
                    ec = self.get_param('cumul_emissions_cost_rate', node, year, tech=tech)
                    ms = self.get_param('total_market_share', node, year, tech=tech)
                    tech_total_emissions_cost = ec * ms * pq
                    tech_total_emissions_cost.num_units = ms * pq
                    value_dict = create_value_dict(tech_total_emissions_cost)
                    graph.nodes[node][year]['technologies'][tech]['total_cumul_emissions_cost'] = value_dict

        value_dict = create_value_dict(total_cumul_emissions_cost)
        graph.nodes[node][year]['total_cumul_emissions_cost'] = value_dict

    def _aggregate_direct_emissions_cost(self, graph, node, year, **kwargs):

        emissions_cost_params = [
            {'rate_param_name': 'emissions_cost_rate', 'total_param_name': 'total_direct_emissions_cost'}
        ]
        pq = self.get_param('provided_quantities', node, year).get_total_quantity()
        for e in emissions_cost_params:
            rate_param = e['rate_param_name']
            total_param = e['total_param_name']
            node_total_direct_emissions_cost = EmissionsCost()
            if 'technologies' in graph.nodes[node][year]:
                for tech in graph.nodes[node][year]['technologies']:
                    total_direct_emissions_cost = EmissionsCost()
                    ms = self.get_param('total_market_share', node, year, tech=tech)
                    direct_emissions_cost = self.get_param(rate_param, node, year, tech=tech)
                    if direct_emissions_cost is not None:
                        total_direct_emissions_cost = direct_emissions_cost * (pq * ms)
                        node_total_direct_emissions_cost += total_direct_emissions_cost
                    value_dict = create_value_dict(total_direct_emissions_cost)
                    graph.nodes[node][year]['technologies'][tech][total_param] = value_dict

            value_dict = create_value_dict(node_total_direct_emissions_cost)
            graph.nodes[node][year][total_param] = value_dict

    def _aggregate_cumul_emissions(self, graph, node, year, **kwargs):
        # Calculate Cumulative Emissions
        if 'technologies' in graph.nodes[node][year]:
            for tech in graph.nodes[node][year]['technologies']:
                calc_cumul_emissions_rate(self, node, year, tech)
        calc_cumul_emissions_rate(self, node, year)

        # Aggregate Cumulative Emissions
        emission_params = ['net_emissions', 'avoided_emissions',
                           'negative_emissions', 'bio_emissions']
        for e in emission_params:
            rate_param = f"cumul_{e}_rate"
            total_param = f"total_cumul_{e}"
            total_cumul_emissions = Emissions()

            comp_type = self.get_param('competition type', node)
            if comp_type in ['root', 'region']:
                structural_children = find_children(graph, node, structural=True)
                for child in structural_children:
                    # Find quantities provided to the node via its structural children
                    total_cumul_emissions += self.get_param(total_param, child, year)

            else:
                pq = self.get_param('provided_quantities', node, year).get_total_quantity()
                cumul_emissions = self.get_param(rate_param, node, year)
                total_cumul_emissions = cumul_emissions * pq

                if 'technologies' in graph.nodes[node][year]:
                    for tech in graph.nodes[node][year]['technologies']:
                        ms = self.get_param('total_market_share', node, year, tech=tech)
                        em = self.get_param(rate_param, node, year, tech=tech)
                        if em is not None:
                            tech_total_cumul_emissions = em * pq * ms
                        else:
                            tech_total_cumul_emissions = Emissions()
                        value_dict = create_value_dict(tech_total_cumul_emissions)
                        graph.nodes[node][year]['technologies'][tech][total_param] = value_dict

            value_dict = create_value_dict(total_cumul_emissions)
            graph.nodes[node][year][total_param] = value_dict

    def _aggregate_direct_emissions(self, graph, node, year, **kwargs):
        # Direct Emissions
        emissions = [
            {'rate_param_name': 'net_emissions_rate', 'total_param_name': 'total_direct_net_emissions'},
            {'rate_param_name': 'avoided_emissions_rate', 'total_param_name': 'total_direct_avoided_emissions'},
            {'rate_param_name': 'negative_emissions_rate', 'total_param_name': 'total_direct_negative_emissions'},
            {'rate_param_name': 'bio_emissions_rate', 'total_param_name': 'total_direct_bio_emissions'}
        ]
        pq = self.get_param('provided_quantities', node, year).get_total_quantity()
        for e in emissions:
            rate_param = e['rate_param_name']
            total_param = e['total_param_name']
            node_total_direct_emissions = Emissions()
            if 'technologies' in graph.nodes[node][year]:
                for tech in graph.nodes[node][year]['technologies']:
                    total_direct_emissions = Emissions()
                    ms = self.get_param('total_market_share', node, year, tech=tech)
                    direct_emissions = self.get_param(rate_param, node, year, tech=tech)
                    if direct_emissions is not None:
                        total_direct_emissions = direct_emissions * pq * ms
                        node_total_direct_emissions += total_direct_emissions
                    value_dict = create_value_dict(total_direct_emissions)
                    graph.nodes[node][year]['technologies'][tech][total_param] = value_dict

            value_dict = create_value_dict(node_total_direct_emissions)
            graph.nodes[node][year][total_param] = value_dict

    def get_parameter_default(self, parameter):
        return self.node_tech_defaults[parameter]

    def get_param(self, param, node, year=None, tech=None, context=None, sub_context=None,
                  return_source=False, do_calc=False, check_exist=False, dict_expected=False):
        """
        Gets a parameter's value from the model, given a specific context (node, year, tech, context, sub-context),
        calculating the parameter's value if needed.

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
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        return_source : bool, default=False
            Whether or not to return the method by which this value was originally obtained.
        do_calc : bool, default=False
            If False, the function will only retrieve the value using the current value in the model,
            inheritance, default, or the previous year's value. It will _not_ calculate the parameter
            value. If True, calculation is allowed.
        check_exist : bool, default=False
            Whether or not to check that the parameter exists as is given the context (without calculation,
            inheritance, or checking past years)
        dict_expected : bool, default=False
            Used to disable the warning get_param is returning a dict. Get_param should normally return a 'single value'
            (float, str, etc.). If the user knows it expects a dict, then this flag is used.

        Returns
        -------
        any :
            The value of the specified `param` at `node`, given the context provided by `year` and
            `tech`.
        str :
            If return_source is `True`, will return a string indicating how the parameter's value
            was originally obtained. Can be one of {model, initialization, inheritance, calculation,
            default, or previous_year}.
        """

        param_val = utils.get_param(self, param, node, year,
                                    tech=tech,
                                    context=context,
                                    sub_context=sub_context,
                                    return_source=return_source,
                                    do_calc=do_calc,
                                    check_exist=check_exist,
                                    dict_expected=dict_expected)

        return param_val

    def set_param(self, val, param, node, year=None, tech=None, context=None, sub_context=None,
                  save=True):
        """
        Sets a parameter's value, given a specific context (node, year, tech, context, sub-context).
        This is intended for when you are using this function outside of model.run to make single changes
        to the model description.

        Parameters
        ----------
        val : any or list of any
            The new value(s) to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        param_val = utils.set_param(self, val, param, node, year,
                                    tech=tech,
                                    context=context,
                                    sub_context=sub_context,
                                    save=save)

        return param_val

    def set_param_internal(self, val, param, node, year=None, tech=None, context=None, sub_context=None):
        """
        Sets a parameter's value, given a specific context (node, year, tech, context, sub_context).
        This is used from within the model.run function and is not intended to make changes to the model
        description externally (see `set_param`).

        Parameters
        ----------
        val : dict
            The new value(s) to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        param_val = utils.set_param_internal(self, val, param, node, year,
                                             tech=tech,
                                             context=context,
                                             sub_context=sub_context)

        return param_val

    def set_param_wildcard(self, val, param, node_regex, year, tech=None, context=None, sub_context=None,
                           save=True):
        """
        Sets a parameter's value, for all contexts (node, year, tech, context, sub_context)
        that satisfy/match the node_regex pattern

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node_regex : str
            The regex pattern of the node (branch format) whose parameter you are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """
        for node in self.graph.nodes:
            if re.search(node_regex, node) != None:
                self.set_param(val, param, node, year, tech, context, sub_context, save)

    def set_param_file(self, filepath):
        """
        Sets parameters' values, for all context (node, year, context, sub_context, and technology)
        from the provided CSV file. See Data_Changes_Tutorial_by_CSV.ipynb for detailed
        description of expected CSV file columns and values.

        Parameters
        ----------
        filepath : str
            This is the path to the CSV file containing all context and value change information
        """
        param_val = utils.set_param_file(self, filepath)

        return param_val

    def set_param_search(self, val, param, node, year=None, tech=None, context=None, sub_context=None,
                         val_operator='==', create_missing=False, row_index=None):
        """
        Sets parameter values for all contexts (node, year, tech, context, sub_context),
        searching through all tech, context, and sub_context keys if necessary.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `context, `sub_context`, and `tech`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
            specified node, param, year, context, and sub_context.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model
            description. If context is `.*`, all possible context keys will be searched at the specified node, param,
            year, sub_context, and tech.
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description.
            If sub_context is `.*`, all possible sub_context keys will be searched at the specified node, param,
            year, context, and tech.
        create_missing : bool, optional
            Will create a new parameter in the model if it is missing. Defaults to False.
        val_operator : str, optional
            This specifies how the value should be set. The possible values are '>=', '<=' and '=='.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the row number in error messages.
        """

        param_val = utils.set_param_search(self, val, param, node, year,
                                           tech=tech,
                                           context=context,
                                           sub_context=sub_context,
                                           val_operator=val_operator,
                                           create_missing=create_missing,
                                           row_index=row_index)

        return param_val

    def create_param(self, val, param, node, year=None, tech=None, context=None, sub_context=None,
                     row_index=None, param_source=None, branch=None):
        """
        Creates parameter in graph, for given context (node, year, tech, context, sub_context),
        and sets the value to val. Returns True if param was created successfully and False otherwise.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            'competition type' can be retrieved without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
            specified node, param, year, context, and sub_context.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model
            description. If context is `.*`, all possible context keys will be searched at the specified node, param,
            year, sub_context, and tech.
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description.
            If sub_context is `.*`, all possible sub_context keys will be searched at the specified node, param,
            year, context, and tech.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the row number in error messages.

        """
        param_val = utils.create_param(self, val, param, node, year,
                                       tech=tech,
                                       context=context,
                                       sub_context=sub_context,
                                       row_index=row_index,
                                       param_source=param_source,
                                       branch=branch)

        return param_val

    def set_param_log(self, output_file=''):
        """
        Writes the saved change history to a CSV specified at `output_file` if provided.

        Parameters
        ----------
        output_file : str, optional
            The output file location where the change history CSV will be saved. If this is left blank,
            the file will be outputed at the current location with the name of the original model description
            and a timestamp in the filename.
        """
        if output_file == '':
            filename = self.model_description_file.split('/')[-1].split('.')[0]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = './change_log_' + filename + '_' + timestamp + '.csv'
        self.change_history.to_csv(output_file, index=False)

    def save_model(self, model_file='', save_changes=True):
        """
        Saves the current model to a pickle file at `model_file` if specified

        Parameters
        ----------
        model_file : str, optional
            The model file location where the pickled model file will be saved. If this is left blank,
            the model will be saved at the current location with the name of the original model description
            and a timestamp in the filename.
        save_changes : bool, optional
            This specifies whether the changes will be written to CSV
        """
        if model_file != '' and not model_file.endswith('.pkl'):
            print('model_file must end with .pkl extension. No model was saved.')
        else:
            if model_file == '':
                filename = self.model_description_file.split('/')[-1].split('.')[0]
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_file = 'model_' + filename + '_' + timestamp + '.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(self, f)
        if save_changes:
            self.set_param_log(output_file='change_log_' + model_file)


def load_model(model_file):
    """
    Loads the model at `model_file`

    Parameters
    ----------
    model_file : str
        The model file location where the pickled model file is saved
    """
    f = open(model_file, 'rb')
    model = pickle.load(f)
    return model
