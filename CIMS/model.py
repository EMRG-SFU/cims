import copy
import warnings
import networkx as nx
import pandas as pd
import re
import time
import pickle
import os.path

from . import graph_utils
from . import old_utils
from . import lcc_calculation
from . import stock_allocation
from . import loop_resolution
from . import tax_foresight
from . import cost_curves
from . import aggregation
from . import visualize
from . import node_utils

from .readers.scenario_reader import ScenarioReader
from .readers.model_reader import ModelReader
from .model_validation import ModelValidator
from .aggregation import quantity_aggregation as qa
from .quantities import ProvidedQuantity, DistributedSupply
from .emissions import EmissionsCost

from .old_utils import create_value_dict, inheritable_params, inherit_parameter

from .utils import parameters as PARAM
from .utils import model_columns as COL


class Model:
    """
    Relevant dataframes and associated information taken from the model description provided in
    `reader`. Also includes methods needed for building and running the Model.

    Parameters
    ----------
    reader : CIMS.reader
        The Reader set up to ingest the description (excel file) for our model.

    Attributes
    ----------
    graph : networkx.DiGraph
        Model Graph populated using the `build_graph` method. Model services are nodes in `graph`,
        with data contained within an associated dictionary. Structural and Request/Provide
        relationships are edges in the `graph`.

    node_dfs : dict {str: pandas.DataFrame}
        Node names (branch notation) are the keys in the dictionary. Associated DataFrames (specified in
        the excel model description) are the values. DataFrames do not include technology information for a node.

    tech_dfs : dict {str: dict {str: pandas.DataFrame}}
        Technology & service information from the excel model description. Node names (branch notation)
        are keys in `tech_dfs` to sub-dictionaries. These sub-dictionaries have technology/service
        names as keys and pandas DataFrames as values. These DataFrames contain information from the
        excel model description.

    supply_nodes : list [str]
        List of supply nodes requested by the demand side of the Model Graph. 
        Populated using the `build_graph` method.

    years : list [str or int]
        List of the years for which the model will be run.

    """

    def __init__(self,
                 csv_init_file_paths,
                 csv_update_file_paths,
                 col_list,
                 year_list,
                 sector_list,
                 default_values_csv_path
                 ):

        self.validator = ModelValidator(
                csv_file_paths = csv_init_file_paths,
                col_list = col_list,
                year_list = year_list,
                sector_list = sector_list,
                scenario_files = csv_update_file_paths,
                default_values_csv_path = default_values_csv_path
                )

        self.graph = nx.DiGraph()

        self._model_reader = ModelReader(
                csv_file_paths = csv_init_file_paths,
                col_list = col_list,
                year_list = year_list,
                sector_list = sector_list,
                default_values_csv_path = default_values_csv_path)

        self._scenario_reader = ScenarioReader(
                csv_file_paths = csv_update_file_paths,
                col_list = col_list,
                year_list = year_list,
                sector_list = sector_list)

        self.root = self._model_reader.root
        self.node_dfs, self.tech_dfs = self._model_reader.get_model_description()
        self.scenario_node_dfs, self.scenario_tech_dfs = self._scenario_reader.get_model_description()
        self.node_tech_defaults = self._model_reader.get_default_params()
        self.step = 5 # ::TODO:: Make this an input or calculate
        self.supply_nodes = []
        self.GHGs = []
        self.emission_types = []
        self.gwp = {}
        self.years = self._model_reader.get_years()
        self.base_year = int(self.years[0])
        self.prices = {}
        self.equilibrium_count = 0

        ## GRAPH BUILDING HERE
        self.build_graph()


        self.dcc_classes = self._dcc_classes()
        self.dic_classes = self._dic_classes()
        self._inherit_parameter_values()
        self._initialize_tax()

        self.show_run_warnings = True
        self.model_description_file_prefix = os.path.commonprefix(self._model_reader.csv_files)
        self.scenario_model_description_file = self._scenario_reader.csv_files

        self.change_history = pd.DataFrame(
            columns=['base_model_description', 
                     COL.parameter.lower(), 
                     'node', 
                     'year', 
                     COL.technology.lower(),
                     COL.context.lower(),
                     COL.sub_context.lower(),
                     'old_value', 
                     'new_value'])

        self.status = 'instantiated'

        # ::TODO:: Now do the stuff that happens otherwise in the `update` method.

        if not isinstance(self._scenario_reader, ScenarioReader):
            raise ValueError("You are attempting to update a model with \
                    something other than a ScenarioReader object.")

        # ::TODO:: What does this do??
        self.graph.max_tree_index[0] = 0
        graph = node_utils.make_or_update_nodes(self.graph, self.scenario_node_dfs, self.scenario_tech_dfs)
        graph = graph_utils.make_or_update_edges(graph, self.scenario_node_dfs, self.scenario_tech_dfs)
        # ::TODO:: What does this do??
        self.graph.cur_tree_index[0] += self.graph.max_tree_index[0]

        self.graph = graph
        self.supply_nodes = graph_utils.get_supply_nodes(graph)
        self.GHGs, self.emission_types, self.gwp = graph_utils.get_ghg_and_emissions(graph, str(self.base_year))
        self.dcc_classes = self._dcc_classes()
        self.dic_classes = self._dic_classes()
        self._inherit_parameter_values()
        self._initialize_tax()
        self.show_run_warnings = True

    def validate_files(self, verbose=False):
        self.validator.validate(verbose=verbose)

    def update(self, scenario_model_reader):
        """
        Create an updated version of self based off another ModelReader.
        Intended for use with a reference + scenario model setup.

        Parameters
        ----------
        scenario_model_reader : CIMS.ModelReader
            An instantiated ModelReader to be used for updating self.

        Returns
        -------
        CIMS.Model :
            An updated version of self.
        """
        if self.status.lower() in ['run initiated', 'run completed']:
            raise ValueError("You've attempted to update a model which has \
                             already been run. To prevent inconsistencies, \
                             this update has not been done.")

        if not isinstance(scenario_model_reader, ScenarioReader):
            raise ValueError("You are attempting to update a model with \
                             something other than a ScenarioReader object.")

        # Make a copy, so we don't alter self
        model = copy.deepcopy(self)

        # Update the model's node_df & tech_dfs
        model.scenario_node_dfs, model.scenario_tech_dfs = \
            scenario_model_reader.get_model_description()

        # Update the nodes & edges in the graph
        self.graph.max_tree_index[0] = 0
        graph = node_utils.make_or_update_nodes(model.graph, model.scenario_node_dfs,
                                                 model.scenario_tech_dfs)
        graph = graph_utils.make_or_update_edges(graph, model.scenario_node_dfs,
                                                 model.scenario_tech_dfs)
        self.graph.cur_tree_index[0] += self.graph.max_tree_index[0]
        model.graph = graph

        # Update the Model's metadata
        model.supply_nodes = graph_utils.get_supply_nodes(graph)

        model.GHGs, model.emission_types, model.gwp = graph_utils.get_ghg_and_emissions(graph,
                                                                                        str(model.base_year))
        model.dcc_classes = model._dcc_classes()
        model.dic_classes = model._dic_classes()

        # Re-initialize the model
        model._inherit_parameter_values()
        model._initialize_tax()

        model.show_run_warnings = True
        model.scenario_model_description_file = scenario_model_reader.csv_files

        return model

    def build_graph(self):
        """

        Builds graph based on the model reader used in instantiation of the class. Stores this graph
        in `self.graph`. Additionally, initializes `self.supply_nodes`.

        Returns
        -------
        None

        """
        graph = nx.DiGraph()
        node_dfs = self.node_dfs
        tech_dfs = self.tech_dfs
        graph.cur_tree_index = [0]
        graph.max_tree_index = [0]
        graph = node_utils.make_or_update_nodes(graph, node_dfs, tech_dfs)
        graph = graph_utils.make_or_update_edges(graph, node_dfs, tech_dfs)
        graph.cur_tree_index[0] += graph.max_tree_index[0]

        self.supply_nodes = graph_utils.get_supply_nodes(graph)
        self.GHGs, self.emission_types, self.gwp = graph_utils.get_ghg_and_emissions(graph,
                                                                                     str(self.base_year))
        self.graph = graph

    def _initialize_tax(self):
        # Initialize Taxes
        for year in self.years:
            # Pass tax to all children for carbon cost
            graph_utils.top_down_traversal(self.graph,
                                           self._init_tax_emissions,
                                           year)
        # Initialize Tax Foresight
        tax_foresight.initialize_tax_foresight(self)

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
            if PARAM.technologies in nodes[node][base_year]:
                for tech in nodes[node][base_year][PARAM.technologies]:
                    dccc = self.get_param(PARAM.dcc_class, node, base_year, tech=tech)
                    if dccc is not None:
                        if dccc in dcc_classes:
                            dcc_classes[dccc].append((node, tech))
                        else:
                            dcc_classes[dccc] = [(node, tech)]

        return dcc_classes

    def _dic_classes(self):
        """
        Iterate through each node and technology in self to create a dictionary mapping Declining
        Capital Cost (DCC) Classes to a list of nodes that belong to that class.

        Returns
        -------
        dict {str: [str]}:
            Dictionary where keys are declining capital cost classes (str) and values are lists of
            nodes (str) belonging to that class.
        """
        dic_classes = {}

        nodes = self.graph.nodes
        base_year = str(self.base_year)
        for node in nodes:
            if PARAM.technologies in nodes[node][base_year]:
                for tech in nodes[node][base_year][PARAM.technologies]:
                    dicc = self.get_param(PARAM.dic_class, node, base_year, tech=tech)
                    if dicc is not None:
                        if dicc in dic_classes:
                            dic_classes[dicc].append((node, tech))
                        else:
                            dic_classes[dicc] = [(node, tech)]

        return dic_classes

    def run(self, equilibrium_threshold=0.05, num_equilibrium_iterations=2, min_iterations=2,
            max_iterations=10, show_warnings=True, print_eq=False):
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
            an equilibrium. TODO: Document why this has changed to 2.

        max_iterations : int, optional
            The maximum number of times to iterate between supply and demand in an attempt to reach
            an equilibrium. If max_iterations is reached, a warning will be raised, iteration for
            that year will stop, and iteration for the next year will begin.

        verbose : bool, optional
            Whether or not to have verbose printing during iterations. If true, supply node prices are
            printed at the end of each iteration.

        Returns
        -------
            Nothing is returned, but `self.graph` will be updated with the resulting prices,
            provided_quantities, etc calculated for each year.

        """
        self.show_run_warnings = show_warnings
        self.status = 'Run initiated'

        self.loops = graph_utils.find_loops(self.graph, warn=True)

        demand_nodes = graph_utils.get_demand_side_nodes(self.graph)
        supply_nodes = graph_utils.get_supply_side_nodes(self.graph)

        for year in self.years:
            print(f"***** ***** year: {year} ***** *****")

            # Initialize Basic Variables
            equilibrium = False
            self.equilibrium_count = 0
            iteration = 1

            # Initialize Graph Values
            self.initialize_graph(self.graph, year)
            while self.equilibrium_count < num_equilibrium_iterations or \
                    iteration <= min_iterations:
                # Early exit if we reach the maximum number of iterations
                if iteration > max_iterations:
                    warnings.warn(f"Max iterations reached for year {year}. Continuing to next year.")
                    break
                print(f'iter {iteration}')
                # Initialize Iteration Specific Values
                self.iteration_initialization(year)

                # DEMAND
                # ******************
                # Calculate Life Cycle Cost values on demand side
                graph_utils.bottom_up_traversal(nx.subgraph(self.graph, demand_nodes),
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self)

                # Calculate Quantities (Total Stock Needed)
                graph_utils.top_down_traversal(nx.subgraph(self.graph, demand_nodes),
                                               self.stock_allocation_and_retirement,
                                               year)

                # Calculate Service Costs on Demand Side
                graph_utils.bottom_up_traversal(nx.subgraph(self.graph, demand_nodes),
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self)

                # Supply
                # ******************
                # Calculate Service Costs on Supply Side
                graph_utils.bottom_up_traversal(nx.subgraph(self.graph, supply_nodes),
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                cost_curve_min_max=True)
                # Calculate Supply Quantities
                graph_utils.top_down_traversal(nx.subgraph(self.graph, supply_nodes),
                                               self.stock_allocation_and_retirement,
                                               year,
                                               )

                # Calculate Service Costs on Supply Side
                graph_utils.bottom_up_traversal(nx.subgraph(self.graph, supply_nodes),
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                )

                # Check for an Equilibrium -- Across all nodes, not just supply nodes
                # ************************
                # Find the previous prices
                prev_prices = self.prices
                # Go get all the new prices
                new_prices = {node: self.get_param(PARAM.price, node, year, do_calc=True) for node in
                              self.graph.nodes()}

                # Check for an equilibrium in prices
                equilibrium = int(year) == self.base_year or \
                              self.check_equilibrium(prev_prices, new_prices, iteration,
                                                     equilibrium_threshold, print_eq)

                if equilibrium:
                    self.equilibrium_count += 1
                else:
                    self.equilibrium_count = 0

                self.prices = new_prices

                # Next Iteration
                # **************
                iteration += 1

            # Once we've reached an equilibrium, calculate the quantities requested by each node.
            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_requested_quantities,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            supply_nodes=self.supply_nodes)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_direct_emissions,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            supply_nodes=self.supply_nodes)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_cumulative_emissions,
                                            year,
                                            loop_resolution_func=loop_resolution.aggregation_resolution,
                                            supply_nodes=self.supply_nodes)

            graph_utils.bottom_up_traversal(self.graph,
                                            self._aggregate_distributed_supplies,
                                            year)
        self.status = 'Run completed'

    def check_equilibrium(self, prev: dict, new: dict, iteration: int, threshold: float,
                          print_equilibrium_details: bool) -> bool:
        """
        Return False unless an equilibrium has been reached.
            1. Check if prev is empty or year not in previous (first year or first
               iteration)
            2. For every node, check if the relative difference in price exceeds the threshold
                (A) If it does, return False
                (B) Otherwise, keep checking
            3. If all nodes are checked and no relative difference exceeds the threshold, return
               True

        Parameters
        ----------
        prev : Prices from the previous iteration.

        new : Prices from the current iteration.

        threshold : The threshold to use for determining whether an equilibrium has been reached.

        print_equilibrium_details : Whether to print details regarding the nodes responsible for an
        equilibrium not being reached.

        Returns
        -------
        True if all nodes changed less than `threshold`. False otherwise.
        """
        # For every node, check if the relative difference in price exceeds the threshold
        equilibrium_reached = True
        for node in new:
            prev_price = prev[node]
            new_price = new[node]
            if (prev_price is None) or (new_price is None):
                print(f"\tNot at equilibrium: {node} does not have an LCC calculated")
                equilibrium_reached = False
            abs_diff = abs(new_price - prev_price)

            if prev_price == 0:
                if self.show_run_warnings:
                    warnings.warn(f"Previous price is 0 for {node}")
                rel_diff = 0
            else:
                rel_diff = abs_diff / prev_price

            # If any node's relative difference exceeds threshold, equilibrium has not been reached
            if rel_diff > threshold:
                equilibrium_reached = False
                if print_equilibrium_details and iteration > 1:
                    print(
                        f"\tNot at equilibrium: {node} has {rel_diff:.1%} difference between"
                        f" iterations")

        return equilibrium_reached

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
            if PARAM.tax in graph.nodes[parent][year]:
                parent_dict = copy.deepcopy(graph.nodes[parent][year][PARAM.tax])

        # Update parameter source for values from parent
        for ghg in parent_dict:
            for emission_type in parent_dict[ghg]:
                parent_dict[ghg][emission_type][PARAM.param_source] = 'inheritance'

        # Store away tax at current node to overwrite parent tax later
        node_dict = {}
        if PARAM.tax in graph.nodes[node][year]:
            node_dict = copy.deepcopy(graph.nodes[node][year][PARAM.tax])
            # Remove any inherited values from the update
            for ghg in list(node_dict):
                for emission_type in list(node_dict[ghg]):
                    if node_dict[ghg][emission_type][PARAM.param_source] == 'inheritance' or node_dict[ghg][emission_type][PARAM.year_value] is None:
                        node_dict[ghg].pop(emission_type)
                        if len(node_dict[ghg]) == 0:
                            node_dict.pop(ghg)

        # Make final dict where we prioritize keeping node_dict and only unique parent taxes
        final_tax = copy.deepcopy(node_dict)
        for ghg in parent_dict:
            if ghg not in final_tax:
                final_tax[ghg] = {}
            for emission_type in parent_dict[ghg]:
                if emission_type not in final_tax[ghg]:
                    final_tax[ghg][emission_type] = parent_dict[ghg][emission_type]

        if final_tax:
            graph.nodes[node][year][PARAM.tax] = final_tax

    def initialize_graph(self, graph, year):
        """
        Initializes the graph at the start of a simulation year.
        Specifically, initializes (1) price multiplier values and (2) supply nodes' Life Cycle Cost
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
                if PARAM.price_multiplier in graph.nodes[parent][year]:
                    price_multipliers = copy.deepcopy(
                        self.graph.nodes[parent][year][PARAM.price_multiplier])
                    parent_price_multipliers.update(price_multipliers)

            # Grab the price multipliers from the current node (if they exist) and replace the parent price multipliers
            node_price_multipliers = copy.deepcopy(parent_price_multipliers)
            if PARAM.price_multiplier in graph.nodes[node][year]:
                price_multipliers = self.get_param(PARAM.price_multiplier, node, year, dict_expected=True)
                node_price_multipliers.update(price_multipliers)

            # Set Price Multiplier of node in the graph
            graph.nodes[node][year][PARAM.price_multiplier] = node_price_multipliers

        def init_supply_node_lcc(graph, node, year, step=5):
            """
            Function for initializing Life Cycle Cost for a node in a graph, if that node is a supply
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
            Cycle Cost if node is a supply node.
            """

            def calc_lcc_from_children():
                """
                Helper function to calculate a node's Life Cycle Cost from its children.

                Returns
                -------
                Nothing is returned, but the node will be updated with a new Life Cycle Cost value.
                """
                # Find the subtree rooted at the supply node
                descendants = nx.descendants(graph, node) | {node}

                descendant_tree = nx.subgraph(graph, descendants)

                # Calculate the Life Cycle Costs for the sub-tree
                graph_utils.bottom_up_traversal(descendant_tree,
                                                lcc_calculation.lcc_calculation,
                                                year,
                                                self,
                                                root=node)

            if node in self.supply_nodes:
                if PARAM.lcc_financial in graph.nodes[node][year]:
                    if self.get_param(PARAM.lcc_financial, node, year) is None:
                        calc_lcc_from_children()
                elif PARAM.cost_curve_function in graph.nodes[node]:
                    lcc = cost_curves.calc_cost_curve_lcc(self, node, year)
                    graph.nodes[node][year][PARAM.lcc_financial] = old_utils.create_value_dict(lcc, param_source='cost curve function')
                else:
                    # Life Cycle Cost needs to be calculated from children
                    calc_lcc_from_children()

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
            if PARAM.technologies in graph.nodes[node][year]:
                techs = graph.nodes[node][year][PARAM.technologies]
                for tech in techs:
                    tech_data = techs[tech]
                    if PARAM.emissions in tech_data:
                        emission_data = tech_data[PARAM.emissions]
                        for ghg in emission_data:
                            for emission_type in emission_data[ghg]:
                                try:
                                    emission_data[ghg][emission_type][PARAM.year_value] *= gwp[ghg]
                                except KeyError:
                                    continue

            # Emissions from a node
            elif PARAM.emissions in graph.nodes[node][year]:
                emission_data = graph.nodes[node][year][PARAM.emissions]
                for ghg in emission_data:
                    for emission_type in emission_data[ghg]:
                        try:
                            emission_data[ghg][emission_type][PARAM.year_value] *= gwp[ghg]
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
            if PARAM.load_factor not in graph.nodes[node][year]:
                # Check if a load factor was defined at the node's structural parent (its first
                # parent). If so, use this load factor for the node.
                parents = list(graph.predecessors(node))
                if len(parents) > 0:
                    parent = parents[0]
                    if PARAM.load_factor in graph.nodes[parent][year]:
                        val = graph.nodes[parent][year][PARAM.load_factor][PARAM.year_value]
                        units = graph.nodes[parent][year][PARAM.load_factor][PARAM.unit]
                        graph.nodes[node][year][PARAM.load_factor] = old_utils.create_value_dict(val,
                                                                                         unit=units,
                                                                                         param_source='inheritance')

            if PARAM.load_factor in graph.nodes[node][year]:
                # Ensure this load factor is recorded at each of the technologies within the node.
                if PARAM.technologies in graph.nodes[node][year]:
                    tech_data = graph.nodes[node][year][PARAM.technologies]
                    for tech in tech_data:
                        if PARAM.load_factor not in tech_data[tech]:
                            val = graph.nodes[node][year][PARAM.load_factor][PARAM.year_value]
                            units = graph.nodes[node][year][PARAM.load_factor][PARAM.unit]
                            tech_data[tech][PARAM.load_factor] = old_utils.create_value_dict(val,
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
                if PARAM.tax in graph.nodes[parent][year]:
                    parent_dict = graph.nodes[parent][year][PARAM.tax]

            # Store away tax at current node to overwrite parent tax later
            node_dict = {}
            if PARAM.tax in graph.nodes[node][year]:
                node_dict = graph.nodes[node][year][PARAM.tax]

            # Make final dict where we prioritize keeping node_dict and only unique parent taxes
            final_tax = copy.deepcopy(node_dict)
            for ghg in parent_dict:
                if ghg not in final_tax:
                    final_tax[ghg] = {}
                for emission_type in parent_dict[ghg]:
                    if emission_type not in final_tax[ghg]:
                        final_tax[ghg][emission_type] = parent_dict[ghg][emission_type]

            if final_tax:
                graph.nodes[node][year][PARAM.tax] = final_tax

        def init_agg_emissions_cost(graph):
            # Reset the aggregate_emissions_cost at each node
            for n in self.graph.nodes():
                self.graph.nodes[n][year][PARAM.aggregate_emissions_cost_rate] = \
                    create_value_dict({}, param_source='initialization')
                self.graph.nodes[n][year][PARAM.cumul_emissions_cost_rate] = \
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
                                        init_supply_node_lcc,
                                        year)

    def iteration_initialization(self, year):
        # Reset the provided_quantities at each node
        for n in self.graph.nodes():
            self.graph.nodes[n][year][PARAM.provided_quantities] = create_value_dict(ProvidedQuantity(),
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
            The name of the node (branch notation) where stock stock retirement and allocation will
            be performed.

        year: str
            The year to perform stock retirement and allocation.

        Returns
        -------
            Nothing is returned. `self` will be updated to reflect the results of stock retirement
            and new stock competitions.
        """
        comp_type = self.get_param(PARAM.competition_type, node).lower()

        if comp_type in ['tech compete']:
            stock_allocation.all_tech_compete_allocation(self, node, year)
        else:
            stock_allocation.general_allocation(self, node, year)

    def _aggregate_requested_quantities(self, graph, node, year, **kwargs):
        """
        Calculates and records supply quantities attributable to a node in the specified year. Supply
        quantities can be attributed to a node in 3 ways:

        (1) via request/provide relationships - any supply quantity directly requested of the node
        (e.g. Lighting requests services directly from Electricity) and supply quantities that
        are indirectly requested, but can be attributed to the node (e.g. Housing requests
        Electricity via its request of Lighting).

        (2) via structural relationships - supply nodes pass indirect quantities to their structural
        parents, rather than request/provide parents. Additionally, root & region nodes collect
        quantities via their structural children, rather than their request/provide children.

        (3) via weighted aggregate relationships - if specified in the model description, nodes will
        aggregate quantities structurally. For example, if a market node has
        `structural_aggregation` turned on, any quantities (direct or in-direct) from the market
        children aggregate through structural parents (i.e. BC.Natural Gas) instead of the market
        which it has a request/provide relationship with (CAN.Natural Gas).

        This method was built to be used with the bottom-up traversal method, which ensures a node
        is only visited once all its children have been visited (except when needs to break a loop).
        """
        aggregation.aggregate_requested_quantities(self, node, year)

    def _aggregate_direct_emissions(self, graph, node, year, **kwargs):
        # Net Emissions
        aggregation.aggregate_direct_emissions(self, graph, node, year,
                                               rate_param=PARAM.net_emissions_rate,
                                               total_param=PARAM.total_direct_net_emissions)
        # Avoided Emissions
        aggregation.aggregate_direct_emissions(self, graph, node, year,
                                               rate_param=PARAM.avoided_emissions_rate,
                                               total_param=PARAM.total_direct_avoided_emissions)

        # Negative Emissions
        aggregation.aggregate_direct_emissions(self, graph, node, year,
                                               rate_param=PARAM.negative_emissions_rate,
                                               total_param=PARAM.total_direct_negative_emissions)

        # Bio Emissions
        aggregation.aggregate_direct_emissions(self, graph, node, year,
                                               rate_param=PARAM.bio_emissions_rate,
                                               total_param=PARAM.total_direct_bio_emissions)

        # Emissions Cost
        aggregation.aggregate_direct_emissions_cost(self, graph, node, year,
                                                    rate_param=PARAM.emissions_cost_rate,
                                                    total_param=PARAM.total_direct_emissions_cost)

    def _aggregate_cumulative_emissions(self, graph, node, year, **kwargs):
        # Net Emissions
        aggregation.aggregate_cumulative_emissions(
            self, node, year,
            rate_param=PARAM.cumul_net_emissions_rate,
            total_param=PARAM.total_cumul_net_emissions
        )

        # Avoided Emissions
        aggregation.aggregate_cumulative_emissions(
            self, node, year,
            rate_param=PARAM.cumul_avoided_emissions_rate,
            total_param=PARAM.total_cumul_avoided_emissions
        )

        # Negative Emissions
        aggregation.aggregate_cumulative_emissions(
            self, node, year,
            rate_param=PARAM.cumul_negative_emissions_rate,
            total_param=PARAM.total_cumul_negative_emissions
        )

        # Bio Emissions
        aggregation.aggregate_cumulative_emissions(
            self, node, year,
            rate_param=PARAM.cumul_bio_emissions_rate,
            total_param=PARAM.total_cumul_bio_emissions
        )

        # Emissions Cost
        aggregation.aggregate_cumulative_emissions_cost(
            self, node, year,
            rate_param=PARAM.cumul_emissions_cost_rate,
            total_param=PARAM.total_cumul_emissions_cost
        )

    def _aggregate_distributed_supplies(self, graph, node, year, **kwargs):
        aggregation.aggregate_distributed_supplies(self, node, year)

    def get_parameter_default(self, parameter):
        return self.node_tech_defaults[parameter]

    def get_param(self, param, node, year=None, tech=None, context=None, sub_context=None,
                  return_source=False, do_calc=False, check_exist=False, dict_expected=False):
        """
        Gets a parameter's value from the model, given a specific context (node,
        year, tech, context, sub-context), calculating the parameter's value if
        needed.

        This will not re-calculate the parameter's value, but will only retrieve
        values which are already stored in the model or obtained via
        inheritance, default values, or estimation using the previous
        year's value. If return_source is True, this function will also,
        return how this value was originally obtained (e.g. via calculation)

        Parameters
        ----------
        param : str
            The name of the parameter whose value is being retrieved.
        node : str
            The name of the node (branch format) whose parameter you are
            interested in retrieving.
        year : str, optional
            The year which you are interested in. `year` is not required
            for parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type).
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level. 
            `tech` is required to get any parameter that is stored within a 
            technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the 
            `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 
            `sub_context` column in the model description
        return_source : bool, default=False
            Whether or not to return the method by which this value was 
            originally obtained.
        do_calc : bool, default=False
            If False, the function will only retrieve the value using the
            current value in the model, inheritance, default, or the previous
            year's value. It will _not_ calculate the parameter value. If True,
            calculation is allowed.
        check_exist : bool, default=False
            Whether or not to check that the parameter exists as is given the
            context (without calculation, inheritance, or checking past years)
        dict_expected : bool, default=False
            Used to disable the warning get_param is returning a dict. Get_param
            should normally return a single value (float, str, etc.). If the
            user knows it expects a dict, then this flag is used.

        Returns
        -------
        any :
            The value of the specified `param` at `node`, given the provided
            `year` and `tech`.
        str :
            If return_source is `True`, returns a string indicating how the
            parameter's value was originally obtained {model, initialization, 
            inheritance, calculation, default, or previous_year}.
        """

        param_val = old_utils.get_param(self, param, node, year,
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
        Sets a parameter's value, given a specific context (node, year, tech,
        context, sub-context). This is intended for when you are using this 
        function outside of model.run to make single changes to the model description.

        Parameters
        ----------
        val : any or list of any
            The new value(s) to be set at the specified `param` at `node`, 
            given the `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are
            interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for
            parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type)
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level. 
            `tech` is required to get any parameter that is stored within a
            technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the
            `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the
            `sub_context` column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log
            csv where True means the change will be saved and False means it
            will not be saved
        """

        param_val = old_utils.set_param(self, val, param, node, year,
                                    tech=tech,
                                    context=context,
                                    sub_context=sub_context,
                                    save=save)

        return param_val

    def set_param_internal(self, val, param, node, year=None, tech=None, context=None,
                           sub_context=None):
        """
        Sets a parameter's value, given a specific context (node, year, tech, 
        context, sub_context). This is used from within the model.run function
        and is not intended to make changes to the model description externally
        (see `set_param`).

        Parameters
        ----------
        val : dict
            The new value(s) to be set at the specified `param` at `node`, given
            the provided `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are
            interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for
            parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type).
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level. `tech`
            is required to get any parameter that is stored within a technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the
            `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 
            `sub_context` column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the
            change_log csv where True means the change will be saved and False
            means it will not be saved
        """

        param_val = old_utils.set_param_internal(self, val, param, node, year,
                                             tech=tech,
                                             context=context,
                                             sub_context=sub_context)

        return param_val

    def set_param_wildcard(self, val, param, node_regex, year, tech=None, context=None,
                           sub_context=None,
                           save=True):
        """
        Sets a parameter's value, for all contexts (node, year, tech, context, 
        sub_context) that satisfy/match the node_regex pattern

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given
            the provided `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node_regex : str
            The regex pattern of the node (branch format) whose parameter you
            are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for
            parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type). 
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level. 
            `tech` is required to get any parameter that is stored within a
            technology.
        context : str, optional
            Used when there is context available in the node. Analogous to the
            `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the
            `sub_context` column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log
            csv where True means the change will be saved and False means it
            will not be saved
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
        param_val = old_utils.set_param_file(self, filepath)

        return param_val

    def set_param_search(self, val, param, node, year=None, tech=None, context=None,
                         sub_context=None,
                         val_operator='==', create_missing=False, row_index=None):
        """
        Sets parameter values for all contexts (node, year, tech, context, sub_context),
        searching through all tech, context, and sub_context keys if necessary.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given
            the provided `year`, `context, `sub_context`, and `tech`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are
            interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for
            parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type).
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level. `tech`
            is required to get any parameter that is stored within a technology.
            If tech is `.*`, all possible tech keys will be searched at the
            specified node, param, year, context, and sub_context.
        context : str, optional
            Used when there is context available in the node. Analogous to the 
            `context` column in the model description. If context is `.*`, all
            possible context keys will be searched at the specified node, param,
            year, sub_context, and tech.
        sub_context : str, optional
            Must be used only if context is given. Analogous to the
            `sub_context` column in the model description. If sub_context is
            `.*`, all possible sub_context keys will be searched at the
            specified node, param, year, context, and tech.
        create_missing : bool, optional
            Will create a new parameter in the model if it is missing. Defaults
            to False.
        val_operator : str, optional
            This specifies how the value should be set. The possible values are 
            `>=`, `<=` and `==`.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the
            row number in error messages.
        """

        param_val = old_utils.set_param_search(self, val, param, node, year,
                                           tech=tech,
                                           context=context,
                                           sub_context=sub_context,
                                           val_operator=val_operator,
                                           create_missing=create_missing,
                                           row_index=row_index)

        return param_val

    def create_param(self, val, param, node, year=None, tech=None, context=None, sub_context=None,
                     row_index=None, param_source=None, target=None):
        """
        Creates parameter in graph, for given context (node, year, tech,
        context, sub_context), and sets the value to val. Returns True if
        param was created successfully and False otherwise.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given
            the provided `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are
            interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for
            parameters specified at the node level and which by definition
            cannot change year to year (e.g. competition type).
        tech : str, optional
            The name of the technology you are interested in. `tech` is not
            required for parameters that are specified at the node level.
            `tech` is required to get any parameter that is stored within a
            technology. If tech is `.*`, all possible tech keys will be searched
            at the specified node, param, year, context, and sub_context.
        context : str, optional
            Used when there is context available in the node. Analogous to the
            `context` column in the model description. If context is `.*`, all
            possible context keys will be searched at the specified node, param,
            year, sub_context, and tech.
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 
            `sub_context` column in the model description. If sub_context is 
            `.*`, all possible sub_context keys will be searched at the 
            specified node, param, year, context, and tech.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the
            row number in error messages.
        """
        param_val = old_utils.create_param(self, val, param, node, year,
                                       tech=tech,
                                       context=context,
                                       sub_context=sub_context,
                                       target=target,
                                       row_index=row_index,
                                       param_source=param_source)

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
            filename = self.model_description_file_prefix.split('/')[-1].split('.')[0]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f"./change_log_{filename}_{timestamp}.csv"
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
                filename = self.model_description_file_prefix.split('/')[-1].split('.')[0]
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_file = f"model_{filename}_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self, f)
        if save_changes:
            self.set_param_log(output_file=f"change_log_{model_file}")

    def visualize_prices_change_over_time(
            self,
            out_file="supply_prices_over_years.png",
            show=False):
        """Creates a visualization of supply prices over time as a multi-line
        graph. A wrapper for the visualize.visualize_prices_change_over_time()
        function.

        Parameters
        ----------
        out_file : str, optional
            Filepath to the location where the visualization will be saved, by
            default "supply_prices_over_years.png".
        show : bool, optional
            If True, displays the generated figure, by default False
        """
        visualize.visualize_prices_change_over_time(
            self, out_file=out_file, show=show)

    def visualize_price_comparison_with_benchmark(
            self,
            benchmark_file='./benchmark/prices.csv',
            out_file='price_comparison_to_baseline.png',
            show=False):
        """Creates a visualization comparing prices with their benchmark values.
        A wrapper for the visualize.visualize_price_comparison_with_benchmark()
        function.

        Parameters
        ----------
        benchmark_file : str, optional
            The location of the CSV file containing benchmark values for each
            supply node, by default tests/data/benchmark_prices.csv.
        out_file : str, optional
            Filepath to the location where the visualization will be saved, by
            default price_comparison_to_baseline.png.
        show : bool, optional
            If True, displays the generated figure, by default False
        """
        visualize.visualize_price_comparison_with_benchmark(
            self, benchmark_file=benchmark_file, out_file=out_file, show=show)

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
