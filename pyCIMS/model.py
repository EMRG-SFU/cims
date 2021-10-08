import copy
import math
import warnings
import networkx as nx
import pandas as pd
import re
import time
import pickle
import operator

from . import graph_utils
from . import utils
from . import lcc_calculation
from . import stock_allocation

from .quantities import ProvidedQuantity, RequestedQuantity
from .emissions import Emissions, EmissionRates
from .utils import create_value_dict
from .quantity_aggregation import find_children, get_quantities_to_record


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
        self.equilibrium_fuels = []
        self.GHGs = []
        self.emission_types = []
        self.gwp = {}
        self.years = model_reader.get_years()
        self.base_year = int(self.years[0])

        self.prices = {}

        self.build_graph()
        self.dcc_classes = self._dcc_classes()

        self.show_run_warnings = True

        self.model_description_file = model_reader.infile
        self.change_history = pd.DataFrame(
            columns=['base_model_description', 'node', 'year', 'technology', 'parameter',
                     'sub_parameter', 'old_value', 'new_value'])

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
                    dccc = self.graph.nodes[node][base_year]['technologies'][tech]['Capital cost_declining_Class']['context']
                    if dccc is not None:
                        if dccc in dcc_classes:
                            dcc_classes[dccc].append((node, tech))
                        else:
                            dcc_classes[dccc] = [(node, tech)]

        return dcc_classes

    def run(self, equilibrium_threshold=0.05, min_iterations=1, max_iterations=10, show_warnings=True):
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

                # Check for an Equilibrium
                # ************************
                # Find the previous prices
                prev_prices = self.prices

                # Go get all the new prices
                new_prices = {fuel: self.get_param('Life Cycle Cost', fuel, year)
                              for fuel in self.equilibrium_fuels}

                equilibrium = min_iterations <= iteration and \
                              (int(year) == self.base_year or
                               self.check_equilibrium(prev_prices,
                                                      new_prices,
                                                      equilibrium_threshold))

                self.prices = new_prices

                # Next Iteration
                # **************
                iteration += 1

            # Once we've reached an equilibrium, calculate the quantities requested by each node.
            graph_utils.bottom_up_traversal(self.graph,
                                            self.calc_requested_quantities,
                                            year)

            graph_utils.bottom_up_traversal(self.graph,
                                            self.aggregate_emissions,
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
                    price_multipliers = copy.deepcopy(self.graph.nodes[parent][year]['Price Multiplier'])
                    parent_price_multipliers.update(price_multipliers)

            # Grab the price multipliers from the current node (if they exist) and replace the parent price multipliers
            node_price_multipliers = copy.deepcopy(parent_price_multipliers)
            if 'Price Multiplier' in graph.nodes[node][year]:
                price_multipliers = self.get_param('Price Multiplier', node, year, return_keys=True)
                node_price_multipliers.update(price_multipliers)

            # Set Price Multiplier of node in the graph
            graph.nodes[node][year]['Price Multiplier'] = node_price_multipliers

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
                if "Life Cycle Cost" in graph.nodes[node][year]:
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
                elif 'cost_curve_function' in graph.nodes[node]:
                    if int(year) == self.base_year:
                        lcc = lcc_calculation.calc_cost_curve_lcc(self, node, year)
                        service_name = node.split('.')[-1]
                        graph.nodes[node][year]["Life Cycle Cost"] = {
                            service_name: utils.create_value_dict(lcc,
                                                                  param_source='initialization')}
                    else:
                        last_year = str(int(year) - self.step)
                        service_name = node.split('.')[-1]
                        last_year_value = self.get_param('Life Cycle Cost', node, last_year)
                        graph.nodes[node][year]["Life Cycle Cost"] = {
                            service_name: utils.create_value_dict(last_year_value,
                                                                  param_source='cost curve function')}

                else:
                    # Life Cycle Cost needs to be calculated from children
                    calc_lcc_from_children()
                    lcc_dict = graph.nodes[node][year]['Life Cycle Cost']
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
                    if 'Emissions' in tech_data:
                        emission_data = tech_data['Emissions']
                        for ghg in emission_data:
                            for emission_type in emission_data[ghg]:
                                try:
                                    emission_data[ghg][emission_type]['year_value'] *= gwp[ghg]
                                except KeyError:
                                    continue

            # Emissions from a node
            elif 'Emissions' in graph.nodes[node][year]:
                emission_data = graph.nodes[node][year]['Emissions']
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
            if 'Load Factor' not in graph.nodes[node][year]:
                # Check if a load factor was defined at the node's structural parent (its first
                # parent). If so, use this load factor for the node.
                parents = list(graph.predecessors(node))
                if len(parents) > 0:
                    parent = parents[0]
                    if 'Load Factor' in graph.nodes[parent][year]:
                        val = graph.nodes[parent][year]['Load Factor']['year_value']
                        units = graph.nodes[parent][year]['Load Factor']['unit']
                        graph.nodes[node][year]['Load Factor'] = utils.create_value_dict(val,
                                                                                         unit=units,
                                                                                         param_source='inheritance')

            if 'Load Factor' in graph.nodes[node][year]:
                # Ensure this load factor is recorded at each of the technologies within the node.
                if 'technologies' in graph.nodes[node][year]:
                    tech_data = graph.nodes[node][year]['technologies']
                    for tech in tech_data:
                        if 'Load Factor' not in tech_data[tech]:
                            val = graph.nodes[node][year]['Load Factor']['year_value']
                            units = graph.nodes[node][year]['Load Factor']['unit']
                            tech_data[tech]['Load Factor'] = utils.create_value_dict(val,
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
            parent_tax = {}
            if len(parents) > 0:
                parent = parents[0]
                if 'Tax' in graph.nodes[parent][year]:
                    parent_dict = self.get_param('Tax', parent, year)
                    parent_tax.update(parent_dict)

            # Store away tax at current node to overwrite parent tax later
            node_tax = {}
            if 'Tax' in graph.nodes[node][year]:
                node_tax = graph.nodes[node][year]['Tax']

            # Make final dict where we prioritize keeping node_tax and only unique parent taxes
            final_tax = copy.deepcopy(node_tax)
            for ghg in parent_tax:
                if ghg not in final_tax:
                    final_tax[ghg] = {}
                for emission_type in parent_tax[ghg]:
                    if emission_type not in final_tax[ghg]:
                        final_tax[ghg][emission_type] = parent_tax[ghg][emission_type]

            graph.nodes[node][year]['Tax'] = final_tax

        graph_utils.top_down_traversal(graph,
                                       init_node_price_multipliers,
                                       year)
        graph_utils.top_down_traversal(graph,
                                       init_convert_to_CO2e,
                                       year,
                                       self.gwp)
        graph_utils.top_down_traversal(graph,
                                       init_load_factor,
                                       year)
        graph_utils.top_down_traversal(graph,
                                       init_tax_emissions,
                                       year)
        graph_utils.bottom_up_traversal(graph,
                                        init_fuel_lcc,
                                        year)

    def iteration_initialization(self, year):
        # Reset the provided_quantities at each node
        for n in self.graph.nodes():
            self.graph.nodes[n][year]['provided_quantities'] = create_value_dict(ProvidedQuantity(),
                                                                                 param_source='initialization')

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
        comp_type = self.get_param('competition type', node)

        # Market acts the same as tech compete
        if comp_type == 'market':
            comp_type = 'tech compete'

        if comp_type in ['tech compete', 'node tech compete']:
            stock_allocation.all_tech_compete_allocation(self, node, year)
        else:
            stock_allocation.general_allocation(self, node, year)

    def get_tech_parameter_default(self, parameter):
        return self.technology_defaults[parameter]

    def get_node_parameter_default(self, parameter, competition_type):
        return self.node_defaults[competition_type][parameter]

    def get_param(self, param, node, year=None, tech=None, sub_param=None, return_source=False, check_exist=False, return_keys=False):
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
        check_exist : bool, default=False
            Whether or not to check that the parameter exists as is given the context (without calculation, 
            inheritance, or checking past years)

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
                                             retrieve_only=True, check_exist=check_exist)

        else:
            param_val = utils.get_node_param(param, self, node, year, sub_param=sub_param,
                                             return_source=return_source,
                                             retrieve_only=True, check_exist=check_exist, return_keys=return_keys)

        return param_val


    def get_param_test(self, param, node, year, context=None, sub_param=None, tech=False,
                       return_source=False, check_exist=False):
        if tech:
            param_val = utils.get_tech_param_test(param, self, node, year, tech=context, sub_param=sub_param,
                                             return_source=return_source,
                                             retrieve_only=True, check_exist=check_exist)

        else:
            param_val = utils.get_node_param_test(param, self, node, year, context=context, sub_param=sub_param,
                                             return_source=return_source,
                                             retrieve_only=True, check_exist=check_exist)

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

        if self.get_param("competition type", node) in ['root', 'region']:
            structural_children = find_children(graph, node, structural=True)
            for child in structural_children:
                # Find quantities provided to the node via its structural children
                child_requested_quant = self.get_param("requested_quantities",
                                                       child, year).get_total_quantities_requested()
                for child_rq_node, child_rq_amount in child_requested_quant.items():
                    # Record requested quantities
                    requested_quantity.record_requested_quantity(child_rq_node,
                                                                 child,
                                                                 child_rq_amount)

        elif 'technologies' in graph.nodes[node][year]:
            for tech in graph.nodes[node][year]['technologies']:
                tech_requested_quantity = RequestedQuantity()
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
                self.graph.nodes[node][year]['technologies'][tech]["requested_quantities"] = \
                    tech_requested_quantity

        else:
            req_prov_children = find_children(graph, node, year, request_provide=True)
            for child in req_prov_children:
                quantities_to_record = get_quantities_to_record(self, child, node, year)

                # Record requested quantities
                for providing_node, child, attributable_amount in quantities_to_record:
                    requested_quantity.record_requested_quantity(providing_node,
                                                                 child,
                                                                 attributable_amount)

        self.graph.nodes[node][year]["requested_quantities"] = \
            utils.create_value_dict(requested_quantity, param_source='calculation')

    def aggregate_emissions(self, graph, node, year):
        net_emissions = Emissions()
        cap_emissions = Emissions()
        bio_emissions = Emissions()
        total_emissions_cost = 0

        # get emissions that originate at the node
        if 'net_emission_rates' in self.graph.nodes[node][year]:
            net_emission_rates = self.get_param('net_emission_rates', node, year)
            cap_emission_rates = self.get_param('captured_emission_rates', node, year)
        else:
            net_emission_rates = EmissionRates()
            cap_emission_rates = EmissionRates()

        if 'bio_emission_rates' in self.graph.nodes[node][year]:
            bio_emission_rates = self.get_param('bio_emission_rates', node, year)
        else:
            bio_emission_rates = EmissionRates()

        if 'Emissions cost' in self.graph.nodes[node][year]:
            emissions_cost = self.get_param('Emissions cost', node, year)
        else:
            emissions_cost = 0

        total_units = self.get_param('provided_quantities', node, year).get_total_quantity()
        net_emissions += Emissions(net_emission_rates.multiply_rates(total_units))
        cap_emissions += Emissions(cap_emission_rates.multiply_rates(total_units))
        bio_emissions += Emissions(bio_emission_rates.multiply_rates(total_units))
        total_emissions_cost += emissions_cost * total_units

        # Get Other Emissions
        if 'technologies' in graph.nodes[node][year]:
            # Get emissions from technologies
            for tech in graph.nodes[node][year]['technologies']:
                tech_net_emissions = Emissions()
                tech_cap_emissions = Emissions()
                tech_bio_emissions = Emissions()
                tech_total_emissions_cost = 0

                # Get emissions originating at the technology
                tech_market_share = self.get_param('total_market_share', node, year, tech)
                tech_units = tech_market_share * total_units
                tech_net_emission_rates = self.get_param("net_emission_rates", node, year, tech)
                tech_cap_emission_rates = self.get_param("captured_emission_rates", node, year, tech)
                tech_bio_emission_rates = self.get_param("bio_emission_rates", node, year, tech)
                tech_emissions_cost = self.get_param("Emissions cost", node, year, tech)
                if tech_net_emission_rates is not None:
                    tech_net_emissions = Emissions(tech_net_emission_rates.multiply_rates(tech_units))
                    tech_cap_emissions = Emissions(tech_cap_emission_rates.multiply_rates(tech_units))
                    net_emissions += tech_net_emissions
                    cap_emissions += tech_cap_emissions

                if tech_bio_emission_rates is not None:
                    tech_bio_emissions = Emissions(tech_bio_emission_rates.multiply_rates(tech_units))
                    bio_emissions += tech_bio_emissions

                if tech_emissions_cost is not None:
                    tech_total_emissions_cost = tech_emissions_cost * tech_units
                    total_emissions_cost += tech_total_emissions_cost

                # Get emissions originating from the technology's request/provide children
                req_prov_children = find_children(graph, node, year, request_provide=True)

                for child in req_prov_children:
                    if child not in self.fuels:
                        child_quantities = self.get_param('provided_quantities', child, year)
                        child_total_quantity = child_quantities.get_total_quantity()
                        if child_total_quantity != 0:
                            child_net_emissions = self.get_param("net_emissions", child, year)
                            child_cap_emissions = self.get_param("captured_emissions", child, year)
                            child_bio_emissions = self.get_param("bio_emissions", child, year)
                            child_total_emissions_cost = self.get_param("total_emissions_cost", child, year)

                            quant_provided_to_tech = child_quantities.get_quantity_provided_to_tech(node, tech)
                            if quant_provided_to_tech > 0:
                                proportion = quant_provided_to_tech / child_total_quantity

                                proportional_child_net_emissions = child_net_emissions * proportion
                                proportional_child_cap_emissions = child_cap_emissions * proportion
                                proportional_child_total_emissions_cost = child_total_emissions_cost * proportion

                                net_emissions += proportional_child_net_emissions
                                cap_emissions += proportional_child_cap_emissions
                                total_emissions_cost += proportional_child_total_emissions_cost

                                tech_net_emissions += proportional_child_net_emissions
                                tech_cap_emissions += proportional_child_cap_emissions
                                tech_total_emissions_cost += proportional_child_total_emissions_cost

                                if child_bio_emissions is not None:
                                    proportional_child_bio_emissions = child_bio_emissions * proportion
                                    bio_emissions += proportional_child_bio_emissions
                                    tech_bio_emissions += proportional_child_bio_emissions

                # Save tech-specific emissions to the model
                self.graph.nodes[node][year]['technologies'][tech]['net_emissions'] = \
                    utils.create_value_dict(tech_net_emissions, param_source='calculation')

                self.graph.nodes[node][year]['technologies'][tech]['captured_emissions'] = \
                    utils.create_value_dict(tech_cap_emissions, param_source='calculation')

                self.graph.nodes[node][year]['technologies'][tech]['bio_emissions'] = \
                    utils.create_value_dict(tech_bio_emissions, param_source='calculation')

                self.graph.nodes[node][year]['technologies'][tech]['total_emissions_cost'] = \
                    utils.create_value_dict(tech_total_emissions_cost, param_source='calculation')

        elif self.get_param("competition type", node) in ['root', 'region']:
            # Retrieve emissions from the node's structural children
            structural_children = find_children(graph, node, structural=True)

            # For each structural child, add its emissions to the region/root
            for child in structural_children:
                net_emissions += self.get_param("net_emissions", child, year)
                cap_emissions += self.get_param("captured_emissions", child, year)
                bio_emissions += self.get_param("bio_emissions", child, year)
                total_emissions_cost += self.get_param('total_emissions_cost', child, year)
        else:
            # Get emissions from req/provide children
            req_prov_children = find_children(graph, node, year, request_provide=True)

            for child in req_prov_children:
                if child not in self.fuels:
                    child_quantities = self.get_param('provided_quantities', child, year)
                    child_total_quantity = child_quantities.get_total_quantity()
                    if child_total_quantity != 0:
                        child_net_emissions = self.get_param("net_emissions", child, year)
                        child_cap_emissions = self.get_param("captured_emissions", child, year)
                        child_bio_emissions = self.get_param("bio_emissions", child, year)
                        child_total_emissions_cost = self.get_param("total_emissions_cost", child, year)

                        quant_provided_to_node = child_quantities.get_quantity_provided_to_node(node)
                        if quant_provided_to_node > 0:
                            proportion = quant_provided_to_node / child_total_quantity

                            proportional_child_net_emissions = child_net_emissions * proportion
                            proportional_child_cap_emissions = child_cap_emissions * proportion
                            proportional_child_total_emissions_cost = child_total_emissions_cost * proportion

                            net_emissions += proportional_child_net_emissions
                            cap_emissions += proportional_child_cap_emissions
                            total_emissions_cost += proportional_child_total_emissions_cost

                            if child_bio_emissions is not None:
                                proportional_child_bio_emissions = child_bio_emissions * proportion
                                bio_emissions += proportional_child_bio_emissions

        # Save the emissions to the node's data
        self.graph.nodes[node][year]['net_emissions'] = \
            utils.create_value_dict(net_emissions, param_source='calculation')

        self.graph.nodes[node][year]['captured_emissions'] = \
            utils.create_value_dict(cap_emissions, param_source='calculation')

        self.graph.nodes[node][year]['bio_emissions'] = \
            utils.create_value_dict(bio_emissions, param_source='calculation')

        self.graph.nodes[node][year]['total_emissions_cost'] = \
            utils.create_value_dict(total_emissions_cost, param_source='calculation')

    def set_param(self, val, param, node, year=None, tech=None, sub_param=None, save=True):
        """
        Sets a parameter's value, given a specific context (node, year, technology, and
        sub-parameter). This is intended for when you are using this function outside of model.run to
        make single changes to the model dscription

        Parameters
        ----------
        val : any or list of any
            The new value(s) to be set at the specified `param` at `node`, given the context provided by 
            `year`, `tech` and `sub_param`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for parameters specified at
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
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        def set_node_param_script(new_val, param, model, node, year, sub_param=None, save=True):
            """
            Queries a model to set a parameter value at a given node, given a specified context
            (year & sub-parameter).

            Parameters
            ----------
            new_val : any
                The new value to be set at the specified `param` at `node`, given the context provided by
                `year` and `sub_param`.
            param : str
                The name of the parameter whose value is being set.
            model : pyCIMS.Model
                The model containing the parameter value of interest.
            node : str
                The name of the node (branch format) whose parameter you are interested in set.
            year : str
                The year which you are interested in. `year` must be provided for all parameters stored at
                the technology level, even if the parameter doesn't change year to year.
            sub_param : str, optional
                This is a rarely used parameter for specifying a nested key. Most commonly used when
                `get_param()` would otherwise return a dictionary where a nested value contains the
                parameter value of interest. In this case, the key corresponding to that value can be
                provided as a `sub_param`
            save : bool, optional
                This specifies whether the change should be saved in the change_log csv where True means
                the change will be saved and False means it will not be saved
            """
            # Set Parameter from Description
            # ******************************
            # If the parameter's value is in the model description for that node & year (if the year has
            # been defined), use it.
            if year:
                data = model.graph.nodes[node][year]
            else:
                data = model.graph.nodes[node]
            if param in data:
                val = data[param]
                # If the value is a dictionary, use its nested result
                if isinstance(val, dict):
                    if sub_param:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[sub_param], dict) and 'year_value' in val[sub_param]:
                            prev_val = val[sub_param]['year_value']
                            val[sub_param]['year_value'] = new_val
                        else:
                            prev_val = val[sub_param]
                            val[sub_param] = new_val
                    elif 'year_value' in val:
                        prev_val = val['year_value']
                        val['year_value'] = new_val
                    elif None in val:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[None], dict) and 'year_value' in val[None]:
                            prev_val = val[None]['year_value']
                            val[None]['year_value'] = new_val
                        else:
                            prev_val = val[None]
                            val[None] = new_val
                    elif len(val.keys()) == 1:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if 'year_value' in val[list(val.keys())[0]]:
                            prev_val = val[list(val.keys())[0]]['year_value']
                            val[list(val.keys())[0]]['year_value'] = new_val
                        else:
                            prev_val = val[list(val.keys())[0]]
                            val[list(val.keys())[0]] = new_val
                else:
                    prev_val = data[param]
                    data[param] = new_val

                # Save Change
                # ******************************
                # Append the change made to model.change_history DataFrame if save is set to True
                if save:
                    filename = model.model_description_file.split('/')[-1].split('.')[0]
                    change_log = {'base_model_description': filename, 'node': node, 'year': year, 'technology': None,
                                  'parameter': param, 'sub_parameter': sub_param, 'old_value': prev_val,
                                  'new_value': new_val}
                    model.change_history = model.change_history.append(pd.Series(change_log), ignore_index=True)
            else:
                print('No param ' + str(param) + ' at node ' + str(node) + ' for year ' + str(
                    year) + '. No new value was set for this.')

        def set_tech_param_script(new_val, param, model, node, year, tech, sub_param=None, save=True):
            """
            Queries a model to set a parameter value at a given node & technology, given a specified
            context (year & sub_param).

            Parameters
            ----------
            new_val : any
                The new value to be set at the specified `param` at `node`, given the context provided by
                `year`, `tech` and `sub_param`.
            param : str
                The name of the parameter whose value is being set.
            model : pyCIMS.Model
                The model containing the parameter value of interest.
            node : str
                The name of the node (branch format) whose parameter you are interested in set.
            year : str
                The year which you are interested in. `year` must be provided for all parameters stored at
                the technology level, even if the parameter doesn't change year to year.
            tech : str
                The name of the technology you are interested in.
            sub_param : str, optional
                This is a rarely used parameter for specifying a nested key. Most commonly used when
                `get_param()` would otherwise return a dictionary where a nested value contains the
                parameter value of interest. In this case, the key corresponding to that value can be
                provided as a `sub_param`
            save : bool, optional
                This specifies whether the change should be saved in the change_log csv where True means
                the change will be saved and False means it will not be saved
            """
            # Set Parameter from Description
            # ******************************
            # If the parameter's value is in the model description for that node, year, & technology, use it
            data = model.graph.nodes[node][year]['technologies'][tech]
            if param in data:
                val = data[param]
                # If the value is a dictionary, use its nested result
                if isinstance(val, dict):
                    if sub_param:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[sub_param], dict) and ('year_value' in val[sub_param]):
                            prev_val = val[sub_param]['year_value']
                            val[sub_param]['year_value'] = new_val
                        else:
                            prev_val = val[sub_param]
                            val[sub_param] = new_val
                    elif None in val:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[None], dict) and ('year_value' in val[None]):
                            prev_val = val[None]['year_value']
                            val[None]['year_value'] = new_val
                        else:
                            prev_val = val[None]
                            val[None] = new_val
                    else:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if 'year_value' in val:
                            prev_val = data[param]['year_value']
                            data[param]['year_value'] = new_val
                else:
                    prev_val = data[param]
                    data[param] = new_val

                # Save Change
                # ******************************
                # Append the change made to model.change_history DataFrame if save is set to True
                if save:
                    filename = model.model_description_file.split('/')[-1].split('.')[0]
                    change_log = {'base_model_description': filename, 'node': node, 'year': year, 'technology': tech,
                                  'parameter': param, 'sub_parameter': sub_param, 'old_value': prev_val,
                                  'new_value': new_val}
                    model.change_history = model.change_history.append(pd.Series(change_log), ignore_index=True)
            else:
                print('No param ' + str(param) + ' at node ' + str(node) + ' for year ' + str(
                    year) + '. No new value was set for this.')

        # Checks whether year or val is a list. If either of them is a list, the other must also be a list
        # of the same length
        if isinstance(val, list) or isinstance(year, list):
            if not isinstance(val, list):
                print('Values must be entered as a list.')
                return
            elif not isinstance(year, list):
                print('Years must be entered as a list.')
                return
            elif len(val) != len(year):
                print('The number of values does not match the number of years. No changes were made.')
                return
        else:
            # changing years and vals to lists
            year = [year]
            val = [val]
        for i in range(len(year)):
            try:
                self.get_param(param, node, year[i], tech, sub_param, check_exist=True)
            except:
                print(f"Unable to access parameter at "
                      f"get_param({param}, {node}, {year}, {tech}, {sub_param}). \n"
                      f"Corresponding value was not set to {val[i]}.")
                continue
            if tech:
                set_tech_param_script(val[i], param, self, node, year[i], tech, sub_param, save)

            else:
                set_node_param_script(val[i], param, self, node, year[i], sub_param, save)

    def set_param_internal(self, val, param, node, year=None, tech=None, sub_param=None, save=True):
        """
        Sets a parameter's value, given a specific context (node, year, technology, and
        sub-parameter). This is used from within the model.run function and is not intended to make
        changes to the model description externally (see `set_param`).

        Parameters
        ----------
        val : dict
            The new value(s) to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech` and `sub_param`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str or list, optional
            The year(s) which you are interested in. `year` is not required for parameters specified at
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
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        # Checks whether year or val is a list. If either of them is a list, the other must also be a list
        # of the same length
        if isinstance(val, list) or isinstance(year, list):
            if not isinstance(val, list):
                print('Values must be entered as a list.')
                return
            elif not isinstance(year, list):
                print('Years must be entered as a list.')
                return
            elif len(val) != len(year):
                print('The number of values does not match the number of years. No changes were made.')
                return
        else:
            # changing years and vals to lists
            year = [year]
            val = [val]
        for i in range(len(year)):

            tech_data = self.graph.nodes[node][year[i]]["technologies"][tech]
            if param in tech_data:
                if tech:
                    utils.set_tech_param(val[i], param, self, node, year[i], tech, sub_param)

                else:
                    utils.set_node_param(val[i], param, self, node, year[i], sub_param)
            else:
                self.graph.nodes[node][year[i]]["technologies"][tech].update({str(param): val[i]})

    def set_param_wildcard(self, val, param, node_regex, year, tech=None, sub_param=None, save=True):
        """
        Sets a parameter's value, for all context (node, year, technology, and
        sub-parameter) that satisfy/matches the node_regex pattern

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by 
            `year`, `tech` and `sub_param`.
        param : str
            The name of the parameter whose value is being set.
        node_regex : str
            The regex pattern of the node (branch format) whose parameter you are interested in matching.
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
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """
        for node in self.graph.nodes:
            if re.search(node_regex, node) != None:
                self.set_param(val, param, node, year, tech, sub_param, save)

    def set_param_file(self, filepath):
        """
        Sets parameters' values, for all context (node, year, technology, and
        sub-parameter) from the provided CSV file. See Data_Changes_Tutorial_by_CSV.ipynb for detailed
        description of expected CSV file columns and values.

        Parameters
        ----------
        filepath : str
            This is the path to the CSV file containing all context and value change information
        """

        if not filepath.endswith('.csv'):
            print('filepath must be in csv format')
            return

        df = pd.read_csv(filepath, delimiter=',')
        df = df.fillna('None')

        ops = {
            '>': operator.gt,
            '>=': operator.ge,
            '==': operator.eq,
            '<': operator.lt,
            '<=': operator.le
        }

        for index, row in df.iterrows():
            # *********
            # Set necessary variables from dataframe row
            # *********
            node = row['node'] if row['node'] != 'None' else None
            node_regex = row['node_regex'] if row['node_regex'] != "None" else None
            param = row['param'] if row['param'] != 'None' else None
            tech = row['tech'] if row['tech'] != 'None' else None
            sub_param = row['sub_param'] if row['sub_param'] != 'None' else None
            year_operator = row['year_operator'] if row['year_operator'] != 'None' else None
            year = row['year'] if row['year'] != 'None' else None
            val_operator = row['val_operator'] if row['val_operator'] != 'None' else None
            val = row['val'] if row['val'] != 'None' else None
            search_param = row['search_param'] if row['search_param'] != 'None' else None
            search_operator = row['search_operator'] if row['search_operator'] != 'None' else None
            search_pattern = row['search_pattern'] if row['search_pattern'] != 'None' else None
            create_missing = row['create_if_missing'] if row['create_if_missing'] != 'None' else None

            # *********
            # Changing years and vals to lists
            # *********
            if year:
                year_int = int(year)
                years = [x for x in self.years if ops[year_operator](int(x), year_int)]
                vals = [val] * len(years)
            else:
                years = [year]
                vals = [val]

            # *********
            # Intial checks on the data
            # *********
            if node == None:
                if node_regex == None:
                    print(f"Row {index}: : neither node or node_regex values were indicated. "
                          f"Skipping this row.")
                    continue
            elif node == '.*':
                if search_param == None or search_operator == None or search_pattern == None:
                    print(f"Row {index}: since node = '.*', search_param, search_operator, and "
                          f"search_pattern must not be empty. Skipping this row.")
                    continue
            else:
                if node_regex:
                    print(f"Row index: both node and node_regex values were indicated. Please "
                          f"specify only one. Skipping this row.")
                    continue
            if year_operator not in list(ops.keys()):
                print(f"Row {index}: year_operator value not one of >, >=, <, <=, ==. Skipping this"
                      f"row.")
                continue
            if val_operator not in ['>=', '<=', '==']:
                print(f"Row {index}: val_operator value not one of >=, <=, ==. Skipping this row.")
                continue
            if search_operator not in [None, '==']:
                print(f"Row {index}: search_operator value must be either empty or ==. Skipping "
                      f"this row.")
                continue
            if create_missing == None:
                print(f"Row {index}: create_if_missing is empty. This value must be either True or"
                      f"False. Skipping this row.")
                continue

            # *********
            # Check the node type ('.*', None, or otherwise) and search through corresponding nodes if necessary
            # *********
            if node == '.*':
                # check if node satisfies search_param, search_operator, search_pattern conditions
                for node_tmp in self.graph.nodes:
                    if self.get_param(search_param, node_tmp).lower() == search_pattern.lower():
                        for idx, year in enumerate(years):
                            val_tmp = vals[idx]
                            self.set_param_search(val_tmp, param, node_tmp, year, tech, sub_param, val_operator,
                                                  create_missing, index)
            elif node == None:
                # check if node satisfies node_regex conditions
                for node_tmp in self.graph.nodes:
                    if re.search(node_regex, node_tmp) != None:
                        for idx, year in enumerate(years):
                            val_tmp = vals[idx]
                            self.set_param_search(val_tmp, param, node_tmp, year, tech, sub_param, val_operator,
                                                  create_missing, index)
            else:
                # node is exactly specified so use as is
                for idx, year in enumerate(years):
                    val_tmp = vals[idx]
                    self.set_param_search(val_tmp, param, node, year, tech, sub_param, val_operator, create_missing,
                                          index)

    def set_param_search(self, val, param, node, year=None, tech=None, sub_param=None, val_operator='==',
                         create_missing=False, row_index=None):
        """
        Sets parameter values, for all context (node, year, technology, and
        sub-parameter), searching through all tech and sub_param keys if necessary.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by 
            `year`, `tech` and `sub_param`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            "competition type" can be retreived without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
            specified node, param, and year.
        sub_param : str, optional
            This is a rarely used parameter for specifying a nested key. Most commonly used when
            `get_param()` would otherwise return a dictionary where a nested value contains the
            parameter value of interest. In this case, the key corresponding to that value can be
            provided as a `sub_param`. If sub_param is `.*`, all possible sub_param keys will be searched at the
            specified node, param, tech, and year.
        create_missing : bool, optional
            Will create a new parameter in the model if it is missing. Defaults to False.
        val_operator : str, optional
            This specifies how the value should be set. The possible values are '>=', '<=' and '=='.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the row number in error messages.
        """

        def get_val_operated(val, param, node, year, tech, sub_param, val_operator, row_index, create_missing):
            try:
                prev_val = self.get_param(param=param, node=node, year=year, tech=tech, sub_param=sub_param,
                                          check_exist=True)
                if val_operator == '>=':
                    val = max(val, prev_val)
                elif val_operator == '<=':
                    val = min(val, prev_val)
            except Exception as e:
                if create_missing:
                    print(f"Row {row_index + 1}: Creating parameter at ({param}, {node}, {year}, "
                          f"{tech}, {sub_param}).")
                    tmp = self.create_param(val=val, param=param, node=node, year=year, tech=tech, sub_param=sub_param,
                                            row_index=row_index)
                    if not tmp:
                        return None
                else:
                    print(f"Row {row_index + 1}: Unable to access parameter at get_param({param}, "
                          f"{node}, {year}, {tech}, {sub_param}). Corresponding value was not set"
                          f"to {val}.")
                    return None
            return val

        if tech == '.*':
            try:
                # search through all technologies in node
                techs = list(self.graph.nodes[node][year]['technologies'].keys())
            except:
                return
            for tech_tmp in techs:
                if sub_param == '.*':
                    try:
                        # search through all sub_parameters in node given tech
                        sub_params = list(self.get_param(param=param, node=node, year=year, tech=tech_tmp).keys())
                    except:
                        continue
                    for sub_param_tmp in sub_params:
                        val_tmp = get_val_operated(val, param, node, year, tech_tmp, sub_param_tmp, val_operator,
                                                   row_index, create_missing)
                        if val_tmp:
                            self.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                           sub_param=sub_param_tmp)
                # use sub_param as is if it is not .*
                else:
                    val_tmp = get_val_operated(val, param, node, year, tech_tmp, sub_param, val_operator, row_index,
                                               create_missing)
                    if val_tmp:
                        self.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                       sub_param=sub_param)
        else:
            if sub_param == '.*':
                try:
                    # search through all sub_parameters in node given tech
                    sub_params = list(self.get_param(param=param, node=node, year=year, tech=tech).keys())
                except:
                    return
                for sub_param_tmp in sub_params:
                    val_tmp = get_val_operated(val, param, node, year, tech, sub_param_tmp, val_operator, row_index,
                                               create_missing)
                    if val_tmp:
                        self.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                       sub_param=sub_param_tmp)
            # use sub_param as is if it is not .*
            else:
                val_tmp = get_val_operated(val, param, node, year, tech, sub_param, val_operator, row_index,
                                           create_missing)
                if val_tmp:
                    self.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech, sub_param=sub_param)

    def create_param(self, val, param, node, year=None, tech=None, sub_param=None, row_index=None):
        """
        Creates parameter in graph, for given context (node, year, technology, and sub-parameter),
        and sets the value to val. Returns True if param was created successfully and False otherwise.

        Parameters
        ----------
        val : any
            The new value to be set at the specified `param` at `node`, given the context provided by 
            `year`, `tech` and `sub_param`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in matching.
        year : str, optional
            The year which you are interested in. `year` is not required for parameters specified at
            the node level and which by definition cannot change year to year. For example,
            "competition type" can be retreived without specifying a year.
        tech : str, optional
            The name of the technology you are interested in. `tech` is not required for parameters
            that are specified at the node level. `tech` is required to get any parameter that is
            stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
            specified node, param, and year.
        sub_param : str, optional
            This is a rarely used parameter for specifying a nested key. Most commonly used when
            `get_param()` would otherwise return a dictionary where a nested value contains the
            parameter value of interest. In this case, the key corresponding to that value can be
            provided as a `sub_param`. If sub_param is `.*`, all possible sub_param keys will be searched at the
            specified node, param, tech, and year.
        row_index : int, optional
            The index of the current row of the CSV. This is used to print the row number in error messages.

        Returns
        -------
        Boolean
        """
        # Print error message and return False if node not found
        if node not in self.graph.nodes:
            print("Row " + str(row_index + 1) + ': Unable to access node ' + str(
                node) + '. Corresponding value was not set to ' + str(val) + ".")
            return False

        if year:
            if year not in self.graph.nodes[node]:
                self.graph.nodes[node][year] = {}
            data = self.graph.nodes[node][year]
        else:
            data = self.graph.nodes[node]

        val_dict = create_value_dict(val, param_source='model')

        # *********
        # If there is a tech specified, check if it exists and create context (tech, param, sub-param) accordingly
        # *********
        if tech:
            # add technology if it does not exist
            if tech not in data:
                if sub_param:
                    sub_param_dict = {sub_param: val_dict}
                    param_dict = {param: sub_param_dict}
                else:
                    param_dict = {param: val_dict}
                data['technologies'][tech] = param_dict
            # add param if it does not exist
            elif param not in data['technologies'][tech]:
                if sub_param:
                    sub_param_dict = {sub_param: val_dict}
                    data['technologies'][tech][param] = sub_param_dict
                else:
                    data['technologies'][tech][param] = val_dict
            # add sub-param if it does not exist
            elif sub_param not in data['technologies'][tech][param]:
                data['technologies'][tech][param][sub_param] = val_dict

        # *********
        # Check if param exists and create context (param, sub-param) accordingly
        # *********
        elif param not in data:
            if sub_param:
                sub_param_dict = {sub_param: val_dict}
                data[param] = sub_param_dict
            else:
                data[param] = val_dict

        # *********
        # Check if sub-param exists and create context (param, sub-param) accordingly
        # *********
        elif sub_param not in data[param]:
            data[param][sub_param] = val_dict
        return True

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
