import networkx as nx
import copy as copy
import re
import random


class Model:
    def __init__(self, reader):
        """
        Construct the Model from reader.
        :param reader: pyCIMS.Reader.
        """
        self.graph = nx.DiGraph()
        self.node_dfs, self.tech_dfs = reader.get_model_description()
        self.fuels = ['Electricity', 'Natural Gas', 'Solar', 'Wind']
        self.years = ['2000', '2005', '2010']

    def build_graph(self):
        def is_year(cn):
            """Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string"""
            # unit test: assert is_year, 1900
            re_year = re.compile(r'^[0-9]{4}$')
            return bool(re_year.match(str(cn)))

        def make_nodes(type_col):
            def add_node_data(current_node):
                # Copy the current node dataframe
                current_node_df = copy.deepcopy(self.node_dfs[current_node])

                # Add a node to the graph
                self.graph.add_node(current_node)

                # Store whether the node type (supply, demand, or standard)
                typ = list(current_node_df[type_col])[0]
                if typ:
                    self.graph.nodes[current_node]['type'] = typ.lower()
                else:
                    self.graph.nodes[current_node]['type'] = 'standard'
                # Drop Demand column
                current_node_df = current_node_df.drop(type_col, axis=1)

                # Store node's competition type. (If there is one)
                comp_list = list(current_node_df[current_node_df['Parameter'] == 'Competition type']['Value'])
                if len(set(comp_list)) == 1:
                    comp_type = comp_list[0]
                    self.graph.nodes[current_node]['competition_type'] = comp_type.lower()
                elif len(set(comp_list)) > 1:
                    print("TOO MANY COMPETITION TYPES")
                # Get rid of competition type row
                current_node_df = current_node_df[current_node_df['Parameter'] != 'Competition type']

                # For the remaining data, group by year.
                years = [c for c in current_node_df.columns if is_year(c)]          # Get Year Columns
                non_years = [c for c in current_node_df.columns if not is_year(c)]  # Get Non-Year Columns

                for y in years:
                    year_df = current_node_df[non_years + [y]]
                    year_dict = {}
                    for param, src, branch, unit, val, year_value in zip(*[year_df[c] for c in year_df.columns]):
                        if param in year_dict.keys():
                            pass
                        else:
                            year_dict[param] = {}

                        dct = {'source': src,
                               'branch': branch,
                               'unit': unit,
                               'year_value': year_value}

                        year_dict[param][val] = dct

                    # Add data to node
                    self.graph.nodes[current_node][y] = year_dict

            def add_tech_data(node_name, tech_name):
                t_df = copy.deepcopy(self.tech_dfs[node_name][tech_name])

                # Remove the row that indicates this is a service or technology.
                t_df = t_df[~t_df['Parameter'].isin(['Service', 'Technology'])]

                # Remove the Demand? column
                t_df = t_df.drop('Demand?', axis=1)

                # VERY SIMILAR to what we do for nodes. But not quite. Because we don't use the value column anymore
                # For the remaining rows, group data by year.
                years = [c for c in t_df.columns if is_year(c)]             # Get Year Columns
                non_years = [c for c in t_df.columns if not is_year(c)]     # Get Non-Year Columns

                for y in years:
                    year_df = t_df[non_years + [y]]
                    year_dict = {}

                    for parameter, source, branch, unit, value, year_value in zip(
                            *[year_df[c] for c in year_df.columns]):
                        dct = {'source': source,
                               'branch': branch,
                               'unit': unit,
                               'year_value': year_value}

                        if parameter in year_dict.keys():
                            if type(year_dict[parameter]) is list:
                                year_dict[parameter] = year_dict[parameter].append(dct)
                            else:
                                year_dict[parameter] = [year_dict[parameter], dct]
                        else:
                            year_dict[parameter] = dct

                    # Add technologies key if needed
                    if 'technologies' not in self.graph.nodes[node_name][y].keys():
                        self.graph.nodes[node_name][y]['technologies'] = {}

                    # Add the technology specific data for that year
                    self.graph.nodes[node_name][y]['technologies'][tech_name] = year_dict

            # Add each node and its associated data to the Graph
            for n in self.node_dfs.keys():
                add_node_data(n)

            for node in self.tech_dfs:
                # Add technologies key to node data
                for tech in self.tech_dfs[node]:
                    add_tech_data(node, tech)

        def make_edges():
            def add_edges(node_name, df):
                # Find edges based on Requester/Provider relationships
                # ----------------------------------------------------
                # Find all nodes node is requesting services from
                providers = df[df['Parameter'] == 'Service requested']['Branch'].unique()
                rp_edges = [(node_name, p) for p in providers]
                self.graph.add_edges_from(rp_edges)

                # Add them to the graph
                for e in rp_edges:
                    try:
                        types = self.graph.edges[e]['type']
                        if 'request_provide' not in types:
                            self.graph.edges[e]['type'] += ['request_provide']
                    except KeyError:
                        self.graph.edges[e]['type'] = ['request_provide']

                # Find edges based on branch
                # --------------------------
                # Find the node's parent
                parent = '.'.join(node_name.split('.')[:-1])
                s_edges = []
                if parent:
                    s_edges += [(parent, node_name)]
                self.graph.add_edges_from(s_edges)

                # Add them to the graph
                for e in s_edges:
                    try:
                        types = self.graph.edges[e]['type']
                        if 'structure' not in types:
                            self.graph.edges[e]['type'] += ['structure']
                    except KeyError:
                        self.graph.edges[e]['type'] = ['structure']

            for node in self.node_dfs:
                add_edges(node, self.node_dfs[node])

            for node in self.tech_dfs:
                for tech in self.tech_dfs[node]:
                    add_edges(node, self.tech_dfs[node][tech])

        def initialize():
            pass

        make_nodes(type_col='Demand?')
        make_edges()
        initialize()

    def run(self, equilibrium_threshold=0.05):
        def run_year(year):
            def traverse_graph(sub_graph, node_process_func):
                # Find the root of the sub-graph
                root = [n for n, d in sub_graph.in_degree() if d == 0][0]

                # Find the distance from the root to each node in the sub-graph
                dist_from_root = nx.single_source_shortest_path_length(sub_graph, root)

                # Start the traversal
                sg_cur = copy.deepcopy(sub_graph)
                visited = []

                while len(sg_cur.nodes) > 0:
                    active_front = [n for n, d in sg_cur.in_degree if d == 0]

                    if len(active_front) > 0:
                        # Choose a node on the active front
                        n_cur = active_front[0]
                        # Process that node in the sub-graph
                        node_process_func(n_cur)
                    else:
                        # Resolve a loop
                        candidates = {n: dist_from_root[n] for n in sg_cur}
                        n_cur = min(candidates, key=lambda x: candidates[x])
                        # Process chosen node in the sub-graph, using estimated values from their parents
                        node_process_func(n_cur, with_estimates=True)

                    visited.append(n_cur)
                    sg_cur.remove_node(n_cur)

                    # Return the updated sub graph

            def get_subgraph(node_types):
                nodes = [n for n, a in self.graph.nodes(data=True) if a['type'] in node_types]
                sub_g = self.graph.subgraph(nodes).copy()
                return sub_g

            def calc_demand():
                def demand_process_func(node, with_estimates=False):
                    self.graph.nodes[node]['demand'] = {f: random.randint(10, 50) for f in self.fuels}
                    pass

                # Find the demand sub-graph
                g_demand = get_subgraph(['demand', 'standard'])

                # Traverse the sub-graph, processing as we encounter nodes
                traverse_graph(g_demand, demand_process_func)

                # Aggregate demand
                demand_by_fuel = self.aggregate(g_demand, [year, 'demand'], agg_func=sum)

                return demand_by_fuel

            def calc_supply():
                def supply_process_func(node, with_estimates=False):
                    pass

                # Find the supply sub-graph
                g_supply = get_subgraph(['supply', 'standard'])

                # Traverse the sub-graph processing as we encounter nodes
                traverse_graph(g_supply, supply_process_func)

                # Find the prices by fuel (Note: Unsure. This might actually happen in the previous step...)
                fuel_prices = {}

                return fuel_prices

            def equilibrium_check(dict1, dict2, threshold):
                for fuel in dict1:
                    abs_diff = abs(dict1[fuel] - dict2[fuel])
                    rel_diff = abs_diff / dict1[fuel]
                    if rel_diff > threshold:
                        return False

                return True

            prev_demand = {}
            prev_supply = {}
            equilibrium = False
            while not equilibrium:
                curr_demand = calc_demand()
                curr_supply = calc_supply()
                equilibrium = equilibrium_check(prev_demand, curr_demand, equilibrium_threshold) and \
                              equilibrium_check(prev_supply, curr_supply, equilibrium_threshold)

                prev_demand = curr_demand
                prev_supply = curr_supply

            # TODO: Add finishing procedures. Ex. Storing resulting prices and demands
            # Some finishing things... Lik

        for y in self.years:
            run_year(y)

    def search_nodes(self, search_term):
        """Search nodes to see if there is one that contains the search term in the final component of its name"""
        def search(name):
            components = name.split('.')
            last_comp = components[-1]
            return search_term.lower() in last_comp.lower()

        return [n for n in self.graph.nodes if search(n)]

    def aggregate(self, sub_graph, agg_key, agg_func=sum):
        """
        Sum agg_val across all nodes in a given subgraph.
        :param sub_graph: nx.Graph. The sub-graph to be aggregated over.
        :param agg_key: List of str. The key list needed to access the values to be aggregated. Will be used to
        :param agg_func:
        :return:
        """

        def get_val(dict, key_list):
            value = dict
            for k in key_list:
                value = value[k]
            return value

        values_by_node = [get_val(data, agg_key) for name, data in sub_graph.nodes(data=True)]

        all_values = [(k, v) for values in values_by_node for k, v in values.items()]

        # Make value lists, separated by key
        value_lists = {}
        for k, v in all_values:
            try:
                value_lists[k].append(v)
            except KeyError:
                value_lists[k] = [v]

        # Aggregate each list by applying agg_function
        aggregates = {k: agg_func(v) for k, v in value_lists.items()}

        return aggregates
