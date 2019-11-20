import networkx as nx
import copy as copy
import re


class Model:
    def __init__(self, node_dfs, tech_dfs):
        """ Constructor """
        self.graph = nx.DiGraph()
        self.node_dfs = node_dfs
        self.tech_dfs = tech_dfs

        self.fuels = ['Electricity', 'Natural Gas', 'Solar', 'Wind']

        self._make_nodes(type_col='Demand?')
        self._make_edges()

    def _make_nodes(self, type_col='Demand?'):
        def is_year(cn):
            """Check if input int or str is 4 digits [0-9] between begin ^ and end $ of string"""
            # unit test: assert is_year, 1900
            re_year = re.compile(r'^[0-9]{4}$')
            return bool(re_year.match(str(cn)))

        def add_node_data(current_node):
            """

            :param current_node:
            :return:
            """
            # Copy the current node dataframe
            current_node_df = copy.deepcopy(self.node_dfs[current_node])

            # 1. we are going to create a node in the graph
            self.graph.add_node(current_node)

            # 2. We will store the Supply/Demand Type of node. This is a special case.
            typ = list(current_node_df[type_col])[0]
            if typ:
                self.graph.nodes[current_node]['type'] = typ.lower()
            else:
                self.graph.nodes[current_node]['type'] = 'standard'
            # Drop Demand column
            current_node_df = current_node_df.drop(type_col, axis=1)

            # 3. We will store the Competition Type of the node at the node level. This is another special case.
            comp_list = list(current_node_df[current_node_df['Parameter'] == 'Competition type']['Value'])
            if len(set(comp_list)) == 1:
                comp_type = comp_list[0]
                self.graph.nodes[current_node]['competition_type'] = comp_type.lower()
            elif len(set(comp_list)) > 1:
                print("TOO MANY COMPETITION TYPES")
            # Get rid of competition type row
            current_node_df = current_node_df[current_node_df['Parameter'] != 'Competition type']

            # 4. For the remaining rows, group data by year.
            # Get Year Columns
            years = [c for c in current_node_df.columns if is_year(c)]

            # Get Non-Year Columns
            non_years = [c for c in current_node_df.columns if not is_year(c)]

            # For each year:
            for y in years:
                year_df = current_node_df[non_years + [y]]
                year_dict = {}
                for parameter, source, branch, unit, value, year_value in zip(*[year_df[c] for c in year_df.columns]):
                    if parameter in year_dict.keys():
                        pass
                    else:
                        year_dict[parameter] = {}

                    dct = {'source': source,
                           'branch': branch,
                           'unit': unit,
                           'year_value': year_value}
                    #             # Clean Dict
                    #             clean_dict = {k: v for k, v in dct.items() if v is not None}

                    year_dict[parameter][value] = dct

                # Add data to node
                self.graph.nodes[current_node][y] = year_dict

        def add_tech_data(node_name, tech_name):
            # TODO: I think we need to differentiate between Technologies and Services.

            t_df = copy.deepcopy(self.tech_dfs[node_name][tech_name])

            # 1. Remove the row that indicates this is a service or technology.
            t_df = t_df[~t_df['Parameter'].isin(['Service', 'Technology'])]

            # 2. Remove the Demand? column
            t_df = t_df.drop('Demand?', axis=1)

            # VERY SIMILAR to what we do for nodes. But not quite. Because we don't use the value column anymore
            # 4. For the remaining rows, group data by year.
            # Get Year Columns
            years = [c for c in t_df.columns if is_year(c)]

            # Get Non-Year Columns
            non_years = [c for c in t_df.columns if not is_year(c)]

            # For each year:
            for y in years:
                year_df = t_df[non_years + [y]]
                year_dict = {}

                for parameter, source, branch, unit, value, year_value in zip(*[year_df[c] for c in year_df.columns]):
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

    def _make_edges(self):
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

    def search_nodes(self, search_term):
        """Search nodes to see if there is one that contains the search term in the final component of its name"""
        def search(name):
            components = name.split('.')
            last_comp = components[-1]
            return search_term.lower() in last_comp.lower()

        return [n for n in self.graph.nodes if search(n)]

    def process_demand(self, year):
        def traverse_sub_graph(sub_g):

            def process_node(node, with_estimates=False):
                pass

            def process_node(node, with_estimates=False):
                # FAKE FUNCTION
                # TODO: CHANGE FUNCTION

                data = self.graph.nodes[node][year]
                if 'technologies' in data.keys():
                    demands = {f: 0 for f in self.fuels}
                    for tech, data in data['technologies'].items():
                        if not type(data['Service requested']) is list:
                            sr = [data['Service requested']]
                        else:
                            sr = data['Service requested']
                        # Get List of Requested Fuels
                        requested = [(r['branch'].split('.')[-1], r['year_value']) for r in sr]

                        for r, yv in requested:
                            is_fuel = r in self.fuels
                            # print(r, yv, is_fuel)
                            if is_fuel:
                                demands[r] += yv
                            else:
                                pass

                    self.graph.nodes[node][year]['demand'] = demands

            # Find the root of the sub-graph
            root = [n for n, d in sub_g.in_degree() if d == 0][0]

            # Find the distance from the root to each node in the sub-graph
            dist_from_root = nx.single_source_shortest_path_length(sub_g, root)

            # Start the traversal
            sg_cur = copy.deepcopy(sub_g)
            visited = []

            while len(sg_cur.nodes) > 0:
                active_front = [n for n, d in sg_cur.in_degree if d == 0]

                if len(active_front) > 0:
                    # Choose a node on the active front
                    n_cur = active_front[0]
                    # Process that node in the sub-graph
                    process_node(n_cur)
                else:
                    # Resolve a loop
                    candidates = {n: dist_from_root[n] for n in sg_cur}
                    n_cur = min(candidates, key=lambda x: candidates[x])
                    # Process chosen node in the sub-graph, using estimated values from their parents
                    process_node(n_cur, with_estimates=True)

                visited.append(n_cur)
                sg_cur.remove_node(n_cur)

                # Return the updated sub graph

        def aggregate_demand(sub_g, year='2000'):
            # Sum the demand across all nodes in the sub-graph

            # Find all the demands
            # node_year = [v for k, v in nx.get_node_attributes(sub_g, year).items()]
            demands_by_node = [v['demand'] for k, v in nx.get_node_attributes(sub_g, year).items() if
                               'demand' in v.keys()]
            # demands_by_node = [v['demand'] for k, v in nx.get_node_attributes(sub_g, year).items()]

            all_demands = [(k, v) for demands in demands_by_node for k, v in demands.items()]

            # Sum over the demands
            summed_demands = {}
            for fuel, demand in all_demands:
                try:
                    summed_demands[fuel] += demand
                except KeyError:
                    summed_demands[fuel] = demand

            return summed_demands

        # Find the demand sub-graph
        d_nodes = [n for n, a in self.graph.nodes(data=True) if a['type'] in ('demand', 'standard')]
        g_demand = self.graph.subgraph(d_nodes).copy()

        # Traverse the sub-graph, processing as we encounter nodes
        traverse_sub_graph(g_demand)

        # Aggregate demand
        demand_by_fuel = aggregate_demand(g_demand)

        return demand_by_fuel

    def process_supply(self):
        pass
