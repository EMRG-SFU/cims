import networkx as nx
import copy
import random
import pickle

pickle_in = open('/Users/jilliana/Documents/Consulting/pyCIMs/pycims_prototype/temp/graph.pickle', 'rb')
g_all = pickle.load(pickle_in)

g_nodes = [n for n, a in g_all.nodes(data=True) if 'type' in a.keys()]  # Find nodes that have data
g = g_all.subgraph(g_nodes).copy()

fuels = ['Electricity', 'Natural Gas', 'Solar', 'Wind']


def search_nodes(g, search_term):
    """Search nodes to see if there is one that contains the search term in the final component of its name"""

    def search(name):
        components = name.split('.')
        last_comp = components[-1]

        return search_term.lower() in last_comp.lower()

    return [n for n in g.nodes if search(n)]


def process_node(node, graph, year='2000'):
    # FAKE FUNCTION
    # TODO: CHANGE FUNCTION

    data = graph.nodes[node][year]
    if 'technologies' in data.keys():
        demands = {f: 0 for f in fuels}
        for tech, data in data['technologies'].items():
            if not type(data['Service requested']) is list:
                sr = [data['Service requested']]
            else:
                sr = data['Service requested']
            # Get List of Requested Fuels
            requested = [(r['branch'].split('.')[-1], r['year_value']) for r in sr]

            for r, yv in requested:
                is_fuel = r in fuels
                # print(r, yv, is_fuel)
                if is_fuel:
                    demands[r] += yv
                else:
                    pass

        graph.nodes[node][year]['demand'] = demands


def process_node_with_estimates(n):
    pass


def process_demand(g):
    def traverse_sub_graph(sub_g):
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
                process_node(n_cur, sub_g)
            else:
                # Resolve a loop
                candidates = {n: dist_from_root[n] for n in sg_cur}
                n_cur = min(candidates, key=lambda x: candidates[x])
                # Process chosen node in the sub-graph, using estimated values from their parents
                process_node_with_estimates(n_cur)
            visited.append(n_cur)
            sg_cur.remove_node(n_cur)

            # Return the updated sub graph

    def aggregate_demand(sub_g, year='2000'):
        # Sum the demand across all nodes in the sub-graph

        # Find all the demands
        node_year = [v for k, v in nx.get_node_attributes(sub_g, year).items()]
        demands_by_node = [v['demand'] for k, v in nx.get_node_attributes(sub_g, year).items() if 'demand' in v.keys()]
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
    d_nodes = [n for n, a in g.nodes(data=True) if a['type'] in ('demand', 'standard')]
    g_demand = g.subgraph(d_nodes).copy()

    # Traverse the sub-graph, processing as we encounter nodes
    traverse_sub_graph(g_demand)

    # Aggregate demand
    demand_by_fuel = aggregate_demand(g_demand)

    return demand_by_fuel


print(process_demand(g))