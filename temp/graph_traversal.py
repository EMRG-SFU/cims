import networkx as nx
import copy
import random


g = nx.DiGraph()

# Add nodes
standard_nodes = ['Canada', 'Alberta']
demand_nodes = ['Residential', 'Building Type', 'Dish Washing', 'Shell', 'Clothes Washing', 'Space Heating',
                'Water Heating', 'Furnace']
supply_nodes = ['Natural Gas', 'Solar', 'Wind', 'Electricity', 'Generation', 'Base Load', 'Conventional', 'Renewables']
g.add_nodes_from(standard_nodes, node_type='standard')
g.add_nodes_from(demand_nodes, node_type='demand')
g.add_nodes_from(supply_nodes, node_type='supply')

# Add edges
g.add_edge('Canada', 'Alberta')
g.add_edge('Alberta', 'Residential')
g.add_edge('Alberta', 'Natural Gas')
g.add_edge('Alberta', 'Solar')
g.add_edge('Alberta', 'Wind')
g.add_edge('Alberta', 'Electricity')
g.add_edge('Residential', 'Building Type')
g.add_edge('Electricity', 'Generation')
g.add_edge('Building Type', 'Dish Washing')
g.add_edge('Building Type', 'Shell')
g.add_edge('Building Type', 'Clothes Washing')
g.add_edge('Generation', 'Base Load')
g.add_edge('Shell', 'Space Heating')
g.add_edge('Shell', 'Water Heating')
g.add_edge('Base Load', 'Conventional')
g.add_edge('Base Load', 'Renewables')
g.add_edge('Space Heating', 'Furnace')


def process_node(n, g):
    # FAKE FUNCTION
    if n in ['Dish Washing', 'Space Heating', 'Furnace', 'Water Heating', 'Clothes Washing']:
        fuels = ['Natural Gas', 'Solar', 'Wind', 'Electricity']
        demand = {f: random.randint(10, 100) for f in fuels}
        g.nodes[n]['demand'] = demand


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
            print(n_cur)
            visited.append(n_cur)
            sg_cur.remove_node(n_cur)

            # Return the updated sub graph

    def aggregate_demand(sub_g):
        # Sum the demand across all nodes in the sub-graph

        # Find all the demands
        demands_by_node = [v for k, v in nx.get_node_attributes(sub_g, 'demand').items()]
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
    d_nodes = [n for n, a in g.nodes(data=True) if a['node_type'] in ('demand', 'standard')]
    g_demand = g.subgraph(d_nodes).copy()

    # Traverse the sub-graph, processing as we encounter nodes
    traverse_sub_graph(g_demand)

    # Aggregate demand
    demand_by_fuel = aggregate_demand(g_demand)

    return demand_by_fuel
