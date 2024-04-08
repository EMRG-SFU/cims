"""
Module containing loop resolution functions.
"""
import networkx as nx


def min_distance_from_root(list_of_cycles, distances_from_root, **kwargs):
    # TODO: Document function
    candidates = {node: distances_from_root[node] for cycle in list_of_cycles for node in cycle}
    next_node = min(candidates, key=lambda x: candidates[x])

    return next_node


def max_distance_from_root(list_of_cycles, distances_from_root, **kwargs):
    # TODO: Document function
    candidates = {node: distances_from_root[node] for cycle in list_of_cycles for node in cycle}
    next_node = max(candidates, key=lambda x: candidates[x])
    return next_node


def aggregation_resolution(list_of_cycles, distances_from_root, **kwargs):
    # TODO: Document function
    lists_of_edges = [list(zip(nodes, (nodes[1:]+nodes[:1]))) for nodes in list_of_cycles]
    edges = [edge for cycle in lists_of_edges for edge in cycle]
    g = nx.DiGraph(edges)
    g.remove_nodes_from([n for n in g.nodes if n in kwargs['supply_nodes']])
    leaf_nodes = [n for n, d in g.out_degree if d == 0]

    candidates = {node: distances_from_root[node] for node in leaf_nodes}
    next_node = max(candidates, key=lambda x: candidates[x])
    return next_node
