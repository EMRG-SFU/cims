import warnings
import networkx as nx

from . import loop_resolution

def find_next_node(degrees):
    for node, degree in degrees:
        if degree == 0:
            return node


def find_loops(graph, warn=False):
    loops = list(nx.simple_cycles(graph))
    if warn and len(loops) > 0:
        warning_str = f"Found {len(loops)} loops (see model.loops for the full list)"
        warnings.warn(warning_str)
    return loops


def top_down_traversal(graph, node_process_func, *args, root=None,
                       loop_resolution_func=loop_resolution.min_distance_from_root, **kwargs):
    """
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

    A node is only visited once its parents have been visited. In the case of a loop (where every
    node has at least one parent who hasn't been visited) the node closest to the `sub_graph` root
    will be visited and processed using the values held over from the last iteration.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to use to find the root (if not provided) and the distance from the root

    node_process_func : function (nx.DiGraph, str) -> None
        The function to be applied to each node in `sub_graph`. Doesn't return anything but should
        have an effect on the node data within `sub_graph`.

    Returns
    -------
    None

    """
    # Find the root of the sub-graph
    if not root:
        possible_roots = [n for n, d in graph.in_degree() if d == 0]
        possible_roots.sort(key=lambda n: len(n))
        root = possible_roots[0]

    # Find the distance from the root to each node in the sub-graph
    dist_from_root = nx.single_source_shortest_path_length(graph, root)

    # Start the traversal
    sub_graph = graph
    sg_cur = sub_graph.copy()

    while len(sg_cur.nodes) > 0:
        n_cur = find_next_node(sg_cur.in_degree)
        if n_cur is not None:
            node_process_func(sub_graph, n_cur, *args, **kwargs)
        else:
            # Resolve a loop
            cycles = find_loops(sg_cur)
            n_cur = loop_resolution_func(cycles, dist_from_root)

            # Process chosen node in the sub-graph, using estimated values from their parents
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        sg_cur.remove_node(n_cur)


def bottom_up_traversal(graph, node_process_func, *args, root=None,
                        loop_resolution_func=loop_resolution.max_distance_from_root, **kwargs):
    """
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

    A node is only visited once its children have been visited. In the case of a loop (where every
    node has at least one parent who hasn't been visited) the node furthest from the `sub_graph`
    root will be visited and processed using the values held over from the last iteration.

    TODO: Properly document
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.


    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to use to find the root (if not provided) and the distance from the root

    node_process_func : function (nx.DiGraph, str) -> None
        The function to be applied to each node in `sub_graph`. Doesn't return anything but should
        have an effect on the node data within `sub_graph`.

    Returns
    -------
    None

    """
    # If root hasn't been provided, find the sub-graph's root
    if not root:
        possible_roots = [n for n, d in graph.in_degree() if d == 0]
        possible_roots.sort(key=lambda n: len(n))
        root = possible_roots[0]

    # Find the distance from the root to each node in the sub-graph
    dist_from_root = nx.single_source_shortest_path_length(graph, root)

    # Start the traversal
    sub_graph = graph
    sg_cur = sub_graph.copy()

    while len(sg_cur.nodes) > 0:
        n_cur = find_next_node(sg_cur.out_degree)
        if n_cur is not None:
            node_process_func(sub_graph, n_cur, *args, **kwargs)
        else:
            # Resolve a loop
            cycles = find_loops(sg_cur)
            n_cur = loop_resolution_func(cycles, dist_from_root, **kwargs)

            # Process chosen node in the sub-graph, using estimated values from their parents
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        sg_cur.remove_node(n_cur)