from __future__ import annotations  # For Type Hinting
import copy
import warnings
import networkx as nx
from typing import List


from . import old_utils
from . import loop_resolution
from . import cost_curves

from .utils import parameters as PARAM
from .utils import model_columns as COL

# **************************
# 1- Perform action in graph
# **************************
def _find_value_in_ancestors(graph, node, parameter, year=None):
    """
    Find a parameter's value at a given node or its structural ancestors.

    First attempts to locate a parameter at a given node. If the parameter does not exist at that
    node, a recursive call is made to find the parameter's value at `node`'s parent, if one exists.
    If no parent exists None is returned.

    Parameters
    ----------
    graph : networkx.Graph
        The graph where `node` resides.
    node : str
        The name of the node to begin our search from. Must be contained within
        `graph`. (e.g. `CIMS.Canada.Alberta`)
    parameter : str
        The name of the parameter whose value is being found. (e.g. 
        `Price Multiplier`)
    year : str, optional
        The year associated with sub-dictionary to search at `node`. Default is None, which implies
        that year sub-dictionaries should be searched. Instead, only search for `parameter` in
        `node`s top level data.

    Returns
    -------
    Any
        The value associated with `parameter` if a value can be found at `node` or one of its
        ancestors. Otherwise None
    """
    data = graph.nodes[node]
    parent = parent_name(node, return_empty=True)

    # Look at the Node/Year
    if year:
        year_data = graph.nodes[node][year]
        if parameter in year_data.keys():
            return year_data[parameter]

    # Look at the Node
    if parameter in data.keys():
        return data[parameter]

    # Look at the Parent
    if parent:
        return _find_value_in_ancestors(graph, parent, parameter, year)

    # Worst Case, return None


def parent_name(curr_node, return_empty=False):
    """
    curr_node is current node name (str)
    CAUTION: when curr_node is tree root, returns root (when retur_empty is False)
    """
    parent = '.'.join(curr_node.split('.')[:-1])
    if parent:
        return_val = parent
    elif return_empty:
        return_val = ""
    else:
        return_val = curr_node

    return return_val


def get_supply_nodes(graph):
    """
    Find the nodes which have been specified as supply nodes in the model description.
    Returns
    -------
    List
        A list containing the names of supply nodes.
    """
    supply_nodes = []
    for node in graph.nodes:
        if graph.nodes[node][PARAM.is_supply]:
            supply_nodes.append(node)
    return supply_nodes


def get_ghg_and_emissions(graph, year):
    """
    Return 2 lists consisting of all the GHGs (CO2, CH4, etc.) and all the emission types (Process, Fugitive, etc.)
    Return 1 dictionary containing the GHGs as keys and GWPs as values
    :param DiGraph graph: graph to search for all emissions
    :param str year: year to find emissions, will likely be base year
    :return: list of GHGs and a list of emission types
    """

    ghg = []
    emission_type = []
    gwp = {}
    for node, data in graph.nodes(data=True):

        # Emissions from a node with technologies
        if PARAM.technologies in data[year]:
            techs = data[year][PARAM.technologies]
            for tech in techs:
                tech_data = data[year][PARAM.technologies][tech]
                if PARAM.emissions in tech_data or PARAM.emissions_removal in tech_data:
                    if PARAM.emissions in tech_data:
                        ghg_list = data[year][PARAM.technologies][tech][PARAM.emissions]
                    else:
                        ghg_list = data[year][PARAM.technologies][tech][PARAM.emissions_removal]

                    node_ghg = [ghg for ghg in ghg_list]
                    node_emission_type = [emission_type for emission_record in ghg_list.values() for
                                          emission_type in emission_record]

                    ghg = list(set(ghg + node_ghg))
                    emission_type = list(set(emission_type + node_emission_type))

        # Emissions from a supply node
        elif PARAM.emissions in data[year] or PARAM.emissions_removal in data[year]:
            if PARAM.emissions in data[year]:
                ghg_dict = data[year][PARAM.emissions]
            else:
                ghg_dict = data[year][PARAM.emissions_removal]

            node_ghg = [ghg for ghg in ghg_dict.keys()]

            node_emission_type = [emission_type for emission_record in ghg_dict.values() for
                                  emission_type in emission_record]

            ghg = list(set(ghg + node_ghg))
            emission_type = list(set(emission_type + node_emission_type))

        #GWP from CIMS node
        if PARAM.emissions_gwp in data[year]:
            for ghg2 in data[year][PARAM.emissions_gwp]:
                gwp[ghg2] = data[year][PARAM.emissions_gwp][ghg2][PARAM.year_value]

    return ghg, emission_type, gwp


def get_demand_side_nodes(graph: nx.DiGraph) -> List[str]:
    """
    Find the nodes to use for demand-side traversals. The returned list of nodes
    will include all nodes whose `node_type` attribute is not `supply`.

    Parameters
    ----------
    graph :
        The graph whose demand tree will be returned.

    Returns
    -------
    A subgraph of graph which includes only non-supply nodes.
    """
    # Find Supply Nodes
    supply_nodes = set([n for n, d in graph.nodes(data=True) if (PARAM.is_supply in d) and d[PARAM.is_supply]])

    # Find the structural descendants of supply_nodes
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structural' in d[PARAM.edge_type]]
    structural_graph = graph.edge_subgraph(structural_edges)

    descendants = set()
    for supply in supply_nodes:
        supply_node_structural_descendants = nx.descendants(structural_graph, supply)
        descendants = descendants.union(supply_node_structural_descendants)

    # Return all the nodes which are neither supply_nodes nor their descendants
    return list(set(graph.nodes).difference(set(supply_nodes).union(descendants)))


def get_supply_side_nodes(graph: nx.DiGraph) -> List[str]:
    """
    Find the nodes to use for supply-side traversals. The returned list of nodes will include all
    supply nodes, the structural ancestors of the supply nodes, and the descendants of the supply nodes.

    Parameters
    ----------
    graph :
        The graph whose supply tree will be returned.

    Returns
    -------
    A subgraph of graph which includes only supply nodes, their structural ancestors, and their descendants
    """
    # Find Supply Nodes
    supply_nodes = [n for n, d in graph.nodes(data=True) if (PARAM.is_supply in d) and d[PARAM.is_supply]]

    # Find the structural ancestors of the supply nodes
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structural' in d[PARAM.edge_type]]
    structural_graph = graph.edge_subgraph(structural_edges)

    structural_ancestors = set()
    descendants = set()
    for supply in supply_nodes:
        structural_ancestors = structural_ancestors.union(nx.ancestors(structural_graph, supply))
        descendants = descendants.union(nx.descendants(graph, supply))

    return supply_nodes + list(structural_ancestors) + list(descendants)


# **************************
# 2 - TRAVERSALS
# **************************
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


def find_next_node(degrees):
    for node, degree in degrees:
        if degree == 0:
            return node

# ****************************
# 3 - Making the graph - Edges
# ****************************
def make_or_update_edges(graph, node_dfs, tech_dfs):
    """
    Add edges to `graph` using information in `node_dfs` and `tech_dfs`.

    Returns
    -------
    networkx.Graph
        An updated `graph` that contains all edges defined in `node_dfs` and `tech_dfs`.

    """
    # Add Structural Edges
    graph = add_structural_edges_from_dfs(graph, node_dfs, tech_dfs)        
    
    # Add Request/Provide Edges
    graph = add_request_provide_edges_from_dfs(graph, node_dfs, tech_dfs)

    # Add Aggregation Edges
    # The request/provide & structural edges must be in place before the
    # aggregation edges can be added.
    graph = add_aggregation_edges_from_dfs(graph, node_dfs)

    return graph


def add_or_update_edge(graph, edge, edge_data):
    if not isinstance(edge_data[PARAM.edge_type], list):
        edge_data[PARAM.edge_type] = [edge_data[PARAM.edge_type]]

    if graph.has_edge(*edge):
        # Update the edge's list of types
        new_edge_types = list(set(graph.edges[edge][PARAM.edge_type] + edge_data[PARAM.edge_type]))
        edge_data[PARAM.edge_type] = new_edge_types
    else:
        # Add the new edge
        graph.add_edge(*edge)

    graph.edges[edge].update(edge_data)


def add_structural_edges_from_dfs(graph, node_dfs, tech_dfs):
    # From Node DFs
    for node in node_dfs:
        graph = add_edges_of_one_type(graph, 
                                      node, 
                                      node_dfs[node], 
                                      edge_type='structural')

    # From Tech DFs
    for node in tech_dfs:
        for tech in tech_dfs[node]:
            graph = add_edges_of_one_type(graph, 
                                          node, 
                                          tech_dfs[node][tech], 
                                          edge_type='structural')
    
    return graph


def add_request_provide_edges_from_dfs(graph, node_dfs, tech_dfs):
    # From Node DFs
    for node in node_dfs:
        graph = add_edges_of_one_type(graph, 
                                      node, 
                                      node_dfs[node], 
                                      edge_type='request_provide')

    # From Tech DFs
    for node in tech_dfs:
        for tech in tech_dfs[node]:
            graph = add_edges_of_one_type(graph, 
                                          node, 
                                          tech_dfs[node][tech], 
                                          edge_type='request_provide')
    
    return graph


def add_aggregation_edges_from_dfs(graph, node_dfs):
    # From Node DFs
    for node in node_dfs:
        graph = add_edges_of_one_type(graph, 
                                      node, 
                                      node_dfs[node], 
                                      edge_type='aggregation')
    
    return graph


def add_edges_of_one_type(graph, node, df, edge_type):
    # 1. Copy the graph
    graph = copy.copy(graph)

    # 2. Find the Edges
    edges_to_add = find_edges(graph, node, df, edge_type)

    # 3. Add or update the edges to the graph
    for edge, edge_data in edges_to_add:
        add_or_update_edge(graph, edge, edge_data)

    return graph


def find_edges(graph, node, df, edge_type):
    edges = []
    if edge_type == 'structural':
        # Find edge based on branch structure.
        # e.g. If our node was CIMS.Canada.Alberta.Residential we create an edge Alberta->Residential
        parent = '.'.join(node.split('.')[:-1])
        if parent:
            edge = (parent, node)
            edge_data = {PARAM.edge_type: 'structural'}
            edges.append((edge, edge_data))
    
    elif edge_type == 'request_provide':
        providers = df[df[COL.parameter] == PARAM.service_requested][COL.target].unique()
        edges += [((node, p), {PARAM.edge_type: 'request_provide'}) for p in providers]

    elif edge_type == 'aggregation':

        agg_children = df[df[COL.parameter] == PARAM.aggregation_requested][COL.target].unique()

        for agg_child in agg_children:
            # For each `aggregation requested`` line at node, add the following edges
            #   (1) 1 weighted edge between node & aggregation requested target (i.e. node -> child)
            #   (2) 0 weighted edge between all other parents of the aggregation requested target & the aggregation requested target (i.e. other parent of child -> child)

            # (1) node -> child {aggregation_weight: 1}
            edges.append(((node, agg_child), {PARAM.edge_type: 'aggregation', PARAM.aggregation_weight: 1}))

            for agg_child_parent in graph.predecessors(agg_child):
                if agg_child_parent == node:
                    continue
                elif graph.edges[(agg_child_parent, agg_child)].get(PARAM.aggregation_weight) == 1:
                    pass # already set as an aggregating node, don't zero it out
                else:
                    # (2) other parents -> child {aggregation_weight: 0}
                    edges.append(((agg_child_parent, agg_child), {PARAM.edge_type: 'aggregation', PARAM.aggregation_weight: 0}))

    else:
        raise ValueError(
            f"{edge_type=} not recognized. Please use \"structure\", \"request_provide\", or \"aggregation\".")
        
    return edges


# ****************************
# 4 - Other
# ****************************
def find_loops(graph, warn=False):
    loops = list(nx.simple_cycles(graph))
    if warn and len(loops) > 0:
        warning_str = f"Found {len(loops)} loops (see model.loops for the full list)"
        warnings.warn(warning_str)
    return loops
