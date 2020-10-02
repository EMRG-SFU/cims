import networkx as nx
import copy
import warnings

# from utils import is_year
from . import utils

'''
1- Perform action in graph
'''


def find_value(graph, node, parameter, year=None):
    """
    Find a parameter's value at a given node or its structural ancestors.

    First attempts to locate a parameter at a given node. If the parameter does not exist at that node, a recursive call
    is made to find the parameter's value at `node`'s parent, if one exists. If no parent exists None is returned.
    Parameters
    ----------
    graph : networkx.Graph
        The graph where `node` resides.
    node : str
        The name of the node to begin our search from. Must be contained within `graph`. (e.g. 'pyCIMS.Canada.Alberta')
    parameter : str
        The name of the parameter whose value is being found. (e.g. 'Sector type')
    year : str, optional
        The year associated with sub-dictionary to search at `node`. Default is None, which implies that year
        sub-dictionaries should be searched. Instead, only search for `parameter` in `node`s top level data.

    Returns
    -------
    Any
        The value associated with `parameter` if a value can be found at `node` or one of its ancestors. Otherwise None
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
        return find_value(graph, parent, parameter, year)

    # Worst Case, return None
    else:
        return None


def find_node_value(graph, node, parameter, year=None, worst_case=True):
    """
    Find a parameter's value at a given node.

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

    # Worst Case, return None
    if worst_case:
        return None
    else:
        # error when the field doesn't exist if `worst_case` is set to False
        raise KeyError


def get_parent(g, curr_node, year):
    """
    g is graph
    curr_node is current node name (str)
    """
    parent = '.'.join(curr_node.split('.')[:-1])
    parent_node = g.nodes[parent][year]
    return parent_node


def parent_name(curr_node, return_empty=False):
    """
    curr_node is current node name (str)
    CAUTION: when curr_node is tree root, returns root (when retur_empty is False)
    """
    parent = '.'.join(curr_node.split('.')[:-1])
    if parent:
        return parent
    elif return_empty:
        return ""
    else:
        return curr_node


def child_name(g, curr_node, return_empty=False):
    """
    curr_node is current node name (str)
    CAUTION: when curr_node is tree leaf, returns leaf (when retur_empty is False)

    MAYBE year parameter should be included
    """
    child = [i for i in g.successors(curr_node)]

    if child:
        return child

    elif return_empty:
        return []
    else:
        return [curr_node]


def get_fuels(graph, years):
    """ Find the names of nodes supplying fuel.

    This is any node which (1) has a Node type "Supply" and (2) whose competition type contains
    Sector (either Sector or Sector No Tech).

    Returns
    -------
    list of str
        A list containing the names of nodes which supply fuels.
    """
    fuels = []
    for n, d in graph.nodes(data=True):
        is_supply = d['type'].lower() == 'supply'
        is_sector = 'sector' in d['competition type'].lower()
        if is_supply & is_sector:
            fuels.append(n)
    return fuels


def get_subgraph(graph, node_types):
    """
    Find the sub-graph of `graph` that only includes nodes whose type is in `node_types`.

    Parameters
    ----------
    node_types : list of str
        A list of node types ('standard', 'supply', or 'demand') to include in the returned sub-graph.

    Returns
    -------
    networkx.DiGraph or networkX.Graph
        The returned graph is a sub-graph of `graph`. A node is only included if its type is one
        of `node_types`. A edge is only included if it connects two nodes found in the returned graph.
    """
    nodes = [n for n, a in graph.nodes(data=True) if a['type'] in node_types]
    sub_g = graph.subgraph(nodes)
    return sub_g


"""
2 - TRAVERSALS
"""


def top_down_traversal(sub_graph, node_process_func, *args, **kwargs):
    """
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

    A node is only visited once its parents have been visited. In the case of a loop (where every
    node has at least one parent who hasn't been visited) the node closest to the `sub_graph` root
    will be visited and processed using the values held over from the last iteration.

    Parameters
    ----------
    sub_graph : networkx.DiGraph
        The graph to be traversed.

    node_process_func : function (nx.DiGraph, str) -> None
        The function to be applied to each node in `sub_graph`. Doesn't return anything but should have an
        effect on the node data within `sub_graph`.

    Returns
    -------
    None

    """
    # Find the root of the sub-graph
    possible_roots = [n for n, d in sub_graph.in_degree() if d == 0]
    possible_roots.sort(key=lambda n: len(n))
    root = possible_roots[0]

    # Find the distance from the root to each node in the sub-graph
    dist_from_root = nx.single_source_shortest_path_length(sub_graph, root)

    original_leaves = [n for n, d in sub_graph.out_degree if d == 0]

    for node_name in sub_graph:
        if node_name in original_leaves:
            sub_graph.nodes[node_name]["is_leaf"] = True
        else:
            sub_graph.nodes[node_name]["is_leaf"] = False

    # Start the traversal
    sg_cur = sub_graph.copy()

    while len(sg_cur.nodes) > 0:
        active_front = [n for n, d in sg_cur.in_degree if d == 0]

        if len(active_front) > 0:
            # Choose a node on the active front
            n_cur = active_front[0]
            # Process that node in the sub-graph
            node_process_func(sub_graph, n_cur, *args, **kwargs)
        else:
            warnings.warn("Found a Loop -- ")
            # Resolve a loop
            candidates = {n: dist_from_root[n] for n in sg_cur}
            n_cur = min(candidates, key=lambda x: candidates[x])
            # Process chosen node in the sub-graph, using estimated values from their parents
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        sg_cur.remove_node(n_cur)


def bottom_up_traversal(sub_graph, node_process_func, *args, **kwargs):
    """
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.

    A node is only visited once its children have been visited. In the case of a loop (where every
    node has at least one parent who hasn't been visited) the node furthest from the `sub_graph`
    root will be visited and processed using the values held over from the last iteration.

    TODO: Properly document
    Visit each node in `sub_graph` applying `node_process_func` to each node as its visited.


    Parameters
    ----------
    sub_graph : networkx.DiGraph
        The graph to be traversed.

    node_process_func : function (nx.DiGraph, str) -> None
        The function to be applied to each node in `sub_graph`. Doesn't return anything but should
        have an effect on the node data within `sub_graph`.

    Returns
    -------
    None

    """

    # Find the root of the sub-graph
    possible_roots = [n for n, d in sub_graph.in_degree() if d == 0]
    possible_roots.sort(key=lambda n: len(n))
    root = possible_roots[0]

    # Find the distance from the root to each node in the sub-graph
    dist_from_root = nx.single_source_shortest_path_length(sub_graph, root)

    original_leaves = [n for n, d in sub_graph.out_degree if d == 0]

    for node_name in sub_graph:
        if node_name in original_leaves:
            sub_graph.nodes[node_name]["is_leaf"] = True
        else:
            sub_graph.nodes[node_name]["is_leaf"] = False

    # Start the traversal
    sg_cur = sub_graph.copy()

    while len(sg_cur.nodes) > 0:
        active_front = [n for n, d in sg_cur.out_degree if d == 0]

        if len(active_front) > 0:
            # Choose a node on the active front
            n_cur = active_front[0]
            # Process that node in the sub-graph
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        else:
            warnings.warn("Found a Loop")
            # Resolve a loop
            candidates = {n: dist_from_root[n] for n in sg_cur}
            n_cur = max(candidates, key=lambda x: candidates[x])
            # Process chosen node in the sub-graph, using estimated values from their parents
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        sg_cur.remove_node(n_cur)


"""
3 - Making the graph
"""


def add_node_data(graph, current_node, node_dfs, *args, **kwargs): # args and kwargs for including new fields (e.g. LCC, Service Cost, etc)
    """ Add and populate a new node to `graph`

    Parameters
    ----------
    current_node : str
        The name of the node (branch format) to add.

    Returns
    -------
    networkx.Graph
        `graph` with `node` added, along with its associated data.
    """
    # 1 Copy the current graph & the current node's dataframe
    current_node_df = copy.copy(node_dfs[current_node])
    graph = copy.copy(graph)

    # 2 Add an empty node to the graph
    graph.add_node(current_node)

    # 3 Find node type (supply, demand, or standard)
    typ = list(current_node_df[current_node_df['Parameter'].str.lower() == 'node type']['Value'])
    if len(typ) > 0:
        graph.nodes[current_node]['type'] = typ[0].lower()
    else:
        # If type isn't in the node's df, try to find it in the ancestors
        val = find_value(graph, current_node, 'type')
        graph.nodes[current_node]['type'] = val if val else 'standard'
    # Drop Demand row
    current_node_df = current_node_df[current_node_df['Parameter'] != 'Node type']

    # 4 Find node's competition type. (If there is one)
    comp_list = list(current_node_df[current_node_df['Parameter'] == 'Competition type']['Value'])
    if len(set(comp_list)) == 1:
        comp_type = comp_list[0]
        graph.nodes[current_node]['competition type'] = comp_type.lower()
    elif len(set(comp_list)) > 1:
        print("TOO MANY COMPETITION TYPES")
    # Get rid of competition type row
    current_node_df = current_node_df[current_node_df['Parameter'] != 'Competition type']

    # 5 For the remaining data, group by year.
    years = [c for c in current_node_df.columns if utils.is_year(c)]          # Get Year Columns
    non_years = [c for c in current_node_df.columns if not utils.is_year(c)]  # Get Non-Year Columns

    for y in years:
        year_df = current_node_df[non_years + [y]]
        year_dict = {}
        for param, val, branch, src, unit, nothing, year_value in zip(*[year_df[c] for c in year_df.columns]):
            dct = {'source': src,
                   'branch': branch,
                   'unit': unit,
                   'year_value': year_value}

            # TODO: Switch away from dictionary of dictionary I think
            # if param in year_dict.keys():
            #     if isinstance(year_dict[param], list):
            #         year_dict[param].append(dct)
            #     else:
            #         year_dict[param] = [year_dict[param], dct]
            # else:
            #     year_dict[param] = dct
            if param in year_dict.keys():
                pass
            else:
                year_dict[param] = {}
            year_dict[param][val] = dct

        # Add data to node
        graph.nodes[current_node][y] = year_dict

    # 7 Return the new graph
    return graph


def add_tech_data(graph, node, tech_dfs, tech):
    """
    Add and populate a new technology to `node`'s data within`graph`
    Parameters
    ----------
    node : str
        The name of the node the new technology data will reside in.
    tech : str
        The name of the technology being added to the graph.
    Returns
    -------
    networkx.Graph
        `graph` with the data for `tech` contained within `node`'s node data

    """
    # 1 Copy the current graph & the current tech's dataframe
    t_df = copy.copy(tech_dfs[node][tech])
    graph = copy.copy(graph)

    # 2 Remove the row that indicates this is a service or technology.
    t_df = t_df[~t_df['Parameter'].isin(['Service', 'Technology'])]

    # 3 Group data by year & add to the tech's dictionary
    # NOTE: This is very similar to what we do for nodes (above). However, it differs because
    # we aren't using the value column (its redundant here).
    years = [c for c in t_df.columns if utils.is_year(c)]             # Get Year Columns
    non_years = [c for c in t_df.columns if not utils.is_year(c)]     # Get Non-Year Columns

    for y in years:
        year_df = t_df[non_years + [y]]
        year_dict = {}

        for parameter, value, branch, source, unit, nothing, year_value in zip(*[year_df[c] for c in year_df.columns]):
            dct = {'value': value,
                   'source': source,
                   'branch': branch,
                   'unit': unit,
                   'year_value': year_value}

            if parameter in year_dict.keys():
                if isinstance(year_dict[parameter], list):
                    year_dict[parameter].append(dct)
                else:
                    year_dict[parameter] = [year_dict[parameter], dct]
            else:
                year_dict[parameter] = dct

        # Add technologies key (to the node's data) if needed
        if 'technologies' not in graph.nodes[node][y].keys():
            graph.nodes[node][y]['technologies'] = {}

        # Add the technology specific data for that year
        graph.nodes[node][y]['technologies'][tech] = year_dict

    # 4 Return the new graph
    return graph


def add_edges(graph, node, df):
    """ Add edges associated with `node` to `graph` based on data in `df`.

    Edges are added to the graph based on: (1) if a node is requesting a service provided by
    another node or (2) the relationships implicit in the branch structure used to identify a node. When an
    edge is added to the graph, we also store the edge type ('request_provide', 'structure') in the edge
    attributes. An edge may have more than one type.

    Parameters
    ----------
    node : str
        The name of the node we are creating edges for. Should already be a node within graph.

    df : pandas.DataFrame
        The DataFrame we will use to create edges for `node`.

    Returns
    -------
    networkx.Graph
        An updated version of graph with edges associated with `node` added to the graph.
    """
    # 1 Copy the graph
    graph = copy.copy(graph)

    # 2 Find edges based on requester/provider relationships
    #   These are the edges that exist because one node requests a service to another node
    # Find all nodes node is requesting services from
    providers = df[df['Parameter'] == 'Service requested']['Branch'].unique()

    rp_edges = [(node, p) for p in providers]
    graph.add_edges_from(rp_edges)

    # Add them to the graph
    for e in rp_edges:
        try:
            types = graph.edges[e]['type']
            if 'request_provide' not in types:
                graph.edges[e]['type'] += ['request_provide']
        except KeyError:
            graph.edges[e]['type'] = ['request_provide']

    # 3 Find edge based on branch structure.
    #   e.g. If our node was pyCIMS.Canada.Alberta.Residential we create an edge Alberta->Residential
    # Find the node's parent
    parent = '.'.join(node.split('.')[:-1])
    if parent:
        s_edge = (parent, node)
        graph.add_edge(s_edge[0], s_edge[1])
        # Add the edges type
        try:
            types = graph.edges[s_edge]['type']
            if 'structure' not in types:
                graph.edges[s_edge]['type'] += ['structure']
        except KeyError:
            graph.edges[s_edge]['type'] = ['structure']

    # 4 Return resulting graph
    return graph


def make_edges(graph, node_dfs, tech_dfs):
    """
    Add edges to `graph` using information in `node_dfs` and `tech_dfs`.

    Returns
    -------
    networkx.Graph
        An updated `graph` that contains all edges defined in `node_dfs` and `tech_dfs`.

    """
    for node in node_dfs:
        graph = add_edges(graph, node, node_dfs[node])

    for node in tech_dfs:
        for tech in tech_dfs[node]:
            graph = add_edges(graph, node, tech_dfs[node][tech])

    return graph


def make_nodes(graph, node_dfs, tech_dfs):
    """
    Add nodes to `graph` using `node_dfs` and `tech_dfs`.

    Returns
    -------
    networkx.Graph
        An updated graph that contains all nodes and technologies in node_dfs and tech_dfs.
    """
    # 1 Copy graph
    new_graph = copy.copy(graph)

    # 2 Add nodes to the graph
    # The strategy of trying to add a node to the graph, and then adding it to a "to add" list
    # if that doesn't work, deals with the possibility that nodes may be defined out of order.
    node_dfs_to_add = list(node_dfs.keys())
    while len(node_dfs_to_add) > 0:
        n = node_dfs_to_add.pop(0)
        try:
            new_graph = add_node_data(graph, n, node_dfs)

        except KeyError as e:
            node_dfs_to_add.append(n)

    # 3 Add technologies to the graph
    for node in tech_dfs:
        # Add technologies key to node data
        for tech in tech_dfs[node]:
            new_graph = add_tech_data(graph, node, tech_dfs, tech)

    # Return the graph
    return new_graph
