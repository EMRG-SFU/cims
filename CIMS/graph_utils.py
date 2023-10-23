from __future__ import annotations  # For Type Hinting
import copy
import warnings
import networkx as nx
from typing import List


from . import utils
from . import loop_resolution
from . import cost_curves


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
        The name of the node to begin our search from. Must be contained within `graph`. (e.g.
        'CIMS.Canada.Alberta')
    parameter : str
        The name of the parameter whose value is being found. (e.g. 'Sector type')
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


def get_fuels(graph):
    """
    Find the nodes which have been specified as fuels in the model description.
    Returns
    -------
    List
        A list containing the names of nodes which supply fuels and markets.
    """
    fuels = []
    for node in graph.nodes:
        if graph.nodes[node]['is fuel']:
            fuels.append(node)
    return fuels


def get_GHG_and_Emissions(graph, year):
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
        if 'technologies' in data[year]:
            techs = data[year]['technologies']
            for tech in techs:
                tech_data = data[year]['technologies'][tech]
                if 'emissions' in tech_data or 'emissions_removal' in tech_data:
                    if 'emissions' in tech_data:
                        ghg_list = data[year]['technologies'][tech]['emissions']
                    else:
                        ghg_list = data[year]['technologies'][tech]['emissions_removal']

                    node_ghg = [ghg for ghg in ghg_list]
                    node_emission_type = [emission_type for emission_record in ghg_list.values() for
                                          emission_type in emission_record]

                    ghg = list(set(ghg + node_ghg))
                    emission_type = list(set(emission_type + node_emission_type))

        # Emissions from a supply node
        elif 'emissions' in data[year] or 'emissions_removal' in data[year]:
            if 'emissions' in data[year]:
                ghg_dict = data[year]['emissions']
            else:
                ghg_dict = data[year]['emissions_removal']

            node_ghg = [ghg for ghg in ghg_dict.keys()]

            node_emission_type = [emission_type for emission_record in ghg_dict.values() for
                                  emission_type in emission_record]

            ghg = list(set(ghg + node_ghg))
            emission_type = list(set(emission_type + node_emission_type))

        #GWP from CIMS node
        if 'emissions gwp' in data[year]:
            for ghg2 in data[year]['emissions gwp']:
                gwp[ghg2] = data[year]['emissions gwp'][ghg2]['year_value']

    return ghg, emission_type, gwp


def get_demand_nodes(graph: nx.DiGraph) -> List[str]:
    """
    Find the nodes to use for demand-side traversals. The returned list of nodes will include all
    nodes whose "node type" attribute is not "supply.

    Parameters
    ----------
    graph :
        The graph whose demand tree will be returned.

    Returns
    -------
    A subgraph of graph which includes only non-supply nodes.
    """
    # Find Fuels
    fuels = set([n for n, d in graph.nodes(data=True) if ('is fuel' in d) and d['is fuel']])

    # Find the structural descendants of fuels
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structure' in d['type']]
    structural_graph = graph.edge_subgraph(structural_edges)

    descendants = set()
    for fuel in fuels:
        fuel_structural_descendants = nx.descendants(structural_graph, fuel)
        descendants = descendants.union(fuel_structural_descendants)

    # Return all the nodes which are neither fuels nor their descendants
    return list(set(graph.nodes).difference(set(fuels).union(descendants)))


def get_supply_nodes(graph: nx.DiGraph) -> List[str]:
    """
    Find the nodes to use for supply-side traversals. The returned list of nodes will include all
    nodes whose "node type" is "supply" and any node which is a structural ancestor of these supply
    nodes.

    Parameters
    ----------
    graph :
        The graph whose supply tree will be returned.

    Returns
    -------
    A subgraph of graph which includes only supply nodes & their structural ancestors.
    """
    # Find Fuels
    fuels = [n for n, d in graph.nodes(data=True) if ('is fuel' in d) and d['is fuel']]

    # Find the structural ancestors of the supply nodes
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structure' in d['type']]
    structural_graph = graph.edge_subgraph(structural_edges)

    ancestors = set()
    descendants = set()
    for fuel in fuels:
        structural_ancestors = nx.ancestors(structural_graph, fuel)
        ancestors = ancestors.union(structural_ancestors)

        structural_descendants = nx.descendants(structural_graph, fuel)
        descendants = descendants.union(structural_descendants)

    return fuels + list(ancestors) + list(descendants)


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
            warnings.warn("Found a Loop -- ")
            # Resolve a loop
            cycles = nx.simple_cycles(sg_cur)
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
            warnings.warn("Found a Loop")
            # Resolve a loop
            cycles = nx.simple_cycles(sg_cur)
            n_cur = loop_resolution_func(cycles, dist_from_root, **kwargs)

            # Process chosen node in the sub-graph, using estimated values from their parents
            node_process_func(sub_graph, n_cur, *args, **kwargs)

        sg_cur.remove_node(n_cur)


def find_next_node(degrees):
    for node, degree in degrees:
        if degree == 0:
            return node


# **************************
# 3 - Making the graph
# **************************
def add_node_data(graph, current_node, node_dfs):
    # args and kwargs for including new fields (e.g. LCC, Service Cost, etc)
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

    # 3 Find whether node is a fuel
    is_fuel_rows = current_node_df[current_node_df['Parameter'] == 'is fuel']['Context']
    is_fuel = ((is_fuel_rows.all() and not is_fuel_rows.empty) or
               (('is fuel' in graph.nodes[current_node]) and (graph.nodes[current_node]['is fuel'])))
    graph.nodes[current_node]['is fuel'] = is_fuel
    # Drop fuel row
    current_node_df = current_node_df[current_node_df['Parameter'] != 'is fuel']

    # 4 Find node's competition type. (If there is one)
    comp_list = list(current_node_df[current_node_df['Parameter'] == 'competition type']['Context'])
    if len(set(comp_list)) == 1:
        comp_type = comp_list[0].lower()
        graph.nodes[current_node]['competition type'] = comp_type
    elif len(set(comp_list)) > 1:
        raise ValueError("TOO MANY COMPETITION TYPES")
    elif 'competition type' in graph.nodes[current_node]:
        comp_type = graph.nodes[current_node]['competition type']

    # Get rid of competition type row
    current_node_df = current_node_df[current_node_df['Parameter'] != 'competition type']

    # 5 Find the cost curve function
    if comp_type in ['fuel - cost curve annual', 'fuel - cost curve cumulative']:
        cc_func = cost_curves.build_cost_curve_function(current_node_df)
        graph.nodes[current_node]['cost_curve_function'] = cc_func

        # Get rid of cost curve rows
        cost_curve_params = ['cost curve quantity', 'cost curve price']
        current_node_df = current_node_df[~current_node_df['Parameter'].isin(cost_curve_params)]

    # 6 For the remaining data, group by year.
    years = [c for c in current_node_df.columns if utils.is_year(c)]          # Get Year Columns
    non_years = [c for c in current_node_df.columns if not utils.is_year(c)]  # Get Non-Year Columns

    non_year_data = [current_node_df[c] for c in non_years]
    for year in years:
        current_year_data = non_year_data + [current_node_df[year]]
        if year in graph.nodes[current_node]:
            year_dict = graph.nodes[current_node][year]
        else:
            year_dict = {}
        for param, context, sub_context, branch, source, unit, _, year_value in zip(*current_year_data):
            dct = {'context': context,
                   'sub_context': sub_context,
                   'branch': branch,
                   'source': source,
                   'unit': unit,
                   'year_value': year_value,
                   'param_source': 'model'}

            if param not in year_dict:
                year_dict[param] = {}

            # If a Context value is present, there are 3 possibilities for what needs to happen
            if context:
                # 1. We need to place our information in a nested dictionary, keyed by the
                # context and the sub-context.
                if sub_context:
                    if context not in year_dict[param]:
                        year_dict[param][context] = {}
                    year_dict[param][context][sub_context] = dct

                # 2. We need to place the information in a dictionary, keyed by only context. In
                #    these cases, sub_context isn't defined, but there will be values in year_
                #    value that we need to record.
                elif year_value is not None:
                    if 'year_value' in year_dict[param]:
                        year_dict[param] = {year_dict[param]['context']: year_dict[param]}
                    year_dict[param][context] = dct

                # No year value has been specified, but there are other instances of this parameter
                # already saved to the model for this node
                elif year_dict[param]:
                    # Check that it's not a base dictionary.
                    if 'year_value' in year_dict[param]:
                        year_dict[param] = {year_dict[param]['context']: year_dict[param]}
                    year_dict[param][context] = dct

                # 3. Context contains the value we actually want to record. Additionally, this
                # value will remain constant across all years.
                else:
                    if context in year_dict[param]:
                        raise ValueError(
                            f'Multiple values have been set for {param}. Please rectify'
                            f'this.')
                    # 1. year_value isn't present, so context is the actual value we want to
                    # record.
                    dct['context'] = context
                    year_dict[param] = dct
            else:
                year_dict[param] = dct

        # 6 Add data to node
        graph.nodes[current_node][year] = year_dict

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
    t_df = t_df[~t_df['Parameter'].isin(['service', 'technology'])]

    # 3 Group data by year & add to the tech's dictionary
    # NOTE: This is very similar to what we do for nodes (above). However, it differs because
    # we aren't using the value column (its redundant here).
    years = [c for c in t_df.columns if utils.is_year(c)]             # Get Year Columns
    non_years = [c for c in t_df.columns if not utils.is_year(c)]     # Get Non-Year Columns

    non_year_data = [t_df[c] for c in non_years]
    for year in years:
        current_year_data = non_year_data + [t_df[year]]
        try:
            year_dict = graph.nodes[node][year]['technologies'][tech]
        except KeyError:
            year_dict = {}

        for param, context, sub_context, branch, source, unit, _, year_value in zip(*current_year_data):
            dct = {'context': context,
                   'sub_context': sub_context,
                   'branch': branch,
                   'source': source,
                   'unit': unit,
                   'year_value': year_value,
                   'param_source': 'model'}

            # If the parameter isn't in the year_dict yet, add it
            if param not in year_dict:
                year_dict[param] = {}

            # If Context is present, there are 3 possibilities for what needs to happen
            if context:
                # 1. We need to place our information in a nested dictionary, keyed by the
                # context and the sub-context.
                if sub_context:
                    if context not in year_dict[param]:
                        year_dict[param][context] = {}
                    year_dict[param][context][sub_context] = dct

                # 2. We need to place the information in a dictionary, keyed by only context. In
                #    these cases, sub_context isn't defined, but there will be values in year_
                #    value that we need to record.
                elif year_value is not None:
                    year_dict[param][context] = dct
                # 3. Context contains the value we actually want to record. Additionally, this
                # value will remain constant across all years.
                else:
                    if context in year_dict[param]:
                        raise ValueError(
                            f'Multiple values have been set for {param}. Please rectify'
                            f'this.')
                    dct['context'] = context
                    year_dict[param] = dct
            else:
                year_dict[param] = dct

        # Add technologies key (to the node's data) if needed
        if 'technologies' not in graph.nodes[node][year].keys():
            graph.nodes[node][year]['technologies'] = {}

        # Add the technology specific data for that year
        graph.nodes[node][year]['technologies'][tech] = year_dict

    # 4 Return the new graph
    return graph


def add_edges(graph, node, df):
    """ Add edges associated with `node` to `graph` based on data in `df`.

    Edges are added to the graph based on: (1) if a node is requesting a service provided by
    another node or (2) the relationships implicit in the branch structure used to identify a node.
    When an edge is added to the graph, we also store the edge type ('request_provide', 'structure')
    in the edge attributes. An edge may have more than one type.

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
    providers = df[df['Parameter'] == 'service requested']['Branch'].unique()

    rp_edges = [(node, p) for p in providers]
    graph.add_edges_from(rp_edges)

    # Add them to the graph
    for edge in rp_edges:
        try:
            types = graph.edges[edge]['type']
            if 'request_provide' not in types:
                graph.edges[edge]['type'] += ['request_provide']
        except KeyError:
            graph.edges[edge]['type'] = ['request_provide']

    # 3 Find edge based on branch structure.
    # e.g. If our node was CIMS.Canada.Alberta.Residential we create an edge Alberta->Residential
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


def make_or_update_edges(graph, node_dfs, tech_dfs):
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


def make_or_update_nodes(graph, node_dfs, tech_dfs):
    """
    Add nodes to `graph` using `node_dfs` and `tech_dfs`. If node already exists, update it.

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
        node_data = node_dfs_to_add.pop(0)
        if node_data in graph.nodes:
            new_graph = add_node_data(graph, node_data, node_dfs)
        else:
            try:
                new_graph = add_node_data(graph, node_data, node_dfs)

            except KeyError:
                node_dfs_to_add.append(node_data)

    # 3 Add technologies to the graph
    for node in tech_dfs:
        # Add technologies key to node data
        for tech in tech_dfs[node]:
            new_graph = add_tech_data(graph, node, tech_dfs, tech)

    # Return the graph
    return new_graph
