from __future__ import annotations  # For Type Hinting
import copy
import warnings
import networkx as nx
from typing import List


from . import utils
from . import loop_resolution
from . import cost_curves

IS_SUPPLY_PARAM = 'is supply'
TREE_IDX_PARAM = 'tree index'
COMP_TYPE_PARAM = 'competition type'
EMISSIONS_GWP_PARAM = 'emissions gwp'

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
        if graph.nodes[node][IS_SUPPLY_PARAM]:
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
        if EMISSIONS_GWP_PARAM in data[year]:
            for ghg2 in data[year][EMISSIONS_GWP_PARAM]:
                gwp[ghg2] = data[year][EMISSIONS_GWP_PARAM][ghg2]['year_value']

    return ghg, emission_type, gwp


def get_demand_side_nodes(graph: nx.DiGraph) -> List[str]:
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
    # Find Supply Nodes
    supply_nodes = set([n for n, d in graph.nodes(data=True) if (IS_SUPPLY_PARAM in d) and d[IS_SUPPLY_PARAM]])

    # Find the structural descendants of supply_nodes
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structural' in d['type']]
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
    supply_nodes = [n for n, d in graph.nodes(data=True) if (IS_SUPPLY_PARAM in d) and d[IS_SUPPLY_PARAM]]

    # Find the structural ancestors of the supply nodes
    structural_edges = [(s, t) for s, t, d in graph.edges(data=True) if 'structural' in d['type']]
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
# 3 - Making the graph - Nodes
# ****************************
def find_loops(graph, warn=False):
    loops = list(nx.simple_cycles(graph))
    if warn and len(loops) > 0:
        warning_str = f"Found {len(loops)} loops (see model.loops for the full list)"
        warnings.warn(warning_str)
    return loops

def add_node_data(graph, current_node, node_dfs):
    # args and kwargs for including new fields (e.g. LCC, Service Cost, etc)
    """ Add and populate a new node to `graph`

    Parameters
    ----------
    current_node : str
        The name of the node (branch notation) to add.

    Returns
    -------
    None. But has the consequence of adding node (and its associated data) to 
    graph. 
    """
    # 1 Copy the current graph & the current node's dataframe
    current_node_df = copy.copy(node_dfs[current_node])

    # 2 Add an empty node to the graph
    graph.add_node(current_node)

    # 2.1 Add index for use in the results viewer file
    if TREE_IDX_PARAM not in graph.nodes[current_node]:
        graph.max_tree_index[0] = max(graph.max_tree_index[0], current_node_df.index[0].item())
        graph.nodes[current_node][TREE_IDX_PARAM] = current_node_df.index[0].item() + graph.cur_tree_index[0]

    # 3 Set boolean node constants
    # 3.1 is supply
    graph = _add_node_constant(graph, current_node_df, current_node, IS_SUPPLY_PARAM)
    current_node_df = current_node_df[current_node_df['Parameter'] != IS_SUPPLY_PARAM]
    # 3.2 structural aggregation
    graph = _add_node_constant(graph, current_node_df, current_node, 'structural_aggregation')
    current_node_df = current_node_df[current_node_df['Parameter'] != 'structural_aggregation']
    # 3.3 competition type
    graph = _add_node_constant(graph, current_node_df, current_node, COMP_TYPE_PARAM, required=True)
    current_node_df = current_node_df[current_node_df['Parameter'] != COMP_TYPE_PARAM]

    # 4 Set node's region and sector
    region_list = []
    for item in current_node_df['Region']:
        if item not in region_list and item != None:
            region_list.append(item)
    try:
        graph.nodes[current_node]['region'] = region_list[0]
    except IndexError:
        pass
    current_node_df = current_node_df.drop(columns=['Region'])

    if graph.nodes[current_node][COMP_TYPE_PARAM] not in ['root', 'region']:
        sector_list = []
        for item in current_node_df['Sector']:
            if item not in sector_list and item != None:
                sector_list.append(item)
        try:
            graph.nodes[current_node]['sector'] = sector_list[0]
        except IndexError:
            pass
    current_node_df = current_node_df.drop(columns=['Sector'])
    
    # 5 Find the cost curve function
    comp_type = graph.nodes[current_node][COMP_TYPE_PARAM]
    if comp_type in ['supply - cost curve annual', 'supply - cost curve cumulative']:
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
        year_data_to_update = non_year_data + [current_node_df[year]]
        existing_year_dict = _get_current_year_dict(graph, current_node, year)
        updated_year_dict = _update_year_dict(existing_year_dict, year_data_to_update)
        graph.nodes[current_node][year] = updated_year_dict

    # 7 Return the new graph
    return graph


def _get_current_year_dict(graph, node, year, tech=None):
    if year in graph.nodes[node]:
        node_year_dict = graph.nodes[node][year]

        if tech is not None:
            if 'technologies' not in node_year_dict:
                year_dict = {}
            elif tech not in node_year_dict['technologies']:
                year_dict = {}
            else:
                year_dict = node_year_dict['technologies'][tech]
        else:
            year_dict = node_year_dict
    else:
        year_dict = {}

    return year_dict 


def _get_existing_value(year_dict, param, context, sub_context):
    if context: 
        if sub_context:
            existing_value = year_dict.get(param, {}).get(context, {}).get(sub_context, {}).get('year_value')
        else:
            context_result = year_dict.get(param, {}).get(context, {})
            if isinstance(context_result, dict):
                existing_value = year_dict.get(param, {}).get(context, {}).get('year_value')
            else:
                existing_value = year_dict.get(param, {}).get('year_value')
    else:
        existing_value = year_dict.get(param, {}).get('year_value')

    return existing_value


# def _update(year_dict, param, value_dict, context=None, sub_context=None):
    if context and sub_context:
        year_dict[param][context][sub_context] = value_dict
    elif context:
        year_dict[param][context] = value_dict
    else:
        year_dict[param] = value_dict
    return year_dict
    

def _allowable_update(existing_value, update_value):
    # If the value to update is None
        # if there exists a value -> don't update
        # if there doesn't exist a value -> update
    # If the value to update is "None" -> change value to None & update
    # Otherwise -> update as is
    if isinstance(update_value, dict):
        year_value = update_value['year_value']
        has_existing_val = bool(existing_value)
    else:
        year_value = update_value
        has_existing_val = existing_value is not None
    
    if year_value is None:
        if has_existing_val:
            return False, None
        else:
            return True, update_value
    elif isinstance(year_value, str) and year_value.lower() == 'none':
        if isinstance(update_value, dict):
            update_value['year_value'] = None
            return True, update_value
        else:
            return True, None
    else:
        return True, update_value


def _update_year_dict(existing_year_dict, update_data):
    year_dict = copy.deepcopy(existing_year_dict)
 
    for _, _, param, context, sub_context, target, source, unit, year_value \
        in zip(*update_data):
        value_dict = {
            'context': context,
            'sub_context': sub_context,
            'target': target,
            'source': source,
            'unit': unit,
            'year_value': year_value,
            'param_source': 'model'
            }
        if param not in year_dict:
            year_dict[param] = {}

        # Determine whether update should proceed
        existing_value = _get_existing_value(year_dict, param, context, sub_context)
        update_ok, value_dict = _allowable_update(existing_value, value_dict)
        if not update_ok:
            continue
        
        # If a Context value is present, there are 3 possibilities for what needs to happen
        if context:
            
            # 1. We place our value dictionary inside a nested dictionary keyed
            #    by context & sub-context.
            if sub_context:
                if context not in year_dict[param]:
                    year_dict[param][context] = {}
                year_dict[param][context][sub_context] = value_dict

            # 2. We place our value dictionary keyed only by context.
            elif (value_dict['year_value'] is not None) or \
                ((param in existing_year_dict) and ('year_value' not in existing_year_dict[param])):
                year_dict[param][context] = value_dict

            # 3. We save context as the year_value, which will remain constant
            #    across all years.
            else:
                value_dict['context'] = context
                year_dict[param] = value_dict
        else:
            year_dict[param] = value_dict

    return year_dict


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
    t_df = tech_dfs[node][tech]

    graph.max_tree_index[0] = max(graph.max_tree_index[0], t_df.index[0].item())

    # 2 Remove the row that indicates this is a service or technology.
    t_df = t_df[t_df['Parameter'] != 'technology']

    # 3 Group data by year & add to the tech's dictionary
    # NOTE: This is very similar to what we do for nodes (above). However, it differs because
    # we aren't using the value column (its redundant here).
    years = [c for c in t_df.columns if utils.is_year(c)]             # Get Year Columns
    non_years = [c for c in t_df.columns if not utils.is_year(c)]     # Get Non-Year Columns
    non_year_data = [t_df[c] for c in non_years]
    for year in years:
        year_data_to_update = non_year_data + [t_df[year]]
        existing_year_dict = _get_current_year_dict(graph, node, year, tech=tech)
        updated_year_dict = _update_year_dict(existing_year_dict, year_data_to_update)

        # Add technologies key (to the node's data) if needed
        if 'technologies' not in graph.nodes[node][year].keys():
            graph.nodes[node][year]['technologies'] = {}
        # Add the technology-specific data for that year
        graph.nodes[node][year]['technologies'][tech] = updated_year_dict

        # Add index for use in the results viewer file
        if TREE_IDX_PARAM not in graph.nodes[node][year]['technologies'][tech]:
            graph.nodes[node][year]['technologies'][tech][TREE_IDX_PARAM] = t_df.index[0].item() + graph.cur_tree_index[0]

    # 4 Return the new graph
    return graph


# def add_edges(graph, node, df):
#     """ Add edges associated with `node` to `graph` based on data in `df`.

#     Edges are added to the graph based on: (1) if a node is requesting a service provided by
#     another node or (2) the relationships implicit in the branch structure used to identify a node.
#     When an edge is added to the graph, we also store the edge type ('request_provide', 'structure',
#     or 'aggregation') in the edge attributes. An edge may have more than one type.

#     Parameters
#     ----------
#     node : str
#         The name of the node we are creating edges for. Should already be a node within graph.

#     df : pandas.DataFrame
#         The DataFrame we will use to create edges for `node`.

#     Returns
#     -------
#     networkx.Graph
#         An updated version of graph with edges associated with `node` added to the graph.
#     """
#     # 1 Copy the graph
#     graph = copy.copy(graph)

#     # 2 Find edges based on requester/provider relationships
#     #   These are the edges that exist because one node requests a service to another node
#     # Find all nodes node is requesting services from
#     providers = df[df['Parameter'] == 'service requested']['Target'].unique()

#     rp_edges = [(node, p) for p in providers]
#     graph.add_edges_from(rp_edges)

#     # Add them to the graph
#     for edge in rp_edges:
#         try:
#             types = graph.edges[edge]['type']
#             if 'request_provide' not in types:
#                 graph.edges[edge]['type'] += ['request_provide']
#         except KeyError:
#             graph.edges[edge]['type'] = ['request_provide']

#     # 3 Find edge based on branch structure.
#     # e.g. If our node was CIMS.Canada.Alberta.Residential we create an edge Alberta->Residential
#     # Find the node's parent
#     parent = '.'.join(node.split('.')[:-1])
#     if parent:
#         s_edge = (parent, node)
#         graph.add_edge(s_edge[0], s_edge[1])
#         # Add the edges type
#         try:
#             types = graph.edges[s_edge]['type']
#             if 'structure' not in types:
#                 graph.edges[s_edge]['type'] += ['structure']
#         except KeyError:
#             graph.edges[s_edge]['type'] = ['structure']

#     # 4 Find Aggregation Edges
#     if ('structural_aggregation' in graph.nodes[node]) and (graph.nodes[node]['structural_aggregation']):
#         graph = add_aggregation_edges(graph, rp_edges)

#     # 5 Return resulting graph
#     return graph



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
        try:
            add_node_data(graph, node_data, node_dfs)
        except KeyError:
            node_dfs_to_add.append(node_data)

    # 3 Add technologies to the graph
    for node in tech_dfs:
        # Add technologies key to node data
        for tech in tech_dfs[node]:
            new_graph = add_tech_data(graph, node, tech_dfs, tech)

    # Return the graph
    return new_graph


def _standardize_param_value(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val


def _add_node_constant(graph, node_df, node, parameter, required=False):
    parameter_list = list(node_df[node_df['Parameter'] == parameter]['Context'])

    if len(set(parameter_list)) == 1:
        parameter_val = _standardize_param_value(parameter_list[0])
        if parameter in graph.nodes[node]:
            update_ok, parameter_val = _allowable_update(graph.nodes[node][parameter], parameter_val)
            if not update_ok:
                return graph 
    elif len(set(parameter_list)) > 1:
        raise ValueError(f"{parameter} has too many values at {node}.")
    elif parameter in graph.nodes[node]:
        parameter_val = graph.nodes[node][parameter]
    elif required:
        raise ValueError(f"Required {parameter} value not found at {node}")
    else:
        parameter_val = None

    # Add constant to graph
    graph.nodes[node][parameter] = parameter_val

    return graph


# ****************************
# 4 - Making the graph - Edges
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
    if not isinstance(edge_data['type'], list):
        edge_data['type'] = [edge_data['type']]

    if graph.has_edge(*edge):
        # Update the edge's list of types
        new_edge_types = list(set(graph.edges[edge]['type'] + edge_data['type']))
        edge_data['type'] = new_edge_types
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
            edge_data = {'type': 'structural'}
            edges.append((edge, edge_data))
    
    elif edge_type == 'request_provide':
        providers = df[df['Parameter'] == 'service requested']['Target'].unique()
        edges += [((node, p), {'type': 'request_provide'}) for p in providers]

    elif edge_type == 'aggregation':

        agg_children = df[df['Parameter'] == 'aggregation requested']['Target'].unique()

        for agg_child in agg_children:
            # For each "aggregation requested" line at node, add the following edges
            #   (1) 1 weighted edge between node & aggregation requested target (i.e. node -> child)
            #   (2) 0 weighted edge between all other parents of the aggregation requested target & the aggregation requested target (i.e. other parent of child -> child)

            # (1) node -> child {aggregation_weight: 1}
            edges.append(((node, agg_child), {'type': 'aggregation', 'aggregation_weight': 1}))

            for agg_child_parent in graph.predecessors(agg_child):
                if agg_child_parent == node:
                    continue
                elif graph.edges[(agg_child_parent, agg_child)].get('aggregation_weight') == 1:
                    pass # already set as an aggregating node, don't zero it out
                else:
                    # (2) other parents -> child {aggregation_weight: 0}
                    edges.append(((agg_child_parent, agg_child), {'type': 'aggregation', 'aggregation_weight': 0}))

    else:
        raise ValueError(
            f"{edge_type=} not recognized. Please use \"structure\", \"request_provide\", or \"aggregation\".")
        
    return edges