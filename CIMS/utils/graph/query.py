from typing import List

from CIMS import vintage_weighting
from CIMS.utils.parameter import list as PARAM
from CIMS.utils.parameter.construction import create_value_dict
from ..parameter import list as PARAM
import networkx as nx


def search_nodes(search_term, graph):
    """
    Search `graph` to find the nodes which contain `search_term` in the node name's final component.
    Not case sensitive.

    Parameters
    ----------
    search_term : str
        The search term.

    Returns
    -------
    list [str]
        A list of node names (branch notation) whose last component contains `search_term`.
    """
    def search(name):
        components = name.split('.')
        last_comp = components[-1]
        return search_term.lower() in last_comp.lower()

    return [n for n in graph.nodes if search(n)]


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


def get_services_requested(model, node, year, tech=None, use_vintage_weighting=False):
    if tech:
        if PARAM.service_requested not in model.graph.nodes[node][year][PARAM.technologies][tech]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year][PARAM.technologies][tech][PARAM.service_requested]
            if use_vintage_weighting:
                weighted_services = {}
                for target in services_requested:
                    weighted_req_ratio = vintage_weighting.calculate_vintage_weighted_parameter(PARAM.service_requested, model, node, year, tech=tech, context=target)
                    weighted_services[target] = create_value_dict(year_val=weighted_req_ratio, target=target, param_source='vintage_weighting')
                services_requested = weighted_services

    else:
        if PARAM.service_requested not in model.graph.nodes[node][year]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year][PARAM.service_requested]

    return services_requested
