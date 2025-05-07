import copy

from ..parameter import list as PARAM
from ..model_description import column_list as COL

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


def add_edges_of_one_type(graph, node, df, edge_type):
    # 1. Copy the graph
    graph = copy.copy(graph)

    # 2. Find the Edges
    edges_to_add = find_edges(graph, node, df, edge_type)

    # 3. Add or update the edges to the graph
    for edge, edge_data in edges_to_add:
        add_or_update_edge(graph, edge, edge_data)

    return graph


def add_aggregation_edges_from_dfs(graph, node_dfs):
    # From Node DFs
    for node in node_dfs:
        graph = add_edges_of_one_type(graph,
                                      node,
                                      node_dfs[node],
                                      edge_type='aggregation')

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