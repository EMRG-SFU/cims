from ..utils.parameter.construction import create_value_dict
from ..utils.parameter import list as PARAM

def find_req_prov_children(graph, node, year, tech=None):
    """
    Find a node/tech's children with which there exists a request/provide relationship."""
    req_prov_children = []

    if tech:
        if PARAM.service_requested in graph.nodes[node][year][PARAM.technologies][tech]:
            targets = graph.nodes[node][year][PARAM.technologies][tech][PARAM.service_requested]
            req_prov_children = [t for t in targets]
    else:
        node_children = graph.successors(node)
        req_prov_children = [c for c in node_children
                             if 'request_provide' in graph.get_edge_data(node, c)[PARAM.edge_type]]

    return req_prov_children


def find_structural_children(graph, node):
    """
    Find a node's structural children in graph.
    """
    children = graph.successors(node)
    structural_children = [c for c in children
                           if 'structural' in graph.get_edge_data(node, c)[PARAM.edge_type]]
    return structural_children


def find_aggregation_children(graph, node):
    """
    Find a node's aggregation children in graph.
    """
    children = graph.successors(node)
    agg_children = [c for c in children
                    if 'aggregation' in graph.get_edge_data(node, c)[PARAM.edge_type]]

    return agg_children


def find_children(graph, node, year=None, tech=None,
                  structural=False, request_provide=False, aggregation=False):
    """
    Find node/tech's children, connected by the specified relationship types.
    """
    children = set()
    if structural:
        children = children.union(set(find_structural_children(graph, node)))

    if request_provide:
        children = children.union(set(find_req_prov_children(graph, node, year, tech)))

    if aggregation:
        children = children.union(set(find_aggregation_children(graph, node)))

    return list(children)


def _add_children_for_aggregation(children, parent_node, parent_tech, aggregation_type):
    child_dicts = {}
    for child in children:
        child_dicts[child] = \
            {'parent_node': parent_node,
             'parent_tech': parent_tech,
             'child_node': child,
             'aggregation_type': aggregation_type}

    return child_dicts


def find_children_for_aggregation(model, node, year, include_self=False):
    children_for_aggregation = []

    competition_type = model.get_param(PARAM.competition_type, node)

    ###
    if competition_type in ['root', 'region']:
        temp_children = {}

        structural_children = find_children(model.graph, node, structural=True)
        temp_children.update(_add_children_for_aggregation(structural_children,
                                                           parent_node=node,
                                                           parent_tech=None,
                                                           aggregation_type='structural'))


        aggregation_children = find_children(model.graph, node, aggregation=True)
        temp_children.update(_add_children_for_aggregation(
            aggregation_children, parent_node=node, parent_tech=None,
            aggregation_type='aggregation'))

        children_for_aggregation += temp_children.values()

    elif PARAM.technologies in model.graph.nodes[node][year]:
        for tech in model.graph.nodes[node][year][PARAM.technologies]:
            temp_children = {}

            req_prov_children = find_children(model.graph, node, year, tech, request_provide=True)
            temp_children.update(
                _add_children_for_aggregation(req_prov_children, parent_node=node, parent_tech=tech,
                                              aggregation_type='request_provide'))

            aggregation_children = find_children(model.graph, node, aggregation=True)
            temp_children.update(
                _add_children_for_aggregation(aggregation_children, parent_node=node,
                                              parent_tech=tech,
                                              aggregation_type='aggregation'))

            children_for_aggregation += temp_children.values()

    else:
        temp_children = {}
        req_prov_children = find_children(model.graph, node, year, request_provide=True)
        temp_children.update(
            _add_children_for_aggregation(req_prov_children, parent_node=node, parent_tech=None,
                                          aggregation_type='request_provide'))

        aggregation_children = find_children(model.graph, node, aggregation=True)
        temp_children.update(
            _add_children_for_aggregation(aggregation_children, parent_node=node, parent_tech=None,
                                          aggregation_type='aggregation'))

        children_for_aggregation += temp_children.values()

    if include_self:
        if PARAM.technologies in model.graph.nodes[node][year]:
            for tech in model.graph.nodes[node][year][PARAM.technologies]:
                children_for_aggregation += _add_children_for_aggregation(
                    [None], parent_node=node, parent_tech=tech,
                    aggregation_type='self').values()
        else:
            children_for_aggregation += _add_children_for_aggregation(
                [None], parent_node=node, parent_tech=None,
                aggregation_type='self').values()

    return children_for_aggregation


def record_aggregate_values(model, node, year, aggregate_values, rate_param, base_class):
    if (node, None) not in aggregate_values:
        aggregate_values[(node, None)] = base_class()

    for (node, tech), aggregate_val in aggregate_values.items():
        if tech is None:
            model.graph.nodes[node][year][rate_param] = create_value_dict(aggregate_val)
        else:
            model.graph.nodes[node][year][PARAM.technologies][tech][rate_param] = (
                create_value_dict(aggregate_val))
