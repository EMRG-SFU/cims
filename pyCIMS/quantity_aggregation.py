from .quantities import RequestedQuantity


def find_req_prov_children(graph, node, year, tech=None):
    req_prov_children = []

    if tech:
        if 'Service requested' in graph.nodes[node][year]['technologies'][tech]:
            services = graph.nodes[node][year]['technologies'][tech]['Service requested']
            req_prov_children = [data['branch'] for data in services.values()]
    else:
        node_children = graph.successors(node)
        req_prov_children = [c for c in node_children
                             if 'request_provide' in graph.get_edge_data(node, c)['type']]

    return req_prov_children


def find_structural_children(graph, node):
    children = graph.successors(node)
    structural_children = [c for c in children
                           if 'structure' in graph.get_edge_data(node, c)['type']]
    return structural_children


def find_children(graph, node, year=None, tech=None, types=['request_provide', 'structural']):
    children = set()

    if 'request_provide' in types:
        children = children.union(set(find_req_prov_children(graph, node, year, tech)))

    if 'structural' in types:
        children = children.union(set(find_structural_children(graph, node)))

    return list(children)


def find_indirect_quantities(model, child, node, year, tech=None):
    quantities_to_record = []

    child_provided_quantities = model.get_param('provided_quantities', child, year)

    if tech is None:
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_node(node)
    else:
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_tech(node, tech)

    try:
        child_total_quantity_provided = child_provided_quantities.get_total_quantity()
        if child_total_quantity_provided != 0:
            proportion = quantity_provided_to_node_tech / child_total_quantity_provided
            child_requested_quant = model.get_param('requested_quantities', child, year)
            for child_rq_node, amount in child_requested_quant.get_total_quantities_requested().items():
                quantities_to_record.append((child_rq_node, child, proportion*amount))
    except KeyError:
        print(f"Continuing b/c of a loop -- {node}")

    return quantities_to_record


def get_quantities_to_record(model, child, node, year, tech=None):
    child_provided_quant = model.get_param("provided_quantities", child, year)
    quantities_to_record = []

    # Find the quantities provided by child to the node/tech
    if tech is None:
        quantity_provided_to_node_tech = child_provided_quant.get_quantity_provided_to_node(node)
    else:
        quantity_provided_to_node_tech = child_provided_quant.get_quantity_provided_to_tech(node,
                                                                                            tech)
    # Return early if there isn't a positive quantity
    if quantity_provided_to_node_tech <= 0:
        return quantities_to_record

    if child in model.fuels:
        # Record quantities provided directly to the node/tech from child
        quantities_to_record.append((child, child, quantity_provided_to_node_tech))
    else:
        # Record quantities provided indirectly to the node/tech from child
        quantities_to_record = find_indirect_quantities(model, child, node, year, tech)

    return quantities_to_record
