"""
Module providing helper functions for calculating a node's requested quantities (i.e. amounts of
fuel which can be attributed to a node).
"""


def find_req_prov_children(graph, node, year, tech=None):
    """
    Find a node/tech's children with which there exists a request/provide relationship."""
    req_prov_children = []

    if tech:
        if 'service requested' in graph.nodes[node][year]['technologies'][tech]:
            services = graph.nodes[node][year]['technologies'][tech]['service requested']
            req_prov_children = [data['branch'] for data in services.values()]
    else:
        node_children = graph.successors(node)
        req_prov_children = [c for c in node_children
                             if 'request_provide' in graph.get_edge_data(node, c)['type']]

    return req_prov_children


def find_structural_children(graph, node):
    """
    Find a node's structural children in graph.
    """
    children = graph.successors(node)
    structural_children = [c for c in children
                           if 'structure' in graph.get_edge_data(node, c)['type']]
    return structural_children


def find_children(graph, node, year=None, tech=None, structural=False, request_provide=False):
    """
    Find node/tech's children, connected by the specified relationship types.
    """
    children = set()
    if structural:
        children = children.union(set(find_structural_children(graph, node)))

    if request_provide:
        children = children.union(set(find_req_prov_children(graph, node, year, tech)))

    return list(children)


def get_quantities_to_record(model, child, node, year, tech=None):
    """
    Find the list of quantities to be recorded for a node/tech based on it's request for services
    from child.
    """
    child_provided_quantities = model.get_param("provided_quantities", child, year=year)
    quantities_to_record = []

    # Find the quantities provided by child to the node/tech
    # Note, the result of get_total_quantity() will not equal the sum across
    # self.provided_quantities values when distributed supply is greater than the sum of
    # positive provided quantities.
    if tech is None:
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_node(node)
    else:
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_tech(node,                                                                                 tech)
    # Return early if there isn't a positive quantity
    if quantity_provided_to_node_tech <= 0:
        return quantities_to_record

    if child in model.fuels:
        # Record quantities provided directly to the node/tech from child
        quantities_to_record.append((child, child, quantity_provided_to_node_tech))
    else:
        # Find the indirectly requested quantities attributable to node/tech by way of its
        # request of services from child.
        try:
            child_total_quantity_provided = child_provided_quantities.get_total_quantity()
            if child_total_quantity_provided != 0:
                proportion = child_provided_quantities.calculate_proportion(node, tech)
                child_requested_quant = model.get_param('requested_quantities', child, year=year)
                quantities_requested = child_requested_quant.get_total_quantities_requested()
                for child_rq_node, amount in quantities_requested.items():
                    quantities_to_record.append((child_rq_node, child, proportion * amount))
        except KeyError:
            print(f"Continuing b/c of a loop -- {node}")

    return quantities_to_record


def get_direct_distributed_supply(model, node, year, tech=None):
    """
    Find the distributed supply originating at the node/tech.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the information of interest.
    node : str
        The node whose distributed supply we are interested in finding.
    year : str
        The year in which we want to find
    tech : str, optional
        Optional. If supplied, the tech whose distributed supply we are finding. Otherwise,
        we will look for distributed supply at the node itself.

    Returns
    -------
    List[(str, float)]
        A list of tuples is returned. Each tuple has two entries. The first correspond to the
        service who is being provided through distributed supply. The second is the amount of that
        service being provided.

    """
    children = find_req_prov_children(model.graph, node, year, tech)

    distributed_supply = []

    for child in children:
        child_provided_quantities = model.get_param("provided_quantities", child, year=year)

        # Find the quantities provided by child to the node/tech
        if tech is None:
            quantity_provided_to_node_tech = \
                child_provided_quantities.get_quantity_provided_to_node(node)
        else:
            quantity_provided_to_node_tech = \
                child_provided_quantities.get_quantity_provided_to_tech(node, tech)

        if quantity_provided_to_node_tech < 0:
            # Record quantities provided directly to the node/tech from child
            distributed_supply.append((child, -1 * quantity_provided_to_node_tech))

    return distributed_supply
