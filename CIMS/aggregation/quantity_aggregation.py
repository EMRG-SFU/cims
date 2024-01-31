"""
Module providing helper functions for calculating a node's requested quantities (i.e. amounts of
fuel which can be attributed to a node).
"""
from ..quantities import RequestedQuantity
from .aggregation_utils import find_req_prov_children


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
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_node(
            node)
    else:
        quantity_provided_to_node_tech = child_provided_quantities.get_quantity_provided_to_tech(
            node, tech)
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


def find_aggregation_quantities(model, year, children_for_aggregation):
    quantities_for_aggregation = []

    # [{child_node, parent_node, parent_tech, aggregate_type}]
    for agg_info in children_for_aggregation:
        parent_node = agg_info['parent_node']
        parent_tech = agg_info['parent_tech']
        child_node = agg_info['child_node']
        agg_type = agg_info['aggregation_type']
        agg_info['quantities'] = []

        if agg_type == 'structural':
            # all quantities requested of child, should be recorded as requested of node
            child_requested_quantities = model.get_param(
                'requested_quantities', child_node, year).get_total_quantities_requested()
            for providing_node, request_amount in child_requested_quantities.items():
                agg_info['quantities'].append((providing_node, request_amount))

        elif agg_type == 'aggregation':
            agg_weight = model.graph.edges[(parent_node, child_node)]['aggregation_weight']

            # all quantities requested of child, should be multiplied by the aggregation weight and
            # recorded as requested of node
            quantities = model.get_param('requested_quantities', child_node,
                                         year).get_total_quantities_requested()
            for providing_node, request_amount in quantities.items():
                agg_info['quantities'].append((providing_node, agg_weight * request_amount))

        elif agg_type == 'request_provide':
            quantities = get_quantities_to_record(model, child_node, parent_node, year,
                                                  tech=parent_tech)
            for providing_node, _, request_amount in quantities:
                agg_info['quantities'].append((providing_node, request_amount))
        else:
            raise ValueError

        quantities_for_aggregation.append(agg_info)

    # [{parent_node, parent_tech, child_node, aggregate_type, quantities: [(providing node, amount)]}]
    return quantities_for_aggregation


def aggregate_quantities(node, quantities_for_aggregation):
    aggregated_quantities = {}

    if len(quantities_for_aggregation) == 0:
        aggregated_quantities[(node, None)] = RequestedQuantity()

    for quantity_dict in quantities_for_aggregation:
        parent_node = quantity_dict['parent_node']
        parent_tech = quantity_dict['parent_tech']
        child_node = quantity_dict['child_node']
        quantities = quantity_dict['quantities']

        # Make sure there are RequestedQuantity() objects to add to
        if (parent_node, None) not in aggregated_quantities:
            aggregated_quantities[(parent_node, None)] = RequestedQuantity()
        if parent_tech is not None and (parent_node, parent_tech) not in aggregated_quantities:
            aggregated_quantities[(parent_node, parent_tech)] = RequestedQuantity()

        # Add to the RequestQuantity() objects
        for providing_node, amount in quantities:
            aggregated_quantities[(parent_node, None)].record_requested_quantity(providing_node,
                                                                                 child_node, amount)
            if parent_tech is not None:
                aggregated_quantities[(parent_node, parent_tech)].record_requested_quantity(
                    providing_node, child_node, amount)
    return aggregated_quantities


def record_quantities(model, aggregate_quantities, year):
    for (node, tech), requested_quant in aggregate_quantities.items():
        if tech is None:
            model.graph.nodes[node][year]['requested_quantities'] = requested_quant
        else:
            model.graph.nodes[node][year]['technologies'][tech][
                'requested_quantities'] = requested_quant
