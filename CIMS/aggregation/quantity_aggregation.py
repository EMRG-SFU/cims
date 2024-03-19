"""
Module providing helper functions for calculating a node's requested quantities (i.e. amounts of
fuel which can be attributed to a node).
"""
from ..quantities import RequestedQuantity
from .aggregation_utils import record_aggregate_values, find_children_for_aggregation


def aggregate_requested_quantities(model, node, year):
    """
    Calculates and records fuel quantities attributable to a node in the specified year. Fuel
    quantities can be attributed to a node in 3 ways:

    (1) via request/provide relationships - any fuel quantity directly requested of the node
    (e.g. Lighting requests services directly from Electricity) and fuel quantities that
    are indirectly requested, but can be attributed to the node (e.g. Housing requests
    Electricity via its request of Lighting).

    (2) via structural relationships - fuel nodes pass indirect quantities to their structural
    parents, rather than request/provide parents. Additionally, root & region nodes collect
    quantities via their structural children, rather than their request/provide children.

    (3) via weighted aggregate relationships - if specified in the model description, nodes will
    aggregate quantities structurally. For example, if a market node has
    "structural_aggregation" turned on, any quantities (direct or in-direct) from the market
    children aggregate through structural parents (i.e. BC.Natural Gas) instead of the market
    which it has a request/provide relationship with (CAN.Natural Gas).

    This method was built to be used with the bottom-up traversal method, which ensures a node
    is only visited once all its children have been visited (except when needs to break a loop).
    """
    # Find children
    children_for_aggregration = find_children_for_aggregation(model, node, year)

    # Find Quantities
    quantities = _find_aggregation_quantities(model, year, children_for_aggregration)

    # Aggregate Quantities
    agg_quantities = _aggregate_quantities(node, quantities)

    # Record Quantities
    record_aggregate_values(model, node, year, agg_quantities, 'requested_quantities', RequestedQuantity)
    # _record_quantities(model, agg_quantities, year)


def _get_quantities_to_record(model, child, node, year, tech=None):
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


def _find_aggregation_quantities(model, year, children_for_aggregation):
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
            quantities = _get_quantities_to_record(model, child_node, parent_node, year,
                                                   tech=parent_tech)
            for providing_node, _, request_amount in quantities:
                agg_info['quantities'].append((providing_node, request_amount))
        else:
            raise ValueError

        quantities_for_aggregation.append(agg_info)

    # [{parent_node, parent_tech, child_node, aggregate_type, quantities: [(providing node, amount)]}]
    return quantities_for_aggregation


def _aggregate_quantities(node, quantities_for_aggregation):
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
