from ..emissions import Emissions, EmissionsCost
from ..utils import create_value_dict
from .aggregation_utils import find_children_for_aggregation


def _abstract_direct_emission_aggregation(base_emission_class, model, graph, node, year,
                                          rate_param, total_param):
    provided_quantities = model.get_param('provided_quantities', node, year).get_total_quantity()

    node_total = base_emission_class()

    if 'technologies' in graph.nodes[node][year]:
        for tech in graph.nodes[node][year]['technologies']:
            tech_total = base_emission_class()
            total_ms = model.get_param('total_market_share', node, year, tech=tech)
            direct = model.get_param(rate_param, node, year, tech=tech)
            if direct is not None:
                total_direct = direct * provided_quantities * total_ms
                node_total += total_direct
            value_dict = create_value_dict(tech_total)
            graph.nodes[node][year]['technologies'][tech][total_param] = value_dict
    value_dict = create_value_dict(node_total)
    graph.nodes[node][year][total_param] = value_dict


def aggregate_direct_emissions(model, graph, node, year, rate_param, total_param):
    _abstract_direct_emission_aggregation(Emissions, model, graph, node, year,
                                          rate_param, total_param)


def aggregate_direct_emissions_cost(model, graph, node, year, rate_param, total_param):
    _abstract_direct_emission_aggregation(EmissionsCost, model, graph, node, year,
                                          rate_param, total_param)


def aggregate_cumulative_emissions(model, _, node, year, rate_param, total_param):
    # Find children to aggregate across
    children_for_aggregation = find_children_for_aggregation(model, node, year, include_self=True)

    # Calculate Cumulative Emission Rates
    cumulative_emission_rate = _find_cumulative_emission_rates(model, year,
                                                               children_for_aggregation, rate_param)

    # Aggregate Cumulative Emission Rates
    aggregate_emissions = _aggregate_cumulative_emission_rates(model, year,
                                                               cumulative_emission_rate)

    # Record Cumulative Emission Rates
    _record_cumulative_emission_rates(model, node, year, aggregate_emissions, rate_param)

    # Record Total Cumulative Emissions
    _record_total_cumulative_emissions(model, node, year, rate_param, total_param)


def _find_cumulative_emission_rates(model, year, children_for_aggregation, cumul_rate_param):
    values_to_aggregate = []

    # [{child_node, parent_node, parent_tech, aggregate_type}]
    for agg_info in children_for_aggregation:
        agg_type = agg_info['aggregation_type']

        if agg_type == 'self':
            # Record Emissions generated at the technology itself -- e.g. net_emissions_rate
            base_rate_param = cumul_rate_param.split('cumul_')[-1]
            aggregate_value = _find_cumulative_rate_via_self(
                model, agg_info['parent_node'], year, agg_info['parent_tech'], base_rate_param)

        elif agg_type == 'structural':
            # All emissions at the child node should be recorded at the parent node
            aggregate_value = _find_cumulative_rate_via_structural_edge(
                model, agg_info['child_node'], year, cumul_rate_param)

        elif agg_type == 'aggregation':
            aggregate_value = _find_cumulative_rate_via_aggregation_edge(
                model, agg_info['parent_node'], agg_info['child_node'], year, cumul_rate_param)

        elif agg_type == 'request_provide':
            aggregate_value = _find_cumulative_rate_via_request_provide_edge(
                model, agg_info, year, cumul_rate_param)

        else:
            raise ValueError

        agg_info['aggregate_values'] = aggregate_value
        values_to_aggregate.append(agg_info)

    # [{parent_node, parent_tech, child_node, aggregate_type, aggregate_values: [Emissions]}]
    return values_to_aggregate


def _aggregate_cumulative_emission_rates(model, year, emissions_for_aggregation):
    total_cumulative_emissions = {}

    for emission_dict in emissions_for_aggregation:
        parent_node = emission_dict['parent_node']
        parent_tech = emission_dict['parent_tech']
        agg_values = emission_dict['aggregate_values']

        if (parent_node, None) not in total_cumulative_emissions:
            total_cumulative_emissions[(parent_node, None)] = Emissions()
        if (parent_tech is not None) and (
                parent_node, parent_tech) not in total_cumulative_emissions:
            total_cumulative_emissions[(parent_node, parent_tech)] = Emissions()

        if parent_tech is not None:
            total_market_share = model.get_param('total_market_share', parent_node, year,
                                                 tech=parent_tech)
            total_cumulative_emissions[(parent_node, None)] += agg_values * total_market_share
            total_cumulative_emissions[(parent_node, parent_tech)] += agg_values
        else:
            total_cumulative_emissions[(parent_node, None)] += agg_values

    return total_cumulative_emissions


def _record_cumulative_emission_rates(model, node, year, aggregate_emissions, rate_param):
    if (node, None) not in aggregate_emissions:
        aggregate_emissions[(node, None)] = Emissions()

    for (node, tech), emissions in aggregate_emissions.items():
        if tech is None:
            model.graph.nodes[node][year][rate_param] = create_value_dict(emissions)

        else:
            model.graph.nodes[node][year]['technologies'][tech][rate_param] = (
                create_value_dict(emissions))


def _record_total_cumulative_emissions(model, node, year, rate_param, total_param):
    provided_quantities = model.get_param('provided_quantities', node, year).get_total_quantity()
    emissions_rate = model.get_param(rate_param, node, year)
    model.graph.nodes[node][year][total_param] = create_value_dict(emissions_rate *
                                                                   provided_quantities)


def _find_cumulative_rate_via_self(model, node, year, tech, base_rate_param):
    direct_emissions = model.get_param(base_rate_param, node, year,
                                       tech=tech)
    if direct_emissions is None:
        direct_emissions = Emissions()

    return direct_emissions


def _find_cumulative_rate_via_structural_edge(model, node, year, cumulative_rate_param):
    return model.get_param(cumulative_rate_param, node, year)


def _find_cumulative_rate_via_aggregation_edge(model, parent_node, child_node, year,
                                               cumulative_rate_param):
    # all emissions requested of the child node should be multiplied by the aggregation
    # weight and recorded at the parent node
    agg_weight = model.graph.edges[(parent_node, child_node)]['aggregation_weight']

    return model.get_param(cumulative_rate_param, child_node, year) * agg_weight


def _find_cumulative_rate_via_request_provide_edge(model, agg_info, year, cumulative_rate_param):
    parent_node = agg_info['parent_node']
    parent_tech = agg_info['parent_tech']
    child_node = agg_info['child_node']

    emission_rates = Emissions()

    req_ratio = model.get_param('service requested', parent_node, year, tech=parent_tech,
                                context=child_node.split('.')[-1])

    child_emissions_rate = model.get_param(cumulative_rate_param, child_node, year)
    if child_emissions_rate is not None:
        emission_rates += child_emissions_rate * req_ratio

    # If the child produces emissions (e.g. requests a fuel which has emissions), we must
    # multiply the child's direct emission rate by the service_request ratio.
    direct_emission_rate_param = cumulative_rate_param.split('cumul_')[-1]
    if direct_emission_rate_param in model.graph.nodes[child_node][year]:
        child_direct_emissions_rate = model.get_param(direct_emission_rate_param, child_node, year)
        emission_rates += child_direct_emissions_rate * req_ratio

    return emission_rates
