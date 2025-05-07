from ..emissions import Emissions, EmissionsCost
from ..utils.parameter import list as PARAM
from ..utils.parameter.construction import create_value_dict


def _abstract_direct_emission_aggregation(base_emission_class, model, graph, node, year,
                                          rate_param, total_param):
    provided_quantities = model.get_param(PARAM.provided_quantities, node, year).get_total_quantity()

    node_total = base_emission_class()

    if PARAM.technologies in graph.nodes[node][year]:
        for tech in graph.nodes[node][year][PARAM.technologies]:
            tech_total = base_emission_class()
            total_ms = model.get_param(PARAM.total_market_share, node, year, tech=tech)
            direct = model.get_param(rate_param, node, year, tech=tech)
            if direct is not None:
                total_direct = direct * provided_quantities * total_ms
                node_total += total_direct
            value_dict = create_value_dict(tech_total)
            graph.nodes[node][year][PARAM.technologies][tech][total_param] = value_dict
    value_dict = create_value_dict(node_total)
    graph.nodes[node][year][total_param] = value_dict


def aggregate_direct_emissions(model, graph, node, year, rate_param, total_param):
    _abstract_direct_emission_aggregation(Emissions, model, graph, node, year,
                                          rate_param, total_param)


def aggregate_direct_emissions_cost(model, graph, node, year, rate_param, total_param):
    _abstract_direct_emission_aggregation(EmissionsCost, model, graph, node, year,
                                          rate_param, total_param)
