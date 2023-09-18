"""
Module to provide the calculation of recycled revenues from emissions.
"""
from .emissions import calc_cumul_emissions_cost_rate


def calc_recycled_revenues(model, node, year, tech=None):
    """
    Calculate recycled revenue (dollars) for one unit of the service provided by the node/tech.
    Based on the aggregate emission rate (emissions per unit, including the emissions used in
    requested services & technologies) and recycled revenue rate.

    Parameters
    ----------
    model : CIMS.Model
        The model containing node
    node : str
        The name of the node (in branch form) for which recycled revenues will be calculated.
    year : str
        The year for which recycled revenues will be calculated.
    tech : str, default=None
        The name of the technology for which recycled revenues will be calculated. If None, recycled
        revenues are calculated at the node level.

    Returns
    -------
    float
        The recycled revenue for one unit of the service provided by the node/tech.
    """
    # Retrieve the recycling rates
    if tech is not None:
        if 'recycled revenues' not in model.graph.nodes[node][year]['technologies'][tech]:
            recycling_rates = {}
        else:
            recycling_rates = model.get_param('recycled revenues',
                                              node, year, tech=tech, dict_expected=True)
    else:
        if 'recycled revenues' not in model.graph.nodes[node][year]:
            recycling_rates = {}
        else:
            recycling_rates = model.get_param('recycled revenues', node, year, dict_expected=True)

    # Retrieve the aggregate emissions cost at the node/tech
    calc_cumul_emissions_cost_rate(model, node, year, tech)

    aggregate_emissions_cost = model.get_param('cumul_emissions_cost_rate',
                                               node, year, tech=tech, dict_expected=True)

    # Apply the recycling rates to the aggregate emissions to find the recycled revenues
    total_recycled_revenue = 0
    for ghg in aggregate_emissions_cost.summarize():
        for emission_type, emissions_cost in aggregate_emissions_cost.summarize()[ghg].items():
            try:
                recycling_rate = recycling_rates[ghg][emission_type]['year_value']
                total_recycled_revenue += emissions_cost * recycling_rate
            except KeyError:
                pass

    for source_branch in aggregate_emissions_cost.emissions_cost:
        for ghg in aggregate_emissions_cost.emissions_cost[source_branch]:
            for emission_type in aggregate_emissions_cost.emissions_cost[source_branch][ghg]:
                try:
                    recycling_rate = recycling_rates[ghg][emission_type]['year_value']
                    aggregate_emissions_cost.emissions_cost[source_branch][ghg][emission_type]['year_value'] = \
                        aggregate_emissions_cost.emissions_cost[source_branch][ghg][emission_type]['year_value'] * \
                        (1 - recycling_rate)
                except KeyError:
                    pass

    if tech is not None:
        model.graph.nodes[node][year]['technologies'][tech]['cumul_emissions_cost_rate'] = \
            aggregate_emissions_cost
    else:
        model.graph.nodes[node][year]['cumul_emissions_cost_rate'] = \
            aggregate_emissions_cost

    return total_recycled_revenue
