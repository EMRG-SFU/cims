"""
Module containing the classes and functions related to emissions and emissions costs.
"""
from . import utils


class EmissionRates:
    """Class for storing the emission rates of technologies. An emission rate is the amount of
       emissions per unit of a technology."""

    def __init__(self, emission_rates=None):
        self.emission_rates = emission_rates if emission_rates is not None else {}

    def multiply_rates(self, amount):
        """
        Multiply, each emission rate in self.emission_rates by amount. Amount will usually be the
        total number of units of a given tech. Then rate * units will give the total emissions for
        that tech.

        Parameters
        ----------
        amount : float or int
            The value to multiply each emission rate by.

        Returns
        -------
        dict :
            Returns a dictionary where each emission rate has been multiplied by amount.
        """

        emission_totals = {}
        for source_branch in self.emission_rates:
            if source_branch not in emission_totals:
                emission_totals[source_branch] = {}
            for ghg in self.emission_rates[source_branch]:
                if ghg not in emission_totals[source_branch]:
                    emission_totals[source_branch][ghg] = {}
                for emission_type in self.emission_rates[source_branch][ghg]:
                    prev_val = self.emission_rates[source_branch][ghg][emission_type]['year_value']
                    emission_totals[source_branch][ghg][emission_type] = \
                        utils.create_value_dict(prev_val * amount)

        return emission_totals

    def summarize_rates(self):
        """
        Sum emission rates across GHG/Emission Type pairs. This removes the differentiation between
        which node an emission rate originated from.

        Returns
        -------
        dict :
            Returns a dictionary containing the summarized version of self.emission_rates. This
            summary dictionary will follow the form {GHG: {Emission Type: rate}}.
        """
        summary_rates = {}
        for source in self.emission_rates:
            for ghg in self.emission_rates[source]:
                if ghg not in summary_rates:
                    summary_rates[ghg] = {}
                for emission_type in self.emission_rates[source][ghg]:
                    if emission_type not in summary_rates[ghg]:
                        summary_rates[ghg][emission_type] = 0
                    summary_rates[ghg][emission_type] += \
                        self.emission_rates[source][ghg][emission_type]['year_value']
        return summary_rates


class Emissions:
    """Class for """

    def __init__(self, emissions=None):
        self.emissions = emissions if emissions is not None else {}

    def __add__(self, other):
        result = Emissions()

        # Start by recording all the emissions from self in our result
        for source_branch in self.emissions:
            if source_branch not in result.emissions:
                result.emissions[source_branch] = {}
            for ghg in self.emissions[source_branch]:
                if ghg not in result.emissions[source_branch]:
                    result.emissions[source_branch][ghg] = {}
                for emission_type in self.emissions[source_branch][ghg]:
                    emission_amount = self.emissions[source_branch][ghg][emission_type][
                        'year_value']
                    result.emissions[source_branch][ghg][emission_type] = utils.create_value_dict(
                        emission_amount)

        # Then, go through each of the emissions in other and add those to the result
        for source_branch in other.emissions:
            if source_branch not in result.emissions:
                result.emissions[source_branch] = {}
            for ghg in other.emissions[source_branch]:
                if ghg not in result.emissions[source_branch]:
                    result.emissions[source_branch][ghg] = {}
                for emission_type in other.emissions[source_branch][ghg]:
                    if emission_type not in result.emissions[source_branch][ghg]:
                        result.emissions[source_branch][ghg][
                            emission_type] = utils.create_value_dict(0)
                    emission_amount = other.emissions[source_branch][ghg][emission_type][
                        'year_value']
                    result.emissions[source_branch][ghg][emission_type][
                        'year_value'] += emission_amount

        return result

    def __mul__(self, other):
        result = Emissions()

        # Start by recording all the emissions from self in our result
        for source_branch in self.emissions:
            if source_branch not in result.emissions:
                result.emissions[source_branch] = {}
            for ghg in self.emissions[source_branch]:
                if ghg not in result.emissions[source_branch]:
                    result.emissions[source_branch][ghg] = {}
                for emission_type in self.emissions[source_branch][ghg]:
                    emission_amount = self.emissions[source_branch][ghg][emission_type][
                                          'year_value'] * other
                    result.emissions[source_branch][ghg][emission_type] = utils.create_value_dict(
                        emission_amount)

        return result

    def summarize_emissions(self):
        """
        Sum emissions across GHG/Emission Type pairs. This removes the  differentiation between
        which node an emission originated from.

        Returns
        -------
        dict :
            Returns a dictionary containing the summarized version of self.emissions. This summary
            dictionary will follow the form {GHG: {Emission Type: X1}}.
        """
        summary_emissions = {}
        for source in self.emissions:
            for ghg in self.emissions[source]:
                if ghg not in summary_emissions:
                    summary_emissions[ghg] = {}
                for emission_type in self.emissions[source][ghg]:
                    if emission_type not in summary_emissions[ghg]:
                        summary_emissions[ghg][emission_type] = 0
                    summary_emissions[ghg][emission_type] += \
                        self.emissions[source][ghg][emission_type]['year_value']

        return summary_emissions


def calc_aggregate_emission_cost_rate(model, node, year, tech=None):
    """
    Calculates the emissions cost rate (cost per unit) for a node/tech, including the emission costs
    from child techs/services that are normally embedded within those LCCs.

    There is slightly different logic, depending on the kind of node we are at. There are three
    possible locations:
        (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        (2) At a node with techs -- Weighted emissions cost from techs
        (3) At a node without techs -- Emissions Cost from Non Fuel children
    """
    if tech is not None:
        # (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        agg_emissions_cost = {}

        # Emission Cost @ Tech
        if 'emissions_cost_dict' in model.graph.nodes[node][year]['technologies'][tech]:
            tech_emissions_cost = model.get_param('emissions_cost_dict', node, year,
                                                  tech=tech, dict_expected=True)
            agg_emissions_cost = _add_tech_emission_cost(tech_emissions_cost, agg_emissions_cost)

        # Emissions Cost from any non-fuel children
        services_requested = _get_services_requested(model, node, year, tech=tech)
        agg_emissions_cost = _add_emission_cost_from_non_fuel_children(model, year,
                                                                       services_requested,
                                                                       agg_emissions_cost)

    elif model.get_param('competition type', node) in ['tech compete', 'node tech compete']:
        # (2) At a node with techs -- Weighted emissions cost from techs
        agg_emissions_cost = {}

        # Weighted emissions cost from techs
        for technology in model.graph.nodes[node][year]['technologies']:
            tech_emissions_cost = model.get_param('aggregate_emissions_cost_rates', node, year,
                                                  tech=technology, dict_expected=True)
            market_share = model.get_param('total_market_share', node, year, tech=technology)

            for ghg in tech_emissions_cost:
                if ghg not in agg_emissions_cost:
                    agg_emissions_cost[ghg] = {}

                for emission_type in tech_emissions_cost[ghg]:
                    if emission_type not in agg_emissions_cost[ghg]:
                        agg_emissions_cost[ghg][emission_type] = 0
                    agg_emissions_cost[ghg][emission_type] += \
                        tech_emissions_cost[ghg][emission_type] * market_share

    else:
        # (3) At a node without techs -- Emissions Cost from Non Fuel children
        agg_emissions_cost = {}

        # Emissions Cost from Non Fuel children
        services_requested = _get_services_requested(model, node, year)
        agg_emissions_cost = _add_emission_cost_from_non_fuel_children(model, year,
                                                                       services_requested,
                                                                       agg_emissions_cost)

    # Save the Aggregate Emission Cost Rates
    val_dict = utils.create_value_dict(year_val=agg_emissions_cost, param_source='calculation')
    if tech:
        model.set_param_internal(val_dict, 'aggregate_emissions_cost_rates', node, year, tech)
    else:
        model.graph.nodes[node][year]['aggregate_emissions_cost_rates'] = val_dict


def _get_services_requested(model, node, year, tech=None):
    if tech:
        if 'Service requested' not in model.graph.nodes[node][year]['technologies'][tech]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year]['technologies'][tech][
                'Service requested']
    else:
        if 'Service requested' not in model.graph.nodes[node][year]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year]['Service requested']

    return services_requested


def _add_emission_cost_from_non_fuel_children(model, year, services_requested, agg_emissions_cost):
    for req_data in services_requested.values():
        child = req_data['branch']
        if child not in model.fuels:
            req_ratio = req_data['year_value']
            something = model.get_param('aggregate_emissions_cost_rates', child, year)
            for ghg in something:
                if ghg not in agg_emissions_cost:
                    agg_emissions_cost[ghg] = {}
                for emission_type in something[ghg]:
                    if emission_type not in agg_emissions_cost[ghg]:
                        agg_emissions_cost[ghg][emission_type] = 0
                    agg_emissions_cost[ghg][emission_type] += something[ghg][
                                                                  emission_type] * req_ratio

    return agg_emissions_cost


def _add_tech_emission_cost(tech_emissions_cost, agg_emissions_cost):
    for e_node in tech_emissions_cost:
        for ghg in tech_emissions_cost[e_node]:
            if ghg not in agg_emissions_cost:
                agg_emissions_cost[ghg] = {}

            for emission_type in tech_emissions_cost[e_node][ghg]:
                if emission_type not in agg_emissions_cost[ghg]:
                    agg_emissions_cost[ghg][emission_type] = 0
                agg_emissions_cost[ghg][emission_type] += \
                    tech_emissions_cost[e_node][ghg][emission_type]['year_value']

    return agg_emissions_cost
