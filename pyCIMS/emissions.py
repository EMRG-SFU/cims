"""
Module containing the classes and functions related to emissions and emissions costs.
"""
import copy
from numpy import linspace
from pyCIMS import utils


class EmissionsCost:
    """
    Class for storing the emission cost per unit of. Emissions cost takes into account the
    emissions cost of children nodes, services, and technologies.
    """

    def __init__(self, emissions_cost=None, num_units=1):
        """
        Initializes an EmissionsCost object.

        Parameters
        ----------
        emissions_cost : dict
            The dictionary containing the detailed emissions cost by source node (fuel), GHG, and
            emission_type.

        num_units : float, Optional
            The number of units the EmissionsCost is spread over. If num_units is 1, EmissionsCost
            is effectively a rate ($/unit).
        """
        self.emissions_cost = emissions_cost if emissions_cost is not None else {}
        self.num_units = num_units

    def __add__(self, other):
        """
        Adds two EmissionsCost objects together by combining their emissions_cost (dictionaries)
        attributes through addition. Any source/GHG/emission type combination that exists in only
        one of the EmissionsCost objects will be contained in the final result.

        Parameters
        ----------
        other : EmissionsCost

        Returns
        -------
        EmissionsCost
        """
        # Start by recording all the emissions from self in the result
        result = copy.deepcopy(self)

        # Then, go through each of the emissions in other and add those to the result
        for source_branch in other.emissions_cost:
            if source_branch not in result.emissions_cost:
                result.emissions_cost[source_branch] = {}
            for ghg in other.emissions_cost[source_branch]:
                if ghg not in result.emissions_cost[source_branch]:
                    result.emissions_cost[source_branch][ghg] = {}
                for emission_type in other.emissions_cost[source_branch][ghg]:
                    if emission_type not in result.emissions_cost[source_branch][ghg]:
                        result.emissions_cost[source_branch][ghg][
                            emission_type] = utils.create_value_dict(0)
                    emission_amount = other.emissions_cost[source_branch][ghg][emission_type][
                        'year_value']
                    result.emissions_cost[source_branch][ghg][emission_type][
                        'year_value'] += emission_amount
        return result

    def __mul__(self, other):
        """
        Multiplies each value in the emissions_cost attribute (which is a nested dictionary) by
        other.

        Parameters
        ----------
        other : float

        Returns
        -------
        EmissionsCost
        """
        if not (isinstance(other, float) or isinstance(other, int)):
            print(type(other))
            raise ValueError

        result = EmissionsCost()

        for source_branch in self.emissions_cost:
            if source_branch not in result.emissions_cost:
                result.emissions_cost[source_branch] = {}
            for ghg in self.emissions_cost[source_branch]:
                if ghg not in result.emissions_cost[source_branch]:
                    result.emissions_cost[source_branch][ghg] = {}
                for emission_type in self.emissions_cost[source_branch][ghg]:
                    prev_val = self.emissions_cost[source_branch][ghg][emission_type][
                        'year_value']
                    result.emissions_cost[source_branch][ghg][emission_type] = \
                        utils.create_value_dict(prev_val * other)

        return result

    def summarize(self):
        summary_rates = {}
        for source_branch in self.emissions_cost:
            for ghg in self.emissions_cost[source_branch]:
                if ghg not in summary_rates:
                    summary_rates[ghg] = {}
                for emission_type in self.emissions_cost[source_branch][ghg]:
                    if emission_type not in summary_rates[ghg]:
                        summary_rates[ghg][emission_type] = 0
                    summary_rates[ghg][emission_type] += \
                        self.emissions_cost[source_branch][ghg][emission_type]['year_value']

        return summary_rates

    def total_emissions_cost(self):
        total = 0

        for source_branch in self.emissions_cost:
            for ghg in self.emissions_cost[source_branch]:
                for emission_type in self.emissions_cost[source_branch][ghg]:
                    total += self.emissions_cost[source_branch][ghg][emission_type]['year_value']

        return total


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


def calc_per_unit_emissions_cost(model, node, year, tech=None):
    """
    Calculates the per unit emissions cost for a node/tech, including the emission costs
    from child techs/services that are normally embedded within those LCCs.

    There is slightly different logic, depending on the kind of node we are at. There are three
    possible locations:
        (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        (2) At a node with techs -- Weighted emissions cost from techs
        (3) At a node without techs -- Emissions Cost from Non Fuel children
    """
    if tech is not None:
        # (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        agg_emissions_cost = EmissionsCost()

        # Emission Cost @ Tech
        if 'emissions_cost_dict' in model.graph.nodes[node][year]['technologies'][tech]:
            tech_emissions_cost = model.get_param('emissions_cost_dict',
                                                  node, year, tech=tech)
            agg_emissions_cost = agg_emissions_cost + tech_emissions_cost

        # Emissions Cost from any non-fuel children
        services_requested = utils.get_services_requested(model, node, year, tech=tech)
        agg_emissions_cost = _add_emissions_cost_from_non_fuel_children(model, year,
                                                                       services_requested,
                                                                       agg_emissions_cost)

    elif model.get_param('competition type', node) in ['tech compete', 'node tech compete']:
        # (2) At a node with techs -- Weighted emissions cost from techs
        agg_emissions_cost = EmissionsCost()
        # Weighted emissions cost from techs
        for technology in model.graph.nodes[node][year]['technologies']:
            tech_emissions_cost = model.get_param('per_unit_emissions_cost', node, year,
                                                  tech=technology, dict_expected=True)
            market_share = model.get_param('total_market_share', node, year, tech=technology)
            agg_emissions_cost = agg_emissions_cost + (tech_emissions_cost * market_share)

    else:
        # (3) At a node without techs -- Emissions Cost from Non Fuel children
        agg_emissions_cost = EmissionsCost()
        services_requested = utils.get_services_requested(model, node, year)

        # Emissions Cost from Non Fuel children
        agg_emissions_cost = _add_emissions_cost_from_non_fuel_children(model, year,
                                                                       services_requested,
                                                                       agg_emissions_cost)

    # Save the Aggregate Emission Cost Rates
    new_val_dict = utils.create_value_dict(year_val=agg_emissions_cost, param_source='calculation')

    if tech:
        model.set_param_internal(new_val_dict, 'per_unit_emissions_cost', node, year, tech)
    else:
        model.graph.nodes[node][year]['per_unit_emissions_cost'] = new_val_dict


def _add_emissions_cost_from_non_fuel_children(model, year, services_requested, agg_emissions_cost):
    """
    Go through each of the requested services and find the emissions cost that can be attributed to
    the requesting node. Add these to agg_emissions_cost.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the relevant data.
    year : str
        The year of interest.
    services_requested : dict
        A dictionary containing requested services (service & request ratio).
    agg_emissions_cost : EmissionsCost
        The emissions cost(s) associated with the requesting node.

    Returns
    -------
    EmissionsCost

    """
    for req_data in services_requested.values():
        child = req_data['branch']
        if child not in model.fuels:
            req_ratio = max(req_data['year_value'], 0)
            child_emissions_cost = model.get_param('per_unit_emissions_cost', child, year,
                                                   dict_expected=True)

            agg_emissions_cost = agg_emissions_cost + (child_emissions_cost * req_ratio)

    return agg_emissions_cost


def calc_emissions_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str,
                        allow_foresight=False) -> float:
    """
    Calculates the emission cost at a node.

    Total, gross, captured, and net emissions are all calculated and combined to find the final
    emission cost. This total emissions cost is returned by the function and stored in the model.
    Net, captured, and biomass emission rates are also stored in the model.

    To see how the calculation works, see the file 'Emissions_tax_example.xlsx':
    https://gitlab.rcg.sfu.ca/mlachain/pycims_prototype/-/issues/22#note_6489

    Parameters
    ----------
    model : The model containing all values needed for calculating emissions cost.
    node : The node to calculate emissions cost for.
    year : The year to calculate emissions cost for.
    tech : The technology to calculate emissions cost for.
    allow_foresight : Whether or not to allow non-myopic carbon cost foresight methods.
    Returns
    -------
    float : the total emission cost. Has the side effect of updating the Emissions Cost,
            net_emission_rates, captured_emission_rates, and bio_emission_rates in the model.
    """

    fuels = model.fuels

    # No tax rate at all or node is a fuel
    if 'tax' not in model.graph.nodes[node][year] or node in fuels:
        return 0

    # Initialize all taxes and emission removal rates to 0
    # example of item in tax_rates -> {'CO2': {'Combustion': 5}}
    tax_rates = {ghg: {em_type: utils.create_value_dict(0) for em_type in model.emission_types} for
                 ghg in model.GHGs}
    removal_rates = copy.deepcopy(tax_rates)

    # Grab correct tax values
    all_taxes = model.get_param('tax', node, year, dict_expected=True)
    for ghg in all_taxes:
        for emission_type in all_taxes[ghg]:
            if ghg not in tax_rates:
                tax_rates[ghg] = {}
            tax_rates[ghg][emission_type] = utils.create_value_dict(
                all_taxes[ghg][emission_type]['year_value'])

    # EMISSIONS tech level
    total_emissions = {}
    total = 0
    if 'emissions' in model.graph.nodes[node][year]['technologies'][tech]:
        total_emissions[tech] = {}
        emission_data = model.graph.nodes[node][year]['technologies'][tech]['emissions']

        for ghg in emission_data:
            for emission_type in emission_data[ghg]:
                if ghg not in total_emissions[tech]:
                    total_emissions[tech][ghg] = {}
                total_emissions[tech][ghg][emission_type] = utils.create_value_dict(
                    emission_data[ghg][emission_type]['year_value'])

    # EMISSIONS REMOVAL tech level
    if 'emissions_removal' in model.graph.nodes[node][year]:
        removal_dict = model.graph.nodes[node][year]['emissions_removal']
        for ghg in removal_dict:
            for emission_type in removal_dict[ghg]:
                if ghg not in removal_rates:
                    removal_rates[ghg] = {}
                removal_rates[ghg][emission_type] = utils.create_value_dict(
                    removal_dict[ghg][emission_type]['year_value'])

    # Check all services requested for
    if 'service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['service requested']

        # EMISSIONS child level
        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']
            if 'emissions' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                fuel_emissions = model.graph.nodes[child_node][year]['emissions']
                total_emissions[child_node] = {}
                for ghg in fuel_emissions:
                    for emission_type in fuel_emissions[ghg]:
                        if ghg not in total_emissions[child_node]:
                            total_emissions[child_node][ghg] = {}
                        total_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                fuel_emissions[ghg][emission_type]['year_value'] * req_val)

    gross_emissions = copy.deepcopy(total_emissions)

    if 'service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['service requested']

        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']

            # GROSS EMISSIONS
            if 'emissions_biomass' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                gross_dict = model.graph.nodes[child_node][year]['emissions_biomass']
                for ghg in gross_dict:
                    for emission_type in gross_dict[ghg]:
                        gross_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                gross_dict[ghg][emission_type]['year_value'] * req_val)

            # EMISSIONS REMOVAL child level
            if 'technologies' in model.graph.nodes[child_node][year]:
                child_techs = model.graph.nodes[child_node][year]['technologies']
                for _, tech_data in child_techs.items():
                    if 'emissions_removal' in tech_data:
                        removal_dict = tech_data['emissions_removal']
                        for ghg in removal_dict:
                            for emission_type in removal_dict[ghg]:
                                removal_rates[ghg][emission_type] = \
                                    utils.create_value_dict(
                                        removal_dict[ghg][emission_type]['year_value'])

    # CAPTURED EMISSIONS
    captured_emissions = copy.deepcopy(gross_emissions)
    for node_name in captured_emissions:
        for ghg in captured_emissions[node_name]:
            for emission_type in captured_emissions[node_name][ghg]:
                em_removed = removal_rates[ghg][emission_type]
                captured_emissions[node_name][ghg][emission_type]['year_value'] *= em_removed[
                    'year_value']

    # NET EMISSIONS
    net_emissions = copy.deepcopy(total_emissions)
    for node_name in net_emissions:
        for ghg in net_emissions[node_name]:
            for emission_type in net_emissions[node_name][ghg]:
                net_emissions[node_name][ghg][emission_type]['year_value'] -= \
                    captured_emissions[node_name][ghg][emission_type]['year_value']

    # EMISSIONS COST
    emissions_cost = copy.deepcopy(net_emissions)
    for node_name in emissions_cost:
        for ghg in emissions_cost[node_name]:
            for emission_type in emissions_cost[node_name][ghg]:
                #  Dict to check if emission_type exists in taxes
                if ghg in all_taxes:
                    tax_check = model.get_param('tax', node, year, context=ghg, dict_expected=True)
                else:
                    tax_check = {}

                # Use foresight method to calculate tax
                Expected_EC = 0
                method_dict = model.get_param('foresight method', node, year, dict_expected=True)

                # Replace current tax with foresight method
                method = None
                if method_dict and ghg in method_dict:
                    method = method_dict[ghg]['year_value']

                if (method == 'Myopic') or (method is None) or \
                        (emission_type not in tax_check) or (not allow_foresight):
                    Expected_EC = tax_rates[ghg][emission_type]['year_value']  # same as regular tax

                elif method == 'Discounted':
                    N = int(model.get_param('lifetime', node, year, tech=tech))
                    r_k = model.get_param('discount rate_financial', node, year)

                    # interpolate all tax values
                    tax_vals = []
                    for n in range(int(year), int(year) + N, model.step):
                        if n in model.years:  # go back one step if current year isn't in range
                            cur_tax = model.get_param('tax', node, str(n),
                                                      context=ghg, sub_context=emission_type)
                        else:
                            cur_tax = model.get_param('tax', node, str(n - model.step),
                                                      context=ghg, sub_context=emission_type)
                        if n + model.step in model.years:  # when future years are out of range
                            next_tax = model.get_param('tax', node, str(n + model.step),
                                                       context=ghg, sub_context=emission_type)
                        else:
                            next_tax = cur_tax
                        tax_vals.extend(linspace(cur_tax, next_tax, model.step, endpoint=False))

                    # calculate discounted tax using formula
                    Expected_EC = sum(
                        [tax / (1 + r_k) ** (n - int(year) + 1)
                         for tax, n in zip(tax_vals, range(int(year), int(year) + N))]
                    )
                    Expected_EC *= r_k / (1 - (1 + r_k) ** (-N))

                elif method == 'Average':
                    N = int(model.get_param('lifetime', node, year, tech=tech))

                    # interpolate tax values
                    tax_vals = []
                    for n in range(int(year), int(year) + N, model.step):
                        if str(n) <= max(
                                model.years):  # go back one step if current year isn't in range
                            cur_tax = model.get_param('tax', node, str(n),
                                                      context=ghg, sub_context=emission_type)
                        else:
                            cur_tax = model.get_param('tax', node, max(model.years),
                                                      context=ghg, sub_context=emission_type)
                        if str(n + model.step) in model.years:  # when future years are out of range
                            next_tax = model.get_param('tax', node, str(n + model.step),
                                                       context=ghg, sub_context=emission_type)
                        else:
                            next_tax = cur_tax
                        tax_vals.extend(linspace(cur_tax, next_tax, model.step, endpoint=False))
                    Expected_EC = sum(tax_vals) / N  # take average of all taxes

                else:
                    raise ValueError(
                        'Foresight method not identified, use Myopic, Discounted, or Average')

                emissions_cost[node_name][ghg][emission_type]['year_value'] *= Expected_EC

    # Add everything in nested dictionary together
    for node_name in emissions_cost:
        for ghg in emissions_cost[node_name]:
            for emission_type in emissions_cost[node_name][ghg]:
                total += emissions_cost[node_name][ghg][emission_type]['year_value']

    # BIO EMISSIONS tech level
    bio_emissions = {}
    if 'emissions_biomass' in model.graph.nodes[node][year]['technologies'][tech]:
        bio_emissions[tech] = {}
        bio_emission_data = model.graph.nodes[node][year]['technologies'][tech]['emissions_biomass']

        for ghg in bio_emission_data:
            for emission_type in bio_emission_data[ghg]:
                if ghg not in bio_emissions[tech]:
                    bio_emissions[tech][ghg] = {}
                bio_emissions[tech][ghg][emission_type] = utils.create_value_dict(
                    bio_emission_data[ghg][emission_type]['year_value'])

    # Check all services requested for
    if 'service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['service requested']

        # BIO EMISSIONS child level
        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']
            if 'emissions_biomass' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                fuel_emissions = model.graph.nodes[child_node][year]['emissions_biomass']
                bio_emissions[child_node] = {}
                for ghg in fuel_emissions:
                    for emission_type in fuel_emissions[ghg]:
                        if ghg not in bio_emissions[child_node]:
                            bio_emissions[child_node][ghg] = {}
                        bio_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                fuel_emissions[ghg][emission_type]['year_value'] * req_val)

    # Record emission rates
    model.graph.nodes[node][year]['technologies'][tech]['net_emission_rates'] = \
        EmissionRates(emission_rates=net_emissions)
    model.graph.nodes[node][year]['technologies'][tech]['captured_emission_rates'] = \
        EmissionRates(emission_rates=captured_emissions)
    model.graph.nodes[node][year]['technologies'][tech]['bio_emission_rates'] = \
        EmissionRates(emission_rates=bio_emissions)

    # Record emission costs
    val_dict = utils.create_value_dict(year_val=total, param_source='calculation')
    model.set_param_internal(val_dict, 'emissions cost', node, year, tech)

    model.set_param_internal(EmissionsCost(emissions_cost),
                             'emissions_cost_dict', node, year, tech)

    return total
