"""
Module containing the classes and functions related to emissions and emissions costs.
"""
from __future__ import annotations  # For Type Hinting
from typing import List
import copy
from numpy import linspace
from . import utils


class EmissionsCost:
    """
    Class for storing the emission cost of a node.
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

    def __add__(self, other: EmissionsCost) -> EmissionsCost:
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

    def __mul__(self, other: int or float) -> EmissionsCost:
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
        if not isinstance(other, (float, int)):
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

    def summarize(self) -> dict:
        """
        Aggregate the `EmissionsCost.emissions_cost` dictionary for each GHG and emission type
        combination.

        Returns
        -------

        dict :
            Returns a nested dictionary. The first level keys are GHGs (e.g. CO2). The second level
            keys are emission types (e.g. Combustion). The second level values are floats
            representing the aggregate cost for the GHG/emission type combinations across all source
            fuels.
        """
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

    def total_emissions_cost(self) -> float:
        """
        Find the total emissions cost across all fuels, GHGs, and emission types.

        Returns
        -------
        float :
            The sum of emissions costs across all fuels, GHGs, and emission types in the
            `EmissionCost.emission_cost` dictionary.
        """
        total = 0

        for source_branch in self.emissions_cost:
            for ghg in self.emissions_cost[source_branch]:
                for emission_type in self.emissions_cost[source_branch][ghg]:
                    total += self.emissions_cost[source_branch][ghg][emission_type]['year_value']

        return total


class Emissions:
    """Class for storing the emissions for a particular technology or node."""

    def __init__(self, emissions=None):
        """
        Initializes an Emissions object.

        Parameters
        ----------
        emissions : dict
            The dictionary containing the detailed emissions by source node (fuel), GHG, and
            emission_type.
        """
        self.emissions = emissions if emissions is not None else {}

    def __add__(self, other: Emissions) -> Emissions:
        """
        Adds two Emissions objects together by combining their emissions (dictionaries)
        attributes through addition. Any source/GHG/emission type combination that exists in only
        one of the Emissions objects will be contained in the final result.

        Parameters
        ----------
        other : Emissions

        Returns
        -------
        Emissions
        """
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

    def __mul__(self, other: int or float) -> Emissions:
        """
        Multiplies each value in the emissions attribute (which is a nested dictionary) by
        other.

        Parameters
        ----------
        other : int or float

        Returns
        -------
        Emissions
        """
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

    def summarize(self) -> dict:
        """
        Aggregate the `Emissions.emissions` dictionary for each GHG and emission type combination.

        Returns
        -------

        dict :
            Returns a nested dictionary. The first level keys are GHGs (e.g. CO2). The second level
            keys are emission types (e.g. Combustion). The second level values are floats
            representing the aggregate cost for the GHG/emission type combinations across all source
            fuels.
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


def calc_cumul_emissions_cost_rate(model: 'pyCIMS.Model', node: str, year: str,
                                   tech: str = None) -> None:
    """
    Calculates the per unit emissions cost for a node or tech. This includes the emissions costs
    from child techs/services that are embedded within the LCCs of those techs/services and are
    excluded from "direct" rates.

    There is slightly different logic, depending on the kind of node we are at. There are three
    possible locations:
        (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        (2) At a node with techs -- Weighted emissions cost from techs
        (3) At a node without techs -- Emissions Cost from Non Fuel children

    Finally, if we are at a node or a tech that previously had stock but no longer has stock, the
    cumulative emissions cost rate at the node will be 0.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the node and information of interest
    node : str
        The node whose cumulative emissions cost rate is being calculated
    year : str
        The year which the cumulative emissions cost rate is being calculated for
    tech : str, optional
        An optional parameter that specifies the technology whose cumulative emissions rate is being
        calculated. If this parameter is not specified, but a node has technologies than the
        cumulative emissions rate is an aggregation across the techs.

    Returns
    -------
    None :
        Nothing is returned. Instead, the node's cumul_emissions_cost_rate calculated and saved in
        the model.
    """
    pq, src = model.get_param('provided_quantities', node, year, tech=tech, return_source=True)
    if tech is not None:
        # (1) At a technology -- Emission Cost @ Tech + Emissions Cost from any non-fuel children
        agg_emissions_cost = EmissionsCost()

        # Emission Cost @ Tech
        if 'emissions_cost_rate' in model.graph.nodes[node][year]['technologies'][tech]:
            tech_emissions_cost = model.get_param('emissions_cost_rate',
                                                  node, year, tech=tech)
            agg_emissions_cost = agg_emissions_cost + tech_emissions_cost

        # Emissions Cost from any non-fuel children
        services_requested = utils.get_services_requested(model, node, year, tech=tech)
        agg_emissions_cost += _find_indirect_emissions_cost(model, year, services_requested)

    elif utils.prev_stock_existed(model, node, year) and (pq is not None) and (
            src == 'calculation') and (pq.get_total_quantity() <= 0):
        agg_emissions_cost = EmissionsCost()
    elif model.get_param('competition type', node) in ['tech compete', 'node tech compete']:
        # (2) At a node with techs -- Weighted emissions cost from techs
        agg_emissions_cost = EmissionsCost()
        # Weighted emissions cost from techs
        for technology in model.graph.nodes[node][year]['technologies']:
            tech_emissions_cost = model.get_param('cumul_emissions_cost_rate', node, year,
                                                  tech=technology, dict_expected=True)
            market_share = model.get_param('total_market_share', node, year, tech=technology)
            agg_emissions_cost = agg_emissions_cost + (tech_emissions_cost * market_share)

    else:
        # (3) At a node without techs -- Emissions Cost from Non Fuel children
        agg_emissions_cost = EmissionsCost()
        services_requested = utils.get_services_requested(model, node, year)

        # Emissions Cost from Non Fuel children
        agg_emissions_cost += _find_indirect_emissions_cost(model, year, services_requested)

    # Save the Aggregate Emission Cost Rates
    new_val_dict = utils.create_value_dict(year_val=agg_emissions_cost, param_source='calculation')

    if tech:
        model.set_param_internal(new_val_dict, 'cumul_emissions_cost_rate', node, year, tech)
    else:
        model.graph.nodes[node][year]['cumul_emissions_cost_rate'] = new_val_dict


def calc_cumul_emissions_rate(model: 'pyCIMS.Model', node: str, year: str,
                              tech: str = None) -> None:
    """
    Calculates the per unit emissions for a node/tech, including the emissions from child
    techs/services that are excluded from "direct" rates.

    There is slightly different logic, depending on the kind of node we are at. There are three
    possible locations:
        (1) At a technology -- Emissions @ Tech + Emissions from any non-fuel children
        (2) At a node with techs -- Weighted emissions from techs
        (3) At a node without techs -- Emissions from Non Fuel children

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the node and information of interest
    node : str
        The node whose cumulative emissions cost rate is being calculated
    year : str
        The year which the cumulative emissions cost rate is being calculated for
    tech : str, optional
        An optional parameter that specifies the technology whose cumulative emissions rate is being
        calculated. If this parameter is not specified, but a node has technologies than the
        cumulative emissions rate is an aggregation across the techs. 

    Returns
    -------
    None :
        Nothing is returned. Instead, the node's cumul_emissions_cost_rate calculated and saved in
        the model.
    """

    pq, src = model.get_param('provided_quantities', node, year, tech=tech, return_source=True)
    emissions = ['net_emissions_rate', 'avoided_emissions_rate', 'negative_emissions_rate',
                 'bio_emissions_rate']
    for rate_param in emissions:
        cumul_rate_param = f'cumul_{rate_param}'
        if tech is not None:
            # (1) At a technology -- Emission @ Tech + Emissions from any non-fuel children
            agg_emissions = Emissions()

            # Emission @ Tech
            if rate_param in model.graph.nodes[node][year]['technologies'][tech]:
                tech_emissions = model.get_param(rate_param, node, year, tech=tech)
                agg_emissions = agg_emissions + tech_emissions

            # Emissions from any non-fuel children
            services_requested = utils.get_services_requested(model, node, year, tech=tech)
            agg_emissions += _find_indirect_emissions(model, year, services_requested,
                                                      emissions_param=cumul_rate_param)

        elif utils.prev_stock_existed(model, node, year) and (pq is not None) and (
                src == 'calculation') and (pq.get_total_quantity() <= 0):
            agg_emissions = Emissions()
        elif model.get_param('competition type', node) in ['tech compete', 'node tech compete']:
            # (2) At a node with techs -- Weighted emissions from techs
            agg_emissions = Emissions()
            # Weighted emissions from techs
            for technology in model.graph.nodes[node][year]['technologies']:
                tech_emissions = model.get_param(cumul_rate_param, node, year,
                                                 tech=technology, dict_expected=True)
                if tech_emissions is not None:
                    market_share = model.get_param('total_market_share', node, year,
                                                   tech=technology)
                    agg_emissions = agg_emissions + (tech_emissions * market_share)

        else:
            # (3) At a node without techs -- Emissions from Non Fuel children
            agg_emissions = Emissions()
            services_requested = utils.get_services_requested(model, node, year)

            # Emissions from Non Fuel children
            agg_emissions += _find_indirect_emissions(model, year, services_requested,
                                                      emissions_param=cumul_rate_param)

        # Save the Aggregate Emission Rates
        new_val_dict = utils.create_value_dict(year_val=agg_emissions, param_source='calculation')

        if tech:
            model.set_param_internal(new_val_dict, cumul_rate_param, node, year, tech)
        else:
            model.graph.nodes[node][year][cumul_rate_param] = new_val_dict


def _find_indirect_emissions_cost(model: "pyCIMS.Model", year: str,
                                  services_requested: List[dict]) -> EmissionsCost:
    """
    Go through each of the requested services and find the emissions cost that can be attributed to
    the requesting node. These are called the indirect emissions costs.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the relevant data.
    year : str
        The year of interest.
    services_requested : dict
        A dictionary containing requested services (service & request ratio).

    Returns
    -------
    EmissionsCost :
        The sum of indirect EmissionsCost objects across all requested services.
    """
    indirect_emissions_cost = EmissionsCost()
    for req_data in services_requested.values():
        child = req_data['branch']
        if child not in model.fuels:
            req_ratio = req_data['year_value']
            child_emissions_cost = model.get_param('cumul_emissions_cost_rate', child, year,
                                                   dict_expected=True)

            indirect_emissions_cost += child_emissions_cost * req_ratio

    return indirect_emissions_cost


def _find_indirect_emissions(model: 'pyCIMS.Model', year: str, services_requested: List[dict],
                             emissions_param: str) -> Emissions:
    """
    Go through each of the requested services and find the emissions that can be attributed to
    the requesting node. These are called the indirect emissions.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the relevant data.
    year : str
        The year of interest.
    services_requested : dict
        A dictionary containing requested services (service & request ratio).

    Returns
    -------
    Emissions :
        The sum of indirect Emission objects across all requested services.

    """
    indirect_emissions = Emissions()
    for req_data in services_requested.values():
        child = req_data['branch']
        if child not in model.fuels:
            req_ratio = req_data['year_value']
            child_emissions = model.get_param(emissions_param, child, year,
                                              dict_expected=True)
            if child_emissions is not None:
                indirect_emissions += child_emissions * req_ratio

    return indirect_emissions


def calc_emissions_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str,
                        allow_foresight=False) -> float:
    """
    Calculates the emission cost at a node.

    Total, gross, avoided, negative, and net emissions are all calculated and combined to find
    the final emission cost. This total emissions cost is returned by the function and stored
    in the model.
    Net, avoided, negative, and biomass emission rates are also stored in the model.

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
            net_emissions_rate, avoided_emissions_rate, negative_emissions_rate, and
            bio_emissions_rate in the model.
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

    # GROSS EMISSIONS tech level
    gross_emissions = {}
    gross_bio_emissions = {}
    total = 0
    if 'emissions' in model.graph.nodes[node][year]['technologies'][tech]:
        gross_emissions[tech] = {}
        emission_data = model.graph.nodes[node][year]['technologies'][tech]['emissions']

        for ghg in emission_data:
            for emission_type in emission_data[ghg]:
                if ghg not in gross_emissions[tech]:
                    gross_emissions[tech][ghg] = {}
                gross_emissions[tech][ghg][emission_type] = utils.create_value_dict(
                    emission_data[ghg][emission_type]['year_value'])

    # EMISSIONS REMOVAL @ the tech
    if 'emissions_removal' in model.graph.nodes[node][year]['technologies'][tech]:
        removal_dict = model.graph.nodes[node][year]['technologies'][tech]['emissions_removal']
        for ghg in removal_dict:
            for emission_type in removal_dict[ghg]:
                if ghg not in removal_rates:
                    removal_rates[ghg] = {}
                removal_rates[ghg][emission_type] = utils.create_value_dict(
                    removal_dict[ghg][emission_type]['year_value'])

    # Check all services requested for
    if 'service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['service requested']

        # Child level
        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']

            # GROSS EMISSIONS
            if 'emissions' in model.graph.nodes[child_node][year] and \
                    child_node in fuels and req_val > 0:
                gross_emissions[child_node] = {}
                emission_data = model.graph.nodes[child_node][year]['emissions']

                for ghg in emission_data:
                    for emission_type in emission_data[ghg]:
                        if ghg not in gross_emissions[child_node]:
                            gross_emissions[child_node][ghg] = {}
                        gross_emissions[child_node][ghg][emission_type] = utils.create_value_dict(
                                emission_data[ghg][emission_type]['year_value'] * req_val)

            # GROSS BIOMASS EMISSIONS
            if 'emissions_biomass' in model.graph.nodes[child_node][year] and \
                    child_node in fuels and req_val > 0:
                gross_bio_emissions[child_node] = {}
                bio_emission_data = model.graph.nodes[child_node][year]['emissions_biomass']

                for ghg in bio_emission_data:
                    for emission_type in bio_emission_data[ghg]:
                        if ghg not in gross_bio_emissions[child_node]:
                            gross_bio_emissions[child_node][ghg] = {}
                        gross_bio_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                bio_emission_data[ghg][emission_type]['year_value'] * req_val)

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

    # AVOIDED EMISSIONS
    avoided_emissions = copy.deepcopy(gross_emissions)
    for node_name in avoided_emissions:
        for ghg in avoided_emissions[node_name]:
            for emission_type in avoided_emissions[node_name][ghg]:
                em_removed = removal_rates[ghg][emission_type]
                avoided_emissions[node_name][ghg][emission_type]['year_value'] *= em_removed[
                    'year_value']

    # NEGATIVE EMISSIONS
    negative_emissions = copy.deepcopy(gross_emissions)
    for node_name in negative_emissions:
        for ghg in negative_emissions[node_name]:
            for emission_type in negative_emissions[node_name][ghg]:
                try:
                    em_removed = removal_rates[ghg][emission_type]
                    negative_emissions[node_name][ghg][emission_type]['year_value'] = \
                        gross_bio_emissions[node_name][ghg][emission_type]['year_value'] * \
                        em_removed['year_value']
                except KeyError:
                    negative_emissions[node_name][ghg][emission_type]['year_value'] = 0

    # NET EMISSIONS
    net_emissions = copy.deepcopy(gross_emissions)
    for node_name in net_emissions:
        for ghg in net_emissions[node_name]:
            for emission_type in net_emissions[node_name][ghg]:
                net_emissions[node_name][ghg][emission_type]['year_value'] -= \
                    avoided_emissions[node_name][ghg][emission_type]['year_value'] + \
                    negative_emissions[node_name][ghg][emission_type]['year_value']

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
                expected_ec = 0
                method_dict = model.get_param('foresight method', node, year, dict_expected=True)

                # Replace current tax with foresight method
                method = None
                if method_dict and ghg in method_dict:
                    method = method_dict[ghg]['year_value']

                if (method == 'Myopic') or (method is None) or \
                        (emission_type not in tax_check) or (not allow_foresight):
                    expected_ec = tax_rates[ghg][emission_type]['year_value']  # same as regular tax

                elif method == 'Discounted':
                    lifetime = int(model.get_param('lifetime', node, year, tech=tech))
                    r_k = model.get_param('discount rate_financial', node, year)

                    # interpolate all tax values
                    tax_vals = []
                    for year_n in range(int(year), int(year) + lifetime, model.step):
                        if year_n in model.years:  # go back one step if current year isn't in range
                            cur_tax = model.get_param('tax', node, str(year_n),
                                                      context=ghg, sub_context=emission_type)
                        else:
                            cur_tax = model.get_param('tax', node, str(year_n - model.step),
                                                      context=ghg, sub_context=emission_type)
                        if year_n + model.step in model.years:  # when future years are out of range
                            next_tax = model.get_param('tax', node, str(year_n + model.step),
                                                       context=ghg, sub_context=emission_type)
                        else:
                            next_tax = cur_tax
                        tax_vals.extend(linspace(cur_tax, next_tax, model.step, endpoint=False))

                    # calculate discounted tax using formula
                    expected_ec = sum(
                        [tax / (1 + r_k) ** (n - int(year) + 1)
                         for tax, n in zip(tax_vals, range(int(year), int(year) + lifetime))]
                    )
                    expected_ec *= r_k / (1 - (1 + r_k) ** (-lifetime))

                elif method == 'Average':
                    lifetime = int(model.get_param('lifetime', node, year, tech=tech))

                    # interpolate tax values
                    tax_vals = []
                    for year_n in range(int(year), int(year) + lifetime, model.step):
                        if str(year_n) <= max(
                                model.years):  # go back one step if current year isn't in range
                            cur_tax = model.get_param('tax', node, str(year_n),
                                                      context=ghg, sub_context=emission_type)
                        else:
                            cur_tax = model.get_param('tax', node, max(model.years),
                                                      context=ghg, sub_context=emission_type)
                        if str(year_n + model.step) in model.years:  # future years are out of range
                            next_tax = model.get_param('tax', node, str(year_n + model.step),
                                                       context=ghg, sub_context=emission_type)
                        else:
                            next_tax = cur_tax
                        tax_vals.extend(linspace(cur_tax, next_tax, model.step, endpoint=False))
                    expected_ec = sum(tax_vals) / lifetime  # take average of all taxes

                else:
                    raise ValueError(
                        'Foresight method not identified, use Myopic, Discounted, or Average')

                emissions_cost[node_name][ghg][emission_type]['year_value'] *= expected_ec

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
    model.graph.nodes[node][year]['technologies'][tech]['net_emissions_rate'] = \
        Emissions(emissions=net_emissions)
    model.graph.nodes[node][year]['technologies'][tech]['avoided_emissions_rate'] = \
        Emissions(emissions=avoided_emissions)
    model.graph.nodes[node][year]['technologies'][tech]['negative_emissions_rate'] = \
        Emissions(emissions=negative_emissions)
    model.graph.nodes[node][year]['technologies'][tech]['bio_emissions_rate'] = \
        Emissions(bio_emissions)

    # Record emission costs
    val_dict = utils.create_value_dict(year_val=total, param_source='calculation')
    model.set_param_internal(val_dict, 'emissions cost', node, year, tech)

    model.set_param_internal(EmissionsCost(emissions_cost),
                             'emissions_cost_rate', node, year, tech)

    return total
