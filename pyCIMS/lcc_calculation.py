"""
This module contains the functions for LCC Calculations for the pyCIMS model.
"""
import warnings
import math
from .emissions import EmissionRates
from . import utils
from typing import TYPE_CHECKING


def lcc_calculation(sub_graph, node, year, model):
    """
    Determines economic parameters for `node` in `year` and stores the values in the sub_graph
    at the appropriate node. Specifically,

    Determines the node's:
    * Total Life Cycle Cost (weighted using total market share across all technologies)
    * Sum of Life Cycle Costs raised to the negative variance

    Determines each of the node's technology's:
    * Service cost
    * CRF
    * Full capital cost
    * Life Cycle Cost

    Initializes new_market_share and total_market_share for technologies where market share
    is exogenously defined.

    Parameters
    ----------
    sub_graph : NetworkX.Graph
        The subgraph where parameters will be stored.

    node : str
        The name of the node whose parameters we are calculating.

    year : str
        The year for which we are calculating parameters.

    Returns
    -------
        None. Produces side effects of updating the node in sub_graph to have parameter values.
    """
    # Check if the node has an exogenously defined Life Cycle Cost
    if 'Life Cycle Cost' in sub_graph.nodes[node][year]:
        lcc, lcc_source = model.get_param('Life Cycle Cost', node, year, return_source=True)
        if lcc_source == 'model':
            return

    # Check if the node is a tech compete node:
    if model.get_param('competition type', node) in ['tech compete', 'node tech compete', 'market']:
        total_lcc_v = 0.0
        v = model.get_param('Heterogeneity', node, year)

        # Get all of the technologies in the node
        node_techs = sub_graph.nodes[node][year]["technologies"].keys()

        # For every tech in the node, retrieve or compute required economic values
        for tech in node_techs:
            # Service Cost
            # ************
            annual_service_cost, sc_source = model.get_or_calc_param('Service cost',
                                                                     node, year, tech,
                                                                     return_source=True)
            val_dict = {'year_value': annual_service_cost,
                        "branch": str(node),
                        'param_source': sc_source}
            model.set_param_internal(val_dict, 'Service cost', node, year, tech)

            # CRF
            # ************
            crf, crf_source = model.get_or_calc_param("CRF",
                                                      node, year, tech,
                                                      return_source=True)
            val_dict = {'year_value': crf, 'param_source': crf_source}
            model.set_param_internal(val_dict, 'CRF', node, year, tech)

            # LCC (financial)
            # ************
            lcc, lcc_source = model.get_or_calc_param('Life Cycle Cost',
                                                      node, year, tech,
                                                      return_source=True)
            val_dict = {'year_value': lcc, 'param_source': lcc_source}
            model.set_param_internal(val_dict, 'Life Cycle Cost', node, year, tech)

            # Complete LCC
            # ************
            complete_lcc, complete_lcc_source = model.get_or_calc_param('Complete Life Cycle Cost',
                                                                        node, year, tech,
                                                                        return_source=True)
            val_dict = {'year_value': complete_lcc, 'param_source': complete_lcc_source}
            model.set_param_internal(val_dict, 'Complete Life Cycle Cost', node, year, tech)

            # If the technology is available in this year, add to the total LCC^-v value.
            first_year_avail= model.get_param('Available', node, str(model.base_year), tech)
            first_year_unavail = model.get_param('Unavailable', node, str(model.base_year), tech)
            if first_year_avail <= int(year) < first_year_unavail:
                # Life Cycle Cost ^ -v
                if lcc < 0.01:
                    # When lcc < 0.01, we will approximate it's weight using a TREND line
                    w1 = 0.1 ** (-1 * v)
                    w2 = 0.01 ** (-1 * v)
                    slope = (w2 - w1) / (0.01 - 0.1)
                    weight = slope * lcc + (w1 - slope * 0.1)
                else:
                    weight = lcc ** (-1 * v)

                total_lcc_v += weight

        # Set sum of Life Cycle Cost raised to negative variance
        val_dict = utils.create_value_dict(total_lcc_v, param_source='calculation')
        sub_graph.nodes[node][year]["total_lcc_v"] = val_dict

        # Weighted Life Cycle Cost
        # ************************
        weighted_lccs = 0
        # For every tech, use a exogenous or previously calculated market share to calculate Life
        # Cycle Cost
        for tech in node_techs:
            # Determine whether Market share is exogenous or not
            ms, ms_source = model.get_param('Market share', node, year, tech, return_source=True)
            ms_exogenous = ms_source == 'model'

            # Determine what market share to use for weighing Life Cycle Costs
            # If market share is exogenous, set new & total market share to exogenous value
            if ms_exogenous:
                val_dict = utils.create_value_dict(ms, param_source='calculation')
                model.set_param_internal(val_dict, 'new_market_share', node, year, tech)
                model.set_param_internal(val_dict, 'total_market_share', node, year, tech)
            market_share = model.get_param('total_market_share', node, year, tech)

            # Weight Life Cycle Cost and Add to Node Total
            # ********************************************
            curr_lcc = model.get_param('Life Cycle Cost', node, year, tech)
            weighted_lccs += market_share * curr_lcc

        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {
            service_name: utils.create_value_dict(weighted_lccs, param_source='calculation')}

    else:
        # When calculating a service cost for a technology or node using the "Fixed Ratio" decision
        # rule, multiply the Life Cycle Costs of the service required by its "Service Requested"
        # line value. Sometimes, the Service Requested line values act as percent shares that add up
        # to 1 for a given fixed ratio decision node. Other times, they do not and the Service
        # Requested Line values sum to numbers greater or less than 1.
        service_cost, sc_source = model.get_or_calc_param('Service cost', node, year,
                                                          return_source=True)
        service_name = node.split('.')[-1]
        sub_graph.nodes[node][year]["Life Cycle Cost"] = {
            service_name: utils.create_value_dict(service_cost, param_source=sc_source)}


def calc_financial_lcc(model: "pyCIMS.Model", node: str, year: str, tech: str) -> float:
    """
    Calculate the Financial Life Cycle Cost (called "Life Cycle Cost" in the model & model
    description). This LCC does not contain intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial LCC.
    node : The name of the node for which financial LCC will be calculated.
    year : The year to calculate financial LCC for.
    tech : The technology to calculate financial LCC for.

    Returns
    -------
    float : The financial LCC, calculated as fLCC = fUC + fAC + SC + EC

    See Also
    --------
    calc_complete_lcc: Calculates complete LCC, which includes intangible costs.
    """
    upfront_cost = model.get_or_calc_param('Financial Upfront cost', node, year, tech)
    annual_cost = model.get_or_calc_param('Financial Annual cost', node, year, tech)
    annual_service_cost = model.get_or_calc_param('Service cost', node, year, tech)
    emissions_cost = model.get_or_calc_param('Emissions cost', node, year, tech)
    lcc = upfront_cost + annual_cost + annual_service_cost + emissions_cost
    return lcc


def calc_complete_lcc(model: "pyCIMS.Model", node: str, year: str, tech: str) -> float:
    """
    Calculate Complete Life Cycle Cost. This LCC includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete LCC.
    node : The name of the node for which complete LCC will be calculated.
    year : The year to calculate complete LCC for.
    tech : The technology to calculate complete LCC for.

    Returns
    -------
    float : The complete LCC, calculated as cLCC = cUC + cAC + SC + EC

    See Also
    --------
    calc_financial_lcc: Calculates financial LCC, which does not include intangible costs.
    """
    complete_upfront_cost = model.get_or_calc_param('Complete Upfront cost', node, year, tech)
    complete_annual_cost = model.get_or_calc_param('Complete Annual cost', node, year, tech)
    annual_service_cost = model.get_or_calc_param('Service cost', node, year, tech)
    emissions_cost = model.get_or_calc_param('Emissions cost', node, year, tech)

    complete_lcc = complete_upfront_cost + complete_annual_cost + annual_service_cost + \
                   emissions_cost

    return complete_lcc


def calc_emissions_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
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

    Returns
    -------
    float : the total emission cost. Has the side effect of updating the Emissions Cost,
            net_emission_rates, captured_emission_rates, and bio_emission_rates in the model.
    """
    def duplicate_tax_dict(dct):
        new_dict = {}
        for ghg in dct:
            if ghg not in new_dict:
                new_dict[ghg] = {}
            for emission_type in dct[ghg]:
                original_val = dct[ghg][emission_type]['year_value']
                new_dict[ghg][emission_type] = utils.create_value_dict(original_val)
        return new_dict

    def duplicate_emissions_dict(dct):
        new_dict = {}

        for source_branch in dct:
            if source_branch not in new_dict:
                new_dict[source_branch] = {}
            for ghg in dct[source_branch]:
                if ghg not in new_dict[source_branch]:
                    new_dict[source_branch][ghg] = {}
                for emission_type in dct[source_branch][ghg]:
                    original_val = dct[source_branch][ghg][emission_type]['year_value']
                    new_dict[source_branch][ghg][emission_type] = \
                        utils.create_value_dict(original_val)
        return new_dict

    fuels = model.fuels

    # No tax rate at all or node is a fuel
    if 'Tax' not in model.graph.nodes[node][year] or node in fuels:
        return 0

    # Initialize all taxes and emission removal rates to 0
    # example of item in tax_rates -> {'CO2': {'Combustion': 5}}
    tax_rates = {ghg: {em_type: utils.create_value_dict(0) for em_type in model.emission_types} for
                 ghg in model.GHGs}
    removal_rates = duplicate_tax_dict(tax_rates)

    # Grab correct tax values
    all_taxes = model.get_param('Tax', node, year)  # returns a dict
    for ghg in all_taxes:
        for emission_type in all_taxes[ghg]:
            if ghg not in tax_rates:
                tax_rates[ghg] = {}
            tax_rates[ghg][emission_type] = utils.create_value_dict(
                all_taxes[ghg][emission_type]['year_value'])

    # EMISSIONS tech level
    total_emissions = {}
    total = 0
    if 'Emissions' in model.graph.nodes[node][year]['technologies'][tech]:
        total_emissions[tech] = {}
        emission_data = model.graph.nodes[node][year]['technologies'][tech]['Emissions']

        for ghg in emission_data:
            for emission_type in emission_data[ghg]:
                if ghg not in total_emissions[tech]:
                    total_emissions[tech][ghg] = {}
                total_emissions[tech][ghg][emission_type] = utils.create_value_dict(
                    emission_data[ghg][emission_type]['year_value'])

    # EMISSIONS REMOVAL tech level
    if 'Emissions removal' in model.graph.nodes[node][year]:
        removal_dict = model.graph.nodes[node][year]['Emissions removal']
        for ghg in removal_dict:
            for emission_type in removal_dict[ghg]:
                if ghg not in removal_rates:
                    removal_rates[ghg] = {}
                removal_rates[ghg][emission_type] = utils.create_value_dict(
                    removal_dict[ghg][emission_type]['year_value'])

    # Check all services requested for
    if 'Service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['Service requested']

        # EMISSIONS child level
        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']
            if 'Emissions' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                fuel_emissions = model.graph.nodes[child_node][year]['Emissions']
                total_emissions[child_node] = {}
                for ghg in fuel_emissions:
                    for emission_type in fuel_emissions[ghg]:
                        if ghg not in total_emissions[child_node]:
                            total_emissions[child_node][ghg] = {}
                        total_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                fuel_emissions[ghg][emission_type]['year_value'] * req_val)

    gross_emissions = duplicate_emissions_dict(total_emissions)

    if 'Service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['Service requested']

        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']

            # GROSS EMISSIONS
            if 'Emissions biomass' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                gross_dict = model.graph.nodes[child_node][year]['Emissions biomass']
                for ghg in gross_dict:
                    for emission_type in gross_dict[ghg]:
                        gross_emissions[child_node][ghg][emission_type] = \
                            utils.create_value_dict(
                                gross_dict[ghg][emission_type]['year_value'] * req_val)

            # EMISSIONS REMOVAL child level
            if 'technologies' in model.graph.nodes[child_node][year]:
                child_techs = model.graph.nodes[child_node][year]['technologies']
                for _, tech_data in child_techs.items():
                    if 'Emissions removal' in tech_data:
                        removal_dict = tech_data['Emissions removal']
                        for ghg in removal_dict:
                            for emission_type in removal_dict[ghg]:
                                removal_rates[ghg][emission_type] = \
                                    utils.create_value_dict(
                                        removal_dict[ghg][emission_type]['year_value'])

    # CAPTURED EMISSIONS
    captured_emissions = duplicate_emissions_dict(gross_emissions)
    for node_name in captured_emissions:
        for ghg in captured_emissions[node_name]:
            for emission_type in captured_emissions[node_name][ghg]:
                em_removed = removal_rates[ghg][emission_type]
                captured_emissions[node_name][ghg][emission_type]['year_value'] *= em_removed[
                    'year_value']

    # NET EMISSIONS
    net_emissions = duplicate_emissions_dict(total_emissions)
    for node_name in net_emissions:
        for ghg in net_emissions[node_name]:
            for emission_type in net_emissions[node_name][ghg]:
                net_emissions[node_name][ghg][emission_type]['year_value'] -= \
                    captured_emissions[node_name][ghg][emission_type]['year_value']

    # EMISSIONS COST
    emissions_cost = duplicate_emissions_dict(net_emissions)
    for node_name in emissions_cost:
        for ghg in emissions_cost[node_name]:
            for emission_type in emissions_cost[node_name][ghg]:
                tax_name = tax_rates[ghg][emission_type]
                emissions_cost[node_name][ghg][emission_type]['year_value'] *= tax_name[
                    'year_value']

    # Add everything in nested dictionary together
    for node_name in emissions_cost:
        for ghg in emissions_cost[node_name]:
            for emission_type in emissions_cost[node_name][ghg]:
                total += emissions_cost[node_name][ghg][emission_type]['year_value']

    # BIO EMISSIONS tech level
    bio_emissions = {}
    if 'Emissions biomass' in model.graph.nodes[node][year]['technologies'][tech]:
        bio_emissions[tech] = {}
        bio_emission_data = model.graph.nodes[node][year]['technologies'][tech]['Emissions biomass']

        for ghg in bio_emission_data:
            for emission_type in bio_emission_data[ghg]:
                if ghg not in bio_emissions[tech]:
                    bio_emissions[tech][ghg] = {}
                bio_emissions[tech][ghg][emission_type] = utils.create_value_dict(
                    bio_emission_data[ghg][emission_type]['year_value'])

    # Check all services requested for
    if 'Service requested' in model.graph.nodes[node][year]['technologies'][tech]:
        data = model.graph.nodes[node][year]['technologies'][tech]['Service requested']

        # BIO EMISSIONS child level
        for child_info in data.values():
            req_val = child_info['year_value']
            child_node = child_info['branch']
            if 'Emissions biomass' in model.graph.nodes[child_node][
                year] and child_node in fuels and req_val > 0:
                fuel_emissions = model.graph.nodes[child_node][year]['Emissions biomass']
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
    model.set_param_internal(val_dict, 'Emissions cost', node, year, tech)

    return total


def calc_complete_upfront_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates complete upfront cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete upfront cost.
    node : The node to calculate complete upfront cost for.
    year : The year to calculate complete upfront cost for.
    tech : The technology to calculate complete upfront cost for.

    Returns
    -------
    float : The complete upfront cost, defined as
            cUC = (CC - AllocC, + UIC_fixed + UIC_declining) / output * CRF

    See Also
    --------
    calc_financial_upfront_cost
    """
    crf = model.get_or_calc_param("CRF", node, year, tech)
    capital_cost = model.get_or_calc_param('Capital cost', node, year, tech)
    allocated_cost = model.get_param('Allocated cost', node, year, tech)
    fixed_uic = model.get_or_calc_param('Upfront intangible cost_fixed', node, year, tech)
    declining_uic = calc_declining_uic(model, node, year, tech)
    output = model.get_param('Output', node, year, tech)

    complete_uc = (capital_cost - allocated_cost + fixed_uic + declining_uic) / output * crf

    return complete_uc


def calc_financial_upfront_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates financial upfront cost, which does not include intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial upfront cost.
    node : The node to calculate financial upfront cost for.
    year : The year to calculate financial upfront cost for.
    tech : The technology to calculate financial upfront cost for.

    Returns
    -------
    float : The financial upfront cost, defined as fUC = (CC - AllocC) / output * CRF

    See Also
    --------
    calc_complete_upfront_cost
    """

    crf = model.get_or_calc_param("CRF", node, year, tech)
    capital_cost = model.get_or_calc_param('Capital cost', node, year, tech)
    allocated_cost = model.get_param('Allocated cost', node, year, tech)
    output = model.get_param('Output', node, year, tech)

    financial_uc = (capital_cost - allocated_cost) / output * crf

    return financial_uc


def calc_complete_annual_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates complete annual cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of complete annual cost.
    node : The node to calculate complete annual cost for.
    year : The year to calculate complete annual cost for.
    tech : The technology to calculate complete annual cost for.

    Returns
    -------
    float : The complete annual cost, defined as cAC = (OM + AIC_fixed + AIC_declining) / output

    See Also
    --------
    calc_financial_annual_cost
    """
    output = model.get_param('Output', node, year, tech)
    operating_maintenance_cost = model.get_param('Operating and maintenance cost', node, year, tech)
    fixed_aic = model.get_param('Annual intangible cost_fixed', node, year, tech)
    declining_aic = model.get_or_calc_param('Annual intangible cost_declining', node, year, tech)

    complete_ac = (operating_maintenance_cost +
                   fixed_aic +
                   declining_aic) / output

    return complete_ac


def calc_financial_annual_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates financial annual cost, which includes intangible costs.

    Parameters
    ----------
    model : The model containing component parts of financial annual cost.
    node : The node to calculate financial annual cost for.
    year : The year to calculate financial annual cost for.
    tech : The technology to calculate financial annual cost for.

    Returns
    -------
    float : The financial annual cost, defined as fAC = OM / output

    See Also
    --------
    calc_complete_annual_cost
    """

    output = model.get_param('Output', node, year, tech)
    operating_maintenance_cost = model.get_param('Operating and maintenance cost', node, year, tech)
    financial_ac = operating_maintenance_cost / output
    return financial_ac


def calc_capital_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates capital cost.

    Parameters
    ----------
    model : The model containing component parts of capital cost.
    node : The node to calculate capital cost for.
    year : The year to calculate capital cost for.
    tech : The technology to calculate capital cost for.

    Returns
    -------
    float : Capital cost, defined as CC = max{CC_declining, CC_overnight * CC_declining_limit}.
    """

    cc_overnight = model.get_param('Capital cost_overnight', node, year, tech)
    declining_cc_limit = model.get_param('Capital cost_declining_limit', node, year, tech)
    declining_cc = model.get_or_calc_param("Capital cost_declining", node, year, tech)

    if declining_cc is None:
        capital_cost = cc_overnight
    else:
        capital_cost = max(declining_cc, cc_overnight * declining_cc_limit)
    return capital_cost


def calc_declining_cc(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculates declining capital cost.

    Parameters
    ----------
    model : The model containing component parts of declining capital cost.
    node : The node to calculate declining capital cost for.
    year : The year to calculate declining capital cost for.
    tech : The technology to calculate declining capital cost for.

    Returns
    -------
    float : The declining capital cost.
    """
    dcc_class = model.get_param('Capital cost_declining_Class', node, year, tech,
                                sub_param='context')

    if dcc_class is None:
        cc_declining = None

    else:
        # Progress Ratio
        progress_ratio = model.get_param('Capital cost_declining_Progress Ratio', node, year, tech)
        gcc_t = model.get_or_calc_param('GCC_t', node, year, tech)

        dcc_class_techs = model.dcc_classes[dcc_class]

        # Cumulative New Stock in DCC Class
        # 'Capital cost_declining_cumulative new stock' already given in vkt, so no need to convert
        cns = model.get_param('Capital cost_declining_cumulative new stock', node, year, tech)

        bs_sum = 0
        ns_sum = 0
        for node_k, tech_k in dcc_class_techs:
            # Need to convert stocks for transportation techs to common vkt unit
            unit_convert = model.get_param('Load Factor', node_k, str(model.base_year), tech_k)
            if unit_convert is None:
                unit_convert = 1

            # Base Stock summed over all techs in DCC class (base year only)
            bs_k = model.get_param('base_stock', node_k, str(model.base_year), tech_k)
            if bs_k is not None:
                bs_sum += bs_k / unit_convert

            # New Stock summed over all techs in DCC class and over all previous years
            # (excluding base year)
            year_list = [str(x) for x in
                         range(int(model.base_year) + int(model.step), int(year), int(model.step))]
            for j in year_list:
                ns_jk = model.get_param('new_stock', node_k, j, tech_k)
                ns_sum += ns_jk / unit_convert

        # Calculate Declining Capital Cost
        inner_sums = (cns + bs_sum + ns_sum) / (cns + bs_sum)
        cc_declining = gcc_t * (inner_sums ** math.log(progress_ratio, 2))

    return cc_declining


def calc_gcc(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate GCC, which is the capital cost adjusted for cumulative stock in other countries.

    Parameters
    ----------
    model : The model containing component parts of GCC.
    node : The node to calculate GCC for.
    year : The year to calculate GCC for.
    tech : The technology to calculate GCC for.

    Returns
    -------
    float : The GCC. If year is the base year, GCC = CC_overnight. Otherwise,
            GCC_t = GCC_(t-5) * (1 - AEEI)^5.
    """
    previous_year = str(int(year) - model.step)
    if previous_year in model.years:
        aeei = model.get_param('Capital cost_declining_AEEI', node, year, tech)
        gcc = ((1 - aeei) ** model.step) * \
              calc_gcc(model, node, previous_year, tech)
    else:
        cc_overnight = model.get_param('Capital cost_overnight', node, year, tech)
        gcc = cc_overnight

    return gcc


def calc_declining_uic(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate Upfront Declining Intangible Cost (UIC_declining).

    Parameters
    ----------
    model : The model containing component parts of declining UIC.
    node : The node to calculate declining UIC for.
    year : The year to calculate declining UIC for.
    tech : The technology to calculate declining UIC for.

    Returns
    -------
    float : The declining UIC.
    """
    # Retrieve Exogenous Terms from Model Description
    initial_uic = model.get_param('Upfront intangible cost_declining_initial', node, year, tech)
    rate_constant = model.get_param('Upfront intangible cost_declining_rate', node, year, tech)
    shape_constant = model.get_param('Upfront intangible cost_declining_shape', node, year, tech)

    # Calculate Declining UIC
    if int(year) == int(model.base_year):
        return_uic = initial_uic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech)

        try:
            denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        except OverflowError as overflow:
            print(node, year, shape_constant, rate_constant, prev_nms)
            raise overflow

        prev_uic_declining = calc_declining_uic(model, node, prev_year, tech)
        uic_declining = min(prev_uic_declining, initial_uic / denominator)

        return_uic = uic_declining

    return return_uic


def calc_declining_aic(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate Annual Declining Intangible Cost (declining AIC).

    Parameters
    ----------
    model : The model containing component parts of declining AIC.
    node : The node to calculate declining AIC for.
    year : The year to calculate declining AIC for.
    tech : The technology to calculate declining AIC for.

    Returns
    -------
    float : The declining AIC.
    """
    # Retrieve Exogenous Terms from Model Description
    initial_aic = model.get_param('Annual intangible cost_declining_initial', node, year, tech)
    rate_constant = model.get_param('Annual intangible cost_declining_rate', node, year, tech)
    shape_constant = model.get_param('Annual intangible cost_declining_shape', node, year, tech)

    # Calculate Declining AIC
    if int(year) == int(model.base_year):
        return_val = initial_aic
    else:
        prev_year = str(int(year) - model.step)
        prev_nms = model.get_param('new_market_share', node, prev_year, tech)

        try:
            denominator = 1 + shape_constant * math.exp(rate_constant * prev_nms)
        except OverflowError as overflow:
            print(node, year, shape_constant, rate_constant, prev_nms)
            raise overflow

        prev_aic_declining = calc_declining_aic(model, node, prev_year, tech)
        aic_declining = min(prev_aic_declining, initial_aic / denominator)

        return_val = aic_declining

    return return_val


def calc_crf(model: 'pyCIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate the Capital Recovery Factor (CRF).

    Parameters
    ----------
    model : The model containing component parts of CRF.
    node : The node to calculate CRF for.
    year : The year to calculate CRF for.
    tech : The technology to calculate CRF for.

    Returns
    -------
    float : The CRF, defined as CRF = discount_rate / (1-(1+discount_rate)^-lifespan)
    """
    finance_discount = model.get_param('Discount rate_Financial', node, year, tech)
    payback_period = model.get_param('Capital recovery', node, year, tech)

    if finance_discount == 0:
        warnings.warn('Discount rate_Financial has value of 0 at {} -- {}'.format(node, tech))
        finance_discount = model.get_tech_parameter_default('Discount rate_Financial')

    crf = finance_discount / (1 - (1 + finance_discount) ** (-1.0 * payback_period))

    return crf


def calc_annual_service_cost(model: 'pyCIMS.Model', node: str, year: str, tech: str = None) -> float:
    """
    Find the service cost associated with a given technology.

    For each service being requested:
        i) If the service is a fuel, find the fuel price (Life Cycle Cost) and add it to the
           service cost.
       ii) Otherwise, use the service's financial Life Cycle Cost (already calculated).

    Parameters
    ----------
    model : The model containing component parts of service cost.
    node : The node to calculate service cost for.
    year : The year to calculate service cost for.
    tech : The technology to calculate service cost for.

    Returns
    -------
    float : The service cost, defined as SC = SUM_over_services(fLCC * request_amount)
    """

    def do_sc_calculation(service_requested):
        service_requested_value = service_requested['year_value']
        service_cost = 0
        if service_requested['branch'] in model.fuels:
            fuel_branch = service_requested['branch']

            if 'Life Cycle Cost' in model.graph.nodes[fuel_branch][year]:
                fuel_name = list(model.graph.nodes[fuel_branch][year]['Life Cycle Cost'].keys())[0]
                service_requested_lcc = \
                    model.graph.nodes[fuel_branch][year]['Life Cycle Cost'][fuel_name]['year_value']
            else:
                service_requested_lcc = model.get_node_parameter_default('Life Cycle Cost',
                                                                         'sector')
            try:
                fuel_name = fuel_branch.split('.')[-1]
                price_multiplier = model.graph.nodes[node][year]['Price Multiplier'][fuel_name][
                    'year_value']
            except KeyError:
                price_multiplier = 1
            service_requested_lcc *= price_multiplier
        else:
            service_requested_branch = service_requested['branch']
            if 'Life Cycle Cost' in model.graph.nodes[service_requested_branch][year]:
                service_name = service_requested_branch.split('.')[-1]
                service_requested_lcc = \
                    model.graph.nodes[service_requested_branch][year]['Life Cycle Cost'][
                        service_name]['year_value']
            else:
                # Encountering a non-visited node
                service_requested_lcc = 1

        service_cost += service_requested_lcc * service_requested_value

        return service_cost

    total_service_cost = 0
    graph = model.graph

    if tech is not None:
        data = graph.nodes[node][year]['technologies'][tech]
    else:
        data = graph.nodes[node][year]

    if 'Service requested' in data:
        service_req = data['Service requested']
        for req in service_req.values():
            total_service_cost += do_sc_calculation(req)

    return total_service_cost
