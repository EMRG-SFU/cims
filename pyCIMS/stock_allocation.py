from . import utils
from .model import ProvidedQuantity

import warnings

#############################
# Stock Allocation
#############################
def get_existing_stock_per_tech(model, sub_graph, node, year):
    node_year_data = model.graph.nodes[node][year]

    existing_stock_per_tech = {}
    for tech in node_year_data['technologies']:
        tech_existing = model.calc_existing_stock(sub_graph, node, year, tech)
        existing_stock_per_tech[tech] = tech_existing

    return existing_stock_per_tech


def calc_new_stock_demanded(demand, existing_stock):
    for stock in existing_stock.values():
        demand -= stock
    return demand


def calc_surplus_retirement_proportion(surplus, vintage_specific_existing_stock):
    if vintage_specific_existing_stock <= 0:
        retirement_proportion = 0
    else:
        retirement_proportion = max(0, min(surplus/vintage_specific_existing_stock, 1))
    return retirement_proportion


def retire_surplus_stock(model, node, year, new_stock_demanded, existing_stock):
    surplus = -1 * new_stock_demanded

    # Base Stock Retirement
    total_base_stock = 0
    for tech in existing_stock:
        tech_base_stock = model.get_param('base_stock_remaining', node, year, tech)
        total_base_stock += tech_base_stock

    if total_base_stock > 0:
        retirement_proportion = calc_surplus_retirement_proportion(surplus, total_base_stock)
        for tech in existing_stock:
            tech_base_stock = model.get_param('base_stock_remaining', node, year, tech)
            amount_tech_to_retire = tech_base_stock * retirement_proportion
            existing_stock[tech] -= amount_tech_to_retire
            surplus -= amount_tech_to_retire
            new_stock_demanded += amount_tech_to_retire

    # New Stock Retirement
    possible_purchase_years = [y for y in model.years if (int(y) > model.base_year) &
                                                         (int(y) < int(year))]
    for purchase_year in possible_purchase_years:
        total_new_stock_remaining_pre_surplus = 0
        if surplus > 0:
            for tech in existing_stock:
                tech_remaining_stock_pre_surplus = model.get_param('new_stock_remaining_pre_surplus', node, year, tech)[purchase_year]
                total_new_stock_remaining_pre_surplus += tech_remaining_stock_pre_surplus

        retirement_proportion = calc_surplus_retirement_proportion(surplus, total_new_stock_remaining_pre_surplus)
        for tech in existing_stock:
            tech_data = model.graph.nodes[node][year]['technologies'][tech]
            tech_remaining_stock_pre_surplus = model.get_param('new_stock_remaining_pre_surplus', node, year, tech)[purchase_year]
            amount_tech_to_retire = tech_remaining_stock_pre_surplus * retirement_proportion
            # Remove from existing stock
            existing_stock[tech] -= amount_tech_to_retire
            # Remove from surplus & new stock demanded
            surplus -= amount_tech_to_retire
            new_stock_demanded += amount_tech_to_retire
            # note new stock remaining (post surplus) in the model
            tech_data['new_stock_remaining']['year_value'][purchase_year] -= amount_tech_to_retire


def calculate_new_market_shares(model, node, year):
    new_market_shares_per_tech = {}
    for tech in model.get_param('technologies', node, year):
        new_market_shares_per_tech[tech] = {}
        ms, ms_source = model.get_param('Market share', node, year, tech, return_source=True)

        if ms_source == 'model':  # 'model' -> exogenous
            new_market_share = ms
        else:
            new_market_share = 0
            first_year_available = model.get_param('Available', node, str(model.base_year), tech)
            first_year_unavailable = model.get_param('Unavailable', node, str(model.base_year), tech)
            if first_year_available <= int(year) < first_year_unavailable:
                v = model.get_param('Heterogeneity', node, year)
                tech_lcc = model.get_param('Life Cycle Cost', node, year, tech)
                total_weight = model.get_param('total_lcc_v', node, year)

                # TODO: Instead of calculating this in 2 places, set this value in
                #  lcc_calculation.py. Or here. Not both.
                if tech_lcc < 0.01:
                    # When lcc < 0.01, we will approximate it's weight using a TREND line
                    w1 = 0.1 ** (-1 * v)
                    w2 = 0.01 ** (-1 * v)
                    slope = (w2 - w1) / (0.01 - 0.1)
                    weight = slope * tech_lcc + (w1 - slope * 0.1)
                else:
                    weight = tech_lcc ** (-1 * v)

                new_market_share = weight / total_weight
        model.graph.nodes[node][year]['technologies'][tech]['base_stock'] = utils.create_value_dict(0, param_source='initialization')
        model.graph.nodes[node][year]['technologies'][tech]['new_stock'] = utils.create_value_dict(0, param_source='initialization')
        new_market_shares_per_tech[tech] = new_market_share

    return new_market_shares_per_tech


def calculate_total_market_shares(model, node, year, assessed_demand, existing_stock, new_market_shares):
    total_market_shares_by_tech = {}
    for tech in model.get_param('technologies', node, year):
        try:
            stock = existing_stock[tech]
        except KeyError:
            stock = 0

        tech_total_stock = existing_stock + new_market_shares[tech] * assessed_demand

        if assessed_demand == 0:
            if model.show_run_warnings:
                warnings.warn("Assessed Demand is 0 for {}[{}]".format(node, tech))
            total_market_share = 0
        else:
            total_market_share = tech_total_stock / assessed_demand

        total_market_shares_by_tech[tech] = total_market_share

    return total_market_shares_by_tech


def record_provided_quantities(model, year, node, tech, requested_services, assessed_demand,
                               market_share=1):
    if isinstance(requested_services, dict):
        requested_services = [requested_services]

    for service_data in requested_services:
        service_req_ratio = service_data['year_value']
        quant_requested = market_share * service_req_ratio * assessed_demand
        year_node = model.graph.nodes[service_data['branch']][year]
        if 'provided_quantities' not in year_node.keys():
            year_node['provided_quantities'] = utils.create_value_dict(ProvidedQuantity(),
                                                                       param_source='initialization')
        year_node['provided_quantities']['year_value'].provide_quantity(amount=quant_requested,
                                                                        requesting_node=node,
                                                                        requesting_technology=tech)


def record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares, new_stock_demanded, assessed_demand):
    for tech in model.get_param('technologies', node, year):
        nms = adjusted_new_ms[tech]
        model.graph.nodes[node][year]['technologies'][tech]['new_market_share'] = utils.create_value_dict(nms, param_source='calculation')

        if int(year) == model.base_year:
            model.graph.nodes[node][year]['technologies'][tech]['base_stock'] = utils.create_value_dict(new_stock_demanded * nms, param_source='calculation')
        else:
            model.graph.nodes[node][year]['technologies'][tech]['new_stock'] = utils.create_value_dict(new_stock_demanded * nms, param_source='calculation')

        tms = total_market_shares[tech]
        model.graph.nodes[node][year]['technologies'][tech]['total_market_share'] = utils.create_value_dict(tms, param_source='calculation')

        tech_data = model.get_param('technologies', node, year)[tech]
        if 'Service requested' in tech_data.keys():
            record_provided_quantities(services_being_requested, assessed_demand,
                                       technology=tech, technology_market_share=tms)
        elif 'Service requested' in model.graph.nodes[node][year].keys():
            services_being_requested = [v for k, v in model.graph.nodes[node][year]['Service requested'].items()]
            record_provided_quantities(services_being_requested, assessed_demand)


def tech_compete_allocation(model, sub_graph, node, year):
    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    assessed_demand = model.get_param('provided_quantities', node, year).get_total_quantity()

    # Existing Tech Specific Stocks -- find existing stock remaining after vintage-based retirement
    existing_stock_per_tech = get_existing_stock_per_tech(model, sub_graph, node, year)

    # Retrofits -- TODO

    # Capital Stock Availability -- Find how much new stock must be adopted to meet demand
    new_stock_demanded = calc_new_stock_demanded(assessed_demand, existing_stock_per_tech)

    # Surplus Retirement
    if new_stock_demanded < 0:
        retire_surplus_stock(model, node, year, new_stock_demanded, existing_stock_per_tech)

    # New Tech Competition
    new_market_shares_per_tech = calculate_new_market_shares(model, node, year)

    # Min/Max Market Share Limits
    adjusted_new_ms = apply_min_max_limits(model, node, year, new_market_shares_per_tech)

    # Calculate Total Market Shares
    total_market_shares = calculate_total_market_shares(model, node, year, assessed_demand,
                                                        existing_stock_per_tech, adjusted_new_ms)

    # Record Values in Model
    record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares,
                              new_stock_demanded, assessed_demand)

def node_tech_compete_allocation(model, sub_graph, node, year):
    pass


def general_allocation():
    pass


#############################
# Min/Max Market Share Limits
#############################
def get_min_max_limits(model, node, year):
    """
    Find the minimum & maximum market share limits in a given year for all technologies at a
    specified node in the model.

    Parameters
    ----------
    model : pyCIMS.Model
        The pyCIMS model containing the market share limits you want to retrieve.
    node : str
        The name of the node from which you want to retrieve the market share limits.
    year : str
        The year to retrieve market share limits value for.

    Returns
    -------
    dict :
        A dictionary mapping each technology at node to the a tuple containing the minimum and
        maximum market share limit for the specified year.
    """
    techs = model.graph.nodes[node][year]['technologies']
    min_max_limits = {}
    for tech in techs:
        min_nms = model.get_param('Market share new_Min', node, year, tech)
        max_nms = model.get_param('Market share new_Max', node, year, tech)
        min_max_limits[tech] = (min_nms, max_nms)
    return min_max_limits


def min_max_ms_compliant(new_market_shares, min_max_limits):
    """
    Determines whether a set of new market shares are compliant given the min/max limits for those
    technologies.

    To be compliant, each technologies' new market share must be greater than or equal to its
    minimum limit and less than or equal to its maximum limit.

    Parameters
    ----------
    new_market_shares : dict {str: float}
        The dictionary containing new market shares. Keys in the dictionary are technologies, values
        are proportions of the new stock allocated to that technology ([0, 1]).
    min_max_limits : dict {str: (float, float)}
        The dictionary containing minimum/maximum new market share limits. Keys are technologies,
        values are tuples which contain the minimum and maximum proportions of the new stock which
        can be allocated to that technology.

    Returns
    -------
    bool :
        True if the new market shares comply with the limits defined in min_max_limits. False
        otherwise.
    """
    for tech in new_market_shares:
        min_nms, max_nms = min_max_limits[tech]
        proposed_nms = new_market_shares[tech]

        if proposed_nms < min_nms:
            return False

        if proposed_nms > max_nms:
            return False

    return True


def get_percent_differences(new_market_shares, min_max_limits, return_sorted=True):
    """
    Finds the differences between each technology's new market share and the nearest new market
    share which would comply with the min_max_limits.

    If a new market share is already compliant, this difference will be 0. If the new market share
    is less than the minimum limit, the difference will be positive. If the new market share is
    greater than the maximum limit, the difference will be negative.

    Parameters
    ----------
    new_market_shares : dict
        The dictionary containing new market shares. Keys in the dictionary are technologies, values
        are proportions of the new stock allocated to that technology ([0, 1]).
    min_max_limits : dict
        The dictionary containing minimum/maximum new market share limits. Keys are technologies,
        values are tuples which contain the minimum and maximum proportions of the new stock which
        can be allocated to that technology.
    return_sorted : bool, optional
        Whether to sort the returned list by the absolute difference between the new market share
        and the nearest new market share which would comply with the min_max_limits.

    Returns
    -------
    list :
        A list of list of tuples. Each tuple contains (1) a technologies name and (2) the difference
        between its original new market share and the nearest compliant new market share.
    """
    percent_diffs = []
    for tech in new_market_shares:
        min_nms, max_nms = min_max_limits[tech]
        proposed_nms = new_market_shares[tech]

        if proposed_nms < min_nms:
            percent_diffs.append((tech, proposed_nms - min_nms))
        elif proposed_nms > max_nms:
            percent_diffs.append((tech, proposed_nms - max_nms))
        else:
            percent_diffs.append((tech, 0))

    if return_sorted:
        percent_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

    return percent_diffs


def make_ms_min_max_compliant(initial_nms, min_max):
    """
    Finds the nearest value to make a new market share compliant with minimum and maximum limits.

    Parameters
    ----------
    initial_nms : float
        An initial new market share, which may or may not comply with the minimum and maximum
        new market share limits.
    min_max : tuple of floats
        A tuple containing (1) the minimum new market share limit and (2) the maximum new
        market share limit.

    Returns
    -------
    float :
        The nearest value to initial_nms which is compliant with the minimum and maximum limits.
    """
    min_nms, max_nms = min_max

    if initial_nms < min_nms:
        return min_nms

    if initial_nms > max_nms:
        return max_nms

    return initial_nms


def adjust_new_market_shares(new_market_shares, limit_adjusted_techs):
    """
    Adjust the new market shares of remaining technologies (those that haven't been adjusted based
    on their min/max limits).

    Parameters
    ----------
    new_market_shares : dict
        The dictionary containing new market shares. Keys in the dictionary are technologies, values
        are proportions of the new stock allocated to that technology ([0, 1]).

    limit_adjusted_techs : list of str
        The list of technologies which have been adjusted to comply with their min/max market share
        limits.

    Returns
    -------
    dict :
        An updated version of new_market_shares, where technologies that weren't set using min/max
        limits have been adjusted.
    """
    remaining_techs = [t for t in new_market_shares if t not in limit_adjusted_techs]

    sum_msj = sum([new_market_shares[t] for t in remaining_techs])
    sum_msl = sum([new_market_shares[t] for t in limit_adjusted_techs])
    adjust_amount = 1 - sum_msl
    for remaining_tech in remaining_techs:
        if adjust_amount > 0:
            new_market_share_h = new_market_shares[remaining_tech]
            anms_h = (new_market_share_h / sum_msj) * (1 - sum_msl)
        else:
            anms_h = 0
        new_market_shares[remaining_tech] = anms_h

    return new_market_shares


def find_eligible_market_shares(model, node, year, new_market_shares):
    """
    Finds the technologies whose market shares are eligible for adjustment. To be eligible for
    adjustment, the technology's market share mustn't be exogenously defined and the technology must
    be available in the relevant year.

    Parameters
    ----------
    model : pyCIMS.Model
        The pyCIMS model containing node.
    node : str
        The name of the node housing the market shares which may be eligible for adjustment.
    year : str
        The year containing the market shares of interest.
    new_market_shares : dict
        The dictionary containing new market shares. Keys in the dictionary are technologies, values
        are proportions of the new stock allocated to that technology ([0, 1]).

    Returns
    -------
    dict :
        A filtered version of the new_market_shares dictionary, which only contains technologies
        which are not exogenously defined and are available in the given year.
    """
    eligible_market_shares = {}
    for tech in new_market_shares:
        is_exogenous = utils.is_param_exogenous(model, 'Market share', node, year, tech)

        first_year_available = model.get_param('Available', node, year, tech)
        first_year_unavailable = model.get_param('Unavailable', node, year, tech)
        is_available = first_year_available <= int(year) < first_year_unavailable

        if (not is_exogenous) and is_available:
            eligible_market_shares[tech] = new_market_shares[tech]

    return eligible_market_shares


def apply_min_max_limits(model, node, year, new_market_shares):
    """
    Apply minimum & maximum market share limits to new market share percentages, adjusting final
    percentages to comply with the min/max limits.

    Parameters
    ----------
    model : pyCIMS.Model
        The pyCIMS model containing node.
    node : str
        The name of the node housing the market shares which limits will be applied.
    year : str
        The year containing the market shares of interest.
    new_market_shares : dict
        The dictionary containing new market shares. Keys in the dictionary are technologies, values
        are proportions of the new stock allocated to that technology ([0, 1]).

    Returns
    -------
    dict :
        An updated version of the new_market_shares dictionary, where endogeneous market shares
        comply with min/max market share limits.
    """
    min_max_limits = get_min_max_limits(model, node, year)

    # Only check & adjust new market shares which are not exogenous
    adjusted_nms = find_eligible_market_shares(model, node, year, new_market_shares)

    # Check to see if all New M/S values comply with Min/Max limits. If yes, do nothing. If no
    # continue to next step.
    limit_adjusted_techs = []
    while not min_max_ms_compliant(adjusted_nms, min_max_limits):
        # Apply exogenous Min or Max New M/S limit values on the technology which has the largest
        # % difference between its limit and its initial new market share value.
        percent_differences = get_percent_differences(adjusted_nms,
                                                      min_max_limits,
                                                      return_sorted=True)
        largest_violator = percent_differences[0]
        violator_name = largest_violator[0]
        adjusted_nms[violator_name] = make_ms_min_max_compliant(adjusted_nms[violator_name],
                                                                min_max_limits[violator_name])
        limit_adjusted_techs.append(violator_name)

        # For remaining technologies, calculate their individual Adjusted New M/S for technology(s)
        adjusted_nms = adjust_new_market_shares(adjusted_nms, limit_adjusted_techs)

    updated_nms = new_market_shares.copy()
    updated_nms.update(adjusted_nms)

    return updated_nms
