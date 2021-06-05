import math

from .quantities import ProvidedQuantity
from . import utils


#############################
# Stock Allocation
#############################
def all_tech_compete_allocation(model, node, year):
    comp_type = model.get_param('competition type', node)

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    assessed_demand = model.get_param('provided_quantities', node, year).get_total_quantity()

    # Existing Tech Specific Stocks -- find existing stock remaining after vintage-based retirement
    existing_stock = _get_existing_stock(model, node, year, comp_type)

    # Retrofits -- TODO (Will be implemented later)

    # Capital Stock Availability -- Find how much new stock must be adopted to meet demand
    new_stock_demanded = _calc_new_stock_demanded(assessed_demand, existing_stock)

    # Surplus Retirement
    if new_stock_demanded < 0:
        new_stock_demanded, existing_stock = _retire_surplus_stock(model, node, year,
                                                                   new_stock_demanded,
                                                                   existing_stock)

    # New Tech Competition
    new_market_shares = _calculate_new_market_shares(model, node, year, comp_type)

    # Min/Max Market Share Limits
    adjusted_new_ms = _apply_min_max_limits(model, node, year, new_market_shares)

    # Calculate Total Market Shares
    total_market_shares_per_tech = _calculate_total_market_shares(node,
                                                                  assessed_demand,
                                                                  new_stock_demanded,
                                                                  existing_stock,
                                                                  adjusted_new_ms)

    # TODO: put this back in place and make sure it's working properly
    # assert(round(sum(list(total_market_shares_per_tech.values())), 4) == 1)

    # Record Values in Model
    _record_allocation_results(model, node, year, adjusted_new_ms,
                               total_market_shares_per_tech, new_stock_demanded,
                               assessed_demand)


def general_allocation(model, node, year):
    node_year_data = model.graph.nodes[node][year]

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    if model.get_param('competition type', node) == 'root':
        assessed_demand = 1
    else:
        assessed_demand = model.get_param('provided_quantities', node, year).get_total_quantity()

    # Based on assessed demand, determine the amount this node requests from other services
    if 'technologies' in node_year_data:
        for tech, tech_data in node_year_data['technologies'].items():
            if 'Service requested' in tech_data.keys():
                services_being_requested = tech_data['Service requested']
                t_ms = tech_data['Market share']
                _record_provided_quantities(services_being_requested,
                                            assessed_demand,
                                            model, node, year,
                                            tech=tech,
                                            market_share=t_ms)

    elif 'Service requested' in node_year_data:
        # Calculate the provided_quantities being requested for each of the services
        services_being_requested = [v for k, v in node_year_data['Service requested'].items()]
        _record_provided_quantities(services_being_requested, assessed_demand, model, node, year)


#############################
# Stock Calculation
#############################
def _get_existing_stock(model, node, year, comp_type):
    node_year_data = model.graph.nodes[node][year]
    existing_stock = {}

    if comp_type == 'tech compete':
        for tech in node_year_data['technologies']:
            t_existing = _retire_old_stock(model, node, year, tech)
            existing_stock[(node, tech)] = t_existing

    elif comp_type == 'node tech compete':
        for child in node_year_data['technologies']:
            child_node = model.get_param('Service requested', node, year, child, sub_param='branch')
            child_year_data = model.graph.nodes[child_node][year]
            for tech in child_year_data['technologies']:
                t_existing = _retire_old_stock(model, child_node, year, tech)
                existing_stock[(child_node, tech)] = t_existing

    return existing_stock


def _calc_new_stock_demanded(demand, existing_stock):
    for e_stocks in existing_stock.values():
        demand -= e_stocks
    return demand


#############################
# Retirement
#############################
def _base_stock_retirement(model, node, tech, base_stock, initial_year, current_year, lifespan):
    # Calculate amount of remaining base stock after natural retirements
    remaining_rate = 1 - (int(current_year) - int(initial_year)) / lifespan
    naturally_unretired_base_stock = base_stock * remaining_rate

    # Retrieve amount of base stock in the previous year, after surplus retirement
    prev_year = str(int(current_year) - model.step)
    if int(prev_year) == int(initial_year):
        prev_year_unretired_base_stock = model.get_param('base_stock', node, prev_year, tech)
    else:
        prev_year_unretired_base_stock = model.get_param('base_stock_remaining', node,
                                                         prev_year, tech)

    base_stock_remaining = max(min(naturally_unretired_base_stock,
                                   prev_year_unretired_base_stock), 0)

    return base_stock_remaining


def _purchased_stock_retirement(model, node, tech, purchased_stock, purchased_year, current_year,
                                lifespan, intercept=-11.513):
    prev_year = str(int(current_year) - model.step)

    # Calculate the remaining purchased stock with only natural retirements
    prev_y_exponent = intercept * (1 - (int(prev_year) - int(purchased_year)) / lifespan)
    prev_y_fictional_purchased_stock_remain = purchased_stock / (1 + math.exp(prev_y_exponent))

    # Calculate Adjustment Multiplier
    adj_multiplier = 1

    if int(prev_year) > int(purchased_year):
        prev_y_unretired_new_stock = \
        model.get_param('new_stock_remaining', node, prev_year, tech)[
            purchased_year]

        if prev_y_fictional_purchased_stock_remain > 0:
            adj_multiplier = prev_y_unretired_new_stock / \
                             prev_y_fictional_purchased_stock_remain

    # Update the tech data
    tech_data = model.graph.nodes[node][current_year]['technologies'][tech]
    if 'adjustment_multiplier' not in tech_data:
        tech_data['adjustment_multiplier'] = {}
    tech_data['adjustment_multiplier'][purchased_year] = adj_multiplier

    # Calculate the remaining purchased stock
    exponent = intercept * (1 - (int(current_year) - int(purchased_year)) / lifespan)
    purchased_stock_remaining = purchased_stock / (1 + math.exp(exponent)) * adj_multiplier

    return purchased_stock_remaining


def _retire_old_stock(model, node, year, tech):
    earlier_years = [y for y in model.years if int(y) < int(year)]
    if len(earlier_years) == 0:
        return 0

    # Means we are not on the initial year & we need to calculate remaining base and new stock
    # (existing)
    existing_stock = 0
    remaining_base_stock = 0
    remaining_new_stock_pre_surplus = {}
    for earlier_year in earlier_years:
        tech_lifespan = model.get_param('Lifetime', node, earlier_year, tech)

        # Base Stock
        tech_base_stock_y = model.get_param('base_stock', node, earlier_year, tech)
        remain_base_stock_vintage_y = _base_stock_retirement(model, node, tech,
                                                             tech_base_stock_y, earlier_year, year,
                                                             tech_lifespan)
        remaining_base_stock += remain_base_stock_vintage_y
        existing_stock += remain_base_stock_vintage_y

        # New Stock
        tech_new_stock_y = model.get_param('new_stock', node, earlier_year, tech)
        remain_new_stock = _purchased_stock_retirement(model, node, tech,
                                                       tech_new_stock_y,
                                                       earlier_year, year,
                                                       tech_lifespan)
        remaining_new_stock_pre_surplus[earlier_year] = remain_new_stock
        existing_stock += remain_new_stock

    # Save to Graph
    model.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining'] = \
        utils.create_value_dict(remaining_base_stock, param_source='calculation')
    model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining_pre_surplus'] = \
        utils.create_value_dict(remaining_new_stock_pre_surplus, param_source='calculation')
    # Note: retired stock will be removed later from ['new_stock_remaining']
    model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining'] = \
        utils.create_value_dict(remaining_new_stock_pre_surplus, param_source='calculation')

    if model.get_param('competition type', node) == 'node tech compete':
        parent = '.'.join(node.split('.')[:-1])
        child = node.split('.')[-1]

        # TODO: change to set_param()
        if 'base_stock_remaining' in model.graph.nodes[parent][year]['technologies'][child].keys():
            model.graph.nodes[parent][year]['technologies'][child]['base_stock_remaining'][
                'year_value'] += remaining_base_stock
        else:
            model.graph.nodes[parent][year]['technologies'][child][
                'base_stock_remaining'] = utils.create_value_dict(remaining_base_stock,
                                                                  param_source='calculation')

        # TODO: change to set_param()
        if 'new_stock_remaining_pre_surplus' in model.graph.nodes[parent][year]['technologies'][
            child].keys():
            for vintage_year in remaining_new_stock_pre_surplus:
                model.graph.nodes[parent][year]['technologies'][child][
                    'new_stock_remaining_pre_surplus']['year_value'][vintage_year] += \
                    remaining_new_stock_pre_surplus[vintage_year]
        else:
            model.graph.nodes[parent][year]['technologies'][child][
                'new_stock_remaining_pre_surplus'] = utils.create_value_dict(
                remaining_new_stock_pre_surplus, param_source='calculation')

        # TODO: change to set_param()
        # Note: retired stock will be removed later from ['new_stock_remaining']
        if 'new_stock_remaining' in model.graph.nodes[parent][year]['technologies'][child].keys():
            for vintage_year in remaining_new_stock_pre_surplus:
                model.graph.nodes[parent][year]['technologies'][child]['new_stock_remaining'][
                    'year_value'][vintage_year] += remaining_new_stock_pre_surplus[vintage_year]
        else:
            model.graph.nodes[parent][year]['technologies'][child][
                'new_stock_remaining'] = utils.create_value_dict(remaining_new_stock_pre_surplus,
                                                                 param_source='calculation')

    return existing_stock


def _calc_surplus_retirement_proportion(surplus, vintage_specific_existing_stock):
    if vintage_specific_existing_stock <= 0:
        retirement_proportion = 0
    else:
        retirement_proportion = max(0, min(surplus / vintage_specific_existing_stock, 1))
    return retirement_proportion


def _retire_surplus_base_stock(model, node, year, existing_stock, surplus):
    total_base_stock = 0
    amount_surplus_to_retire = 0
    for node_branch, tech in existing_stock:
        tech_base_stock = model.get_param('base_stock_remaining', node_branch, year, tech)
        total_base_stock += tech_base_stock
    if total_base_stock != 0:
        retirement_proportion = _calc_surplus_retirement_proportion(surplus, total_base_stock)
        for node_branch, tech in existing_stock:
            tech_base_stock = model.get_param('base_stock_remaining', node_branch, year, tech)
            amount_tech_to_retire = tech_base_stock * retirement_proportion

            # Remove from existing stock
            existing_stock[(node_branch, tech)] -= amount_tech_to_retire

            # Add to stock to retire
            amount_surplus_to_retire += amount_tech_to_retire

            # Note early retirement in the model
            model.graph.nodes[node_branch][year]['technologies'][tech]['base_stock_remaining'][
                'year_value'] -= amount_tech_to_retire
            if node_branch != node:
                child = node_branch.split('.')[-1]
                model.graph.nodes[node][year]['technologies'][child]['base_stock_remaining'][
                    'year_value'] -= amount_tech_to_retire

    return amount_surplus_to_retire


def _retire_surplus_new_stock(model, node, year, existing_stock, surplus):
    possible_purchase_years = [y for y in model.years if (int(y) > model.base_year) &
                                                         (int(y) < int(year))]
    amount_surplus_to_retire = 0
    for purchase_year in possible_purchase_years:
        total_new_stock_pre_surplus = 0
        if surplus > 0:
            for node_branch, tech in existing_stock:
                tech_rem_new_stock_pre_surplus = \
                    model.get_param('new_stock_remaining_pre_surplus',
                                    node_branch, year, tech)[purchase_year]
                total_new_stock_pre_surplus += tech_rem_new_stock_pre_surplus

        retirement_proportion = _calc_surplus_retirement_proportion(surplus,
                                                                    total_new_stock_pre_surplus)

        for node_branch, tech in existing_stock:
            t_rem_new_stock_pre_surplus = model.get_param('new_stock_remaining_pre_surplus',
                                                          node_branch, year, tech, )[purchase_year]
            amount_tech_to_retire = t_rem_new_stock_pre_surplus * retirement_proportion

            # Remove from existing stock
            existing_stock[(node_branch, tech)] -= amount_tech_to_retire

            # Remove from surplus & new stock demanded
            surplus -= amount_tech_to_retire
            amount_surplus_to_retire += amount_tech_to_retire

            # note new stock remaining (post surplus) in the model
            model.graph.nodes[node_branch][year]['technologies'][tech]['new_stock_remaining'][
                'year_value'][purchase_year] -= amount_tech_to_retire
            if node_branch != node:
                child = node_branch.split('.')[-1]
                model.graph.nodes[node][year]['technologies'][child]['new_stock_remaining'][
                    'year_value'][purchase_year] -= amount_tech_to_retire

    return amount_surplus_to_retire


def _retire_surplus_stock(model, node, year, new_stock_demanded, existing_stock):
    surplus = -1 * new_stock_demanded

    # Base Stock Retirement
    base_stock_to_retire = _retire_surplus_base_stock(model, node, year, existing_stock,
                                                      new_stock_demanded)
    surplus -= base_stock_to_retire
    new_stock_demanded += base_stock_to_retire

    # New Stock Retirement
    new_stock_to_retire = _retire_surplus_new_stock(model, node, year, existing_stock, surplus)
    surplus -= new_stock_to_retire
    new_stock_demanded += new_stock_to_retire

    return new_stock_demanded, existing_stock


#############################
# Market Share Calculations
#############################
def _calculate_lcc_weight(tech_lcc, heterogeneity):
    if tech_lcc < 0.01:
        # When lcc < 0.01, we will approximate it's weight using a TREND line
        weight_1 = 0.1 ** (-1 * heterogeneity)
        weight_2 = 0.01 ** (-1 * heterogeneity)
        slope = (weight_2 - weight_1) / (0.01 - 0.1)
        weight = slope * tech_lcc + (weight_1 - slope * 0.1)
    else:
        weight = tech_lcc ** (-1 * heterogeneity)

    return weight


def _find_exogenous_market_shares(model, node, year):
    node_year_data = model.graph.nodes[node][year]
    exo_market_shares = {}
    for tech in node_year_data['technologies']:
        market_share, ms_source = model.get_param('Market share', node, year, tech,
                                                  return_source=True)
        if ms_source == 'model':  # model --> exogenous
            exo_market_shares[tech] = market_share
    return exo_market_shares


def _initialize_stocks(model, node, year, tech):
    model.graph.nodes[node][year]['technologies'][tech]['base_stock'] = \
        utils.create_value_dict(0, param_source='initialization')
    model.graph.nodes[node][year]['technologies'][tech]['new_stock'] = \
        utils.create_value_dict(0, param_source='initialization')


def _find_competing_techs(model, node, year, comp_type):
    node_year_data = model.graph.nodes[node][year]
    competing_technologies = []

    if comp_type == 'tech compete':
        for tech in node_year_data['technologies']:
            competing_technologies.append((node, tech))

    elif comp_type == 'node tech compete':
        for child in node_year_data['technologies']:
            child_node = model.get_param('Service requested', node, year, child, sub_param='branch')
            for tech in model.graph.nodes[child_node][year]['technologies']:
                competing_technologies.append((child_node, tech))

    return competing_technologies


def _find_competing_weights(model, year, competing_techs, heterogeneity):
    total_weight = 0
    weights = {}

    for node_branch, tech in competing_techs:
        year_avail = model.get_param('Available', node_branch, str(model.base_year), tech)
        year_unavail = model.get_param('Unavailable', node_branch, str(model.base_year), tech)
        if year_avail <= int(year) < year_unavail:
            tech_lcc = model.get_param('Life Cycle Cost', node_branch, year, tech)
            weight = _calculate_lcc_weight(tech_lcc, heterogeneity)
            weights[(node_branch, tech)] = weight
            total_weight += weight

    return total_weight, weights


def _calculate_new_market_shares(model, node, year, comp_type):
    heterogeneity = model.get_param('Heterogeneity', node, year)

    # Find each of the technologies which will be competed for
    competing_techs = _find_competing_techs(model, node, year, comp_type)

    # Find the weights that we will be using to calculate market share
    total_weight, tech_weights = _find_competing_weights(model, year, competing_techs,
                                                         heterogeneity)

    # Find the new market shares for each tech
    new_market_shares = _find_exogenous_market_shares(model, node, year)
    for tech_child in model.graph.nodes[node][year]['technologies']:
        if tech_child not in new_market_shares:
            new_market_shares[tech_child] = 0

            if comp_type == 'tech compete':
                if (node, tech_child) in tech_weights:
                    new_market_shares[tech_child] = tech_weights[(node, tech_child)] / total_weight

            elif comp_type == 'node tech compete':
                child_node = model.get_param('Service requested', node, year, tech_child,
                                             sub_param='branch')
                child_new_market_shares = _find_exogenous_market_shares(model, child_node, year)
                child_weights = {t: w for (n, t), w in tech_weights.items() if n == child_node}

                for tech in child_weights:
                    if tech not in child_new_market_shares:
                        new_market_share = child_weights[tech] / total_weight
                    else:
                        new_market_share = child_new_market_shares[tech]
                    new_market_shares[tech_child] += new_market_share
        _initialize_stocks(model, node, year, tech_child)

    return new_market_shares


def _calculate_total_market_shares(node, assessed_demand, new_stock_demanded,
                                   existing_stock, adjusted_new_ms):

    # Initialize Total Stock
    total_stocks = {t: 0 for t in adjusted_new_ms}
    # Add existing stocks
    for node_branch, tech in existing_stock:
        if node_branch == node:
            total_stocks[tech] += existing_stock[(node_branch, tech)]
        else:
            child = node_branch.split('.')[-1]
            total_stocks[child] += existing_stock[(node_branch, tech)]
    # Add new stocks
    for tech_child in adjusted_new_ms:
        total_stocks[tech_child] += adjusted_new_ms[tech_child] * new_stock_demanded

    # Market Share
    total_market_shares = {}
    for tech in total_stocks:
        if assessed_demand == 0:
            total_market_shares[tech] = 0
        else:
            total_market_shares[tech] = total_stocks[tech] / assessed_demand

    return total_market_shares


#############################
# Min/Max Market Share Limits
#############################
def _get_min_max_limits(model, node, year):
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


def _min_max_ms_compliant(new_market_shares, min_max_limits):
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


def _get_percent_differences(new_market_shares, min_max_limits, return_sorted=True):
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


def _make_ms_min_max_compliant(initial_nms, min_max):
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


def _adjust_new_market_shares(new_market_shares, limit_adjusted_techs):
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


def _find_eligible_market_shares(model, node, year, new_market_shares):
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


def _apply_min_max_limits(model, node, year, new_market_shares):
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
    min_max_limits = _get_min_max_limits(model, node, year)

    # Only check & adjust new market shares which are not exogenous
    adjusted_nms = _find_eligible_market_shares(model, node, year, new_market_shares)

    # Check to see if all New M/S values comply with Min/Max limits. If yes, do nothing. If no
    # continue to next step.
    limit_adjusted_techs = []
    while not _min_max_ms_compliant(adjusted_nms, min_max_limits):
        # Apply exogenous Min or Max New M/S limit values on the technology which has the largest
        # % difference between its limit and its initial new market share value.
        percent_differences = _get_percent_differences(adjusted_nms,
                                                       min_max_limits,
                                                       return_sorted=True)
        largest_violator = percent_differences[0]
        violator_name = largest_violator[0]
        adjusted_nms[violator_name] = _make_ms_min_max_compliant(adjusted_nms[violator_name],
                                                                 min_max_limits[violator_name])
        limit_adjusted_techs.append(violator_name)

        # For remaining technologies, calculate their individual Adjusted New M/S for technology(s)
        adjusted_nms = _adjust_new_market_shares(adjusted_nms, limit_adjusted_techs)

    updated_nms = new_market_shares.copy()
    updated_nms.update(adjusted_nms)

    return updated_nms


#############################
# Record Values
#############################
def _record_provided_quantities(requested_services, assessed_demand, model, node, year,
                                tech=None, market_share=1):
    if isinstance(requested_services, dict):
        requested_services = [requested_services]

    for service_data in requested_services:
        service_req_ratio = service_data['year_value']
        quant_requested = market_share * service_req_ratio * assessed_demand
        year_node = model.graph.nodes[service_data['branch']][year]
        if 'provided_quantities' not in year_node.keys():
            year_node['provided_quantities'] = \
                utils.create_value_dict(ProvidedQuantity(), param_source='initialization')
        year_node['provided_quantities']['year_value'].provide_quantity(amount=quant_requested,
                                                                        requesting_node=node,
                                                                        requesting_technology=tech)


def _record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares_per_tech,
                               new_stock_demanded, assessed_demand):
    for tech in adjusted_new_ms:
        # New Market Shares
        model.graph.nodes[node][year]['technologies'][tech]['new_market_share'] = \
            utils.create_value_dict(adjusted_new_ms[tech], param_source='calculation')

        # Base Stock
        if int(year) == model.base_year:
            model.graph.nodes[node][year]['technologies'][tech]['base_stock'] = \
                utils.create_value_dict(new_stock_demanded * adjusted_new_ms[tech],
                                        param_source='calculation')
        # New Stock
        else:
            model.graph.nodes[node][year]['technologies'][tech]['new_stock'] = \
                utils.create_value_dict(new_stock_demanded * adjusted_new_ms[tech],
                                        param_source='calculation')

    # Record Total Market Share
    for tech in total_market_shares_per_tech:
        model.graph.nodes[node][year]['technologies'][tech]['total_market_share'] = \
            utils.create_value_dict(total_market_shares_per_tech[tech], param_source='calculation')

    # Send Demand Below
    for tech, tech_data in model.graph.nodes[node][year]['technologies'].items():
        if 'Service requested' in tech_data.keys():
            services_being_requested = tech_data['Service requested']
            # Calculate the provided_quantities being for each of the services
            t_ms = total_market_shares_per_tech[tech]
            _record_provided_quantities(services_being_requested, assessed_demand,
                                        model, node, year, tech=tech, market_share=t_ms)
