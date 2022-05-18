"""
Retrofit module. Contains functions for retrofitting previously adopted stock.
"""
from .allocation_utils import _find_competing_techs, _find_competing_weights, _calculate_lcc_weight
from .market_share_limits import _min_max_ms_compliant, _get_percent_differences, \
    _make_ms_min_max_compliant, _adjust_new_market_shares


def _retrofit_lcc(model, node, year, existing_tech):
    """
    Calculates the LCC to be used by the current technology during a retrofit competition.
    This differs from regular LCC by excluding all upfront costs.

    Parameters
    ----------
    model : pyCIMS.Model
        The model where LCC components are stored. Must contain node.
    node : str
        The name of the node (branch notation) where the LCC components can be found.
    year : str
        The year of interest.
    existing_tech : str
        The existing technology whose LCC is being calculated.

    Returns
    -------
    float :
        The LCC to use for the current technology during a retrofit competition.
    """
    complete_annual_cost = model.get_param('Complete Annual cost', node, year,
                                           tech=existing_tech, do_calc=True)
    annual_service_cost = model.get_param('Service cost', node, year,
                                          tech=existing_tech, do_calc=True)
    emissions_cost = model.get_param('Emissions cost', node, year,
                                     tech=existing_tech, do_calc=True)
    retrofit_complete_lcc = complete_annual_cost + annual_service_cost + emissions_cost
    return retrofit_complete_lcc


def _apply_retrofit_limits(model, year, existing_tech, retrofit_market_shares):
    """
    Ensures the amount of existing stock which is retrofitted falls within the defined retrofit
    limits. If these limits are broken, the market share results are adjusted.

    Parameters
    ----------
    model : pyCIMS.Model
        The model where the limits are stored.
    year : str
        The year for which the retrofit competition has been done.
    existing_tech : tuple (str, str)
        A tuple containing two strings. The first is the node. The second is the technology.
        Together they identify where the existing technology can be found.
    retrofit_market_shares : dict {(str, str): float}
        A dictionary containing the market shares assigned during the retrofit competition.

    Returns
    -------
    dict : {(str, str): float}
        An updated version of retrofit_market_shares, where the amount of existing stock which is
        retrofitted falls within the min/max retrofit limits.
    """
    limits = {}
    for (node, tech) in retrofit_market_shares.keys():
        if (node, tech) == existing_tech:
            if len(retrofit_market_shares) == 1:
                # There are no technologies competing to retrofit, limits don't apply
                limits[(node, tech)] = (0, 1)
            else:
                min_retrofit = model.get_param('Retrofit_Min', node, year, tech=tech)
                existing_tech_max_ms = 1 - min_retrofit
                max_retrofit = model.get_param('Retrofit_Max', node, year, tech=tech)
                existing_tech_min_ms = 1 - max_retrofit
                limits[(node, tech)] = (existing_tech_min_ms, existing_tech_max_ms)
        else:
            limits[(node, tech)] = (0, 1)

    limit_adjusted_techs = []
    if not _min_max_ms_compliant(retrofit_market_shares, limits):
        percent_differences = _get_percent_differences(retrofit_market_shares,
                                                       limits,
                                                       return_sorted=True)
        largest_violator_name = percent_differences[0][0]

        retrofit_market_shares[largest_violator_name] = \
            _make_ms_min_max_compliant(retrofit_market_shares[largest_violator_name],
                                       limits[largest_violator_name])
        limit_adjusted_techs.append(largest_violator_name)

        retrofit_market_shares = _adjust_new_market_shares(retrofit_market_shares,
                                                           limit_adjusted_techs)

    return retrofit_market_shares


def _adjust_retrofit_marketshares(model, year, existing_tech, retrofit_market_shares):
    """
    If an existing_stock was retrofitted, check that each of the "other" technologies adhere to the
    limits specified by their `Market share retro_Max` and `Market share retro_Min` values. These
    limits are compared with the relative market shares amongst the retrofitting technologies.

    For example, suppose a retrofit competition results in 90% market share remaining as existing
    stock, 10% retrofitted to tech B, and 0% retrofitted to tech C. If tech C had a
    `Market share retro_Min` of 50%, than tech B would need to be reduced to 5% and tech C increased
     to 5% (thereby accounting for 50% of the retrofitted stock).

    Parameters
    ----------
    model : pyCIMS.Model
        The model of interest.
    year : str
        The year for which the retrofit competition has been done.
    existing_tech : tuple (str, str)
        A tuple containing two strings. The first is the node. The second is the technology.
        Together they identify where the existing technology can be found.
    retrofit_market_shares : dict {(str, str): float}
        A dictionary containing the market shares assigned during the retrofit competition.

    Returns
    -------
    dict : {(str, str): float}
        An updated version of retrofit_market_shares, where market shares of newly retrofitted
        technologies are adjusted to comply with their min/max limits.
    """
    if len(retrofit_market_shares) == 0:
        return retrofit_market_shares

    # Find the market share amongst all newly retrofitted technologies
    ms_of_all_adopting_techs = 1 - retrofit_market_shares[existing_tech]
    if ms_of_all_adopting_techs == 0:
        # If no retrofits occurred, there is no need to check limits
        return retrofit_market_shares

    # For each newly adopted retrofit technology, calculate its relative market share
    adopting_tech_market_shares = {}
    adopting_tech_ms_limits = {}
    for (node, tech), market_share in retrofit_market_shares.items():
        if (node, tech) == existing_tech:
            pass
        else:
            ms_retrofit_min = model.get_param('Market share retro_Min', node, year, tech=tech)
            ms_retrofit_max = model.get_param('Market share retro_Max', node, year, tech=tech)
            adopting_tech_ms_limits[(node, tech)] = (ms_retrofit_min, ms_retrofit_max)
            adopting_tech_market_shares[(node, tech)] = market_share / ms_of_all_adopting_techs

    # Adjust market shares until they are compliant
    limit_adjusted_techs = []
    while not _min_max_ms_compliant(adopting_tech_market_shares, adopting_tech_ms_limits):
        percent_differences = _get_percent_differences(adopting_tech_market_shares,
                                                       adopting_tech_ms_limits,
                                                       return_sorted=True)
        largest_violator_name = percent_differences[0][0]

        adopting_tech_market_shares[largest_violator_name] = \
            _make_ms_min_max_compliant(adopting_tech_market_shares[largest_violator_name],
                                       adopting_tech_ms_limits[largest_violator_name])
        limit_adjusted_techs.append(largest_violator_name)

        adopting_tech_market_shares = _adjust_new_market_shares(adopting_tech_market_shares,
                                                                limit_adjusted_techs)

    # Compliant marketshares are out of 1, adjust according to total retrofit market share
    adopting_tech_market_shares = {k: v * ms_of_all_adopting_techs
                                   for k, v in adopting_tech_market_shares.items()}
    retrofit_market_shares.update(adopting_tech_market_shares)

    return retrofit_market_shares


def _record_retrofitted_stock(model, node, year, tech, retrofit_amount):
    """
    Update the amount of base stock and new stock remaining in the model based on how much stock
    is retrofitted. The amount of retrofitted stock is first subtracted from base_stock_remaining.
    If no base stock remains, stock is subtracted from new_stock_remaining beginning with the oldest
    stock (this is also reflected in new_stock_remaining_pre_surplus).

    Parameters
    ----------
    model : pyCIMS.Model
        The model to record the retrofit results in.
    node : str
        The name of the node (branch notation) whose stock has been retrofitted.
    year : str
        The year in which the retrofit took place.
    tech : str
        The technology whose stock has been retrofitted.
    retrofit_amount : float
        The amount of stock which has been retrofitted.

    Returns
    -------
    None :
        Returns None. The model is updated to reflect the retrofitted stock.
    """
    if retrofit_amount <= 0:
        return

    # Base stock
    base_stock_remaining = model.get_param('base_stock_remaining', node, year, tech=tech)
    base_stock_retrofitted = min(base_stock_remaining, retrofit_amount)
    retrofit_amount -= base_stock_retrofitted
    model.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining']['year_value'] -= base_stock_retrofitted

    # New Stock
    if retrofit_amount > 0:
        new_stock_remaining = model.get_param('new_stock_remaining', node, year, tech=tech)
        for prev_year in new_stock_remaining:
            y_ns_remaining = new_stock_remaining[prev_year]
            y_ns_retrofitted = min(y_ns_remaining, retrofit_amount)
            retrofit_amount -= y_ns_retrofitted
            model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining']['year_value'][prev_year] -= y_ns_retrofitted
            model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining_pre_surplus']['year_value'][prev_year] -= y_ns_retrofitted


def calc_retrofits(model, node, year, existing_stock):
    """
    For each technology in the existing_stock dictionary, perform a retrofit competition to
    determine what portion of that technology's existing stock will be retrofitted to a new
    technology.

    Parameters
    ----------
    model : pyCIMS.Model
        The model of interest.
    node : str
        The name of the node (branch notation) where the retrofit competition will occur.
    year : str
        The year for which the retrofit competition will occur.
    existing_stock : dict {(str, str): float}
        A dictionary that maps an existing technology (node, tech) to the amount of stock already
        existing for that technology.

    Returns
    -------
    dict : dict {(str, str): float}
        An updated existing_stock dictionary, where existing stock is reduced for technologies which
        were retrofitted.

    dict : dict {(str, str): float}
        A new retrofit_stock dictionary that contains the amount of stock adopted by each technology
        during the retrofit competition.
    """
    comp_type = model.get_param('competition type', node)
    heterogeneity = model.get_param('Heterogeneity', node, year)
    retrofit_stocks = {}
    for existing_node_tech in existing_stock.keys():
        existing_node, existing_tech = existing_node_tech

        # Find Other Competing
        other_competing_techs = _find_competing_techs(model, node, comp_type)
        other_competing_techs.remove(existing_node_tech)
        total_weight, competing_weights = _find_competing_weights(model, year,
                                                                  other_competing_techs,
                                                                  heterogeneity)

        # Add Existing Tech's Weight
        existing_tech_lcc = _retrofit_lcc(model, existing_node, year, existing_tech)
        existing_tech_weight = _calculate_lcc_weight(existing_tech_lcc, heterogeneity)
        total_weight += existing_tech_weight
        competing_weights[existing_node_tech] = existing_tech_weight

        # Find Market shares based off of weights
        retrofit_market_shares = {}
        if comp_type == 'tech compete':
            for tech in competing_weights:
                retrofit_market_shares[tech] = competing_weights[tech] / total_weight
        elif comp_type == 'node tech compete':
            for tech in competing_weights:
                retrofit_market_shares[tech] = competing_weights[tech] / total_weight

        # Adjust based on limits of existing technology
        retrofit_market_shares = _apply_retrofit_limits(model, year, existing_node_tech,
                                                        retrofit_market_shares)

        # Adjust market shares based on limits of techs being retrofitted to
        retrofit_market_shares = _adjust_retrofit_marketshares(model, year, existing_node_tech,
                                                               retrofit_market_shares)

        # Adjust stocks based on retrofit market shares
        for tech, market_share in retrofit_market_shares.items():
            if tech == existing_node_tech:
                pre_retro_existing_stock = existing_stock[existing_node_tech]
                post_retro_existing_stock = market_share * pre_retro_existing_stock
                existing_stock[existing_node_tech] = post_retro_existing_stock
                _record_retrofitted_stock(model, existing_node, year, existing_tech,
                                          pre_retro_existing_stock - post_retro_existing_stock)
                if comp_type == 'node tech compete':
                    _record_retrofitted_stock(model, node, year, existing_node.split('.')[-1],
                                              pre_retro_existing_stock - post_retro_existing_stock)

            else:
                if tech not in retrofit_stocks:
                    retrofit_stocks[tech] = 0
                retrofit_stocks[tech] += existing_stock[existing_node_tech] * market_share

        # note the remaining stock in the model
    return existing_stock, retrofit_stocks
