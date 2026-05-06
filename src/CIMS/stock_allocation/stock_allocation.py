"""
Stock retirement and allocation module. Contains all the core logic for retiring stock (vintage &
surplus) and allocating new stock through a market share competition between technologies.
"""
import math
import copy
import numpy as np

from .retrofits import calc_retrofits
from .macro_economics import calc_stock_total_demanded
from .allocation_utils import _find_competing_techs, _find_competing_weights
from .market_share_limits import apply_min_max_limits, apply_min_max_class_limits
from ..quantities import ProvidedQuantity
from ..vintage_weighting import calculate_vintage_weighted_parameter
from ..utils.parameter import query, list as PARAM, construction

#############################
# Stock Allocation
#############################
def all_tech_compete_allocation(model, node, year):
    """
    Performs stock retirement and allocation for `tech compete` nodes, updating
    the model data to reflect the results.

    Stock retirement and allocation performs (1) Vintage-based requirements, (2) Surplus stock
    retirement, (3) New stock competition between technologies, (4) Market share limit adjustments,
    and (5) Total market share calculations.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to stock retirement and
        allocation.

    node: str
        The name of the node (branch notation) where stock retirement and allocation is performed.

    year: str
        The year to perform stock retirement and allocation.

    Returns
    -------
        Nothing is returned. `model` will be updated to reflect the results of stock retirement and
        new stock competitions.
    """
    comp_type = model.get_param(PARAM.competition_type, node).lower()

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    assessed_demand = calc_stock_total_demanded(model, node, year)

    # Existing Tech Specific Stocks -- find existing stock remaining after vintage-based retirement
    stock_existing = _get_existing_stock(model, node, year, comp_type)

    # Retrofits
    stock_existing, stock_retrofit_added, stock_retrofit = calc_retrofits(model, node, year, stock_existing)

    # Capital Stock Availability -- Find how much new stock must be adopted to meet demand
    stock_new_demanded = _calc_stock_new_demanded(assessed_demand, stock_existing, stock_retrofit_added)

    # Surplus Retirement
    if stock_new_demanded < 0:
        stock_new_demanded, stock_existing, stock_retrofit_added, stock_retrofit = \
            _retire_surplus_stock(model, node, year,
                                  stock_new_demanded, stock_existing,
                                  stock_retrofit_added, stock_retrofit)

    # New Tech Competition
    new_market_shares = _calculate_new_market_shares(model, node, year, comp_type)

    # Min/Max Market Share Class Limits
    adjusted_new_ms = apply_min_max_class_limits(model, node, year, new_market_shares)

    # Min/Max Market Share Limits
    adjusted_new_ms = apply_min_max_limits(model, node, year, adjusted_new_ms)

    # Calculate Total Market Shares
    total_market_shares_per_tech = _calculate_total_market_shares(node,
                                                                  assessed_demand,
                                                                  stock_new_demanded,
                                                                  stock_existing,
                                                                  stock_retrofit_added,
                                                                  adjusted_new_ms)


    # Record Values in Model
    _record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares_per_tech,
                               assessed_demand, stock_new_demanded,
                               stock_retrofit_added, stock_retrofit)


def general_allocation(model, node, year):
    """
    Performs stock retirement and allocation for non tech competition nodes. This includes
    `fixed ratio`, `region`, `sector`, `supply - fixed price`, and `root` competition types.

    No competition is required for any of these types. Instead, any demand is automatically filled
    according to exogenously defined paramters.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to stock retirement and
        allocation.

    node: str
        The name of the node (branch notation) where stock retirement and allocation is performed.

    year: str
        The year to perform stock retirement and allocation.

    Returns
    -------
        Nothing is returned. `model` will be updated to reflect the amounts provided by the node.
    """
    node_year_data = model.graph.nodes[node][year]

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    if model.get_param(PARAM.competition_type, node) == 'root' or model.get_param(PARAM.competition_type,
                                                                            node) == 'fixed amount':
        assessed_demand = 1
    else:
        assessed_demand = calc_stock_total_demanded(model, node, year)

    # Based on assessed demand, determine the amount this node requests from other services
    if PARAM.technologies in node_year_data:
        for tech, tech_data in node_year_data[PARAM.technologies].items():
            if PARAM.service_request in tech_data.keys():
                services_being_requested = tech_data[PARAM.service_request]
                t_ms = tech_data[PARAM.market_share_total][PARAM.year_value]
                _record_provided_quantities(model, node, year, services_being_requested,
                                            assessed_demand, tech=tech, market_share_total=t_ms)

    elif PARAM.service_request in node_year_data:
        # Calculate the provided_quantities being requested for each of the services
        services_being_requested = node_year_data[PARAM.service_request]
        _record_provided_quantities(model, node, year, services_being_requested, assessed_demand)


#############################
# Stock Calculation
#############################
def _get_existing_stock(model, node, year, comp_type):
    """
    Find the amount of stock remaining after vintage specific retirements for each technology
    competing for market share at the node.

    For tech compete nodes, this will be for each technology present at the node. For "node tech
    compete" nodes, this will include all the technologies of services directly requested by the
    node.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to vintage specific retirement
    node : str
        Name of the node (branch notation) to query for existing stock
    year : str
        The year to calculate stock for.
    comp_type : str
        The type of competition occurring at the node. One of {'tech compete'}

    Returns
    -------
        A dictionary mapping competing technologies to the amount of their stock remaining at node
        in the given year.

        The dictionary will follow the structure of `{(parent_node, tech): float}`, where each
        (parent_node, tech) tuple corresponds to a competing technology. For a 'tech compete',
        parent_node will be the same for each tech.
    """
    node_year_data = model.graph.nodes[node][year]
    stock_existing = {}

    for tech in node_year_data[PARAM.technologies]:
        t_existing = _do_natural_retirement(model, node, year, tech, comp_type)
        stock_existing[(node, tech)] = t_existing

    return stock_existing


def _calc_stock_new_demanded(demand, stock_existing, stock_retrofit_added):
    """
    Calculate amount of new stock that will be demanded by subtracting all existing stock from the
    total amount of stock being demanded.

    Parameters
    ----------
    demand : int
    stock_existing : dict
        The stock_existing dictionary returned from _get_existing_stock()

    Returns
    -------
        The amount of new stock demanded.
    """
    for e_stocks in stock_existing.values():
        demand -= e_stocks

    for r_stocks in stock_retrofit_added.values():
        demand -= r_stocks

    return demand


#############################
# Retirement
#############################
def _stock_base_retirement(model, node, tech, initial_year, current_year):
    """
    Calculate the amount of base stock (adopted in initial_year) remaining in current_year, after
    natural retirements.

    Parameters
    ----------
    model : CIMS.Model
        The model used for retrieving data relevant to base stock retirement.
    node : str
        The name of the node (branch notation) for which base stock retirement will be calculated.
    tech : str
        The name of the technology to calculate base stock retirement.
    initial_year : str
        The vintage of base stock being retired.
    current_year : str
        The year in which we want to determine how much base stock is remaining.

    Returns
    -------
    float :
        The amount of base stock adopted in initial_year which remains in current_year, after
        natural retirements are performed.
    """
    lifetime = model.get_param(PARAM.lifetime, node, initial_year, tech=tech)
    stock_base = model.get_param(PARAM.stock_base, node, initial_year, tech=tech)

    # Calculate amount of remaining base stock after natural retirements
    remaining_rate = 1 - (int(current_year) - int(initial_year)) / lifetime
    naturally_unretired_stock_base = stock_base * remaining_rate

    # Retrieve amount of base stock in the previous year, after surplus retirement
    prev_year = str(int(current_year) - model.step)
    if int(prev_year) == int(initial_year):
        prev_year_unretired_stock_base = model.get_param(PARAM.stock_base, node,
                                                         year=prev_year, tech=tech)
    else:
        prev_year_unretired_stock_base = model.get_param(PARAM.stock_base_remaining, node,
                                                         year=prev_year, tech=tech)

    stock_base_remaining = max(min(naturally_unretired_stock_base,
                                   prev_year_unretired_stock_base), 0)

    return stock_base_remaining


def _purchased_stock_retirement(model, node, tech, purchased_year, current_year, intercept):
    """
    Calculate the amount of new stock (adopted in purchased_year) remaining in current_year, after
    natural retirements.

    New stock retirement follows the function

    Parameters
    ----------
    model : CIMS.Model
        The model used for retrieving data relevant to new stock retirement.
    node : str
        The name of the node (branch notation) for which new stock retirement will be calculated.
    tech : str
        The name of the technology to calculate new stock retirement for.
    purchased_year : str
        The vintage of new stock being retired.
    current_year : str
        The year in which we want to determine how much new stock is remaining.
    intercept

    Returns
    -------
    float :
        The amount of new stock adopted in purchased_year which remains in current_year, after
        natural retirements are performed.
    """
    lifetime = model.get_param(PARAM.lifetime, node, purchased_year, tech=tech)
    purchased_stock = model.get_param(PARAM.stock_new, node,purchased_year, tech=tech)
    purchased_stock += model.get_param(PARAM.stock_retrofit_added, node, purchased_year, tech=tech)
    prev_year = str(int(current_year) - model.step)

    # Calculate the remaining purchased stock with only natural retirements
    prev_y_exponent = intercept * (1 - (int(prev_year) - int(purchased_year)) / lifetime)
    prev_y_stock_remain_incl_surplus = purchased_stock / (1 + math.exp(prev_y_exponent))

    # Calculate Adjustment Multiplier
    stock_surplus_adjustment = 1

    if int(prev_year) > int(purchased_year):
        prev_y_stock_remain_excl_surplus = model.get_param(PARAM.stock_new_remaining, node,
                                                     year=prev_year, tech=tech, dict_expected=True)[purchased_year]

        if prev_y_stock_remain_incl_surplus > 0:
            stock_surplus_adjustment = prev_y_stock_remain_excl_surplus / \
                             prev_y_stock_remain_incl_surplus

    # Update the tech data
    tech_data = model.graph.nodes[node][current_year][PARAM.technologies][tech]
    if PARAM.stock_surplus_adjustment not in tech_data:
        tech_data[PARAM.stock_surplus_adjustment] = {}
    tech_data[PARAM.stock_surplus_adjustment][purchased_year] = stock_surplus_adjustment

    # Calculate the remaining purchased stock
    exponent = intercept * (1 - (int(current_year) - int(purchased_year)) / lifetime)
    purchased_stock_remaining = purchased_stock / (1 + math.exp(exponent)) * stock_surplus_adjustment

    return purchased_stock_remaining


def _do_natural_retirement(model, node, year, tech, competition_type):
    """
    Performs natural retirement of tech stock (base & new) at node prior to year.

    Parameters
    ----------
    model : CIMS.Model
        The model used for retrieving and storing data relevant to natural retirement.
    node : str
        The name of the node (branch notation) containing the technology to be retired.
    year : str
        The year to calculate natural retirements up to.
    tech : str
        The name of technology whose stock is being retired.
    competition_type : str
        One of {"tech compete"}.

    Returns
    -------
    float :
        Amount of existing tech stock remaining at a node after natural retirements are performed
        over all years prior.
    """
    earlier_years = [y for y in model.years if int(y) < int(year)]
    stock_existing = 0

    if len(earlier_years) != 0:
        # When we are not on the initial year, calculate remaining base and remaining new stock
        stock_base_remaining = 0
        stock_new_remaining_pre_surplus = {}
        for earlier_year in earlier_years:
            # Base Stock
            stock_base_remain_vintage_y = _stock_base_retirement(model, node, tech,
                                                                 earlier_year, year)
            stock_base_remaining += stock_base_remain_vintage_y
            stock_existing += stock_base_remain_vintage_y

            # New Stock (Including Previous Years' Retrofitted Stock)
            intercept_retirement = model.get_param(PARAM.intercept_retirement, node, year)
            stock_new_remaining = _purchased_stock_retirement(model, node, tech, earlier_year, year,
                                                           intercept=intercept_retirement)
            stock_new_remaining_pre_surplus[earlier_year] = stock_new_remaining
            stock_existing += stock_new_remaining

        # Save to Graph
        model.graph.nodes[node][year][PARAM.technologies][tech][PARAM.stock_base_remaining] = \
            construction.create_value_dict(stock_base_remaining, param_source='calculation')
        model.graph.nodes[node][year][PARAM.technologies][tech][PARAM.stock_new_remaining_pre_surplus] = \
            construction.create_value_dict(stock_new_remaining_pre_surplus, param_source='calculation')
        # Note: retired stock will be removed later from [PARAM.stock_new_remaining]
        model.graph.nodes[node][year][PARAM.technologies][tech][PARAM.stock_new_remaining] = \
            construction.create_value_dict(copy.deepcopy(stock_new_remaining_pre_surplus), param_source='calculation')

    return stock_existing


def _calc_surplus_retirement_proportion(surplus, stock_existing):
    """
    Calculate the proportion of stock_existing to be retired, given the amount of surplus stock.

    Parameters
    ----------
    surplus : float
        The amount of surplus stock

    stock_existing : float
        The amount of existing (aka remaining) stock

    Returns
    -------
    float
        The proportion of stock to be retired to reduce the amount of surplus stock
    """
    if stock_existing <= 0:
        retirement_proportion = 0
    else:
        retirement_proportion = max(0, min(surplus / stock_existing, 1))
    return retirement_proportion


def _retire_surplus_stock_base(model, node, year, stock_existing, surplus):
    """
    Called by `_retire_surplus_stock()` to conduct base-stock specific surplus retirements.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    stock_existing : dict
        A dictionary returned from _get_existing_stock() containing the amount of existing stock
        for each technology competing for marketshare at `node`.
    surplus : float
        The amount of surplus stock that currently exists at `node` in the given `year`.

    Returns
    -------
    float
        The amount of surplus base stock left to retire after the provided existing stock (or some
        portion of it) was retired.
    dict
        An updated version of stock_existing where any retired stocks have been deducted.
    """
    total_stock_base = 0
    amount_surplus_to_retire = 0
    for node_branch, tech in stock_existing:
        tech_stock_base = model.get_param(PARAM.stock_base_remaining, node_branch, year, tech=tech)
        total_stock_base += tech_stock_base
    if total_stock_base != 0:
        retirement_proportion = _calc_surplus_retirement_proportion(surplus, total_stock_base)
        for node_branch, tech in stock_existing:
            tech_stock_base = model.get_param(PARAM.stock_base_remaining, node_branch, year, tech=tech)
            amount_tech_to_retire = tech_stock_base * retirement_proportion

            # Remove from existing stock
            stock_existing[(node_branch, tech)] -= amount_tech_to_retire

            # Add to stock to retire
            amount_surplus_to_retire += amount_tech_to_retire

            # Note early retirement in the model
            model.graph.nodes[node_branch][year][PARAM.technologies][tech][PARAM.stock_base_remaining][
                PARAM.year_value] -= amount_tech_to_retire

    return amount_surplus_to_retire, stock_existing


def _retire_surplus_stock_new(model, node, year, stock_existing, surplus):
    """
    Called by `_retire_surplus_stock()` to conduct new-stock surplus retirements.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    stock_existing : dict
        A dictionary returned from _get_existing_stock() containing the amount of existing stock
        for each technology competing for marketshare at `node`.
    surplus : float
        The amount of surplus stock that currently exists at `node` in the given `year`.

    Returns
    -------
    float
        The amount of surplus new stock left to retire after the provided existing stock (or some
        portion of it) was retired.
    dict
        An updated version of stock_existing where any retired stocks have been deducted.
    """
    possible_purchase_years = [y for y in model.years if (int(y) > model.base_year) &
                                                         (int(y) < int(year))]
    amount_surplus_to_retire = 0
    for purchase_year in possible_purchase_years:
        stock_new_total_pre_surplus = 0
        if surplus > 0:
            for node_branch, tech in stock_existing:
                tech_rem_stock_new_pre_surplus = \
                    model.get_param(PARAM.stock_new_remaining_pre_surplus,
                                    node_branch,
                                    year=year,
                                    tech=tech,
                                    dict_expected=True)[purchase_year]
                stock_new_total_pre_surplus += tech_rem_stock_new_pre_surplus

        retirement_proportion = _calc_surplus_retirement_proportion(surplus,
                                                                    stock_new_total_pre_surplus)

        for node_branch, tech in stock_existing:
            t_rem_stock_new_pre_surplus = model.get_param(PARAM.stock_new_remaining_pre_surplus,
                                                          node_branch,
                                                          year=year,
                                                          tech=tech,
                                                          dict_expected=True)[purchase_year]
            amount_tech_to_retire = t_rem_stock_new_pre_surplus * retirement_proportion

            # Remove from existing stock
            stock_existing[(node_branch, tech)] -= amount_tech_to_retire

            # Remove from surplus & new stock demanded
            surplus -= amount_tech_to_retire
            amount_surplus_to_retire += amount_tech_to_retire

            # Note new stock remaining (post surplus) in the model
            model.graph.nodes[node_branch][year][PARAM.technologies][tech][PARAM.stock_new_remaining][
                PARAM.year_value][purchase_year] -= amount_tech_to_retire

    return amount_surplus_to_retire, stock_existing


def _retire_surplus_added_retrofit_stock(model, node, year, stock_retrofit_added,
                                         stock_retrofit, surplus):
    total_added_stock_retrofit = sum(stock_retrofit_added.values())
    amount_surplus_to_retire = 0
    if total_added_stock_retrofit != 0:
        retirement_proportion = _calc_surplus_retirement_proportion(surplus, total_added_stock_retrofit)
        for node_branch, tech in stock_retrofit_added:
            tech_added_stock_retrofit = stock_retrofit_added[(node_branch, tech)]
            amount_tech_to_retire = tech_added_stock_retrofit * retirement_proportion

            # Remove from retrofit stock
            stock_retrofit_added[(node_branch, tech)] -= amount_tech_to_retire
            stock_retrofit[(node_branch, tech)] -= amount_tech_to_retire

            # Add to stock to retire
            amount_surplus_to_retire += amount_tech_to_retire

    return amount_surplus_to_retire, stock_retrofit_added, stock_retrofit


def _retire_surplus_stock(model, node, year, stock_new_demanded, stock_existing,
                          stock_retrofit_added, stock_retrofit):
    """
    Retires surplus stock, starting with the oldest existing stock first. There is surplus stock if
    fewer than 0 units of new stock have been demanded.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    stock_new_demanded : float
        The amount of new stock demanded of `node` in the given `year`.
    stock_existing : dict
        A dictionary returned from _get_existing_stock() containing the amount of existing stock
        for each technology competing for marketshare at `node`.

    Returns
    -------
    float
        The amount of surplus new stock left to retire after the provided existing stock (or some
        portion of it) was retired.
    dict
        An updated version of stock_existing where any retired stocks have been deducted.
    """
    surplus = -1 * stock_new_demanded

    # Base Stock Retirement
    stock_base_to_retire, stock_existing = \
        _retire_surplus_stock_base(model, node, year, stock_existing, surplus)
    surplus -= stock_base_to_retire
    stock_new_demanded += stock_base_to_retire

    # New Stock Retirement
    new_stock_to_retire, stock_existing = \
        _retire_surplus_stock_new(model, node, year, stock_existing, surplus)
    surplus -= new_stock_to_retire
    stock_new_demanded += new_stock_to_retire

    # Retrofit Stock Retirement
    added_retrofit_stock_to_retire, stock_retrofit_added, stock_retrofit = \
        _retire_surplus_added_retrofit_stock(model, node, year, stock_retrofit_added,
                                             stock_retrofit, surplus)
    surplus -= added_retrofit_stock_to_retire
    stock_new_demanded += added_retrofit_stock_to_retire
    
    assertion_message = (f"node: {node}, base: {stock_base_to_retire}, new: {new_stock_to_retire}, "
        f"retrofit: {added_retrofit_stock_to_retire}, new_demand: {stock_new_demanded}")
    # Use lower tolerance for assert check since quantities (demanded and existing) can be slightly out of sync due to loops
    assert(stock_new_demanded >= 0 or np.isclose(stock_new_demanded, 0, atol=1e-03)), assertion_message

    return stock_new_demanded, stock_existing, stock_retrofit_added, stock_retrofit


#############################
# Market Share Calculations
#############################


def _find_exogenous_market_shares(model, node, year):
    """
    A helper function used by _calculate_new_market_shares() to find exogenously defined market
    shares at a given node in a given year.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving exogenous market shares.
    node :
        The name of the node (branch notation) to query for exogenous market shares.
    year : str
        The year in which to query.

    Returns
    -------
    dict :
        A dictionary mapping technologies (str) to exogenouos market shares (float). Only
        technologies with exogenously defined market shares are included in the dictionary.
    """
    node_year_data = model.graph.nodes[node][year]
    exo_market_shares = {}
    for tech in node_year_data[PARAM.technologies]:
        ms_new, ms_source = model.get_param(PARAM.market_share_new, node, year, tech=tech,
                                                  return_source=True)
        if ms_source in ['model', 'initialization']:  # model or initialization --> exogenous
            exo_market_shares[tech] = ms_new
    return exo_market_shares


def _calculate_new_market_shares(model, node, year, comp_type):
    """
    A helper function called by `all_tech_compete_allocation()` to calculate the new market shares
    for the technologies or services at the specified node. This is where the market share
    competition occurs.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving values relevant to weight calculation.
    node
        The name of the node (branch notation) whose technologies' will compete for market share.
    year : str
        The year to calculate new market share for.
    comp_type : str
        The type of competition occurring at the node. One of {'tech compete'}.

    Returns
    -------
    dict :
        A dictionary mapping the technologies or services within node to their 
        new market shares. Note that while market share competition occurs
        across all competing technologies, new market shares are agregated to
        the tech/service specified at `node` before being returned as a
        dictionary.
    """
    heterogeneity = model.get_param(PARAM.heterogeneity, node, year)

    # Find each of the technologies which will be competed for
    competing_techs = _find_competing_techs(model, node, comp_type)

    # Find the weights that we will be using to calculate market share
    total_weight, tech_weights = _find_competing_weights(model, year, heterogeneity, competing_techs)

    # Find the new market shares for each tech
    new_market_shares = _find_exogenous_market_shares(model, node, year)
    for tech_child in model.graph.nodes[node][year][PARAM.technologies]:
        if tech_child not in new_market_shares:
            new_market_shares[tech_child] = 0

            if comp_type == PARAM.competition_compete:
                if (node, tech_child) in tech_weights:
                    new_market_shares[tech_child] = tech_weights[(node, tech_child)] / total_weight

        # Initialize stocks in the Model
        model.graph.nodes[node][year][PARAM.technologies][tech_child][PARAM.stock_base] = \
            construction.create_value_dict(0, param_source='initialization')
        model.graph.nodes[node][year][PARAM.technologies][tech_child][PARAM.stock_new] = \
            construction.create_value_dict(0, param_source='initialization')
        model.graph.nodes[node][year][PARAM.technologies][tech_child][PARAM.stock_retrofit_added] = \
            construction.create_value_dict(0, param_source='initialization')

    return new_market_shares


def _calculate_total_market_shares(node, assessed_demand, stock_new_demanded,
                                   stock_existing, stock_retrofit_added, adjusted_new_ms):
    """
    A helper function called by `all_tech_compete_allocation()` to calculate total market shares
    for all technologies competing at the specified node. This is where the market share competition
    happens.

    Parameters
    ----------
    node : str
        The name of the node (branch notation) whose technologies/services we want to calculate
        total market share for.
    assessed_demand : float
        The total quantity demanded of `node` (includes existing stock).
    stock_new_demanded : float
        The amount of new stock demanded of `node`.
    stock_existing : dict
        A dictionary mapping each competing technology to the amount of previously adopted stock
        that remains at the node.
    adjusted_new_ms : dict
        Min/Max market share compatible new market shares for each technology or service defined at
        `node`.

    Returns
    -------
    dict :
        A dictionary mapping the technologies or services within `node` to their total market
        shares.
    """
    # Initialize Total Stock
    stock_total = {t: 0 for t in adjusted_new_ms}

    # Add existing stocks
    for node_branch, tech in stock_existing:
        if node_branch == node:
            stock_total[tech] += stock_existing[(node_branch, tech)]

    # Add retrofit stocks
    for node_branch, tech in stock_retrofit_added:
        if node_branch == node:
            stock_total[tech] += stock_retrofit_added[(node_branch, tech)]

    # Add new stocks
    for tech_child in adjusted_new_ms:
        stock_total[tech_child] += adjusted_new_ms[tech_child] * stock_new_demanded

    # Market Share
    total_market_shares = {}
    for tech in stock_total:
        if assessed_demand == 0:
            total_market_shares[tech] = 0
        else:
            total_market_shares[tech] = stock_total[tech] / assessed_demand

    return total_market_shares


#############################
# Record Values
#############################
def _record_provided_quantities(model, node, year, requested_services, assessed_demand, tech=None,
                                market_share_total=1):
    """
    A helper function used by `all_tech_compete_allocation()` and `general_allocation()` to record
    the quantities provided by down-tree services (nodes requested by node) to `node` in `year`.

    Parameters
    ----------
    model : CIMS.Model
        The model where provided quantities will be recorded.
    node : str
        The node which requests the quantities.
    year : str
        The year in which the quantities are being requested.
    requested_services : list or dict
        The services being requested by the given node/tech.
    assessed_demand : float
        The total quantity demanded by node (across all technologies).
    tech : str, optional
        The technology which requests the quantities. Defaults to None.
    market_share_total : float, optional
        The ratio [0, 1] of assessed demand attributed to the requesting node/technology. Defaults
        to 1.

    Returns
    -------
    None :
        Nothing is returned. Instead, the model is updated with the provided quantities.
    """

    for target in requested_services:
        vintage_weighted_service_request_ratio = calculate_vintage_weighted_parameter(
            PARAM.service_request, model, node, year, tech=tech, context=target)
        quant_requested = market_share_total * vintage_weighted_service_request_ratio * assessed_demand
        year_node = model.graph.nodes[target][year]
        if PARAM.provided_quantities not in year_node.keys():
            year_node[PARAM.provided_quantities] = \
                construction.create_value_dict(ProvidedQuantity(), param_source='calculation')
        year_node[PARAM.provided_quantities][PARAM.year_value].provide_quantity(amount=quant_requested,
                                                                        requesting_node=node,
                                                                        requesting_technology=tech)
        year_node[PARAM.provided_quantities][PARAM.param_source] = 'calculation'


def _record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares,
                               assessed_demand, stock_new_demanded,
                               added_stock_retrofit, stock_retrofit):
    """

    Parameters
    ----------
    model : CIMS.Model
        The model where the results of stock allocation (new stock, market shares, etc).
    node : str
        The name of the node (branch form) whose results are being recorded.
    year : str
        The year in which to record results.
    adjusted_new_ms : dict
        The dictionary containing min/max limit compliant new market shares for each of the node's
        technologies/services.
    total_market_shares : dict
        The dictionary containing total market shares for each of the node's technologies/services.
    assessed_demand : float
        The total amount of stock demanded of `node` in the given `year`, including existing stock.
    stock_new_demanded :
        The amount of new stock demanded of `node` in the given `year`.

    Returns
    -------
    None :
        Nothing is returned. Instead, the model is updated with the results of stock allocation.
    """
    for tech in adjusted_new_ms:
        # New Market Shares
        is_exogenous = query.is_param_exogenous(model, PARAM.market_share_new, node, year=year, tech=tech)
        if not is_exogenous:
            new_ms_dict = construction.create_value_dict(adjusted_new_ms[tech], param_source='calculation')
            model.set_param_internal(new_ms_dict, PARAM.market_share_new, node, year, tech)

        # Base Stock
        if int(year) == model.base_year:
            stock_base_dict = construction.create_value_dict(stock_new_demanded * adjusted_new_ms[tech],
                                                      param_source='calculation')
            model.set_param_internal(stock_base_dict, PARAM.stock_base, node, year, tech)

        # New Stock
        else:
            stock_new_dict = construction.create_value_dict(stock_new_demanded * adjusted_new_ms[tech],
                                                     param_source='calculation')
            model.set_param_internal(stock_new_dict, PARAM.stock_new, node, year, tech)

    for tech in total_market_shares:
        # Record Total Market Shares
        is_exogenous = query.is_param_exogenous(model, PARAM.market_share_total, node, year=year, tech=tech)
        if not is_exogenous:
            total_ms_dict = construction.create_value_dict(total_market_shares[tech],
                                                param_source='calculation')
            model.set_param_internal(total_ms_dict, PARAM.market_share_total, node, year, tech)

        # Total Stock
        stock_total_dict = construction.create_value_dict(assessed_demand * total_market_shares[tech],
                                                  param_source='calculation')
        model.set_param_internal(stock_total_dict, PARAM.stock_total, node, year, tech)

    # Retrofit Stock
    comp_type = model.get_param(PARAM.competition_type, node)

    for n, t in added_stock_retrofit:
        # Added retrofit stock
        added_stock_retrofit_dict = construction.create_value_dict(added_stock_retrofit[(n, t)],
                                                      param_source='calculation')
        model.set_param_internal(added_stock_retrofit_dict, PARAM.stock_retrofit_added, n, year, t)

    for n, t in stock_retrofit:
        # Net retrofit stock
        stock_retrofit_dict = construction.create_value_dict(stock_retrofit[(n, t)],
                                                            param_source='calculation')
        model.set_param_internal(stock_retrofit_dict, PARAM.stock_retrofit, n, year, t)

    # Send Demand Below
    for tech, tech_data in model.graph.nodes[node][year][PARAM.technologies].items():
        if PARAM.service_request in tech_data.keys():
            services_being_requested = tech_data[PARAM.service_request]
            # Calculate the provided_quantities being for each of the services
            t_ms = total_market_shares[tech]
            _record_provided_quantities(model, node, year, services_being_requested,
                                        assessed_demand, tech=tech, market_share_total=t_ms)
