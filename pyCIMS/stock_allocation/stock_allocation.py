"""
Stock retirement and allocation module. Contains all the core logic for retiring stock (vintage &
surplus) and allocating new stock through a market share competition between technologies.
"""
import math
from ..quantities import ProvidedQuantity
from pyCIMS import utils
from .retrofits import calc_retrofits
from .allocation_utils import _find_competing_techs, _find_competing_weights
from .market_share_limits import _apply_min_max_limits
import copy


#############################
# Stock Allocation
#############################
def all_tech_compete_allocation(model, node, year):
    """
    Performs stock retirement and allocation for "tech compete" and "node tech compete" nodes,
    updating the model data to reflect the results.

    Stock retirement and allocation performs (1) Vintage-based requirements, (2) Surplus stock
    retirement, (3) New stock competition between technologies, (4) Market share limit adjustments,
    and (5) Total market share calculations.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to stock retirement and
        allocation.

    node: str
        The name of the node (in branch form) where stock retirement and allocation is performed.

    year: str
        The year to perform stock retirement and allocation.

    Returns
    -------
        Nothing is returned. `model` will be updated to reflect the results of stock retirement and
        new stock competitions.
    """
    comp_type = model.get_param('competition type', node)

    if comp_type == 'market':
        comp_type = 'tech compete'

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    assessed_demand = model.get_param('provided_quantities', node, year).get_total_quantity()

    # Existing Tech Specific Stocks -- find existing stock remaining after vintage-based retirement
    existing_stock = _get_existing_stock(model, node, year, comp_type)

    # Retrofits
    existing_stock, retrofit_stock = calc_retrofits(model, node, year, existing_stock)

    # Capital Stock Availability -- Find how much new stock must be adopted to meet demand
    new_stock_demanded = _calc_new_stock_demanded(assessed_demand, existing_stock, retrofit_stock)

    # Surplus Retirement
    if new_stock_demanded < 0:
        new_stock_demanded, existing_stock, retrofit_stock = \
            _retire_surplus_stock(model, node, year,
                                  new_stock_demanded, existing_stock, retrofit_stock)

    # New Tech Competition
    new_market_shares = _calculate_new_market_shares(model, node, year, comp_type)

    # Min/Max Market Share Limits
    adjusted_new_ms = _apply_min_max_limits(model, node, year, new_market_shares)

    # Calculate Total Market Shares
    total_market_shares_per_tech = _calculate_total_market_shares(node,
                                                                  assessed_demand,
                                                                  new_stock_demanded,
                                                                  existing_stock,
                                                                  retrofit_stock,
                                                                  adjusted_new_ms)


    # Record Values in Model
    _record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares_per_tech,
                               assessed_demand, new_stock_demanded, retrofit_stock)


def general_allocation(model, node, year):
    """
    Performs stock retirement and allocation for non tech competition nodes. This includes
    'fixed ratio', 'region', 'sector', 'fuel - fixed price', and 'root' competition types.

    No competition is required for any of these types. Instead, any demand is automatically filled
    according to exogenously defined paramters.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to stock retirement and
        allocation.

    node: str
        The name of the node (in branch form) where stock retirement and allocation is performed.

    year: str
        The year to perform stock retirement and allocation.

    Returns
    -------
        Nothing is returned. `model` will be updated to reflect the amounts provided by the node.
    """
    node_year_data = model.graph.nodes[node][year]

    # Demand Assessment -- find amount demanded of the node by requesting nodes/techs
    if model.get_param('competition type', node) == 'root':
        assessed_demand = 1
    else:
        assessed_demand = model.get_param('provided_quantities', node, year).get_total_quantity()

    # Based on assessed demand, determine the amount this node requests from other services
    if 'technologies' in node_year_data:
        for tech, tech_data in node_year_data['technologies'].items():
            if 'service requested' in tech_data.keys():
                services_being_requested = tech_data['service requested']
                t_ms = tech_data['market share']
                _record_provided_quantities(model, node, year, services_being_requested,
                                            assessed_demand, tech=tech, market_share=t_ms)

    elif 'service requested' in node_year_data:
        # Calculate the provided_quantities being requested for each of the services
        services_being_requested = node_year_data['service requested']
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
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to vintage specific retirement
    node : str
        Name of the node (branch notation) to query for existing stock
    year : str
        The year to calculate stock for.
    comp_type : str
        The type of competition occurring at the node. One of {'node tech compete', 'tech compete'}

    Returns
    -------
        A dictionary mapping competing technologies to the amount of their stock remaining at node
        in the given year.

        The dictionary will follow the structure of `{("parent_node", "tech"): float}`, where each
        (parent_node, tech) tuple corresponds to a competing technology. For a 'tech compete',
        parent_node will be the same for each tech while for 'node tech compete', parent_node will
        correspond to the parent service of the competing technology.
    """
    node_year_data = model.graph.nodes[node][year]
    existing_stock = {}

    if comp_type == 'tech compete':
        for tech in node_year_data['technologies']:
            t_existing = _do_natural_retirement(model, node, year, tech, comp_type)
            existing_stock[(node, tech)] = t_existing

    elif comp_type == 'node tech compete':
        for child in node_year_data['technologies']:
            child_node = model.get_param('service requested', node,
                                         year=year,
                                         tech=child,
                                         dict_expected=True)[child]['branch']

            child_year_data = model.graph.nodes[child_node][year]
            for tech in child_year_data['technologies']:
                t_existing = _do_natural_retirement(model, child_node, year, tech, comp_type)
                existing_stock[(child_node, tech)] = t_existing

    return existing_stock


def _calc_new_stock_demanded(demand, existing_stock, retrofit_stock):
    """
    Calculate amount of new stock that will be demanded by subtracting all existing stock from the
    total amount of stock being demanded.

    Parameters
    ----------
    demand : int
    existing_stock : dict
        The existing_stock dictionary returned from _get_existing_stock()

    Returns
    -------
        The amount of new stock demanded.
    """
    for e_stocks in existing_stock.values():
        demand -= e_stocks

    for r_stocks in retrofit_stock.values():
        demand -= r_stocks

    return demand


#############################
# Retirement
#############################
def _base_stock_retirement(model, node, tech, initial_year, current_year):
    """
    Calculate the amount of base stock (adopted in initial_year) remaining in current_year, after
    natural retirements.

    Parameters
    ----------
    model : pyCIMS.Model
        The model used for retrieving data relevant to base stock retirement.
    node : str
        The name of the node (in branch form) for which base stock retirement will be calculated.
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
    lifetime = model.get_param('lifetime', node, initial_year, tech=tech)
    base_stock = model.get_param('base_stock', node, initial_year, tech=tech)

    # Calculate amount of remaining base stock after natural retirements
    remaining_rate = 1 - (int(current_year) - int(initial_year)) / lifetime
    naturally_unretired_base_stock = base_stock * remaining_rate

    # Retrieve amount of base stock in the previous year, after surplus retirement
    prev_year = str(int(current_year) - model.step)
    if int(prev_year) == int(initial_year):
        prev_year_unretired_base_stock = model.get_param('base_stock', node,
                                                         year=prev_year, tech=tech)
    else:
        prev_year_unretired_base_stock = model.get_param('base_stock_remaining', node,
                                                         year=prev_year, tech=tech)

    base_stock_remaining = max(min(naturally_unretired_base_stock,
                                   prev_year_unretired_base_stock), 0)

    return base_stock_remaining


def _purchased_stock_retirement(model, node, tech, purchased_year, current_year,
                                intercept=-11.513):
    """
    Calculate the amount of new stock (adopted in purchased_year) remaining in current_year, after
    natural retirements.

    New stock retirement follows the function

    Parameters
    ----------
    model : pyCIMS.Model
        The model used for retrieving data relevant to new stock retirement.
    node : str
        The name of the node (in branch form) for which new stock retirement will be calculated.
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
    lifetime = model.get_param('lifetime', node, purchased_year, tech=tech)
    purchased_stock = model.get_param('new_stock', node,purchased_year, tech=tech)
    purchased_stock += model.get_param('retrofit_stock', node, purchased_year, tech=tech)
    prev_year = str(int(current_year) - model.step)

    # Calculate the remaining purchased stock with only natural retirements
    prev_y_exponent = intercept * (1 - (int(prev_year) - int(purchased_year)) / lifetime)
    prev_y_fictional_purchased_stock_remain = purchased_stock / (1 + math.exp(prev_y_exponent))

    # Calculate Adjustment Multiplier
    adj_multiplier = 1

    if int(prev_year) > int(purchased_year):
        prev_y_unretired_new_stock = model.get_param('new_stock_remaining', node,
                                                     year=prev_year, tech=tech, dict_expected=True)[purchased_year]

        if prev_y_fictional_purchased_stock_remain > 0:
            adj_multiplier = prev_y_unretired_new_stock / \
                             prev_y_fictional_purchased_stock_remain

    # Update the tech data
    tech_data = model.graph.nodes[node][current_year]['technologies'][tech]
    if 'adjustment_multiplier' not in tech_data:
        tech_data['adjustment_multiplier'] = {}
    tech_data['adjustment_multiplier'][purchased_year] = adj_multiplier

    # Calculate the remaining purchased stock
    exponent = intercept * (1 - (int(current_year) - int(purchased_year)) / lifetime)
    purchased_stock_remaining = purchased_stock / (1 + math.exp(exponent)) * adj_multiplier

    return purchased_stock_remaining


def _do_natural_retirement(model, node, year, tech, competition_type):
    """
    Performs natural retirement of tech stock (base & new) at node prior to year.

    Parameters
    ----------
    model : pyCIMS.Model
        The model used for retrieving and storing data relevant to natural retirement.
    node : str
        The name of the node (in branch form) containing the technology to be retired.
    year : str
        The year to calculate natural retirements up to.
    tech : str
        The name of technology whose stock is being retired.
    competition_type : str
        One of {"tech compete", "node tech compete"}. If "node tech compete", retirement results are
        saved at the parent node.

    Returns
    -------
    float :
        Amount of existing tech stock remaining at a node after natural retirements are performed
        over all years prior.
    """
    earlier_years = [y for y in model.years if int(y) < int(year)]
    existing_stock = 0

    if len(earlier_years) != 0:
        # When we are not on the initial year, calculate remaining base and remaining new stock
        remaining_base_stock = 0
        remaining_new_stock_pre_surplus = {}
        for earlier_year in earlier_years:
            # Base Stock
            remain_base_stock_vintage_y = _base_stock_retirement(model, node, tech,
                                                                 earlier_year, year)
            remaining_base_stock += remain_base_stock_vintage_y
            existing_stock += remain_base_stock_vintage_y

            # New Stock (Including Previous Years' Retrofitted Stock)
            remain_new_stock = _purchased_stock_retirement(model, node, tech, earlier_year, year)
            remaining_new_stock_pre_surplus[earlier_year] = remain_new_stock
            existing_stock += remain_new_stock

        # Save to Graph
        model.graph.nodes[node][year]['technologies'][tech]['base_stock_remaining'] = \
            utils.create_value_dict(remaining_base_stock, param_source='calculation')
        model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining_pre_surplus'] = \
            utils.create_value_dict(remaining_new_stock_pre_surplus, param_source='calculation')
        # Note: retired stock will be removed later from ['new_stock_remaining']
        model.graph.nodes[node][year]['technologies'][tech]['new_stock_remaining'] = \
            utils.create_value_dict(copy.deepcopy(remaining_new_stock_pre_surplus), param_source='calculation')

        if competition_type == 'node tech compete':
            # Add stock data @ parent for "node tech compete" nodes
            parent = '.'.join(node.split('.')[:-1])
            child = node.split('.')[-1]

            child_tech_keys = model.graph.nodes[parent][year]['technologies'][child].keys()
            if 'base_stock_remaining' in child_tech_keys:
                model.graph.nodes[parent][year]['technologies'][child]['base_stock_remaining'][
                    'year_value'] += remaining_base_stock
            else:
                model.graph.nodes[parent][year]['technologies'][child][
                    'base_stock_remaining'] = utils.create_value_dict(remaining_base_stock,
                                                                      param_source='calculation')

            if 'new_stock_remaining_pre_surplus' in child_tech_keys:
                for vintage_year in remaining_new_stock_pre_surplus:
                    model.graph.nodes[parent][year]['technologies'][child][
                        'new_stock_remaining_pre_surplus']['year_value'][vintage_year] += \
                        remaining_new_stock_pre_surplus[vintage_year]
            else:
                model.graph.nodes[parent][year]['technologies'][child][
                    'new_stock_remaining_pre_surplus'] = utils.create_value_dict(
                    copy.deepcopy(remaining_new_stock_pre_surplus), param_source='calculation')

            if 'new_stock_remaining' in child_tech_keys:
                for vintage_year in remaining_new_stock_pre_surplus:
                    model.graph.nodes[parent][year]['technologies'][child]['new_stock_remaining'][
                        'year_value'][vintage_year] += remaining_new_stock_pre_surplus[vintage_year]
            else:
                model.graph.nodes[parent][year]['technologies'][child]['new_stock_remaining'] = \
                    utils.create_value_dict(copy.deepcopy(remaining_new_stock_pre_surplus),
                                            param_source='calculation')

    return existing_stock


def _calc_surplus_retirement_proportion(surplus, existing_stock):
    """
    Calculate the proportion of existing_stock to be retired, given the amount of surplus stock.

    Parameters
    ----------
    surplus : float
        The amount of surplus stock

    existing_stock : float
        The amount of existing (aka remaining) stock

    Returns
    -------
    float
        The proportion of stock to be retired to reduce the amount of surplus stock
    """
    if existing_stock <= 0:
        retirement_proportion = 0
    else:
        retirement_proportion = max(0, min(surplus / existing_stock, 1))
    return retirement_proportion


def _retire_surplus_base_stock(model, node, year, existing_stock, surplus):
    """
    Called by `_retire_surplus_stock()` to conduct base-stock specific surplus retirements.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    existing_stock : dict
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
        An updated version of existing_stock where any retired stocks have been deducted.
    """
    total_base_stock = 0
    amount_surplus_to_retire = 0
    for node_branch, tech in existing_stock:
        tech_base_stock = model.get_param('base_stock_remaining', node_branch, year, tech=tech)
        total_base_stock += tech_base_stock
    if total_base_stock != 0:
        retirement_proportion = _calc_surplus_retirement_proportion(surplus, total_base_stock)
        for node_branch, tech in existing_stock:
            tech_base_stock = model.get_param('base_stock_remaining', node_branch, year, tech=tech)
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

    return amount_surplus_to_retire, existing_stock


def _retire_surplus_new_stock(model, node, year, existing_stock, surplus):
    """
    Called by `_retire_surplus_stock()` to conduct new-stock surplus retirements.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    existing_stock : dict
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
        An updated version of existing_stock where any retired stocks have been deducted.
    """
    possible_purchase_years = [y for y in model.years if (int(y) > model.base_year) &
                                                         (int(y) < int(year))]
    amount_surplus_to_retire = 0
    for purchase_year in possible_purchase_years:
        total_new_stock_pre_surplus = 0
        if surplus > 0:
            for node_branch, tech in existing_stock:
                tech_rem_new_stock_pre_surplus = \
                    model.get_param('new_stock_remaining_pre_surplus',
                                    node_branch,
                                    year=year,
                                    tech=tech,
                                    dict_expected=True)[purchase_year]
                total_new_stock_pre_surplus += tech_rem_new_stock_pre_surplus

        retirement_proportion = _calc_surplus_retirement_proportion(surplus,
                                                                    total_new_stock_pre_surplus)

        for node_branch, tech in existing_stock:
            t_rem_new_stock_pre_surplus = model.get_param('new_stock_remaining_pre_surplus',
                                                          node_branch,
                                                          year=year,
                                                          tech=tech,
                                                          dict_expected=True)[purchase_year]
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

    return amount_surplus_to_retire, existing_stock


def _retire_surplus_retrofit_stock(model, node, year, retrofit_stock, surplus):
    total_retrofit_stock = sum(retrofit_stock.values())
    amount_surplus_to_retire = 0
    if total_retrofit_stock != 0:
        retirement_proportion = _calc_surplus_retirement_proportion(surplus, total_retrofit_stock)
        for node_branch, tech in retrofit_stock:
            tech_retrofit_stock = retrofit_stock[(node_branch, tech)]
            amount_tech_to_retire = tech_retrofit_stock * retirement_proportion

            # Remove from retrofit stock
            retrofit_stock[(node_branch, tech)] -= amount_tech_to_retire

            # Add to stock to retire
            amount_surplus_to_retire += amount_tech_to_retire

    return amount_surplus_to_retire, retrofit_stock


def _retire_surplus_stock(model, node, year, new_stock_demanded, existing_stock, retrofit_stock):
    """
    Retires surplus stock, starting with the oldest existing stock first. There is surplus stock if
    fewer than 0 units of new stock have been demanded.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving and storing data relevant to surplus retirements
    node : str
        Name of the node (branch notation) where surplus stock will be retired from.
    year : str
        The year in which to retire surplus stock.
    new_stock_demanded : float
        The amount of new stock demanded of `node` in the given `year`.
    existing_stock : dict
        A dictionary returned from _get_existing_stock() containing the amount of existing stock
        for each technology competing for marketshare at `node`.

    Returns
    -------
    float
        The amount of surplus new stock left to retire after the provided existing stock (or some
        portion of it) was retired.
    dict
        An updated version of existing_stock where any retired stocks have been deducted.
    """
    surplus = -1 * new_stock_demanded

    # Base Stock Retirement
    base_stock_to_retire, existing_stock = \
        _retire_surplus_base_stock(model, node, year, existing_stock, surplus)
    surplus -= base_stock_to_retire
    new_stock_demanded += base_stock_to_retire

    # New Stock Retirement
    new_stock_to_retire, existing_stock = \
        _retire_surplus_new_stock(model, node, year, existing_stock, surplus)
    surplus -= new_stock_to_retire
    new_stock_demanded += new_stock_to_retire

    # Retrofit Stock Retirement
    retrofit_stock_to_retire, retrofit_stock = \
        _retire_surplus_retrofit_stock(model, node, year, retrofit_stock, surplus)

    surplus -= retrofit_stock_to_retire
    new_stock_demanded += retrofit_stock_to_retire

    assert(round(new_stock_demanded) >= 0)

    return new_stock_demanded, existing_stock, retrofit_stock


#############################
# Market Share Calculations
#############################


def _find_exogenous_market_shares(model, node, year):
    """
    A helper function used by _calculate_new_market_shares() to find exogenously defined market
    shares at a given node in a given year.

    Parameters
    ----------
    model : pyCIMS.Model
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
    for tech in node_year_data['technologies']:
        market_share, ms_source = model.get_param('market share', node, year, tech=tech,
                                                  return_source=True)
        if ms_source == 'model':  # model --> exogenous
            exo_market_shares[tech] = market_share
    return exo_market_shares


def _calculate_new_market_shares(model, node, year, comp_type):
    """
    A helper function called by `all_tech_compete_allocation()` to calculate the new market shares
    for the technologies or services at the specified node. This is where the market share
    competition occurs.

    Parameters
    ----------
    model : pyCIMS.Model
        The model to use for retrieving values relevant to weight calculation.
    node
        The name of the node (branch notation) whose technologies' will compete for market share.
    year : str
        The year to calculate new market share for.
    comp_type : str
        The type of competition occurring at the node. One of {'node tech compete', 'tech compete'}.

    Returns
    -------
    dict :
        A dictionary mapping the technologies or services within node to their new market shares.
        Note that while market share competition occurs across all competing technologies (includes
        techs @ services for Node Tech Compete), new market shares are agregated to the tech/service
        specified at `node` before being returned as a dictionary.
    """
    heterogeneity = model.get_param('heterogeneity', node, year)

    # Find each of the technologies which will be competed for
    competing_techs = _find_competing_techs(model, node, comp_type)

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
                child_node = model.get_param('service requested', node,
                                             year=year,
                                             tech=tech_child,
                                             dict_expected=True)[tech_child]['branch']
                child_new_market_shares = _find_exogenous_market_shares(model, child_node, year)
                child_weights = {t: w for (n, t), w in tech_weights.items() if n == child_node}

                for tech in child_weights:
                    if tech not in child_new_market_shares:
                        new_market_share = child_weights[tech] / total_weight
                    else:
                        new_market_share = child_new_market_shares[tech]
                    new_market_shares[tech_child] += new_market_share
        # Initialize stocks in the Model
        model.graph.nodes[node][year]['technologies'][tech_child]['base_stock'] = \
            utils.create_value_dict(0, param_source='initialization')
        model.graph.nodes[node][year]['technologies'][tech_child]['new_stock'] = \
            utils.create_value_dict(0, param_source='initialization')
        model.graph.nodes[node][year]['technologies'][tech_child]['retrofit_stock'] = \
            utils.create_value_dict(0, param_source='initialization')

    return new_market_shares


def _calculate_total_market_shares(node, assessed_demand, new_stock_demanded,
                                   existing_stock, retrofit_stock, adjusted_new_ms):
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
    new_stock_demanded : float
        The amount of new stock demanded of `node`.
    existing_stock : dict
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
    total_stocks = {t: 0 for t in adjusted_new_ms}

    # Add existing stocks
    for node_branch, tech in existing_stock:
        if node_branch == node:
            total_stocks[tech] += existing_stock[(node_branch, tech)]
        else:
            child = node_branch.split('.')[-1]
            total_stocks[child] += existing_stock[(node_branch, tech)]

    # Add retrofit stocks
    for node_branch, tech in retrofit_stock:
        if node_branch == node:
            total_stocks[tech] += retrofit_stock[(node_branch, tech)]
        else:
            child = node_branch.split('.')[-1]
            total_stocks[child] += retrofit_stock[(node_branch, tech)]

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
# Record Values
#############################
def _record_provided_quantities(model, node, year, requested_services, assessed_demand, tech=None,
                                market_share=1):
    """
    A helper function used by `all_tech_compete_allocation()` and `general_allocation()` to record
    the quantities provided by down-tree services (nodes requested by node) to `node` in `year`.

    Parameters
    ----------
    model : pyCIMS.Model
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
    market_share : float, optional
        The ratio [0, 1] of assessed demand attributed to the requesting node/technology. Defaults
        to 1.

    Returns
    -------
    None :
        Nothing is returned. Instead, the model is updated with the provided quantities.
    """
    for service_data in requested_services.values():
        service_req_ratio = service_data['year_value']
        quant_requested = market_share * service_req_ratio * assessed_demand
        year_node = model.graph.nodes[service_data['branch']][year]
        if 'provided_quantities' not in year_node.keys():
            year_node['provided_quantities'] = \
                utils.create_value_dict(ProvidedQuantity(), param_source='initialization')
        year_node['provided_quantities']['year_value'].provide_quantity(amount=quant_requested,
                                                                        requesting_node=node,
                                                                        requesting_technology=tech)


def _record_allocation_results(model, node, year, adjusted_new_ms, total_market_shares,
                               assessed_demand, new_stock_demanded, retrofit_stocks):
    """

    Parameters
    ----------
    model : pyCIMS.Model
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
    new_stock_demanded :
        The amount of new stock demanded of `node` in the given `year`.

    Returns
    -------
    None :
        Nothing is returned. Instead, the model is updated with the results of stock allocation.
    """
    for tech in adjusted_new_ms:
        # New Market Shares
        new_ms_dict = utils.create_value_dict(adjusted_new_ms[tech], param_source='calculation')
        model.set_param_internal(new_ms_dict, 'new_market_share', node, year, tech)

        # Base Stock
        if int(year) == model.base_year:
            base_stock_dict = utils.create_value_dict(new_stock_demanded * adjusted_new_ms[tech],
                                                      param_source='calculation')
            model.set_param_internal(base_stock_dict, 'base_stock', node, year, tech)

        # New Stock
        else:
            new_stock_dict = utils.create_value_dict(new_stock_demanded * adjusted_new_ms[tech],
                                                     param_source='calculation')
            model.set_param_internal(new_stock_dict, 'new_stock', node, year, tech)

    for tech in total_market_shares:
        # Record Total Market Shares
        total_ms_dict = utils.create_value_dict(total_market_shares[tech],
                                                param_source='calculation')
        model.set_param_internal(total_ms_dict, 'total_market_share', node, year, tech)

        # Total Stock
        total_stock_dict = utils.create_value_dict(assessed_demand * total_market_shares[tech],
                                                  param_source='calculation')
        model.set_param_internal(total_stock_dict, 'total_stock', node, year, tech)

    # Retrofit Stock
    comp_type = model.get_param('competition type', node)

    for n, t in retrofit_stocks:
        retrofit_stock_dict = utils.create_value_dict(retrofit_stocks[(n, t)],
                                                      param_source='calculation')
        model.set_param_internal(retrofit_stock_dict, 'retrofit_stock', n, year, t)

    if comp_type == 'node tech compete':
        # We need to also store summary retrofit information at the Node Tech Compete parent node
        # Create Summary Retrofit Dictionary
        summary_retrofit_stocks = {}
        for n, t in retrofit_stocks:
            child_service = n.split('.')[-1]
            if child_service not in summary_retrofit_stocks:
                summary_retrofit_stocks[child_service] = 0
            summary_retrofit_stocks[child_service] += retrofit_stocks[(n, t)]

        # Save the info to the model
        for child_service in summary_retrofit_stocks:
            retrofit_stock_dict = utils.create_value_dict(summary_retrofit_stocks[child_service],
                                                          param_source='calculation')
            model.set_param_internal(retrofit_stock_dict, 'retrofit_stock', node, year, child_service)

    # Send Demand Below
    for tech, tech_data in model.graph.nodes[node][year]['technologies'].items():
        if 'service requested' in tech_data.keys():
            services_being_requested = tech_data['service requested']
            # Calculate the provided_quantities being for each of the services
            t_ms = total_market_shares[tech]
            _record_provided_quantities(model, node, year, services_being_requested,
                                        assessed_demand, tech=tech, market_share=t_ms)
