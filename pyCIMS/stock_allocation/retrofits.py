from .allocation_utils import _find_competing_techs, _find_competing_weights, _calculate_lcc_weight
from .market_share_limits import _min_max_ms_compliant, _get_percent_differences, \
    _make_ms_min_max_compliant, _adjust_new_market_shares


def _retrofit_lcc(model, node, year, tech):
    complete_annual_cost = model.get_or_calc_param('Complete Annual cost', node, year, tech)
    annual_service_cost = model.get_or_calc_param('Service cost', node, year, tech)
    emissions_cost = model.get_or_calc_param('Emissions cost', node, year, tech)
    retrofit_complete_lcc = complete_annual_cost + annual_service_cost + emissions_cost
    return retrofit_complete_lcc


def _retrofit_existing_tech_weight(model, year, existing_tech, heterogeneity):
    node_branch, tech = existing_tech
    retrofit_lcc = _retrofit_lcc(model, node_branch, year, tech)
    retrofit_weight = _calculate_lcc_weight(retrofit_lcc, heterogeneity)
    return retrofit_weight


def _apply_retrofit_limits(model, year, existing_tech, retrofit_market_shares):
    limits = {}
    for (node, tech), ms in retrofit_market_shares.items():

        if (node, tech) == existing_tech:
            if len(retrofit_market_shares) == 1:
                # There are no technologies competing for retrofit marketshare.
                # So no limits will apply.
                limits[(node, tech)] = (0, 1)
            else:
                min_retrofit = model.get_param('Retrofit_Min', node, year, tech)
                existing_tech_max_ms = 1 - min_retrofit
                max_retrofit = model.get_param('Retrofit_Max', node, year, tech)
                existing_tech_min_ms = 1 - max_retrofit
                limits[(node, tech)] = (existing_tech_min_ms, existing_tech_max_ms)
        else:
            limits[(node, tech)] = (0, 1)

    limit_adjusted_techs = []
    if not _min_max_ms_compliant(retrofit_market_shares, limits):
        percent_differences = _get_percent_differences(retrofit_market_shares,
                                                       limits,
                                                       return_sorted=True)
        largest_violator = percent_differences[0]
        violator_name = largest_violator[0]

        retrofit_market_shares[violator_name] = _make_ms_min_max_compliant(retrofit_market_shares[violator_name],
                                                                           limits[violator_name])
        limit_adjusted_techs.append(violator_name)

        retrofit_market_shares = _adjust_new_market_shares(retrofit_market_shares, limit_adjusted_techs)

    return retrofit_market_shares


def _adjust_retrofit_marketshares(model, year, existing_tech, retrofit_market_shares):
    if len(retrofit_market_shares) == 0:
        return retrofit_market_shares

    ms_of_all_adopting_techs = 1 - retrofit_market_shares[existing_tech]
    if ms_of_all_adopting_techs == 0:
        # In this case, no market share is given to any retrofitting techs. So no limits to check
        return retrofit_market_shares

    adopting_tech_market_shares = {}
    adopting_tech_ms_limits = {}
    for (node, tech), ms in retrofit_market_shares.items():
        if (node, tech) == existing_tech:
            pass
        else:
            ms_retrofit_min = model.get_param('Market share retro_Min', node, year, tech)
            ms_retrofit_max = model.get_param('Market share retro_Max', node, year, tech)
            adopting_tech_ms_limits[(node, tech)] = (ms_retrofit_min, ms_retrofit_max)
            try:
                adopting_tech_market_shares[(node, tech)] = ms / ms_of_all_adopting_techs
            except:
                jillian = 1

    limit_adjusted_techs = []
    while not _min_max_ms_compliant(adopting_tech_market_shares, adopting_tech_ms_limits):
        percent_differences = _get_percent_differences(adopting_tech_market_shares,
                                                       adopting_tech_ms_limits,
                                                       return_sorted=True)
        largest_violator = percent_differences[0]
        violator_name = largest_violator[0]

        adopting_tech_market_shares[violator_name] = _make_ms_min_max_compliant(adopting_tech_market_shares[violator_name],
                                                                                adopting_tech_ms_limits[violator_name])
        limit_adjusted_techs.append(violator_name)

        # We adjust the marketshares, but then we need to ... , but then need to
        # think_make_ms_min_max_compliant norms everything to be out of 1. So we need to adjust
        # what we are passing into that function
        adopting_tech_market_shares = _adjust_new_market_shares(adopting_tech_market_shares,
                                                                 limit_adjusted_techs)

    adopting_tech_market_shares = {k: v * ms_of_all_adopting_techs for k, v in adopting_tech_market_shares.items()}
    retrofit_market_shares.update(adopting_tech_market_shares)

    return retrofit_market_shares


def calc_retrofits(model, node, year, existing_stock):
    """
    In: the existing stock dictionary. {fuel: existing stock}
    Out:
    """
    comp_type = model.get_param('competition type', node)
    heterogeneity = model.get_param('Heterogeneity', node, year)
    retrofit_stocks = {}
    for existing_tech, stock in existing_stock.items():

        # Find Other Competing
        other_competing_techs = _find_competing_techs(model, node, comp_type)
        other_competing_techs.remove(existing_tech)
        total_weight, competing_weights = _find_competing_weights(model,
                                                                  year,
                                                                  other_competing_techs,
                                                                  heterogeneity)

        # Add Existing Tech's Weight
        existing_tech_weight = _retrofit_existing_tech_weight(model, year, existing_tech, heterogeneity)
        total_weight += existing_tech_weight
        competing_weights[existing_tech] = existing_tech_weight

        # Find Market shares based off of weights
        retrofit_market_shares = {}
        if comp_type == 'tech compete':
            for tech in competing_weights:
                retrofit_market_shares[tech] = competing_weights[tech] / total_weight
        elif comp_type == 'node tech compete':
            for tech in competing_weights:
                retrofit_market_shares[tech] = competing_weights[tech] / total_weight

        # Adjust based on limits of existing technology
        retrofit_market_shares = _apply_retrofit_limits(model, year, existing_tech,
                                                        retrofit_market_shares)

        # Adjust market shares based on limits of techs being retrofitted to
        retrofit_market_shares = _adjust_retrofit_marketshares(model, year, existing_tech,
                                                               retrofit_market_shares)

        # Adjust stocks based on retrofit market shares
        for tech, ms in retrofit_market_shares.items():
            if tech == existing_tech:
                existing_stock[existing_tech] *= ms
            else:
                if tech not in retrofit_stocks:
                    retrofit_stocks[tech] = 0
                retrofit_stocks[tech] += existing_stock[existing_tech] * ms

        # Apply to the existing stock
    return existing_stock, retrofit_stocks
