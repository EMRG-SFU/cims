import pprint as pp
from . import utils


def get_min_max_limits(model, node, year):
    techs = model.get_param('technologies', node, year)
    techs = model.graph.nodes[node][year]['technologies']
    min_max_limits = {}
    for tech in techs:
        min_nms = model.get_param('Market share new_Min', node, year, tech)
        max_nms = model.get_param('Market share new_Max', node, year, tech)
        min_max_limits[tech] = (min_nms, max_nms)
    return min_max_limits


def min_max_ms_compliant(new_market_shares, min_max_limits):
    """

    Parameters
    ----------
    new_market_shares
    min_max_limits

    Returns
    -------

    """
    for tech in new_market_shares:
        min_nms, max_nms = min_max_limits[tech]
        proposed_nms = new_market_shares[tech]

        if proposed_nms < min_nms:
            return False
        elif proposed_nms > max_nms:
            return False

    return True


def get_percent_differences(new_market_shares, min_max_limits, return_sorted=True):
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
    min_nms, max_nms = min_max

    if initial_nms < min_nms:
        return min_nms
    elif initial_nms > max_nms:
        return max_nms
    else:
        raise ValueError


def adjust_new_marketshares(new_market_shares, limit_adjusted_techs):
    remaining_techs = [t for t in new_market_shares if t not in limit_adjusted_techs]

    sum_msj = sum([new_market_shares[t] for t in remaining_techs])
    sum_msl = sum([new_market_shares[t] for t in limit_adjusted_techs])
    for remaining_tech in remaining_techs:
        new_market_share_h = new_market_shares[remaining_tech]
        anms_h = (new_market_share_h / sum_msj) * (1 - sum_msl)
        new_market_shares[remaining_tech] = anms_h

    return new_market_shares


def find_eligible_marketshares(model, node, year, new_market_shares):
    eligible_marketshares = {}
    for tech in new_market_shares:
        is_exogenous = utils.is_param_exogenous(model, 'Market share', node, year, tech)

        first_year_available = model.get_param('Available', node, year, tech)
        first_year_unavailable = model.get_param('Unavailable', node, year, tech)
        is_available = first_year_available <= int(year) < first_year_unavailable

        if (not is_exogenous) and is_available:
            eligible_marketshares[tech] = new_market_shares[tech]

    return eligible_marketshares


def apply_min_max_limits(model, node, year, new_market_shares):
    min_max_limits = get_min_max_limits(model, node, year)

    # Only check & adjust new market shares which are not exogenous
    adjusted_nms = find_eligible_marketshares(model, node, year, new_market_shares)

    # Check to see if all New M/S values comply with Min/Max limits. If yes, do nothing. If no
    # continue to next step.
    limit_adjusted_techs = []
    while not min_max_ms_compliant(adjusted_nms, min_max_limits):
        jillian = 1
        # Apply exogenous Min or Max New M/S limit values on the technology which has the largest
        # % difference between its limit and its initial new market share value.
        percent_differences = get_percent_differences(adjusted_nms,
                                                      min_max_limits,
                                                      return_sorted=True)
        largest_violator, perc_diff = percent_differences[0]
        adjusted_nms[largest_violator] = make_ms_min_max_compliant(adjusted_nms[largest_violator],
                                                                   min_max_limits[largest_violator])
        limit_adjusted_techs.append(largest_violator)

        # For remaining technologies, calculate their individual Adjusted New M/S for technology(s)
        adjusted_nms = adjust_new_marketshares(adjusted_nms, limit_adjusted_techs)

    updated_nms = new_market_shares.copy()
    updated_nms.update(adjusted_nms)

    return updated_nms
