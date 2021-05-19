from . import utils


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
