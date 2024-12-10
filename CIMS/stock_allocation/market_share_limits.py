from CIMS import old_utils
from ..utils import parameters as PARAM

#############################
# Market Share Classes
#############################
def apply_min_max_class_limits(model: "CIMS.Model", node: str, year: str, new_market_shares: dict):
    """
    Adjusts new market shares to comply with minimum and maximum class limits
    for a given model, node, and year.

    This function ensures that the market shares of classes comply with the
    exogenously set minimum and maximum limits. It iteratively adjusts the
    market shares of other classes' technologies until all classes satisfy the
    imposed constraints.

    Parameters
    ----------
    model : CIMS.Model
        The model object containing information about the technologies, min/max
        marketshare limits, and their min/max classes.
    node : str
        The node in the model for which the market shares are applied.
    year : str
        The year for which the market shares and limits are being evaluated.
    new_market_shares : dict
        A dictionary where  keys are technology identifiers and values are the
        proposed new market shares for each technology in the given year.

    Returns
    -------
    updated_nms : dict
        A dictionary of updated market shares for the technologies after
        applying minimum and maximum class limits. If no limits are violated,
        the function returns the original `new_market_shares`. Otherwise, the
        values are adjusted to ensure compliance with the limits.

    Notes
    -----
    - Only non-exogenous market shares are adjusted, and the adjustment is
      performed iteratively until all classes comply with the min/max limits or
      no more classes remain to use for adjustments.
    - The market share class with the largest percentage difference between its
      initial and constrained market share value is prioritized for adjustment
      in each iteration.
    """
    # Get marketshare classes
    tech_class_map, class_tech_map = _get_market_share_class_maps(model, node, year)

    # Get min/max limits for each class
    min_max_class_limits = _get_min_max_class_limits(model, node, year)

    if len(min_max_class_limits) == 0:
        return new_market_shares

    # Only check & adjust new market shares which are not exogenous
    adjusted_nms = _find_eligible_market_shares(model, node, year, new_market_shares)
    ms_class_adjusted_nms = _find_ms_class_market_shares(adjusted_nms, tech_class_map)

    # Check to see if all New M/S values comply with Min/Max limits. If yes, do nothing. If no
    # continue to next step.
    limit_adjusted_techs = []
    while not _min_max_ms_class_compliant(ms_class_adjusted_nms, min_max_class_limits):
        # Apply exogenous Min or Max New M/S limit values on the technology which has the largest
        # % difference between its limit and its initial new market share value.
        percent_differences = _get_percent_differences_class(ms_class_adjusted_nms,
                                                       min_max_class_limits,
                                                       return_sorted=True)
        largest_violator = percent_differences[0]
        violator_class = largest_violator[0]
        for tech in class_tech_map[violator_class]:
            if tech in adjusted_nms:
                adjusted_nms[tech] = _make_ms_min_max_class_compliant(
                    ms_class_adjusted_nms[violator_class],
                    min_max_class_limits[tech_class_map[tech]],
                    adjusted_nms[tech]
                )
                limit_adjusted_techs.append(tech)

        # For remaining technologies, calculate their individual Adjusted New M/S for technology(s)
        adjusted_nms = _adjust_new_market_shares(adjusted_nms, limit_adjusted_techs)
        ms_class_adjusted_nms = _find_ms_class_market_shares(adjusted_nms, tech_class_map)

    updated_nms = new_market_shares.copy()
    updated_nms.update(adjusted_nms)

    return updated_nms


def _get_market_share_class_maps(model, node, year):
    tech_ms_class_map = {}
    ms_class_tech_map = {}
    techs = model.graph.nodes[node][year][PARAM.technologies]
    for tech in techs:
        ms_class = model.get_param(PARAM.market_share_class, node, year=year, tech=tech)
        tech_ms_class_map[tech] = ms_class
        ms_class_tech_map.setdefault(ms_class, []).append(tech)

    return tech_ms_class_map, ms_class_tech_map


def _get_min_max_class_limits(model, node, year):
    """
    Find the minimum & maximum market share limits in a given year for all
    market share classes at a node.
    """
    ms_classes = []
    techs = model.graph.nodes[node][year][PARAM.technologies]
    for tech in techs:
        ms_class = model.get_param(PARAM.market_share_class, node, year=year, tech=tech)
        ms_classes.append(ms_class)
    
    min_max_limits = {}
    for ms_class in ms_classes:
        if ms_class is None:
            min_ms = model.get_parameter_default(PARAM.market_share_class_min)
            max_ms = model.get_parameter_default(PARAM.market_share_class_max)
        else:
            min_ms = model.get_param(PARAM.market_share_class_min, node, year=year, context=ms_class)
            max_ms = model.get_param(PARAM.market_share_class_max, node, year=year, context=ms_class)
        min_max_limits[ms_class] = (min_ms, max_ms)

    return min_max_limits


def _get_percent_differences_class(aggregate_nms, min_max_limits, return_sorted=True):
    percent_diffs = []
    for ms_class in aggregate_nms:
        min_nms, max_nms = min_max_limits[ms_class]
        proposed_nms = aggregate_nms[ms_class]

        if proposed_nms < min_nms:
            percent_diffs.append((ms_class, proposed_nms - min_nms))
        elif proposed_nms > max_nms:
            percent_diffs.append((ms_class, proposed_nms - max_nms))
        else:
            percent_diffs.append((ms_class, 0))

    if return_sorted:
        percent_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

    return percent_diffs


def _find_ms_class_market_shares(new_market_shares, ms_class_map):
    aggregate_nms = {ms_class_map[t]: 0 for t in new_market_shares}
    for tech in new_market_shares:
        ms_class = ms_class_map[tech]
        aggregate_nms[ms_class] += new_market_shares[tech]
    return aggregate_nms


def _min_max_ms_class_compliant(ms_class_adjusted_nms, min_max_limits):

    for ms_class in ms_class_adjusted_nms:
        min_ms, max_ms = min_max_limits[ms_class]
        proposed_ms = ms_class_adjusted_nms[ms_class]

        if proposed_ms < min_ms:
            return False
        if proposed_ms > max_ms:
            return False
        
    return True


def _make_ms_min_max_class_compliant(class_nms, min_max_class, nms):
    min_nms, max_nms = min_max_class

    if class_nms < min_nms:
        return nms * min_nms/class_nms

    if class_nms > max_nms:
        return nms * max_nms/class_nms

    return nms



#############################
# Min/Max Market Share Limits
#############################
def apply_min_max_limits(model: "CIMS.Model", node: str, year: str, new_market_shares: dict):
    """
    Adjusts technologies' new market shares to comply with minimum and maximum
    market share limits of a given node and year.

    This function iteratively adjusts the market shares of technologies that
    exceed or fall short of their exogenous minimum and maximum limits. If a
    technology violates the limits, its market share is adjusted, and the
    remaining technologies' market shares are recalculated, prioritizing
    technologies with the largest deviation from their limits.

    Parameters
    ----------
    model : CIMS.Model
        The model object containing information about the technologies and their
        min/max market share limits.
    node : str
        The node in the model for which the market shares are applied.
    year : str
        The year for which the market shares and limits are being evaluated.
    new_market_shares : dict
        A dictionary where keys are technology identifiers and values are the proposed new market
        shares for each technology in the given year. The values should be proportions in the
        range [0, 1].

    Returns
    -------
    updated_nms : dict
        An updated version of the `new_market_shares` dictionary, where
        endogenous market shares comply with the exogenous min/max limits.
        Technologies that do not have classmates for stock redistribution will
        have their market share limits relaxed.

    Notes
    -----
    - Only non-exogenous market shares are adjusted. The adjustment is performed
      iteratively until all technologies comply with their min/max limits, or
      until no classmates remain for stock redistribution.
    - The technology with the largest percentage difference between its initial
      and constrained market share value is prioritized for adjustment in each
      iteration.
    - If no available classmates exist for a violating technology, its min/max
      limit is relaxed.
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

        # Check whether tech has classmates available for stock redistribution
        tech_class_map, class_tech_map = _get_market_share_class_maps(model, node, year)
        class_members = class_tech_map[tech_class_map[violator_name]]
        available_classmates = [c for c in class_members if c not in limit_adjusted_techs + [violator_name]]
        if len(available_classmates):
            adjusted_nms[violator_name] = _make_ms_min_max_compliant(adjusted_nms[violator_name],
                                                                    min_max_limits[violator_name])
            limit_adjusted_techs.append(violator_name)
            # For remaining classmate technologies, calculate their individual
            # Adjusted New M/S for technology(s)
            class_market_share = sum([new_market_shares[t] for t in class_members])
            adjusted_nms = _adjust_new_market_shares(adjusted_nms, limit_adjusted_techs, class_tech_map[tech_class_map[violator_name]], class_market_share)
        else:
            min_max_limits[violator_name] = (0, 1)

    updated_nms = new_market_shares.copy()
    updated_nms.update(adjusted_nms)

    return updated_nms


def _get_min_max_limits(model, node, year):
    """
    Find the minimum & maximum market share limits in a given year for all technologies at a
    specified node in the model.

    Parameters
    ----------
    model : CIMS.Model
        The CIMS model containing the market share limits you want to retrieve.
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
    techs = model.graph.nodes[node][year][PARAM.technologies]
    min_max_limits = {}
    for tech in techs:
        min_nms = model.get_param(PARAM.market_share_new_min, node, year=year, tech=tech)
        max_nms = model.get_param(PARAM.market_share_new_max, node, year=year, tech=tech)
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


#############################
# Shared Code
#############################
def _adjust_new_market_shares(new_market_shares, limit_adjusted_techs, ms_classmates=None, class_market_share=None):
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
    if ms_classmates is None:
        ms_classmates = new_market_shares.keys()
    if class_market_share is None:
        class_market_share = 1

    remaining_techs = [t for t in new_market_shares if (t not in limit_adjusted_techs) and (t in ms_classmates)]

    sum_msj = sum([new_market_shares[t] for t in remaining_techs])
    sum_msl = sum([new_market_shares[t] for t in limit_adjusted_techs])

    adjust_amount = class_market_share - sum_msl
    for remaining_tech in remaining_techs:
        if adjust_amount > 0:
            new_market_share_h = new_market_shares[remaining_tech]
            try:
                # [(initial M/S / sum of Class initial M/S not overridden) * (calculated class m/s - sum of Class min max applied)
                anms_h = (new_market_share_h / sum_msj) * (class_market_share - sum_msl)
            except ZeroDivisionError:
                anms_h = 0
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
    model : CIMS.Model
        The CIMS model containing node.
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
        is_exogenous = old_utils.is_param_exogenous(model, PARAM.market_share, node, year=year, tech=tech)

        first_year_available = model.get_param(PARAM.available, node, year=year, tech=tech)
        first_year_unavailable = model.get_param(PARAM.unavailable, node, year=year, tech=tech)
        is_available = first_year_available <= int(year) < first_year_unavailable

        if (not is_exogenous) and is_available:
            eligible_market_shares[tech] = new_market_shares[tech]

    return eligible_market_shares