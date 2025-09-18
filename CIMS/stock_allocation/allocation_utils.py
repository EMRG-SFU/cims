from ..utils.parameter import list as PARAM
from ..lcc_calculation import calc_lcc_retrofit
import numpy as np

def _find_competing_techs(model, node, comp_type):
    """
    A helper function used by _calculate_new_market_shares() to find all the technologies competing
    for marketshare at a given node & year.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving data.
    node : str
        Name of the node (branch notation) whose competing technologies we want to find.
    comp_type : str
        The type of competition occurring at the node. One of {'tech compete'}.

    Returns
    -------
    list :
        The list of technologies competing for market share at `node`.
        If comp_type is Tech Compete, this will simply be the technologies
        defined at the node. This does not verify the technology is available
        in the given year.

    """
    base_year = str(model.base_year)
    node_year_data = model.graph.nodes[node][base_year]
    competing_technologies = []

    if comp_type == 'tech compete':
        for tech in node_year_data[PARAM.technologies]:
            competing_technologies.append((node, tech))

    return competing_technologies


def _find_competing_weights(model, year, heterogeneity, competing_techs, existing_tech=None):
    """
    A helper function called by _calculate_new_market_shares() and calc_retrofits() to find the total weight and technology-specific weights used during market share competition.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving values relevant to weight calculation.
    year : str
        The year of interest.
    heterogeneity : float
        The heterogeneity value used during market share competition.
    competing_techs : list
        A list returned from _find_competing_techs() that includes all of the technologies competing
        for market share at the given node.
    existing_tech : str, default=None
        The name of the existing technology for which the alternative retrofit lifecycle cost will 
        be calculated and used. If None, the function assumes a new market share competition where 
        the full lifecycle cost is used for all technologies.
    
    Returns
    -------
    float :
        The total weight across all competing_technologies.
    dict :
        A dictionary mapping each technology (represented by a `(node_branch, tech)`) to the weight
        it will have during market share competition.
    """
    log_weight_all = []
    log_weight = {}

    for node_branch, tech in competing_techs:
        tech_lcc = None
        # retrofit existing stock
        if tech == existing_tech:
            tech_lcc = calc_lcc_retrofit(model, node_branch, year, tech)
        # regular tech compete
        else:
            # find competing techs
            year_avail = model.get_param(PARAM.available, node_branch, str(model.base_year), tech=tech)
            year_unavail = model.get_param(PARAM.unavailable, node_branch, str(model.base_year), tech=tech)
            if year_avail <= int(year) < year_unavail:
                tech_lcc = model.get_param(PARAM.lcc_competition, node_branch, year, tech=tech)
        
        if tech_lcc is not None:
            # find softplus transform for all lcc_competition values
            tech_lcc_transform = _stable_transform(tech_lcc)
            # use log weights to avoid overflow errors - log-sum-exp trick
            log_weight[tech] =  (-1 * heterogeneity) * np.log(tech_lcc_transform)
            log_weight_all.append(log_weight[tech])

    total_weight = 0
    weights = {}

    for node_branch, tech in competing_techs:
        year_avail = model.get_param(PARAM.available, node_branch, str(model.base_year), tech=tech)
        year_unavail = model.get_param(PARAM.unavailable, node_branch, str(model.base_year), tech=tech)
        if (tech == existing_tech) or (year_avail <= int(year) < year_unavail):
            # use log weights to avoid overflow errors - log-sum-exp trick
            weight_max = max(log_weight_all)
            weight = np.exp(log_weight[tech] - weight_max)
            weights[(node_branch, tech)] = weight
            total_weight += weight

    return total_weight, weights


def _stable_transform(tech_lcc):
    """
    A helper function of _find_competing_weights(). To handle negative and zero value lcc's, use a piecewise function:
            - when positive or zero, use the softplus transform to reduce the effect of small positive input values. This function is monotonic (strictly increasing) and introduces minimal distortion for lifecycle costs greater than 3.
            - when negative, use a monotonic, smooth equation based on softplus(-x), but approaching zero weight polynomially (rather than exponentially) to preserve behaviour of relative lifecycle costs similar to the positive number space.

    Parameters
    ----------
    tech_lcc : float
        The lifecycle cost associated with a specific technology.

    Returns
    -------
    float :
        softplus transform of lcc_competition
    """
    # The softplus transform function 'softplus(x)' is approximately equal 'x' for large values, but returns infinity for very large values of 'x'. To avoid an overflow error with the np.exp function, use a limit on the input value above where softplus(x) = x.
    if tech_lcc > 20:
        tech_lcc_transform = tech_lcc
    elif tech_lcc < 0:
        # Use an exponent value of 0.7 for smooth weighting approximately equal to behaviour in positive number space
        tech_lcc_transform = (1 + np.log1p( np.exp( -1 * np.clip( tech_lcc, -500, 0)))) ** (-0.7)
    else:
        # # np.log1p(y) = log(1 + y) but more accurate for small y
        # clip prevents underflow error for very negative x
        tech_lcc_transform = np.log1p( np.exp( np.clip( tech_lcc, -500, 20 )))

    return tech_lcc_transform
