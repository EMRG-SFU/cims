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
        The type of competition occurring at the node. One of {'node tech compete', 'tech compete'}.

    Returns
    -------
    list :
        The list of technologies competing for market share at `node`.
        If comp_type is Tech Compete, this will simply be the technologies defined at the node. If
        comp_type is Node Tech Compete, this will include the technologies of the services requested
        by node. This does not verify the technology is available in the given year.

    """
    base_year = str(model.base_year)
    node_year_data = model.graph.nodes[node][base_year]
    competing_technologies = []

    if comp_type == 'tech compete':
        for tech in node_year_data['technologies']:
            competing_technologies.append((node, tech))

    elif comp_type == 'node tech compete':
        for child in node_year_data['technologies']:
            child_node = model.graph.nodes[node][base_year]['technologies'][child]['service requested'][child]['target']
            for tech in model.graph.nodes[child_node][base_year]['technologies']:
                competing_technologies.append((child_node, tech))

    return competing_technologies


def _find_competing_weights(model, year, competing_techs, heterogeneity):
    """
    A helper function called by _calculate_new_market_shares() to find the total weight and
    technology-specific weights used during market share competition.

    Parameters
    ----------
    model : CIMS.Model
        The model to use for retrieving values relevant to weight calculation.
    year : str
        The year of interest.
    competing_techs : list
        A list returned from _find_competing_techs() that includes all of the technologies competing
        for market share at the given node.
    heterogeneity : float
        The heterogeneity value used during market share competition.

    Returns
    -------
    float :
        The total weight across all competing_technologies.
    dict :
        A dictionary mapping each technology (represented by a `(node_branch, tech)`) to the weight
        it will have during market share competition.
    """
    total_weight = 0
    weights = {}

    for node_branch, tech in competing_techs:
        year_avail = model.get_param('available', node_branch, str(model.base_year), tech=tech)
        year_unavail = model.get_param('unavailable', node_branch, str(model.base_year), tech=tech)
        if year_avail <= int(year) < year_unavail:
            tech_lcc = model.get_param('lcc_complete', node_branch, year, tech=tech)
            weight = _calculate_lcc_weight(tech_lcc, heterogeneity)
            weights[(node_branch, tech)] = weight
            total_weight += weight

    return total_weight, weights


def _calculate_lcc_weight(tech_lcc, heterogeneity):
    """
    A helper function of _find_competing_weights() that calculates the weight a technology will be
    assigned during market share competition.

    If the technology's lcc is less than 0.01, we will approximate weight using a calculation
    equivalent to Excel's TREND() function.

    Parameters
    ----------
    tech_lcc : float
        The life cycle cost associated with a specific technology.

    heterogeneity : float
        The heterogeneity value the technology's node will use during market share competition.

    Returns
    -------
    float :
        The weight a technology will have during market share competition.
    """
    if tech_lcc < 0.01:
        weight_1 = 0.1 ** (-1 * heterogeneity)
        weight_2 = 0.01 ** (-1 * heterogeneity)
        slope = (weight_2 - weight_1) / (0.01 - 0.1)
        weight = slope * tech_lcc + (weight_1 - slope * 0.1)
    else:
        weight = tech_lcc ** (-1 * heterogeneity)
    return weight
