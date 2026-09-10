"""
Module containing all declining capital cost & declining intangible cost functionality, used as
part of LCC calculation.
"""
from math import log2, exp
from .utils.parameter import construction
from .utils.parameter import list as PARAM



# ==========================================
# Declining Capital Cost Functions
# ==========================================
def calc_declining_capital_cost(model: 'CIMS.Model', node: str, year: str, tech: str):
    """
    Calculate the declining capital cost for a node. Should only be used for nodes where a DCC class
    has been specified.

    Parameters
    ----------
    model : The model containing all the information needed for calculating declining capital cost
    node : The name of the node whose declining capital cost is being calculated
    year : The year to calculate declining capital cost for
    tech : The name of the technology whose declining capital cost is being calculated

    Returns
    -------
    float : Declining capital cost for the node, tech, and year specified.
    """
    cc_min = _calc_cc_min(model, node, year, tech=tech)
    cc_learning = _calc_cc_learning(model, node, year, tech=tech)
    cc_declining = min(cc_min, cc_learning)

    return cc_declining


def _calc_cc_min(model, node, year, tech):
    year_avail = model.get_param(PARAM.available, node, str(model.base_year), tech=tech)
    learning_min = model.get_param(PARAM.dcc_learning_min, node, year, tech=tech)

    if int(year) == model.base_year or int(year) <= year_avail or learning_min == 0:
        cc_min = model.get_param(PARAM.fcc, node, year, tech=tech)
    else:
        prev_cc_min = model.get_param(PARAM.capital_cost_min, node, str(int(year) - model.step),
                                      tech=tech)
        cc_min = prev_cc_min * (1 - learning_min) ** model.step

    model.set_param_internal(construction.create_value_dict(cc_min, param_source='calculation'),
                             PARAM.capital_cost_min, node, year, tech=tech)

    return cc_min


def _calc_cc_learning(model, node, year, tech):
    cc_fixed = model.get_param(PARAM.fcc, node, year, tech=tech)

    all_stock = _calc_all_stock(model, node, year, tech=tech)

    bc_1 = model.get_param(PARAM.dcc_capacity_1, node, year, tech=tech)
    bc_2 = model.get_param(PARAM.dcc_capacity_2, node, year, tech=tech)
    bc_3 = model.get_param(PARAM.dcc_capacity_3, node, year, tech=tech)

    pr_1 = model.get_param(PARAM.dcc_progress_ratio_1, node, year, tech=tech)
    pr_2 = model.get_param(PARAM.dcc_progress_ratio_2, node, year, tech=tech)
    pr_3 = model.get_param(PARAM.dcc_progress_ratio_3, node, year, tech=tech)

    segment_1 = segment_2 = segment_3 = 1

    if bc_3:
        segment_1 = _dcc_segment(all_stock, pr_1, bc_1, bc_2)
        segment_2 = _dcc_segment(all_stock, pr_2, bc_2, bc_3)
        segment_3 = _dcc_segment(all_stock, pr_3, bc_3)
    elif bc_2:
        segment_1 = _dcc_segment(all_stock, pr_1, bc_1, bc_2)
        segment_2 = _dcc_segment(all_stock, pr_2, bc_2)
    elif bc_1:
        segment_1 = _dcc_segment(all_stock, pr_1, bc_1)

    cc_learning = cc_fixed * segment_1 * segment_2 * segment_3

    return cc_learning


def _dcc_segment(all_stock, pr, bc_A=None, bc_B=None):
    if bc_A:
        if bc_B:
            segment = (min(max(all_stock, bc_A), bc_B) / bc_A) ** log2(pr)
        else:
            segment = (max(all_stock, bc_A) / bc_A) ** log2(pr)
    else:
        segment = 1
    return segment


# ==========================================
# DCC Caching
# ==========================================
# `_calc_all_stock()` depends only on a technology's DCC class and the year -- never on the
# node/tech it was called for. Left uncached it is therefore recomputed, identically, once per
# member of the class, twice per LCC evaluation (financial + competition upfront cost), four
# times per equilibrium iteration, for every iteration of every year. The caches below collapse
# that to one computation per (dcc_class, year).
_CACHE_ATTR = '_dcc_cache'


def reset_dcc_caches(model: 'CIMS.Model') -> None:
    """
    Discard any DCC state cached on `model`.

    Called at the start of `CIMS.Model.run()`. Only needed when the graph has been edited in
    place between runs; a rebuilt or swapped-in set of DCC classes is detected automatically by
    `_get_cache()`.
    """
    if hasattr(model, _CACHE_ATTR):
        delattr(model, _CACHE_ATTR)


def _get_cache(model):
    """
    Return `model`'s DCC cache, rebuilding it if the model's DCC classes have been replaced.

    `Model.dcc_classes` is reassigned whenever the graph is (re)constructed or swapped out for a
    subgraph (see CIMS.calibration.Calibration.SubGraphs), so the cache is tied to the identity
    of the dict it was derived from and is discarded whenever that dict changes.
    """
    cache = getattr(model, _CACHE_ATTR, None)
    if cache is None or cache['dcc_classes'] is not model.dcc_classes:
        cache = {'dcc_classes': model.dcc_classes,
                 'members': {},
                 'year': None,
                 'all_stock': {}}
        setattr(model, _CACHE_ATTR, cache)
    return cache


def _get_class_members(model, dcc_class, cache):
    """
    Return `[(node, tech, unit_convert), ...]` for every technology in `dcc_class`.

    `multiplier_load_factor` (needed to convert transportation stocks to a common vkt unit) is
    read at the base year, and base-year values are fixed once `Model.initialize_graph()` has run
    for the base year. It is therefore looked up once per technology rather than once per
    technology per call.
    """
    members = cache['members'].get(dcc_class)
    if members is None:
        base_year = str(model.base_year)
        members = []
        for node_k, tech_k in model.dcc_classes[dcc_class]:
            # Need to convert stocks for transportation techs to common vkt unit
            unit_convert = model.get_param(PARAM.multiplier_load_factor, node_k, base_year,
                                           tech=tech_k)
            if unit_convert is None:
                unit_convert = 1
            members.append((node_k, tech_k, unit_convert))
        cache['members'][dcc_class] = members
    return members


def _calc_all_stock(model, node, year, tech):
    dcc_class = model.get_param(PARAM.dcc_class, node, year, tech=tech)

    cache = _get_cache(model)
    if cache['year'] != year:
        # Only one year is ever in flight at a time, so the previous year's entries are dead.
        cache['year'] = year
        cache['all_stock'] = {}
    elif dcc_class in cache['all_stock']:
        return cache['all_stock'][dcc_class]

    base_year = int(model.base_year)
    base_year_str = str(base_year)
    step = model.step

    # Range function is exclusive of final year (i.e., up to but not including final year)
    reference_years = [str((j - base_year) // step * step + base_year)
                       for j in range(base_year, int(year))]

    stock_sums = {PARAM.stock_base: 0,
                  PARAM.stock_new: 0}
    for node_k, tech_k, unit_convert in _get_class_members(model, dcc_class, cache):
        # Base Stock summed over all techs in DCC class (base year only)
        bs_k = model.get_param(PARAM.stock_base, node_k, base_year_str, tech=tech_k)
        if bs_k is not None:
            stock_sums[PARAM.stock_base] += bs_k / unit_convert

        for reference_year in reference_years:
            ns_jk = model.get_param(PARAM.stock_new, node_k, reference_year, tech=tech_k)
            stock_sums[PARAM.stock_new] += ns_jk / unit_convert
    all_stock = stock_sums[PARAM.stock_base] + stock_sums[PARAM.stock_new]

    # Cache beyond the base year only. During the base year, stock allocation is still writing
    # stock_base at the base year (stock_allocation._record_allocation_results), so all_stock
    # genuinely changes within that year. From the following year onward every input -- stock_base
    # at the base year, and stock_new at years strictly before `year` -- is final, so the value is
    # constant for the whole of that year's solve, equilibrium iterations included.
    if int(year) > base_year:
        cache['all_stock'][dcc_class] = all_stock

    return all_stock


# ==========================================
# Declining Intangible Cost Functions
# ==========================================
def calc_declining_intangible_cost(model: 'CIMS.Model', node: str, year: str, tech: str) -> float:
    """
    Calculate Annual Declining Intangible Cost (DIC).

    Parameters
    ----------
    model : The model containing component parts of DIC.
    node : The node to calculate DIC for.
    year : The year to calculate DIC for.
    tech : The technology to calculate DIC for.

    Returns
    -------
    float : The DIC.
    """
    # Retrieve Exogenous Terms from Model Description
    dic_initial = model.get_param(PARAM.dic_initial, node, year, tech=tech)
    dic_slope = model.get_param(PARAM.dic_slope, node, year, tech=tech)
    dic_x50 = model.get_param(PARAM.dic_x50, node, year, tech=tech)
    dic_min = model.get_param(PARAM.dic_min, node, year, tech=tech)

    # In base year, dic==dic_0
    if int(year) <= int(model.base_year + model.step):
        return dic_initial

    # Find the tech's NMS & DIC in the previous year
    prev_year = str(int(year) - model.step)
    prev_nms = _find_dic_class_new_market_share(model, node, prev_year, tech=tech)
    prev_dic = model.get_param(PARAM.dic, node, prev_year, tech=tech)

    # Calculate DIC
    dic = min(prev_dic, max(0, dic_min + (dic_initial-dic_min)/
                                         (1+(prev_nms/dic_x50)**dic_slope)))

    return dic


def _find_dic_class_new_market_share(model, node, year, tech):
    """
    Find the total new market share attributed to technologies from the node's DIC class (relative to
    all technologies and nodes competing for market share with technologies within the DIC class)
    """
    dic_class = model.get_param(PARAM.dic_class, node, year, tech=tech)
    if dic_class:
        # We already know there is a DIC class
        dic_class_techs = model.dic_classes[dic_class]

        # DIC Stock
        dic_class_stock = _find_dic_class_stock_new(model, dic_class_techs, year)

        # All Stock
        all_competing_stock = _find_dic_competing_stock_new(model, dic_class_techs, year)

        # New Market Share
        if dic_class_stock == 0:
            dic_nms = 0
        else:
            dic_nms = dic_class_stock / all_competing_stock
    else:
        dic_nms = model.get_param(PARAM.market_share_new, node, year, tech=tech)

    return dic_nms


def _find_dic_class_stock_new(model, dic_techs, year):
    """
    Calculate the new stock from all the technologies in the DIC class.
    """
    new_dic_stock = 0
    for node, tech in dic_techs:
        new_dic_stock += model.get_param(PARAM.stock_new, node, year, tech=tech)
    return new_dic_stock


def _find_dic_competing_stock_new(model, dic_techs, year):
    """
    Calculate the new stock from all the technologies competing for market share with the nodes in
    the DIC class (including the DIC techs).
    """
    dic_nodes = {x[0] for x in dic_techs}

    competing_stocks = {}
    for node in dic_nodes:
        competing_techs = _find_dic_competing_techs(model, node)
        for c_node, c_tech in competing_techs:
            competing_stocks[(c_node, c_tech)] = \
                model.get_param(PARAM.stock_new, c_node, year, tech=c_tech)

    return sum(v for k, v in competing_stocks.items() if v is not None)


def _find_dic_competing_techs(model, node):
    """
    Find all the nodes/technologies competing for stock with the DIC class technologies. For node
    tech compete nodes this includes all the technologies of nodes requested by the parent NTC node.
    """
    base_year = str(model.base_year)
    competing_technologies = []

    # Find all technologies at the node
    if model.get_param(PARAM.competition_type, node) == PARAM.competition_compete:
        for tech in model.graph.nodes[node][base_year][PARAM.technologies]:
            competing_technologies.append((node, tech))

    return set(competing_technologies)
