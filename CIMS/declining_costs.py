"""
Module containing all declining capital cost & declining intangible cost functionality, used as
part of LCC calculation.
"""
from math import log2, exp
from . import utils


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
    year_avail = model.get_param('available', node, str(model.base_year), tech=tech)
    min_learning = model.get_param('dcc_min learning', node, year, tech=tech)

    if int(year) == model.base_year or int(year) <= year_avail or min_learning == 0:
        cc_min = model.get_param('fcc', node, year, tech=tech)
    else:
        prev_cc_min = model.get_param('capital_cost_min', node, str(int(year) - model.step),
                                      tech=tech)
        cc_min = prev_cc_min * (1 - min_learning) ** model.step

    model.set_param_internal(utils.create_value_dict(cc_min, param_source='calculation'),
                             'capital_cost_min', node, year, tech=tech)

    return cc_min


def _calc_cc_learning(model, node, year, tech):
    cc_fixed = model.get_param('fcc', node, year, tech=tech)

    all_stock = _calc_all_stock(model, node, year, tech=tech)
    segment_1 = _dcc_segment_1(model, node, year, tech, all_stock)
    segment_2 = _dcc_segment_2(model, node, year, tech, all_stock)
    segment_3 = _dcc_segment_3(model, node, year, tech, all_stock)

    cc_learning = cc_fixed * segment_1 * segment_2 * segment_3

    return cc_learning


def _calc_all_stock(model, node, year, tech):
    dcc_class = model.get_param('dcc_class', node, year, tech=tech, context='context')
    dcc_class_techs = model.dcc_classes[dcc_class]

    stock_sums = {'base_stock': 0,
                  'new_stock': 0}
    for node_k, tech_k in dcc_class_techs:
        # Need to convert stocks for transportation techs to common vkt unit
        unit_convert = model.get_param('load factor', node_k, str(model.base_year), tech=tech_k)
        if unit_convert is None:
            unit_convert = 1

        # Base Stock summed over all techs in DCC class (base year only)
        bs_k = model.get_param('base_stock', node_k, str(model.base_year), tech=tech_k)
        if bs_k is not None:
            stock_sums['base_stock'] += bs_k / unit_convert

        # New Stock summed over all techs in DCC class and over all previous years (excluding base
        # year)
        year_list = [str(x) for x in
                     range(int(model.base_year) + int(model.step),
                           int(year),
                           int(model.step))]
        for j in year_list:
            ns_jk = model.get_param('new_stock', node_k, j, tech=tech_k)
            stock_sums['new_stock'] += ns_jk / unit_convert

    all_stock = stock_sums['base_stock'] + stock_sums['new_stock']

    return all_stock


def _dcc_segment_1(model, node, year, tech, all_stock):
    bc_1 = model.get_param('dcc_capacity_1', node, year, tech=tech)
    bc_2 = model.get_param('dcc_capacity_2', node, year, tech=tech)
    pr_1 = model.get_param('dcc_progress ratio_1', node, year, tech=tech)
    segment_1 = (min(max(all_stock, bc_1), bc_2) / bc_1) ** log2(pr_1)
    return segment_1


def _dcc_segment_2(model, node, year, tech, all_stock):
    bc_2 = model.get_param('dcc_capacity_2', node, year, tech=tech)
    bc_3 = model.get_param('dcc_capacity_3', node, year, tech=tech)
    pr_2 = model.get_param('dcc_progress ratio_2', node, year, tech=tech)
    segment_2 = (min(max(all_stock, bc_2), bc_3) / bc_2) ** log2(pr_2)
    return segment_2


def _dcc_segment_3(model, node, year, tech, all_stock):
    bc_3 = model.get_param('dcc_capacity_3', node, year, tech=tech)
    pr_3 = model.get_param('dcc_progress ratio_3', node, year, tech=tech)
    segment_3 = (max(all_stock, bc_3) / bc_3) ** log2(pr_3)
    return segment_3


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
    initial_intangible_cost = model.get_param('dic_initial', node, year, tech=tech)
    rate_constant = model.get_param('dic_rate', node, year, tech=tech)
    shape_constant = model.get_param('dic_shape', node, year, tech=tech)

    # Calculate DIC
    if int(year) == int(model.base_year):
        return initial_intangible_cost

    prev_year = str(int(year) - model.step)

    prev_nms = _find_dic_class_new_market_share(model, node, prev_year, tech=tech)

    try:
        denominator = 1 + shape_constant * exp(rate_constant * prev_nms)
    except OverflowError as overflow:
        print(node, year, shape_constant, rate_constant, prev_nms)
        raise overflow

    prev_dic = calc_declining_intangible_cost(model, node, prev_year, tech)
    dic = min(prev_dic, initial_intangible_cost / denominator)

    return dic


def _find_dic_class_new_market_share(model, node, year, tech):
    """
    Find the total new market share attributed to technologies from the node's DIC class (relative to
    all technologies and nodes competing for market share with technologies within the DIC class)
    """
    dic_class = model.get_param('dic_class', node, year, tech=tech, context='context')
    if dic_class:
        # We already know there is a DIC class
        dic_class_techs = model.dic_classes[dic_class]

        # DIC Stock
        dic_class_stock = _find_dic_class_new_stock(model, dic_class_techs, year)

        # All Stock
        all_competing_stock = _find_dic_competing_new_stock(model, dic_class_techs, year)

        # New Market Share
        if dic_class_stock == 0:
            dic_nms = 0
        else:
            dic_nms = dic_class_stock / all_competing_stock
    else:
        dic_nms = model.get_param('new_market_share', node, year, tech=tech)

    return dic_nms


def _find_dic_class_new_stock(model, dic_techs, year):
    """
    Calculate the new stock from all the technologies in the DIC class.
    """
    new_dic_stock = 0
    for node, tech in dic_techs:
        new_dic_stock += model.get_param('new_stock', node, year, tech=tech)
    return new_dic_stock


def _find_dic_competing_new_stock(model, dic_techs, year):
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
                model.get_param('new_stock', c_node, year, tech=c_tech)

    return sum(v for k, v in competing_stocks.items() if v is not None)


def _find_dic_competing_techs(model, node):
    """
    Find all the nodes/technologies competing for stock with the DIC class technologies. For node
    tech compete nodes this includes all the technologies of nodes requested by the parent NTC node.
    """
    base_year = str(model.base_year)
    competing_technologies = []

    # Find all technologies at the node
    if model.get_param('competition type', node) == 'tech compete':
        for tech in model.graph.nodes[node][base_year]['technologies']:
            competing_technologies.append((node, tech))

    # Find any technologies from Node-Tech-Compete siblings
    parents = [u for u, v in model.graph.in_edges(node)]
    for parent in parents:
        if model.get_param('competition type', parent) == 'node tech compete':
            for sibling in model.graph.nodes[parent][base_year]['technologies']:
                sibling_node = \
                model.graph.nodes[parent][base_year]['technologies'][sibling]['service requested'][
                    sibling]['target']
                for tech in model.graph.nodes[sibling_node][base_year]['technologies']:
                    competing_technologies.append((sibling_node, tech))

    return set(competing_technologies)
