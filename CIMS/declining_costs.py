"""
Module containing all declining capital cost & declining intangible cost functionality, used as
part of LCC calculation.
"""
from math import log2, exp
from . import old_utils
from .utils import parameters as PARAM



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
    min_learning = model.get_param(PARAM.dcc_min_learning, node, year, tech=tech)

    if int(year) == model.base_year or int(year) <= year_avail or min_learning == 0:
        cc_min = model.get_param(PARAM.fcc, node, year, tech=tech)
    else:
        prev_cc_min = model.get_param(PARAM.capital_cost_min, node, str(int(year) - model.step),
                                      tech=tech)
        cc_min = prev_cc_min * (1 - min_learning) ** model.step

    model.set_param_internal(old_utils.create_value_dict(cc_min, param_source='calculation'),
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


def _calc_all_stock(model, node, year, tech):
    dcc_class = model.get_param(PARAM.dcc_class, node, year, tech=tech)
    dcc_class_techs = model.dcc_classes[dcc_class]

    stock_sums = {PARAM.base_stock: 0,
                  PARAM.new_stock: 0}
    for node_k, tech_k in dcc_class_techs:
        # Need to convert stocks for transportation techs to common vkt unit
        unit_convert = model.get_param(PARAM.load_factor, node_k, str(model.base_year), tech=tech_k)
        if unit_convert is None:
            unit_convert = 1

        # Base Stock summed over all techs in DCC class (base year only)
        bs_k = model.get_param(PARAM.base_stock, node_k, str(model.base_year), tech=tech_k)
        if bs_k is not None:
            stock_sums[PARAM.base_stock] += bs_k / unit_convert

        year_list = [x for x in range(int(model.base_year), int(year))]
        for j in year_list:
            reference_year = (j - int(model.base_year)) // model.step * model.step + int(model.base_year)
            ns_jk = model.get_param(PARAM.new_stock, node_k, str(reference_year), tech=tech_k)
            stock_sums[PARAM.new_stock] += ns_jk / unit_convert
    all_stock = stock_sums[PARAM.base_stock] + stock_sums[PARAM.new_stock]

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
        dic_class_stock = _find_dic_class_new_stock(model, dic_class_techs, year)

        # All Stock
        all_competing_stock = _find_dic_competing_new_stock(model, dic_class_techs, year)

        # New Market Share
        if dic_class_stock == 0:
            dic_nms = 0
        else:
            dic_nms = dic_class_stock / all_competing_stock
    else:
        dic_nms = model.get_param(PARAM.new_market_share, node, year, tech=tech)

    return dic_nms


def _find_dic_class_new_stock(model, dic_techs, year):
    """
    Calculate the new stock from all the technologies in the DIC class.
    """
    new_dic_stock = 0
    for node, tech in dic_techs:
        new_dic_stock += model.get_param(PARAM.new_stock, node, year, tech=tech)
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
                model.get_param(PARAM.new_stock, c_node, year, tech=c_tech)

    return sum(v for k, v in competing_stocks.items() if v is not None)


def _find_dic_competing_techs(model, node):
    """
    Find all the nodes/technologies competing for stock with the DIC class technologies. For node
    tech compete nodes this includes all the technologies of nodes requested by the parent NTC node.
    """
    base_year = str(model.base_year)
    competing_technologies = []

    # Find all technologies at the node
    if model.get_param(PARAM.competition_type, node) == 'tech compete':
        for tech in model.graph.nodes[node][base_year][PARAM.technologies]:
            competing_technologies.append((node, tech))

    # Find any technologies from Node-Tech-Compete siblings
    parents = [u for u, v in model.graph.in_edges(node)]
    for parent in parents:
        if model.get_param(PARAM.competition_type, parent) == 'node tech compete':
            for sibling_tech in model.graph.nodes[parent][base_year][PARAM.technologies]:
                sibling_nodes = model.get_param(PARAM.service_requested, parent, base_year, tech=sibling_tech)
                for sibling_node in sibling_nodes:
                    for tech in model.graph.nodes[sibling_node][base_year][PARAM.technologies]:
                        competing_technologies.append((sibling_node, tech))

    return set(competing_technologies)
