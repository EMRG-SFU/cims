"""
Module containing all declining capital cost functionality, used as part of LCC calculation.
"""
from math import log2

import pyCIMS
from . import utils


def calc_declining_capital_cost(model: pyCIMS.Model, node: str, year: str, tech: str):
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
    if int(year) == model.base_year:
        cc_min = model.get_param('capital cost_overnight', node, year, tech=tech)
    else:
        prev_cc_min = model.get_param('capital_cost_min', node, str(int(year) - model.step),
                                      tech=tech)
        mal = model.get_param('mal', node, year, tech=tech)
        cc_min = prev_cc_min * (1 - mal) ** model.step

    model.set_param_internal(utils.create_value_dict(cc_min, param_source='calculation'),
                             'capital_cost_min', node, year, tech=tech)

    return cc_min


def _calc_cc_learning(model, node, year, tech):
    cc_overnight = model.get_param('capital cost_overnight', node, year, tech=tech)

    all_stock = _calc_all_stock(model, node, year, tech=tech)
    segment_1 = _segment_1(model, node, year, tech, all_stock)
    segment_2 = _segment_2(model, node, year, tech, all_stock)
    segment_3 = _segment_3(model, node, year, tech, all_stock)

    cc_learning = cc_overnight * segment_1 * segment_2 * segment_3

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


def _segment_1(model, node, year, tech, all_stock):
    bc_1 = model.get_param('baseline_capacity_1', node, year, tech=tech)
    bc_2 = model.get_param('baseline_capacity_2', node, year, tech=tech)
    pr_1 = model.get_param('progress_ratio_1', node, year, tech=tech)
    segment_1 = (min(max(all_stock, bc_1), bc_2) / bc_1) ** log2(pr_1)
    return segment_1


def _segment_2(model, node, year, tech, all_stock):
    bc_2 = model.get_param('baseline_capacity_2', node, year, tech=tech)
    bc_3 = model.get_param('baseline_capacity_3', node, year, tech=tech)
    pr_2 = model.get_param('progress_ratio_2', node, year, tech=tech)
    segment_2 = (min(max(all_stock, bc_2), bc_3) / bc_2) ** log2(pr_2)
    return segment_2


def _segment_3(model, node, year, tech, all_stock):
    bc_3 = model.get_param('baseline_capacity_3', node, year, tech=tech)
    pr_3 = model.get_param('progress_ratio_3', node, year, tech=tech)
    segment_3 = (max(all_stock, bc_3) / bc_3) ** log2(pr_3)
    return segment_3
