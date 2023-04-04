"This module contains functions related to cost-curve price calculations"
from statistics import mean
from . import utils


def calc_cost_curve_lcc(model: "pyCIMS.Model", node: str, year: str,
                        cost_curve_min_max: bool = False):
    """
    Calculate a node's LCC using its cost curve function (stored in the node level data).
    Depending on the node's competition type, annual or cumulative provided quantity values will be
    used in the call to the cost curve interpolation function.

    Note, cost curve LCC is only used for fuels, so this LCC is the same as the financial LCC we use
    for other nodes.

    Parameters
    ----------
    model : The model containing node.
    node : The name of the node for which LCC  will be calculated.
    year : The year to calculate LCC for.
    cost_curve_min_max : Whether to re-calculate the cost curve price

    Returns
    -------
    float : LCC (financial) calculated from the node's cost curve function.
    """
    comp_type = model.get_param('competition type', node).lower()
    if comp_type == 'fuel - cost curve annual':
        min_year = year
        max_year = year
    elif comp_type == 'fuel - cost curve cumulative':
        min_year = model.base_year
        max_year = year
    else:
        raise ValueError("Unrecognized cost curve calculation competition type")

    quantity = calc_cost_curve_quantity(model, node, min_year, max_year)
    cc_func = model.get_param('cost_curve_function', node)

    service_name = node.split('.')[-1]
    previous_lcc, prev_src = model.get_param('financial life cycle cost', node, year,
                                             context=service_name, return_source=True)

    # TODO: Make sure this works with cumulative cost curves as well --
    #       minimum will be value from previous year.
    if cost_curve_min_max:
        cc_min = find_cost_curve_min(model, node, year)
        cc_max = find_cost_curve_max(model, node, year)

        expected_lcc = max(x for x in [float(cc_func(quantity)), cc_min] if x is not None)
        expected_lcc = min(x for x in [expected_lcc, cc_max] if x is not None)

        if expected_lcc > previous_lcc:
            update_cost_curve_min(model, node, year, previous_lcc, cc_min)
        elif expected_lcc < previous_lcc:
            update_cost_curve_max(model, node, year, previous_lcc, cc_max)

        if prev_src == 'cost curve function':
            lcc = mean((previous_lcc, expected_lcc))
        else:
            lcc = expected_lcc

    elif prev_src == 'default':
        if (comp_type == 'fuel - cost curve cumulative') and (int(year) > model.base_year):
            prev_year = str(int(year) - model.step)
            prev_year_lcc = model.get_param('financial life cycle cost', node, prev_year,
                                            context=service_name)
            lcc = max(prev_year_lcc, float(cc_func(quantity)))
        else:
            lcc = float(cc_func(quantity))
    else:
        lcc = previous_lcc

    return lcc


def find_cost_curve_min(model, node, year):
    cc_min, cc_min_src = model.get_param('cost_curve_lcc_min', node, year, return_source=True)
    if cc_min_src == 'previous_year':
        cc_min = None
    return cc_min


def find_cost_curve_max(model, node, year):
    cc_max, cc_max_src = model.get_param('cost_curve_lcc_max', node, year, return_source=True)
    if cc_max_src == 'previous_year':
        cc_max = None
    return cc_max


def update_cost_curve_min(model, node, year, previous_lcc, cost_curve_min):
    cc_min = max(x for x in [previous_lcc, cost_curve_min] if x is not None)
    model.set_param_internal(utils.create_value_dict(cc_min, param_source='cost curve function'),
                             'cost_curve_lcc_min', node, year)


def update_cost_curve_max(model, node, year, previous_lcc, cost_curve_max):
    cc_max = min(x for x in [previous_lcc, cost_curve_max] if x is not None)
    model.set_param_internal(utils.create_value_dict(cc_max, param_source='cost curve function'),
                             'cost_curve_lcc_max', node, year)


def calc_cost_curve_quantity(model: "pyCIMS.Model", node: str, min_year: str, max_year: str):
    """
    Calculate the total quantity provided by node from min_year to max_year (inclusive).
    This serves as a helper function for calc_cost_curve_lcc.

    Parameters
    ----------
    model : The model containing node.
    node : The name of the node for which total quantity will be calculated.
    min_year : The first year to include in the sum of total quantities.
    max_year : The last year to include in the sum of total quantities.

    Returns
    -------
    float : Total quantity provided by node from min_year to max_year (inclusive).
    """
    total_quantity = 0
    for year in range(int(min_year), int(max_year) + 1, model.step):
        if 'provided_quantities' in model.graph.nodes[node][str(year)]:
            year_provided_quant = model.get_param('provided_quantities', node, str(year))
            total_quantity += year_provided_quant.get_total_quantity()
    return total_quantity


def build_cost_curve_function(node_df):
    years = [c for c in node_df.columns if utils.is_year(c)]

    # Get quantities
    cc_quant_line = node_df[node_df['Parameter'] == 'cost curve quantity']
    cc_quants = [cc_quant_line[y].iloc[0] for y in years]

    # Get prices
    cc_price_line = node_df[node_df['Parameter'] == 'cost curve price']
    cc_prices = [cc_price_line[y].iloc[0] for y in years]

    # Create interpolator
    cc_func = utils.create_cost_curve_func(cc_quants, cc_prices)

    return cc_func
