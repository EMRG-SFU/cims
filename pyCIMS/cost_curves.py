"This module contains functions related to cost-curve price calculations"
from statistics import mean
from . import utils


def calc_cost_curve_lcc(model: "pyCIMS.Model", node: str, year: str):
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
    expected_lcc = float(cc_func(quantity))
    service_name = node.split('.')[-1]
    previous_lcc, prev_src = model.get_param('financial life cycle cost', node, year,
                                             context=service_name, return_source=True)
    if prev_src == 'cost curve function':
        lcc = mean((previous_lcc, expected_lcc))
    else:
        lcc = expected_lcc
    print(f"{round(previous_lcc, 2):10}\t{round(quantity,2):10}\t{round(expected_lcc,2):10}\t{round(lcc,2):10}")
    return lcc


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
