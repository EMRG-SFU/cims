"""This module contains functions related to cost-curve price calculations"""
from statistics import mean
from scipy.interpolate import interp1d

from .utils import parameters as PARAM
from .utils import model_columns as COL
from . import old_utils


def calc_cost_curve_lcc(model: "CIMS.Model", node: str, year: str,
                        cost_curve_min_max: bool = False):
    """
    Calculate a node's LCC using its cost curve function (stored in the node level data).
    Depending on the node's competition type, annual or cumulative provided quantity values will be
    used in the call to the cost curve interpolation function.

    Note, cost curve LCC is only used supply nodes, so this LCC is the same as the financial LCC we use
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
    quantity = calc_cost_curve_quantity(model, node, year)
    cc_func = model.get_param(PARAM.cost_curve_function, node)
    cc_lcc = float(cc_func(quantity))
    previous_lcc, prev_src = model.get_param(PARAM.lcc_financial, node, year, return_source=True)

    if cost_curve_min_max:
        lcc = calc_lcc_with_min_max(model, node, year, cc_lcc)

    elif prev_src == 'default':
        comp_type = model.get_param(PARAM.competition_type, node).lower()
        if (comp_type == 'supply - cost curve cumulative') and (int(year) > model.base_year):
            prev_year = str(int(year) - model.step)
            prev_year_lcc = model.get_param(PARAM.lcc_financial, node, prev_year)
            lcc = max(prev_year_lcc, cc_lcc)
        else:
            lcc = cc_lcc

    else:
        lcc = previous_lcc

    return lcc


def calc_lcc_with_min_max(model, node, year, cc_lcc):
    """
    Calculate cost curve LCC using min/max limits updated between iterations.

    Parameters
    ----------
    model : CIMS.Model
        The model to fetch paramters from.
    node : str
        The node whose LCC is being calculated.
    year : str
        The year for which to calculate LCC.
    cc_lcc : float
        The LCC estimated from directly applying the Cost Curve function to quantity.

    Returns
    -------
    float :
        The LCC calculated from applying minimums, maximums, and averaging to cc_lcc.
    Additionally, the cost_curve_lcc_min and cost_curve_lcc_max values are updated.
    """
    # Find Previous Iteration's LCC
    prev_lcc, prev_src = model.get_param(PARAM.lcc_financial, node, year, return_source=True)

    # Find Cost Curve min/max
    cc_min, cc_min_src = model.get_param(PARAM.cost_curve_lcc_min, node, year, return_source=True)
    if cc_min_src == 'previous_year':
        cc_min = None
    cc_max, cc_max_src = model.get_param(PARAM.cost_curve_lcc_max, node, year, return_source=True)
    if cc_max_src == 'previous_year':
        cc_max = None

    # Calculate Expected LCC
    expected_lcc = max(x for x in [cc_lcc, cc_min] if x is not None)
    expected_lcc = min(x for x in [expected_lcc, cc_max] if x is not None)

    # Update Cost Curve min/max
    if expected_lcc > prev_lcc:
        cc_min = max(x for x in [prev_lcc, cc_min] if x is not None)
        model.set_param_internal(
            old_utils.create_value_dict(cc_min, param_source='cost curve function'),
            PARAM.cost_curve_lcc_min, node, year)

    elif expected_lcc < prev_lcc:
        cc_max = min(x for x in [prev_lcc, cc_max] if x is not None)
        model.set_param_internal(
            old_utils.create_value_dict(cc_max, param_source='cost curve function'),
            PARAM.cost_curve_lcc_max, node, year)

    # If the last price was calculated with cost curve, provide an average
    if prev_src == 'cost curve function':
        lcc = mean((prev_lcc, expected_lcc))
    else:
        lcc = expected_lcc

    return lcc


def calc_cost_curve_quantity(model: "CIMS.Model", node: str, year: str):
    """
    Calculate the total quantity provided by the node.
    The number of years this is calcualted over depends on whether cost curves are annual or
    cumulative. This serves as a helper function for calc_cost_curve_lcc.

    Parameters
    ----------
    model : CIMS.Model
        The model containing node.
    node : str
        The name of the node for which total quantity will be calculated.
    year : str
        The year in which we are calculating quantity for.

    Returns
    -------
    float :
        Total quantity needed to calcualte cost curve price.
    """
    comp_type = model.get_param(PARAM.competition_type, node).lower()
    if comp_type == 'supply - cost curve annual':
        min_year = year
        max_year = year
    elif comp_type == 'supply - cost curve cumulative':
        min_year = model.base_year
        max_year = year
    else:
        raise ValueError("Unrecognized cost curve calculation competition type")

    total_quantity = 0
    for year_i in range(int(min_year), int(max_year) + 1, model.step):
        if PARAM.provided_quantities in model.graph.nodes[node][str(year_i)]:
            year_provided_quant = model.get_param(PARAM.provided_quantities, node, str(year_i))
            total_quantity += year_provided_quant.get_total_quantity()

    return total_quantity


def build_cost_curve_function(node_df):
    """
    Create an interpolating cost curve function based on the cost curve quantities & prices
    defined in the model.

    Parameters
    ----------
    node_df : pandas.DataFrame
        The dataframe containing the cost curve quantity and cost curve price values to use for
        creating the interpolator.

    Returns
    -------
    scipy.interpolate._interpolate.interp1d
        An interpolator that inputs a quantity & outputs a price.

    """
    years = [c for c in node_df.columns if old_utils.is_year(c)]

    # Get quantities
    cc_quant_line = node_df[node_df[COL.parameter] == PARAM.cost_curve_quantity]
    cc_quants = [cc_quant_line[y].iloc[0] for y in years]

    # Get prices
    cc_price_line = node_df[node_df[COL.parameter] == PARAM.cost_curve_price]
    cc_prices = [cc_price_line[y].iloc[0] for y in years]

    # Create interpolator
    qp_pairs = list(set(zip(cc_quants, cc_prices)))
    qp_pairs.sort(key=lambda x: x[0])
    quantities, prices = zip(*qp_pairs)
    cc_func = interp1d(quantities, prices, bounds_error=False, fill_value=(prices[0], prices[-1]))

    return cc_func
