"""
This module contains the functions for conducting vintage-based weighting
"""
import math

from .utils.parameter import list as PARAM

def _get_vintage_weights(model, node, year, tech):
    """
    Find the distribution (in percentages) of a node/technology's total stock over each possible
    vintage year. Returns a dictionary where keys are vintage years and values are percentages.
    """
    # Total Stock
    stock_total, src = model.get_param(PARAM.stock_total, node, year, tech=tech, return_source=True)
    if (stock_total is None) or (math.isclose(stock_total, 0, abs_tol=1e-3)):
        vintage_weights = {year: 1}
    elif src == 'previous_year':
        stock_by_vintage = {}
        if year == str(model.base_year+model.step):
            stock_by_vintage[year] = model.get_param(PARAM.stock_base, node, year, tech=tech)
        else:
            stock_by_vintage.update(
                model.get_param(PARAM.stock_new_remaining, node, year, tech=tech,
                                dict_expected=True) or {})
            stock_base = model.get_param(PARAM.stock_base_remaining, node, year, tech=tech) or 0
            stock_by_vintage[str(model.base_year)] = stock_base
            stock_by_vintage[year] = model.get_param(PARAM.stock_new, node, year, tech=tech) + \
                                     model.get_param(PARAM.stock_retrofit_added, node, year,
                                                     tech=tech)
        vintage_weights = {k: v / stock_total for k, v in stock_by_vintage.items()}

    else:
        stock_by_vintage = {}
        if year == str(model.base_year):
            stock_by_vintage[year] = model.get_param(PARAM.stock_base, node, year, tech=tech)
        else:
            stock_by_vintage.update(
                model.get_param(PARAM.stock_new_remaining, node, year, tech=tech,
                                dict_expected=True) or {})
            stock_base = model.get_param(PARAM.stock_base_remaining, node, year, tech=tech) or 0
            stock_by_vintage[str(model.base_year)] = stock_base
            stock_by_vintage[year] = model.get_param(PARAM.stock_new, node, year, tech=tech) + \
                                     model.get_param(PARAM.stock_retrofit_added, node, year,
                                                     tech=tech)

        vintage_weights = {k: round(v / stock_total, 5) for k, v in stock_by_vintage.items()}

    return vintage_weights


def calculate_vintage_weighted_parameter(parameter: str, model: "CIMS.Model", node: str,
                                         year: str, tech: str, context: str = None,
                                         target: str = None, default_value=0) -> float:
    """
    Uses vintage-based weighting to calculate the value of a parameter. This function is used for
    peforming vintage-based weighting of financial LCC and quantities requested of children nodes.

    This ensures between year changes of financial LCC components (e.g. upfront cost) are accounted
    for when we calculate the financial LCC value associated with all stock, not just the newest
    stock.

    Similarly, this ensures between year changes of service request ratios (e.g. a technology
    becoming more or less efficient over time) are accounted for when we calculate the total demand
    for these services.

    Parameters
    ----------
    parameter : The name of a numerical parameter whose vintage-weighted value will be calculated
        (e.g. "financial lifecycle cost").
    model : The CIMS.Model storing the data required to calculate the vintage-weighted value of
        the parameter.
    node : The name of the node which contains the technology of interest
    year : The year whose vintage-weighted parameter will be calculated. This will use the new stock
        adopted in year, and all stock remaining from previous years to calculated the
        vintage-weighted value.
    tech : The name of the technology whose vintage-weighted parameter value will be calculated
    context : Optional. The additional context needed to access the parameter value of interest.
    target : Optional. A target node name used to differentiate values for the same parameter
        across multiple service request lines.

    Returns
    -------
    float : The vintage-weighted value of the specified parameter. The value is weighted based on
        the amount of stock from each vintage.
    """
    vintage_weights = _get_vintage_weights(model, node, year, tech)

    assert math.isclose(sum(vintage_weights.values()), 1, rel_tol=0.01)

    weighted_parameter = default_value
    for vintage_year, weight in vintage_weights.items():
        parameter_value = model.get_param(parameter, node, vintage_year, tech=tech,
                                          context=context, target=target)
        weighted_parameter += parameter_value * weight

    return weighted_parameter

