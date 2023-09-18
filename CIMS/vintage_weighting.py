"""
This module contains the functions for conducting vintage-based weighting
"""


def _get_vintage_weights(model, node, year, tech):
    """
    Find the distribution (in percentages) of a node/technology's total stock over each possible
    vintage year. Returns a dictionary where keys are vintage years and values are percentages.
    """
    # Total Stock
    total_stock, src = model.get_param('total_stock', node, year, tech=tech, return_source=True)
    if (total_stock is None) or (total_stock == 0):
        vintage_weights = {year: 1}
    elif src == 'previous_year':
        stock_by_vintage = {}
        if year == str(model.base_year+model.step):
            stock_by_vintage[year] = model.get_param('base_stock', node, year, tech=tech)
        else:
            stock_by_vintage.update(
                model.get_param('new_stock_remaining', node, year, tech=tech,
                                dict_expected=True) or {})
            base_stock = model.get_param('base_stock_remaining', node, year, tech=tech) or 0
            stock_by_vintage[str(model.base_year)] = base_stock
            stock_by_vintage[year] = model.get_param('new_stock', node, year, tech=tech) + \
                                     model.get_param('added_retrofit_stock', node, year,
                                                     tech=tech)
        vintage_weights = {k: v / total_stock for k, v in stock_by_vintage.items()}

    else:
        stock_by_vintage = {}
        if year == str(model.base_year):
            stock_by_vintage[year] = model.get_param('base_stock', node, year, tech=tech)
        else:
            stock_by_vintage.update(
                model.get_param('new_stock_remaining', node, year, tech=tech,
                                dict_expected=True) or {})
            base_stock = model.get_param('base_stock_remaining', node, year, tech=tech) or 0
            stock_by_vintage[str(model.base_year)] = base_stock
            stock_by_vintage[year] = model.get_param('new_stock', node, year, tech=tech) + \
                                     model.get_param('added_retrofit_stock', node, year,
                                                     tech=tech)

        vintage_weights = {k: v / total_stock for k, v in stock_by_vintage.items()}

    return vintage_weights


def calculate_vintage_weighted_parameter(parameter: str, model: "CIMS.Model", node: str,
                                         year: str, tech: str, context: str = None, default_value=0) -> float:
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
        (e.g. "financial life cycle cost").
    model : The CIMS.Model storing the data required to calculate the vintage-weighted value of
        the parameter.
    node : The name of the node which contains the technology of interest
    year : The year whose vintage-weighted parameter will be calculated. This will use the new stock
        adopted in year, and all stock remaining from previous years to calculated the
        vintage-weighted value.
    tech : The name of the technology whose vintage-weighted parameter value will be calculated
    context : Optional. The additional context needed to access the parameter value of interest.

    Returns
    -------
    float : The vintage-weighted value of the specified parameter. The value is weighted based on
        the amount of stock from each vintage.
    """
    vintage_weights = _get_vintage_weights(model, node, year, tech)

    assert (round(sum(vintage_weights.values()), 5) == 1)

    weighted_parameter = default_value
    for vintage_year, weight in vintage_weights.items():
        parameter_value = model.get_param(parameter, node, vintage_year, tech=tech, context=context)
        weighted_parameter += parameter_value * weight

    return weighted_parameter

