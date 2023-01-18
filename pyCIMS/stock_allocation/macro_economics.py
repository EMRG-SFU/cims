"""
Module containing the functions required for performing Macro Economics calculations.
"""


def calc_total_stock_demanded(model, node, year):
    stock_demanded = calc_stock_demanded(model, node, year)
    stock_exported_region_1 = 0
    stock_exported_region_2 = 0

    total_stock_demanded = stock_demanded + stock_exported_region_1 + stock_exported_region_2

    return total_stock_demanded


def calc_stock_demanded(model, node, year):
    """
    Calculate the Stock Demanded for a node in a particular year. The result is used in the
    calc_total_stock_demanded() function.

    Stock Demanded is calculated by applying a macro-multiplier to the stock demanded of node by
    other nodes in the model. The macro-multiplier is calculated using the node's relative price
    and elasticity terms.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the data required for calculation.
    node : str
        The node whose Stock Demanded is being calculated.
    year : str
        The year for which Stock Demanded is being calculated.

    Returns
    -------
    float :
        Returns the Stock Demanded for a node in a particular year.

        Stock Demanded is calculated by applying a macro-multiplier to the stock demanded of node by
        other nodes in the model. The macro-multiplier is calculated using the node's relative price
        and elasticity terms.
    """
    sum_service_stock_requested = model.get_param('provided_quantities', node,
                                                  year).get_total_quantity()

    price_t = max([model.get_param('price', node, year), 0.01])
    price_2000 = max([model.get_param('price', node, str(model.base_year)), 0.01])
    domestic_elasticity = model.get_param('domestic elasticity', node, year)
    macro_multiplier = (price_t / price_2000) ** domestic_elasticity

    stock_demanded = sum_service_stock_requested * macro_multiplier

    return stock_demanded
