"""
Module containing the functions required for performing Macro Economics calculations.
"""
from CIMS import old_utils
from ..utils import parameters as PARAM


def calc_total_stock_demanded(model, node, year):
    """
    Calculate the total stock demanded term, which is a sum of the stock demanded from within the
    node's heirarchy & the stock exported to external regions.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the data required for calculation.
    node : str
        The node whose Total Stock Demanded is being calculated.
    year : str
        The year for which Total Stock Demanded is being calculated.

    Returns
    -------
    float :
        The total stock demanded from the node's hierarchy & across all regions.
    """
    stock_demanded = calc_stock_demanded(model, node, year)
    stock_exported_all_regions = calc_stock_exported(model, node, year)
    total_stock_demanded = stock_demanded + sum(stock_exported_all_regions)

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
    model : CIMS.Model
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
    sum_service_stock_requested = model.get_param(PARAM.provided_quantities, node,
                                                  year).get_total_quantity()

    price_t = max(model.get_param(PARAM.price, node, year), 0.01)
    price_2000 = max(model.get_param(PARAM.price, node, str(model.base_year)), 0.01)
    domestic_elasticity = model.get_param(PARAM.domestic_elasticity, node, year)
    macro_multiplier = (price_t / price_2000) ** domestic_elasticity

    stock_demanded = sum_service_stock_requested * macro_multiplier

    model.set_param_internal(old_utils.create_value_dict(stock_demanded, param_source='calculation'),
                             PARAM.stock_demanded, node, year)

    return stock_demanded


def find_regions(model, node, year):
    """
    Determine which Macro-Economic regions have been specified at a node in a particular year.

    This is done by looking at each of the 5 exogenous parameters which are used
    when calculated Stock Exported.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the data required for calculation.
    node : str
        The node whose regions are being determined.
    year : str
        The year for which regions are being determined.

    Returns
    -------
    list :
        A list of unique regions for which at least one of the Stock Exported related parameters
        are defined.
    """
    stock_export_params = [PARAM.global_price, PARAM.export_subsidy, PARAM.export_benchmark,
                           PARAM.ref_stock_exported, PARAM.export_elasticity]
    regions = []
    for param in stock_export_params:
        param_value = model.get_param(param, node, year, dict_expected=True)
        if isinstance(param_value, dict):
            regions += param_value.keys()

    return set(regions)


def calc_stock_exported(model, node, year):
    """
    Calculate the amount of stock being exported by each region in the model.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the data required for calculation.
    node : str
        The node whose Stock Exported is being calculated.
    year : str
        The year for which Stock Exported is being calculated.

    Returns
    -------
    list [float]:
        Return a list of exported stock values, one for each region. Additionally, save the amount
        of stock exported by each region to the model.
    """
    price_t = max(model.get_param(PARAM.price, node, year), 0.01)
    price_2000 = max(model.get_param(PARAM.price, node, str(model.base_year)), 0.01)

    all_stock_exported = []
    for region in find_regions(model, node, year):
        ref_stock_exported = model.get_param(PARAM.ref_stock_exported, node, year, context=region)

        global_price_t = max(model.get_param(PARAM.global_price, node, year, context=region), 0.01)
        global_price_2000 = max(model.get_param(PARAM.global_price, node, str(model.base_year),
                                                context=region), 0.01)
        export_subsidy_t = model.get_param(PARAM.export_subsidy, node, year, context=region)
        export_subsidy_2000 = model.get_param(PARAM.export_subsidy, node, str(model.base_year),
                                              context=region)
        
        export_benchmark_t= model.get_param(PARAM.export_benchmark, node, year, context=region)
        export_benchmark_2000= model.get_param(PARAM.export_benchmark, node, str(model.base_year), context=region)


        price_term = ((price_t - export_subsidy_t * export_benchmark_t) / global_price_t) / \
        max((price_2000 - export_subsidy_2000 * export_benchmark_2000) / global_price_2000, 0.01)

        export_elasticity = model.get_param(PARAM.export_elasticity, node, year, context=region)
        stock_exported_region = ref_stock_exported * price_term ** export_elasticity

        all_stock_exported.append(stock_exported_region)

        if PARAM.stock_exported not in model.graph.nodes[node][year]:
            model.graph.nodes[node][year][PARAM.stock_exported] = {}

        model.graph.nodes[node][year][PARAM.stock_exported][region] = \
            old_utils.create_value_dict(stock_exported_region,
                                    context=region,
                                    param_source='calculation')

    return all_stock_exported
