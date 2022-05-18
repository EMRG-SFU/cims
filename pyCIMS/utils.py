"""
This module contains utility functions used throughout the pyCIMS package.
"""
import re
import copy
import warnings
from typing import List
from scipy.interpolate import interp1d

from . import lcc_calculation

def is_year(val: str or int) -> bool:
    """ Determines whether `cn` is a year

    Parameters
    ----------
    val : int or str
        The value to check to determine if it is a year.

    Returns
    -------
    bool
        True if `cn` is made entirely of digits [0-9] and is 4 characters in length. False
        otherwise.

    Examples
    --------
    >>> is_year(1900)
    True

    >>> is_year('2010')
    True
    """
    re_year = re.compile(r'^[0-9]{4}$')
    return bool(re_year.match(str(val)))


def search_nodes(search_term, graph):
    """
    Search `graph` to find the nodes which contain `search_term` in the node name's final component.
    Not case sensitive.

    Parameters
    ----------
    search_term : str
        The search term.

    Returns
    -------
    list [str]
        A list of node names (branch format) whose last component contains `search_term`.
    """
    def search(name):
        components = name.split('.')
        last_comp = components[-1]
        return search_term.lower() in last_comp.lower()

    return [n for n in graph.nodes if search(n)]


def create_value_dict(year_val, source=None, sub_context=None, branch=None, unit=None,
                      param_source=None):
    """
    Creates a standard value dictionary from the inner values.
    """
    value_dictionary = {'sub_context': sub_context,
                        'branch': branch,
                        'source': source,
                        'unit': unit,
                        'year_value': year_val,
                        'param_source': param_source
                        }

    return value_dictionary


def dict_has_none_year_value(dictionary):
    """
    Given a dictionary, check if it has a year_value key in any level where the value for the
    year_value key is  None.
    """
    has_none_year_value = False
    if 'year_value' in dictionary.keys():
        year_value = dictionary['year_value']
        if year_value is None:
            has_none_year_value = True
    else:
        for value in dictionary.values():
            if isinstance(value, dict):
                has_none_year_value = has_none_year_value or dict_has_none_year_value(value)
    return has_none_year_value


def is_param_exogenous(model, param, node, year, tech=None):
    """Checks if a parameter is exogenously defined"""
    _, source = model.get_param(param, node, year=year, tech=tech, return_source=True)
    ms_exogenous = source == 'model'
    return ms_exogenous


def create_cost_curve_func(quantities: List[float], prices: List[float]):
    """
    Build an interpolator that uses quantity to interpolate price.
    To be used for cost curve LCC calculations.

    Parameters
    ----------
    quantities : list of quantities.
    prices : list of prices. The length of prices must be equal to the length of quantities.

    Returns
    -------
    scipy.interpolate.interp1d : A 1d interpolator that consumes a quantity to interpolate price.
    """
    qp_pairs = list(set(zip(quantities, prices)))
    qp_pairs.sort(key=lambda x: x[0])
    quantities, prices = zip(*qp_pairs)

    return interp1d(quantities, prices, bounds_error=False, fill_value=(prices[0], prices[-1]))


# ******************
# Parameter Fetching
# ******************
calculation_directory = {
    'GCC_t': lcc_calculation.calc_gcc,
    'Capital cost_declining': lcc_calculation.calc_declining_cc,
    'Capital cost': lcc_calculation.calc_capital_cost,
    'CRF': lcc_calculation.calc_crf,
    'Financial Upfront cost': lcc_calculation.calc_financial_upfront_cost,
    'Complete Upfront cost': lcc_calculation.calc_complete_upfront_cost,
    'Annual intangible cost_declining': lcc_calculation.calc_declining_aic,
    'Financial Annual cost': lcc_calculation.calc_financial_annual_cost,
    'Complete Annual cost': lcc_calculation.calc_complete_annual_cost,
    'Service cost': lcc_calculation.calc_annual_service_cost,
    'Emissions cost': lcc_calculation.calc_emissions_cost,
    'Life Cycle Cost': lcc_calculation.calc_financial_lcc,
    'Complete Life Cycle Cost': lcc_calculation.calc_complete_lcc,
}

inheritable_params = [
    'Price Multiplier',
]


def get_param(model, param, node, year=None, context=None, sub_context=None, tech=None,
              return_source=False, do_calc=False, check_exist=False, dict_expected=False):
    """
    Queries the model to retrieve a parameter value at a given node, given a specified context
    (year & sub-parameter).

    Parameters
    ----------
    param : str
        The name of the parameter whose value is being retrieved.
    model : pyCIMS.Model
        The model containing the parameter value of interest.
    node : str
        The name of the node (branch format) whose parameter you are interested in retrieving.
    year : str
        The year which you are interested in. `year` must be provided for all parameters stored at
        the technology level, even if the parameter doesn't change year to year.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    return_source : bool, default=False
        Whether to return the method by which this value was originally obtained.
    do_calc : bool
        applies calculation formula to calculate the parameter
    check_exist : bool, default=False
        Whether to check that the parameter exists as is given the context (without
        calculation, inheritance, or checking past years)
    dict_expected : bool, default=False
        Used to disable the warning get_param is returning a dict. Get_param should normally return a 'single value'
        (float, str, etc.). If the user knows it expects a dict, then this flag is used.

    Returns
    -------
    float
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
    str
        If return_source is `True`, will return a string indicating how the parameter's value
        was originally obtained. Can be one of {model, initialization, inheritance, calculation,
        default, or previous_year}.
    """
    val = None
    is_exogenous = None

    # Get Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    data = model.graph.nodes[node]
    if year:
        data = data[year]
        if tech:  # assumption: any tech node always requires a year
            data = data['technologies'][tech]

    # Val can be the final return result (float, string, etc) or a dict, check for other params
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if context:
                val = val[context]
                if sub_context:
                    val = val[sub_context]
            elif None in val:
                val = val[None]

    # Grab the year_value in the dictionary if exists
    if isinstance(val, dict) and ('year_value' in val):
        param_source = val['param_source']
        is_exogenous = param_source in ['model', 'initialization']
        val = val['year_value']

    # Raise warning if user isn't using get_param correctly
    if isinstance(val, dict) and not dict_expected:
        warnings.warn("Get Param is returning a dict, considering using more parameters in get_param.")

    if val is not None:
        if not do_calc:
            if return_source:
                return val, param_source
            return val
        elif is_exogenous:
            if return_source:
                return val, param_source
            else:
                return val

    # If check_exist is True, raise an Exception if val has not yet been returned, which means
    # the value at the current context could not be found as is.
    if check_exist:
        raise Exception

    param_source = None
    # Calculate Parameter Value
    # ******************************
    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    if (param in calculation_directory) & do_calc:
        param_calculator = calculation_directory[param]
        val = param_calculator(model, node, year, tech)
        param_source = 'calculation'

    # Inherit Parameter Value
    # ******************************
    # If the value has been defined at a structural parent node for that year, use that value.
    if (param_source is None) and (param in inheritable_params):
        if tech:
            try:
                val, source = model.get_param(param, node, year=year, context=context,
                                              sub_context=sub_context, return_source=True)
                assert (source in ['inheritance', 'model'])
                assert (val is not None)
                param_source = source
            except AssertionError:
                pass
        else:
            # If the value has been defined at a structural ancestor, it should be here with
            # param_source == 'inheritance'
            try:
                val = model.graph.nodes[node][year][param]
                if context:
                    val = val[context]
                    if sub_context:
                        val = val[sub_context]
                if val['param_source'] == 'inheritance':
                    param_source = 'inheritance'
            except KeyError:
                pass

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    if param_source is None:
        if tech:
            if param in model.technology_defaults:
                val = model.get_tech_parameter_default(param)
                param_source = 'default'
        else:
            comp_type = model.get_param('competition type', node)
            if (comp_type in model.node_defaults) and (param in model.node_defaults[comp_type]):
                val = model.get_node_parameter_default(param)
                param_source = 'default'

    # Use a Default Parameter Value (tech)
    # ******************************
    # If there is a default value defined, use this value

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    if param_source is None:
        prev_year = str(int(year) - model.step)
        if int(prev_year) >= model.base_year:
            val = model.get_param(param, node,
                                  year=prev_year,
                                  context=context,
                                  sub_context=sub_context,
                                  tech=tech)
            param_source = 'previous_year'
        else:
            val = None
            param_source = None

    if return_source:
        return val, param_source
    else:
        return val


def set_node_param(new_value, param, model, node, year, sub_param=None, save=True):
    """
    Queries a model to set a parameter value at a given node, given a specified context
    (year & sub-parameter).

    Parameters
    ----------
    new_value : dict
        The new value to be set at the specified `param` at `node`, given the context provided by
        `year` and `sub_param`.
    param : str
        The name of the parameter whose value is being set.
    model : pyCIMS.Model
        The model containing the parameter value of interest.
    node : str
        The name of the node (branch format) whose parameter you are interested in set.
    year : str
        The year which you are interested in. `year` must be provided for all parameters stored at
        the technology level, even if the parameter doesn't change year to year.
    sub_param : str, optional
        This is a rarely used parameter for specifying a nested key. Most commonly used when
        `get_param()` would otherwise return a dictionary where a nested value contains the
        parameter value of interest. In this case, the key corresponding to that value can be
        provided as a `sub_param`
    """
    # Set Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    if year:
        data = model.graph.nodes[node][year]
    else:
        data = model.graph.nodes[node]

    if param in data:
        val = data[param]
        # If the value is a dictionary, use its nested result
        if isinstance(val, dict):
            if sub_param:
                val[sub_param].update(new_value)
            elif None in val:
                # If the value is a dictionary, check if 'year_value' can be accessed.
                if isinstance(val[None], dict):
                    val[None].update(new_value)
                else:
                    val[None].update(new_value)
            elif len(val.keys()) == 1:
                val[list(val.keys())[0]].update(new_value)
        else:
            data[param] = new_value

    else:
        print(f"No param {param} at node {node} for year {year}. No new value was set for this")


def set_tech_param(new_value, param, model, node, year, tech, sub_param=None):
    """
    Queries a model to set a parameter value at a given node & technology, given a specified
    context (year & sub_param).

    Parameters
    ----------
    new_value : dict
        The new value to be set at the specified `param` at `node`, given the context provided by
        `year`, `tech` and `sub_param`.
    param : str
        The name of the parameter whose value is being set.
    model : pyCIMS.Model
        The model containing the parameter value of interest.
    node : str
        The name of the node (branch format) whose parameter you are interested in set.
    year : str
        The year which you are interested in. `year` must be provided for all parameters stored at
        the technology level, even if the parameter doesn't change year to year.
    tech : str
        The name of the technology you are interested in.
    sub_param : str, optional
        This is a rarely used parameter for specifying a nested key. Most commonly used when
        `get_param()` would otherwise return a dictionary where a nested value contains the
        parameter value of interest. In this case, the key corresponding to that value can be
        provided as a `sub_param`
    """
    # Set Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node, year, & technology, use it
    data = model.graph.nodes[node][year]['technologies'][tech]
    if param in data:
        val = data[param]
        # If the value is a dictionary, use its nested result
        if isinstance(val, dict):
            if sub_param:
                # If the value is a dictionary, check if 'year_value' can be accessed.
                if isinstance(val[sub_param], dict):
                    val[sub_param].update(new_value)
                else:
                    val[sub_param].update(new_value)
            elif None in val:
                # If the value is a dictionary, check if 'year_value' can be accessed.
                if isinstance(val[None], dict):
                    val[None].update(new_value)
                else:
                    val[None].update(new_value)
            else:
                data[param].update(new_value)
        else:
            data[param] = new_value

    else:
        print(f"No param {param} at node {node} for year {year}. No new value was set for this")


def inherit_parameter(graph, node, year, param):
    assert(param in inheritable_params)

    parent = '.'.join(node.split('.')[:-1])

    if parent:
        parent_param_val = {}

        if param in graph.nodes[parent][year]:
            param_val = copy.deepcopy(graph.nodes[parent][year][param])
            parent_param_val.update(param_val)

        node_param_val = copy.deepcopy(parent_param_val)
        if param in graph.nodes[node][year]:
            param_val = graph.nodes[node][year][param]
            node_param_val.update(param_val)

        # Update Param Source
        if 'param_source' in node_param_val:
            node_param_val.update({'param_source': 'inheritance'})
        else:
            for context in node_param_val:
                if 'param_source' in node_param_val[context]:
                    node_param_val[context].update({'param_source': 'inheritance'})
                else:
                    for sub_context in node_param_val[context]:
                        node_param_val[context][sub_context].updatet({'param_source': 'inheritance'})

        if node_param_val:
            graph.nodes[node][year][param] = node_param_val