"""
This module contains utility functions used throughout the pyCIMS package.
"""
import re
import copy
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
    _, source = model.get_param(param, node, year, tech, return_source=True)
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
    'Retrofit_Min',
    'Retrofit_Max',
]


def get_node_param(param, model, node, year, sub_param=None,
                   return_source=False, retrieve_only=False, check_exist=False, return_keys=False):
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
    sub_param : str, optional
        This is a rarely used parameter for specifying a nested key. Most commonly used when
        `get_param()` would otherwise return a dictionary where a nested value contains the
        parameter value of interest. In this case, the key corresponding to that value can be
        provided as a `sub_param`
    return_source : bool, default=False
        Whether or not to return the method by which this value was originally obtained.
    retrieve_only : bool, default=False
        If True the function will only retrieve the value using the current value in the model,
        inheritance, default, or the previous year's value. It will _not_ calculate the parameter
        value. If False, calculation is allowed.
    check_exist : bool, default=False
        Whether or not to check that the parameter exists as is given the context (without
        calculation, inheritance, or checking past years)

    Returns
    -------
    any :
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
    str :
        If return_source is `True`, will return a string indicating how the parameter's value
        was originally obtained. Can be one of {model, initialization, inheritance, calculation,
        default, or previous_year}.
    """
    is_exogenous = True

    # Get Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    if year:
        data = model.graph.nodes[node][year]
    else:
        data = model.graph.nodes[node]

    if param in data:
        val = data[param]
        # If the value is a dictionary, check if a base value (float, string, etc) has been nested.
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif None in val:
                val = val[None]
            elif len(val.keys()) == 1:
                if not return_keys:
                    val = list(val.values())[0]

            if 'year_value' in val:
                param_source = val['param_source']
                is_exogenous = param_source in ['model', 'initialization']
                val = val['year_value']

        # Choose which values to return
        if val is not None:
            if retrieve_only:
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
    if (param in calculation_directory) & (not retrieve_only):
        param_calculator = calculation_directory[param]
        val = param_calculator(model, node, year)
        param_source = 'calculation'

    # Inherit Parameter Value
    # ******************************
    # If the value has been defined at a structural parent node, it will be present at the node with
    # param_source == "inheritance"
    if (param_source is None) and (param in inheritable_params):
        try:
            val = model.graph.nodes[node][year][param]
            if sub_param:
                val = val[sub_param]
            assert(val['param_source'] == 'inheritance')
            param_source = val['param_source']
        except KeyError:
            pass

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    comp_type = model.get_param('competition type', node)
    if (param_source is None) and (comp_type in model.node_defaults) and (param in model.node_defaults[comp_type]):
        val = model.get_node_parameter_default(param)
        param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    if param_source is None:
        prev_year = str(int(year) - model.step)
        if prev_year >= str(model.base_year):
            val = model.get_param(param, node, prev_year)
            param_source = 'previous_year'
        else:
            val = None
            param_source = None

    if return_source:
        return val, param_source
    else:
        return val


def get_tech_param(param, model, node, year, tech, sub_param=None,
                   return_source=False, retrieve_only=False, check_exist=False):
    """
    Queries a model to retrieve a parameter value at a given node & technology, given a specified
    context (year & sub-parameter).

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
    tech : str
        The name of the technology you are interested in.
    sub_param : str, optional
        This is a rarely used parameter for specifying a nested key. Most commonly used when
        `get_param()` would otherwise return a dictionary where a nested value contains the
        parameter value of interest. In this case, the key corresponding to that value can be
        provided as a `sub_param`
    return_source : bool, default=False
        Whether or not to return the method by which this value was originally obtained.
    retrieve_only : bool, default=False
        If True the function will only retrieve the value using the current value in the model,
        inheritance, default, or the previous year's value. It will _not_ calculate the parameter
        value. If False, calculation is allowed.
    check_exist : bool, default=False
        Whether or not to check that the parameter exists as is given the context (without
        calculation, inheritance, or checking past years)

    Returns
    -------
    any :
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
    str :
        If return_source is `True`, will return a string indicating how the parameter's value
        was originally obtained. Can be one of {model, initialization, inheritance, calculation,
        default, or previous_year}.
    """
    val = None
    is_exogenous = None
    # Get Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node, year, & technology, use it
    data = model.graph.nodes[node][year]['technologies'][tech]
    if param in data:
        val = data[param]
        # If the value is a dictionary, check if a base value (float, str, etc) has been nested
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif None in val:
                val = val[None]
            if isinstance(val, dict) and ('year_value' in val):
                param_source = val['param_source']
                is_exogenous = param_source in ['model', 'initialization']
                val = val['year_value']

        # As long as the value has been specified, return it. & it is exogenously specified
        if val is not None:
            if retrieve_only:
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
    if param in calculation_directory:
        param_calculator = calculation_directory[param]
        val = param_calculator(model, node, year, tech)
        param_source = 'calculation'

    # Inherit Parameter Value
    # ******************************
    # If the value has been defined at a structural parent node for that year, use this value.
    if (param_source is None) and (param in inheritable_params):
        # Look to the node
        try:
            val, source = model.get_param(param, node, year, sub_param=sub_param, return_source=True)
            assert(source in ['inheritance', 'model'])
            assert(val is not None)
            param_source = source#
        except AssertionError:
            pass

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    if (param_source is None) and (param in model.technology_defaults):
        val = model.get_tech_parameter_default(param)
        param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year.
    if param_source is None:
        prev_year = str(int(year) - model.step)
        if prev_year >= str(model.base_year):
            val = model.get_param(param, node, prev_year)
            param_source = 'previous_year'
        else:
            val = None
            param_source = None

    if return_source:
        return val, param_source
    else:
        return val


def set_node_param(new_value, param, model, node, year, sub_param=None):
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