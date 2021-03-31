import re
from . import lcc_calculation


def is_year(cn: str or int) -> bool:
    """ Determines whether `cn` is a year

    Parameters
    ----------
    cn : int or str
        The value to check to determine if it is a year.

    Returns
    -------
    bool
        True if `cn` is made entirely of digits [0-9] and is 4 characters in length. False otherwise.

    Examples
    --------
    >>> is_year(1900)
    True

    >>> is_year('2010')
    True
    """
    re_year = re.compile(r'^[0-9]{4}$')
    return bool(re_year.match(str(cn)))


def search_nodes(search_term, g):
    """
    Search `graph` to find the nodes which contain `search_term` in the node name's final component. Not case
    sensitive.

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

    return [n for n in g.nodes if search(n)]


def create_value_dict(year_val, source=None, branch=None, unit=None, param_source=None):
    value_dictionary = {'source': source,
                        'branch': branch,
                        'unit': unit,
                        'year_value': year_val,
                        'param_source': param_source
                        }

    return value_dictionary


def dict_has_None_year_value(dictionary):
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
        for v in dictionary.values():
            if isinstance(v, dict):
                has_none_year_value = has_none_year_value or dict_has_None_year_value(v)
    return has_none_year_value


# ******************
# Parameter Fetching
# ******************
calculation_directory = {'GCC_t': lcc_calculation.calc_gcc,
                         'Capital cost_declining': lcc_calculation.calc_declining_cc,
                         'Capital cost': lcc_calculation.calc_capital_cost,
                         'CRF': lcc_calculation.calc_crf,
                         'Upfront cost': lcc_calculation.calc_upfront_cost,
                         'Annual intangible cost_declining': lcc_calculation.calc_declining_aic,
                         'Annual cost': lcc_calculation.calc_annual_cost,
                         'Service cost': lcc_calculation.calc_annual_service_cost,
                         'Life Cycle Cost': lcc_calculation.calc_lcc}

inheritable_params = []


def get_node_param(param, model, node, year, sub_param=None, return_source=False, retrieve_only=False):
    """
    Queries a model to retrieve a parameter value at a given node, given a specified context
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

    Returns
    -------
    any :
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
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
    # If the value has been defined at a structural parent node for that year, use that value.
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True) if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]
        val = get_node_param(param, model, parent, year=year)
        param_source = 'inheritance'

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    elif param in model.node_defaults:
        val = model.get_node_parameter_default(param)
        param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    else:
        prev_year = str(int(year) - model.step)
        val = model.get_param(param, node, prev_year)
        param_source = 'previous_year'

    if return_source:
        return val, param_source
    else:
        return val


def get_tech_param(param, model, node, year, tech, sub_param=None, return_source=False, retrieve_only=False):
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

    Returns
    -------
    any :
        The value of the specified `param` at `node`, given the context provided by `year` and
        `tech`.
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
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True)
                            if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]
        val = model.get_param(param, parent, year, tech, sub_param)
        param_source = 'inheritance'

    # Use a Default Parameter Value
    # ******************************
    # If there is a default value defined, use this value
    elif param in model.technology_defaults:
        val = model.get_tech_parameter_default(param)
        param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year.
    else:
        prev_year = str(int(year) - model.step)
        if int(prev_year) >= model.base_year:
            val = model.get_param(param, node, prev_year, tech, sub_param=sub_param)
            param_source = 'previous_year'

    if return_source:
        return val, param_source
    else:
        return val
