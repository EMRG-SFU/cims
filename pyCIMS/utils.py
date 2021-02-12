import re


def range_available(g, node, tech, add_upper=True):
    """
    NOTE: year not mentioned atm because dataset specifies that it remains constant

    Returns +1 at upper boundary if using range() function
    """
    avail = g.nodes[node]['2000']['technologies'][tech]['Available']['year_value']
    unavail = g.nodes[node]['2000']['technologies'][tech]['Unavailable']['year_value']
    if add_upper:
        return int(avail), int(unavail) + 1
    else:
        return int(avail), int(unavail)


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


def create_value_dict(year_val, source=None, branch=None, unit=None):
    value_dictionary = {'source': source,
                        'branch': branch,
                        'unit': unit,
                        'year_value': year_val
                        }

    return value_dictionary


def get_node_param(param, model, node, year, sub_param):
    calculation_directory = {}
    inheritable_params = []

    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    if year:
        data = model.graph.nodes[node][year]
    else:
        data = model.graph.nodes[node]

    # TODO: Add functionality for getting values from parameter dicts that use None as a key
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif 'year_value' in val:
                val = val['year_value']

    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    elif param in calculation_directory:
        pass

    # If the value has been defined at a structural parent node for that year, use this value.
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True) if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]  # We can do this because there is only ever one structural parent
        val = get_param(param, model, parent, year=year, sub_param=sub_param)

    # If there is a default value defined, use this value
    elif param in model.node_defaults:
        val = model.get_node_parameter_default(param)

    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    else:
        prev_year = str(int(year) - model.step)
        if prev_year == str(model.base_year):
            raise Exception()
        val = get_param(param, model, node, prev_year)

    return val


def get_tech_param(param, model, node, year, tech, sub_param):
    calculation_directory = {}
    inheritable_params = []

    # If the parameter's value is in the model description for that node & year, use it.
    data = model.graph.nodes[node][year]['technologies'][tech]
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif 'year_value' in val:
                val = val['year_value']

        if val is not None:
            return val

    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    if param in calculation_directory:
        pass

    # If the value has been defined at a structural parent node for that year, use this value.
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True) if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]  # We can do this because there is only ever one structural parent
        val = get_param(param, model, parent, year, tech, sub_param)

    # If there is a default value defined, use this value
    elif param in model.technology_defaults:
        val = model.get_tech_parameter_default(param)

    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    else:
        prev_year = str(int(year) - model.step)
        if prev_year == str(model.base_year):
            raise Exception()
        val = get_param(param, model, node, prev_year, tech, sub_param)

    return val


def get_param(param, model, node, year=None, tech=None, sub_param=None):
    if tech:
        param_val = get_tech_param(param, model, node, year, tech, sub_param)

    else:
        param_val = get_node_param(param, model, node, year, sub_param)

    return param_val





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


def new_get_node_param(param, model, node, year, sub_param=None):
    calculation_directory = {}
    inheritable_params = []

    if year:
        data = model.graph.nodes[node][year]
    else:
        data = model.graph.nodes[node]

    # Get Parameter from Description
    # ******************************
    # If the parameter's value is in the model description for that node & year (if the year has
    # been defined), use it.
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif None in val:
                val = val[None]
            elif len(val.keys()) == 1:
                val = list(val.values())[0]
            if 'year_value' in val:
                val = val['year_value']
        if val is not None:
            return val

    # if param in data:
    #     val = data[param]
    #     if sub_param:
    #         val = val[sub_param]

        # As long as (A) val isn't None and (B) val isn't a dictionary with a None year_value
        # if (val is not None) and not (isinstance(val, dict) and dict_has_None_year_value(val)):
        #     return val

    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    if param in calculation_directory:
        pass

    # If the value has been defined at a structural parent node for that year, use this value.
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True) if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]  # We can do this because there is only ever one structural parent
        val = get_param(param, model, parent, year=year)

    # If there is a default value defined, use this value
    elif param in model.node_defaults:
        val = model.get_node_parameter_default(param)

    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    else:
        prev_year = str(int(year) - model.step)
        if prev_year == str(model.base_year):
            raise Exception()
        val = model.get_param(param, node, prev_year)

    return val


def new_get_tech_param(param, model, node, year, tech, sub_param=None):
    calculation_directory = {}
    inheritable_params = []

    # If the parameter's value is in the model description for that node & year, use it.
    data = model.graph.nodes[node][year]['technologies'][tech]
    if param in data:
        val = data[param]
        if isinstance(val, dict):
            if sub_param:
                val = val[sub_param]
            elif None in val:
                val = val[None]
            # elif len(val.keys()) == 1:
            #     val = list(val.values())[0]
            if 'year_value' in val:
                val = val['year_value']
        if val is not None:
            return val
    # if param in data:
    #     val = data[param]
    #     if sub_param:
    #         val = val[sub_param]
    #
    #     # As long as (A) val isn't None and (B) val isn't a dictionary with a None value
    #     if (val is not None) and not (isinstance(val, dict) and dict_has_None_year_value(val)):
    #         return val

    # If there is a calculation for the parameter & the arguments for that calculation are present
    # in the model description for that node & year, calculate the parameter value using this
    # calculation.
    if param in calculation_directory:
        pass

    # If the value has been defined at a structural parent node for that year, use this value.
    elif param in inheritable_params:
        structured_edges = [(s, t) for s, t, d in model.graph.edges(data=True) if 'structure' in d['type']]
        g_structure_edges = model.graph.edge_subgraph(structured_edges)
        parent = g_structure_edges.predecessors(node)[0]  # We can do this because there is only ever one structural parent
        val = model.get_param(param, parent, year, tech, sub_param)

    # If there is a default value defined, use this value
    elif param in model.technology_defaults:
        val = model.get_tech_parameter_default(param)

    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    else:
        prev_year = str(int(year) - model.step)
        val = get_param(param, model, node, prev_year, tech)
        if val is None:
            raise Exception()

    return val
