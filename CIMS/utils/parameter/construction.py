from . import list as PARAM


import copy


def recursive_key_value_filter(value_dict, key, value):
    """
    Recursively filters a nested dictionary by removing entries that contain a
    specified key-value pair.

    This function traverses a nested dictionary and removes any entries that
    contain the specified key with the specified value. If this removal results
    in an empty dictionary for any context or sub-context, that context or
    sub-context is also removed.


    Parameters
    ----------
    value_dict : dict
        The dictionary to be filtered, created by create_value_dict().
    key : str
        The key to check within the dictionary.
    value : any
        The value to check for the specified key. Can be any type, including
        None.

    Returns
    -------
    dict
    The filtered dictionary with value_dicts containing the specified key-value
    pairs removed.

    Examples
    --------
    >>> value_dict_1 = create_value_dict(10, param_source='initialization')
    >>> result = recursive_key_value_filter(value_dict_1, 'param_source', 'initialization')
    >>> print(result)
    {}

    >>> value_dict_2 = {
    ...     "C1": create_value_dict(10, param_source='initialization'), 
    ...     "C2": create_value_dict(None, param_source='default'), 
    ...     "C3": create_value_dict(15, param_source="calculation")
    ... }
    >>> result = recursive_key_value_filter(value_dict_2, 'year_value', None)
    >>> print(result)
    {'C1': {'year_value': 10, 'param_source': 'initialization'}, 'C3': {'year_value': 15, 'param_source': 'calculation'}}
    """
    if PARAM.year_value in value_dict:
        if (value_dict.get(key) == value) or \
           ((value_dict.get(key) is None) and (value is None)):
            return {}
    else:
        value_dict = {context: recursive_key_value_filter(context_dict, key, value) for context, context_dict in value_dict.items()}
        value_dict = {k: v for k, v in value_dict.items() if v}
    return value_dict


def inherit_parameter(model, graph, node, year, param, no_inheritance=False):
    assert param in model.inheritable_params

    if not no_inheritance:
        parent = '.'.join(node.split('.')[:-1])

        if parent:
            parent_param_val = {}
            if param in graph.nodes[parent][year]:
                param_val = copy.deepcopy(graph.nodes[parent][year][param])
                parent_param_val.update(param_val)

            # Update Param Source
            if PARAM.param_source in parent_param_val:
                parent_param_val.update({PARAM.param_source: 'inheritance'})
            else:
                for context in parent_param_val:
                    if PARAM.param_source in parent_param_val[context]:
                        parent_param_val[context].update({PARAM.param_source: 'inheritance'})
                    else:
                        for sub_context in parent_param_val[context]:
                            parent_param_val[context][sub_context].update(
                                {PARAM.param_source: 'inheritance'})

            param_value = copy.deepcopy(parent_param_val)
            if param in graph.nodes[node][year]:
                uninheritable_param_vals = graph.nodes[node][year][param]
                # Remove any previously inherited values or any None values, which
                # allows these parameters to become available for inheritance
                uninheritable_param_vals = recursive_key_value_filter(
                    uninheritable_param_vals, key=PARAM.param_source,
                    value="inheritance")
                uninheritable_param_vals = recursive_key_value_filter(
                    uninheritable_param_vals, key=PARAM.year_value, value=None)

                # Update inherited-values with any node-specific values
                param_value.update(uninheritable_param_vals)

            if param_value:
                graph.nodes[node][year][param] = param_value


def create_value_dict(year_val, source=None, context=None, sub_context=None, target=None, unit=None,
                      param_source=None):
    """
    Creates a standard value dictionary from the inner values.
    """
    value_dictionary = {PARAM.context: context,
                        PARAM.sub_context: sub_context,
                        PARAM.target: target,
                        PARAM.source: source,
                        PARAM.unit: unit,
                        PARAM.year_value: year_val,
                        PARAM.param_source: param_source
                        }

    return value_dictionary


def create_param(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
                 target=None, param_source=None, row_index=None):
    """
    Creates parameter in graph, for given context (node, year, tech, context, sub_context),
    and sets the value to val. Returns True if param was created successfully and False otherwise.

    Parameters
    ----------
    val : any
        The new value to be set at the specified `param` at `node`, given the context provided by
        `year`, `tech`, `context`, and `sub_context`.
    param : str
        The name of the parameter whose value is being set.
    node : str
        The name of the node (branch format) whose parameter you are interested in matching.
    year : str, optional
        The year which you are setting a value for. `year` is not required for
        parameters specified at the node level and which by definition cannot
        change year to year (e.g. competition type).
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
        specified node, param, year, context, and sub_context.
    context : str, optional
        Used when there is context available in the node. Analogous to the `context` column in the model
        description. If context is `.*`, all possible context keys will be searched at the specified node, param,
        year, sub_context, and tech.
    sub_context : str, optional
        Must be used only if context is given. Analogous to the `sub_context` column in the model description.
        If sub_context is `.*`, all possible sub_context keys will be searched at the specified node, param,
        year, context, and tech.
    row_index : int, optional
        The index of the current row of the CSV. This is used to print the row number in error messages.

    Returns
    -------
    Boolean
    """
    # Print error message and return False if node not found
    if node not in model.graph.nodes:
        print("Row " + str(row_index + 1) + ': Unable to access node ' + str(
            node) + '. Corresponding value was not set to ' + str(val) + ".")
        return False

    if year:
        if year not in model.graph.nodes[node]:
            model.graph.nodes[node][year] = {}
        data = model.graph.nodes[node][year]
    else:
        data = model.graph.nodes[node]

    param_source = param_source if param_source is not None else 'model'
    val_dict = create_value_dict(val, param_source=param_source, target=target)

    # *********
    # If there is a tech specified, check if it exists and create context (tech, param, context,
    # sub_context) accordingly
    # *********
    if tech:
        # add technology if it does not exist
        if tech not in data[PARAM.technologies]:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                context_dict = {context: sub_context_dict}
                param_dict = {param: context_dict}
            elif context:
                context_dict = {context: val_dict}
                param_dict = {param: context_dict}
            else:
                param_dict = {param: val_dict}
            data[PARAM.technologies][tech] = param_dict
        # add param if it does not exist
        elif param not in data[PARAM.technologies][tech]:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                context_dict = {context: sub_context_dict}
                data[PARAM.technologies][tech][param] = context_dict
            elif context:
                context_dict = {context: val_dict}
                data[PARAM.technologies][tech][param] = context_dict
            else:
                data[PARAM.technologies][tech][param] = val_dict
        # add context if it does not exist
        elif context not in data[PARAM.technologies][tech][param]:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                data[PARAM.technologies][tech][param][context] = sub_context_dict
            else:
                data[PARAM.technologies][tech][param][context] = val_dict
        # add sub_context if it does not exist
        elif sub_context not in data[PARAM.technologies][tech][param][context]:
            data[PARAM.technologies][tech][param][context][sub_context] = val_dict

    # *********
    # Check if param exists and create context (param, context, sub_context) accordingly
    # *********
    elif param not in data:
        if sub_context:
            sub_context_dict = {sub_context: val_dict}
            context_dict = {context: sub_context_dict}
            data[param] = context_dict
        elif context:
            context_dict = {context: val_dict}
            data[param] = context_dict
        else:
            data[param] = val_dict

    # *********
    # Check if context exists and create context (param, context, sub_context) accordingly
    # *********
    elif context not in data[param]:
        if sub_context:
            sub_context_dict = {sub_context: val_dict}
            data[param][context] = sub_context_dict
        else:
            data[param][context] = val_dict

    # *********
    # Check if sub_context exists and create context (param, context, sub_context) accordingly
    # *********
    elif sub_context not in data[param][context]:
        data[param][context][sub_context] = val_dict

    return True