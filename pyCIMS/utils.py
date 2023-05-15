"""
This module contains utility functions used throughout the pyCIMS package.
"""
import re
import copy
import warnings
import pandas as pd
import operator

from . import lcc_calculation
from . import declining_capital_cost


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


def create_value_dict(year_val, source=None, context=None, sub_context=None, branch=None, unit=None,
                      param_source=None):
    """
    Creates a standard value dictionary from the inner values.
    """
    value_dictionary = {'context': context,
                        'sub_context': sub_context,
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


def get_services_requested(model, node, year, tech=None):
    if tech:
        if 'service requested' not in model.graph.nodes[node][year]['technologies'][tech]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year]['technologies'][tech][
                'service requested']
    else:
        if 'service requested' not in model.graph.nodes[node][year]:
            services_requested = {}
        else:
            services_requested = model.graph.nodes[node][year]['service requested']

    return services_requested


def prev_stock_existed(model, node, year):
    for year in [y for y in model.years if y < year]:
        pq, src = model.get_param('provided_quantities', node, year, return_source=True)
        if pq.get_total_quantity() > 0:
            return True
    return False


# ******************
# Parameter Fetching
# ******************
calculation_directory = {
    'capital cost_declining': declining_capital_cost.calc_declining_capital_cost,
    'capital cost': lcc_calculation.calc_capital_cost,
    'crf': lcc_calculation.calc_crf,
    'uic_declining': lcc_calculation.calc_declining_uic,
    'financial upfront cost': lcc_calculation.calc_financial_upfront_cost,
    'complete upfront cost': lcc_calculation.calc_complete_upfront_cost,
    'aic_declining': lcc_calculation.calc_declining_aic,
    'financial annual cost': lcc_calculation.calc_financial_annual_cost,
    'complete annual cost': lcc_calculation.calc_complete_annual_cost,
    'service cost': lcc_calculation.calc_annual_service_cost,
    'emissions cost': lcc_calculation.calc_emissions_cost,
    'financial life cycle cost': lcc_calculation.calc_financial_lcc,
    'complete life cycle cost': lcc_calculation.calc_complete_lcc,
    'price': lcc_calculation.calc_price,
    'fixed cost rate': lcc_calculation.calc_fixed_cost_rate,
    'price_subsidy': lcc_calculation.calc_price_subsidy
}

# TODO: Move inheritable params to sheet in model description to get with reader
inheritable_params = [
    'price multiplier',
    'discount rate_financial',
    'discount rate_retrofit',
    'retrofit_existing_min',
    'retrofit_existing_max',
    'retrofit_heterogeneity',
]


def inherit_parameter(graph, node, year, param):
    assert param in inheritable_params
    parent = '.'.join(node.split('.')[:-1])

    if parent:
        parent_param_val = {}
        if param in graph.nodes[parent][year]:
            param_val = copy.deepcopy(graph.nodes[parent][year][param])
            parent_param_val.update(param_val)

        # Update Param Source
        if 'param_source' in parent_param_val:
            parent_param_val.update({'param_source': 'inheritance'})
        else:
            for context in parent_param_val:
                if 'param_source' in parent_param_val[context]:
                    parent_param_val[context].update({'param_source': 'inheritance'})
                else:
                    for sub_context in parent_param_val[context]:
                        parent_param_val[context][sub_context].update(
                            {'param_source': 'inheritance'})

        node_param_val = copy.deepcopy(parent_param_val)
        if param in graph.nodes[node][year]:
            param_val = graph.nodes[node][year][param]

            # Remove any previously inherited parameter values
            if 'param_source' in param_val:
                if param_val['param_source'] == 'inheritance':
                    param_val = {}
            elif param_val is not None:
                for context in list(param_val):
                    if 'param_source' in param_val[context]:
                        if param_val[context]['param_source'] == 'inheritance':
                            param_val.pop(context)
                    else:
                        for sub_context in list(param_val[context]):
                            if param_val[context][sub_context]['param_source'] == 'inheritance':
                                param_val[context].pop(sub_context)

            node_param_val.update(param_val)

        if node_param_val:
            graph.nodes[node][year][param] = node_param_val


def get_param(model, param, node, year=None, tech=None, context=None, sub_context=None,
              return_source=False, do_calc=False, check_exist=False, dict_expected=False):
    """
    Gets a parameter's value from the model, given a specific context (node, year, tech, context, sub-context),
    calculating the parameter's value if needed.

    This will not re-calculate the parameter's value, but will only retrieve
    values which are already stored in the model or obtained via inheritance, default values,
    or estimation using the previous year's value. If return_source is True, this function will
    also, return how this value was originally obtained (e.g. via calculation)

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the parameter value of interest.
    param : str
        The name of the parameter whose value is being retrieved.
    node : str
        The name of the node (branch format) whose parameter you are interested in retrieving.
    year : str
        The year which you are interested in. `year` must be provided for all parameters stored at
        the technology level, even if the parameter doesn't change year to year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description
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
                try:
                    val = val[context]
                except KeyError:
                    val = None
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
        warning_message = ("Get Param is returning a dict, considering using more parameters in get_param." +
                      "\nParameter: " + (param if param else "") +
                      "\nNode: " + (node if node else "") +
                      "\nYear: " + (year if year else "") +
                      "\nContext: " + (context if context else "") +
                      "\nSub-context: " + (sub_context if sub_context else "") +
                      "\nTech: " + (tech if tech else ""))
        warnings.warn(warning_message)

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
                assert (source in ['inheritance', 'model', 'default'])
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
        if param in model.node_tech_defaults:
            val = model.get_parameter_default(param)
            param_source = 'default'

    # Use Last Year's Value
    # ******************************
    # Otherwise, use the value from the previous year. (If no base year value, throw an error)
    if param_source is None:
        if year is not None:
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
        else:
            val = None
            param_source = None

    if return_source:
        return val, param_source
    else:
        return val


def set_param(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
              save=True):
    """
    Sets a parameter's value, given a specific context (node, year, tech, context, sub-context).
    This is intended for when you are using this function outside of model.run to make single changes
    to the model description.

    Parameters
    ----------
    model : pyCIMS.Model
        The model containing the parameter value of interest.
    val : any or list of any
        The new value(s) to be set at the specified `param` at `node`, given the context provided by
        `year`, `tech`, `context`, and `sub_context`.
    param : str
        The name of the parameter whose value is being set.
    node : str
        The name of the node (branch format) whose parameter you are interested in set.
    year : str or list, optional
        The year(s) which you are interested in. `year` is not required for parameters specified at
        the node level and which by definition cannot change year to year. For example,
        'competition type' can be retrieved without specifying a year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description
    save : bool, optional
        This specifies whether the change should be saved in the change_log csv where True means
        the change will be saved and False means it will not be saved
    """

    def set_node_param_script(model, new_val, param, node, year, context=None, sub_context=None,
                              save=True):
        """
        Sets a parameter's value, given a specific context (node, year, tech, context, sub-context).
        This is intended for when you are using this function outside of model.run to make single changes
        to the model description.

        Parameters
        ----------
        model : pyCIMS.Model
            The model containing the parameter value of interest.
        new_val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
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
                if context:
                    if sub_context:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[context][sub_context], dict) and 'year_value' in val[context][sub_context]:
                            prev_val = val[context][sub_context]['year_value']
                            val[context][sub_context]['year_value'] = new_val
                        else:
                            prev_val = val[context][sub_context]
                            val[context][sub_context] = new_val
                    else:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[context], dict) and 'year_value' in val[context]:
                            prev_val = val[context]['year_value']
                            val[context]['year_value'] = new_val
                        else:
                            prev_val = val[context]
                            val[context] = new_val
                elif 'year_value' in val:
                    prev_val = val['year_value']
                    val['year_value'] = new_val
                elif None in val:
                    # If the value is a dictionary, check if 'year_value' can be accessed.
                    if isinstance(val[None], dict) and 'year_value' in val[None]:
                        prev_val = val[None]['year_value']
                        val[None]['year_value'] = new_val
                    else:
                        prev_val = val[None]
                        val[None] = new_val
                elif len(val.keys()) == 1:
                    # If the value is a dictionary, check if 'year_value' can be accessed.
                    if 'year_value' in val[list(val.keys())[0]]:
                        prev_val = val[list(val.keys())[0]]['year_value']
                        val[list(val.keys())[0]]['year_value'] = new_val
                    else:
                        prev_val = val[list(val.keys())[0]]
                        val[list(val.keys())[0]] = new_val
            else:
                prev_val = data[param]
                data[param] = new_val

            # Save Change
            # ******************************
            # Append the change made to model.change_history DataFrame if save is set to True
            if save:
                filename = model.model_description_file.split('/')[-1].split('.')[0]
                change_log = {'base_model_description': filename, 'parameter': param, 'node': node,
                              'year': year, 'technology': None, 'context': context, 'sub_context': sub_context,
                              'old_value': prev_val, 'new_value': new_val}
                model.change_history = model.change_history.append(pd.Series(change_log), ignore_index=True)
        else:
            print('No param ' + str(param) + ' at node ' + str(node) + ' for year ' + str(
                year) + '. No new value was set for this.')


    def set_tech_param_script(model, new_val, param, node, year, tech=None, context=None, sub_context=None,
                              save=True):
        """
        Sets a parameter's value, given a specific context (node, year, tech, context, sub-context).
        This is intended for when you are using this function outside of model.run to make single changes
        to the model description.

        Parameters
        ----------
        model : pyCIMS.Model
            The model containing the parameter value of interest.
        new_val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        tech : str
            The name of the technology you are interested in.
        context : str, optional
            Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        # Set Parameter from Description
        # ******************************
        # If the parameter's value is in the model description for that node, year, & technology, use it
        data = model.graph.nodes[node][year]['technologies'][tech]
        if param in data:
            val = data[param]
            # If the value is a dictionary, use its nested result
            if isinstance(val, dict):
                if context:
                    if sub_context:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[context][sub_context], dict) and ('year_value' in val[context][sub_context]):
                            prev_val = val[context][sub_context]['year_value']
                            val[context][sub_context]['year_value'] = new_val
                        else:
                            prev_val = val[context][sub_context]
                            val[context][sub_context] = new_val
                    else:
                        # If the value is a dictionary, check if 'year_value' can be accessed.
                        if isinstance(val[context], dict) and ('year_value' in val[context]):
                            prev_val = val[context]['year_value']
                            val[context]['year_value'] = new_val
                        else:
                            prev_val = val[context]
                            val[context] = new_val
                elif None in val:
                    # If the value is a dictionary, check if 'year_value' can be accessed.
                    if isinstance(val[None], dict) and ('year_value' in val[None]):
                        prev_val = val[None]['year_value']
                        val[None]['year_value'] = new_val
                    else:
                        prev_val = val[None]
                        val[None] = new_val
                else:
                    # If the value is a dictionary, check if 'year_value' can be accessed.
                    if 'year_value' in val:
                        prev_val = data[param]['year_value']
                        data[param]['year_value'] = new_val
            else:
                prev_val = data[param]
                data[param] = new_val

            # Save Change
            # ******************************
            # Append the change made to model.change_history DataFrame if save is set to True
            if save:
                filename = model.model_description_file.split('/')[-1].split('.')[0]
                change_log = {'base_model_description': filename, 'parameter': param, 'node': node,
                              'year': year, 'technology': tech, 'context': context, 'sub_context': sub_context,
                              'old_value': prev_val, 'new_value': new_val}
                model.change_history = model.change_history.append(pd.Series(change_log), ignore_index=True)
        else:
            print('No param ' + str(param) + ' at node ' + str(node) + ' for year ' + str(
                year) + '. No new value was set for this.')

    # Checks whether year or val is a list. If either of them is a list, the other must also be a list
    # of the same length
    if isinstance(val, list) or isinstance(year, list):
        if not isinstance(val, list):
            print('Values must be entered as a list.')
            return
        elif not isinstance(year, list):
            print('Years must be entered as a list.')
            return
        elif len(val) != len(year):
            print('The number of values does not match the number of years. No changes were made.')
            return
    else:
        # changing years and vals to lists
        year = [year]
        val = [val]
    for i in range(len(year)):
        try:
            model.get_param(param, node, year[i], tech=tech, context=context, sub_context=sub_context, check_exist=True)
        except:
            print(f"Unable to access parameter at "
                  f"get_param({param}, {node}, {year}, {tech}, {context}, {sub_context}). \n"
                  f"Corresponding value was not set to {val[i]}.")
            continue
        if tech:
            set_tech_param_script(model, val[i], param, node, year[i], tech, context, sub_context, save)

        else:
            set_node_param_script(model, val[i], param, node, year[i], context, sub_context, save)


def set_param_internal(model, val, param, node, year=None, tech=None, context=None, sub_context=None):
    """
    Sets a parameter's value, given a specific context (node, year, tech, context, sub_context).
    This is used from within the model.run function and is not intended to make changes to the model
    description externally (see `set_param`).

    Parameters
    ----------
    val : dict
        The new value(s) to be set at the specified `param` at `node`, given the context provided by
        `year`, `tech`, `context`, and `sub_context`.
    param : str
        The name of the parameter whose value is being set.
    node : str
        The name of the node (branch format) whose parameter you are interested in set.
    year : str or list, optional
        The year(s) which you are interested in. `year` is not required for parameters specified at
        the node level and which by definition cannot change year to year. For example,
        'competition type' can be retrieved without specifying a year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description
    save : bool, optional
        This specifies whether the change should be saved in the change_log csv where True means
        the change will be saved and False means it will not be saved
    """


    def set_node_param(model, new_value, param, node, year, context=None, sub_context=None):
        """
        Queries a model to set a parameter value at a given node, given a specified context
        (year, context, and sub_context).

        Parameters
        ----------
        model : pyCIMS.Model
            The model containing the parameter value of interest.
        new_value : dict
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        context : str, optional
                Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
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
                if context:
                    if sub_context:
                        if isinstance(val[context][sub_context], dict):
                            val[context][sub_context].update(new_value)
                        else:
                            val[context][sub_context] = new_value
                    else:
                        if isinstance(val[context], dict):
                            val[context].update(new_value)
                        else:
                            val[context] = new_value
                elif None in val:
                    val[None].update(new_value)
                else:
                    data[param].update(new_value)
            else:
                data[param] = new_value

        else:
            print(f"No param {param} at node {node} for year {year}. No new value was set for this")


    def set_tech_param(model, new_value, param, node, year, tech, context=None, sub_context=None):
        """
        Queries a model to set a parameter value at a given node & technology, given a specified
        context (year, context, and sub_context).

        Parameters
        ----------
        model : pyCIMS.Model
            The model containing the parameter value of interest.
        new_value : dict
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch format) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        tech : str
            The name of the technology you are interested in.
        context : str, optional
                Used when there is context available in the node. Analogous to the 'context' column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the 'sub_context' column in the model description
        """
        # Set Parameter from Description
        # ******************************
        # If the parameter's value is in the model description for that node, year, & technology, use it
        data = model.graph.nodes[node][year]['technologies'][tech]
        if param in data:
            val = data[param]
            # If the value is a dictionary, use its nested result
            if isinstance(val, dict):
                if context:
                    if sub_context:
                        if isinstance(val[context][sub_context], dict):
                            val[context][sub_context].update(new_value)
                        else:
                            val[context][sub_context] = new_value
                    else:
                        if isinstance(val[context], dict):
                            val[context].update(new_value)
                        else:
                            val[context] = new_value
                elif None in val:
                        val[None].update(new_value)
                else:
                    data[param].update(new_value)
            else:
                data[param] = new_value

        else:
            print(f"No param {param} at node {node} for year {year}. No new value was set for this")


    # Checks whether year or val is a list. If either of them is a list, the other must also be a list
    # of the same length
    if isinstance(val, list) or isinstance(year, list):
        if not isinstance(val, list):
            print('Values must be entered as a list.')
            return
        elif not isinstance(year, list):
            print('Years must be entered as a list.')
            return
        elif len(val) != len(year):
            print('The number of values does not match the number of years. No changes were made.')
            return
    else:
        # changing years and vals to lists
        year = [year]
        val = [val]
    for i in range(len(year)):
        node_data = model.graph.nodes[node][year[i]]
        if tech:
            if tech in node_data["technologies"]:
                tech_data = model.graph.nodes[node][year[i]]["technologies"][tech]
                if param in tech_data:
                    set_tech_param(model, val[i], param, node, year[i], tech, context, sub_context)
                else:
                    value = val[i]['year_value']
                    param_source = val[i]['param_source'] if 'param_source' in val[i] else None
                    branch = val[i]['branch'] if 'branch' in val[i] else None
                    model.create_param(val=value, param=param, node=node, year=year[i], tech=tech,
                                       context=context, sub_context=sub_context,
                                       param_source=param_source, branch=branch)
            else:
                value = val[i]['year_value']
                param_source = val[i]['param_source'] if 'param_source' in val[i] else None
                branch = val[i]['branch'] if 'branch' in val[i] else None
                model.create_param(val=value, param=param, node=node, year=year[i], tech=tech,
                                   context=context, sub_context=sub_context,
                                   param_source=param_source, branch=branch)
        else:
            if param in node_data:
                set_node_param(model, val[i], param, node, year[i], context, sub_context)
            else:
                value = val[i]['year_value']
                model.create_param(val=value, param=param, node=node, year=year[i],
                                   context=context, sub_context=sub_context, param_source=val[i]['param_source'])


def set_param_file(model, filepath):
    """
    Sets parameters' values, for all context (node, year, context, sub_context, and technology)
    from the provided CSV file. See Data_Changes_Tutorial_by_CSV.ipynb for detailed
    description of expected CSV file columns and values.

    Parameters
    ----------
    filepath : str
        This is the path to the CSV file containing all context and value change information
    """

    if not filepath.endswith('.csv'):
        print('filepath must be in csv format')
        return

    df = pd.read_csv(filepath, delimiter=',')
    df = df.fillna('None')

    ops = {
        '>': operator.gt,
        '>=': operator.ge,
        '==': operator.eq,
        '<': operator.lt,
        '<=': operator.le
    }

    for index, row in df.iterrows():
        # *********
        # Set necessary variables from dataframe row
        # *********
        node = row['node'] if row['node'] != 'None' else None
        node_regex = row['node_regex'] if row['node_regex'] != "None" else None
        param = row['param'] if row['param'] != 'None' else None
        tech = row['tech'] if row['tech'] != 'None' else None
        context = row['context'] if row['context'] != 'None' else None
        sub_context = row['sub_context'] if row['sub_context'] != 'None' else None
        year_operator = row['year_operator'] if row['year_operator'] != 'None' else None
        year = row['year'] if row['year'] != 'None' else None
        val_operator = row['val_operator'] if row['val_operator'] != 'None' else None
        val = row['val'] if row['val'] != 'None' else None
        search_param = row['search_param'] if row['search_param'] != 'None' else None
        search_operator = row['search_operator'] if row['search_operator'] != 'None' else None
        search_pattern = row['search_pattern'] if row['search_pattern'] != 'None' else None
        create_missing = row['create_if_missing'] if row['create_if_missing'] != 'None' else None

        # *********
        # Changing years and vals to lists
        # *********
        if year:
            year_int = int(year)
            years = [x for x in model.years if ops[year_operator](int(x), year_int)]
            vals = [val] * len(years)
        else:
            years = [year]
            vals = [val]

        # *********
        # Intial checks on the data
        # *********
        if node == None:
            if node_regex == None:
                print(f"Row {index}: : neither node or node_regex values were indicated. "
                      f"Skipping this row.")
                continue
        elif node == '.*':
            if search_param == None or search_operator == None or search_pattern == None:
                print(f"Row {index}: since node = '.*', search_param, search_operator, and "
                      f"search_pattern must not be empty. Skipping this row.")
                continue
        else:
            if node_regex:
                print(f"Row index: both node and node_regex values were indicated. Please "
                      f"specify only one. Skipping this row.")
                continue
        if year_operator not in list(ops.keys()):
            print(f"Row {index}: year_operator value not one of >, >=, <, <=, ==. Skipping this"
                  f"row.")
            continue
        if val_operator not in ['>=', '<=', '==']:
            print(f"Row {index}: val_operator value not one of >=, <=, ==. Skipping this row.")
            continue
        if search_operator not in [None, '==']:
            print(f"Row {index}: search_operator value must be either empty or ==. Skipping "
                  f"this row.")
            continue
        if create_missing == None:
            print(f"Row {index}: create_if_missing is empty. This value must be either True or"
                  f"False. Skipping this row.")
            continue

        # *********
        # Check the node type ('.*', None, or otherwise) and search through corresponding nodes if necessary
        # *********
        if node == '.*':
            # check if node satisfies search_param, search_operator, search_pattern conditions
            for node_tmp in model.graph.nodes:
                if model.get_param(search_param, node_tmp).lower() == search_pattern.lower():
                    for idx, year in enumerate(years):
                        val_tmp = vals[idx]
                        model.set_param_search(val_tmp, param, node_tmp, year, tech, context, sub_context,
                                              val_operator, create_missing, index)
        elif node == None:
            # check if node satisfies node_regex conditions
            for node_tmp in model.graph.nodes:
                if re.search(node_regex, node_tmp) != None:
                    for idx, year in enumerate(years):
                        val_tmp = vals[idx]
                        model.set_param_search(val_tmp, param, node_tmp, year, tech, context, sub_context,
                                              val_operator, create_missing, index)
        else:
            # node is exactly specified so use as is
            for idx, year in enumerate(years):
                val_tmp = vals[idx]
                model.set_param_search(val_tmp, param, node, year, tech, context, sub_context,
                                      val_operator, create_missing, index)


def set_param_search(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
                     val_operator='==', create_missing=False, row_index=None):
    """
    Sets parameter values for all contexts (node, year, tech, context, sub_context),
    searching through all tech, context, and sub_context keys if necessary.

    Parameters
    ----------
    val : any
        The new value to be set at the specified `param` at `node`, given the context provided by
        `year`, `context, `sub_context`, and `tech`.
    param : str
        The name of the parameter whose value is being set.
    node : str
        The name of the node (branch format) whose parameter you are interested in matching.
    year : str, optional
        The year which you are interested in. `year` is not required for parameters specified at
        the node level and which by definition cannot change year to year. For example,
        'competition type' can be retrieved without specifying a year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
        specified node, param, year, context, and sub_context.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model
        description. If context is `.*`, all possible context keys will be searched at the specified node, param,
        year, sub_context, and tech.
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description.
        If sub_context is `.*`, all possible sub_context keys will be searched at the specified node, param,
        year, context, and tech.
    create_missing : bool, optional
        Will create a new parameter in the model if it is missing. Defaults to False.
    val_operator : str, optional
        This specifies how the value should be set. The possible values are '>=', '<=' and '=='.
    row_index : int, optional
        The index of the current row of the CSV. This is used to print the row number in error messages.
    """

    def get_val_operated(model, val, param, node, year, tech, context, sub_context, val_operator,
                         row_index, create_missing):
        try:
            prev_val = model.get_param(param, node, year, tech=tech, context=context,
                                      sub_context=sub_context, check_exist=True)
            if val_operator == '>=':
                val = max(val, prev_val)
            elif val_operator == '<=':
                val = min(val, prev_val)
        except Exception as e:
            if create_missing:
                print(f"Row {row_index + 1}: Creating parameter at ({param}, {node}, {year}, {context},"
                      f" {sub_context}, {tech}).")
                tmp = model.create_param(val=val, param=param, node=node, year=year, tech=tech,
                                        context=context, sub_context=sub_context, row_index=row_index)
                if not tmp:
                    return None
            else:
                print(f"Row {row_index + 1}: Unable to access parameter at get_param({param}, "
                      f"{node}, {year}, {tech}, {context}, {sub_context}). Corresponding value was not set"
                      f"to {val}.")
                return None
        return val

    if tech == '.*':
        try:
            # search through all technologies in node
            techs = list(model.graph.nodes[node][year]['technologies'].keys())
        except:
            return
        for tech_tmp in techs:
            if context == '.*':
                try:
                    # search through all contexts in node given tech
                    contexts = list(model.get_param(param, node, year, tech=tech_tmp).keys())
                except:
                    continue
                for context_tmp in contexts:
                    if sub_context == '.*':
                        try:
                            # search through all sub_contexts in node given tech
                            sub_contexts = list(
                                model.get_param(param, node, year, tech=tech_tmp, context=context_tmp).keys())
                        except:
                            continue
                        for sub_context_tmp in sub_contexts:
                            val_tmp = get_val_operated(val, param, node, year, tech_tmp, context_tmp, sub_context_tmp,
                                                       val_operator, row_index, create_missing)
                            if val_tmp:
                                model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                               context=context_tmp, sub_context=sub_context_tmp)

                    # use sub_context as is if it is not .*
                    else:
                        val_tmp = get_val_operated(val, param, node, year, tech_tmp, context_tmp, sub_context,
                                                   val_operator, row_index, create_missing)
                        if val_tmp:
                            model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                           context=context_tmp, sub_context=sub_context)

            # use context as is if it is not .*
            else:
                if sub_context == '.*':
                    try:
                        # search through all sub_contexts in node given tech
                        sub_contexts = list(model.get_param(param, node, year, tech=tech_tmp, context=context).keys())
                    except:
                        continue
                    for sub_context_tmp in sub_contexts:
                        val_tmp = get_val_operated(val, param, node, year, tech_tmp, context, sub_context_tmp,
                                                   val_operator, row_index, create_missing)
                        if val_tmp:
                            model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                           context=context, sub_context=sub_context_tmp)

                # use sub_context as is if it is not .*
                else:
                    val_tmp = get_val_operated(val, param, node, year, tech_tmp, context, sub_context,
                                               val_operator, row_index, create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                       context=context, sub_context=sub_context)

    # use tech as is if it is not .*
    else:
        if context == '.*':
            try:
                # search through all contexts in node given tech
                contexts = list(model.get_param(param, node, year, tech=tech).keys())
            except:
                return
            for context_tmp in contexts:
                if sub_context == '.*':
                    try:
                        # search through all sub_contexts in node given tech
                        sub_contexts = list(model.get_param(param, node, year, tech=tech, context=context_tmp).keys())
                    except:
                        continue
                    for sub_context_tmp in sub_contexts:
                        val_tmp = get_val_operated(val, param, node, year, tech, context_tmp, sub_context_tmp,
                                                   val_operator, row_index, create_missing)
                        if val_tmp:
                            model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                           context=context_tmp, sub_context=sub_context_tmp)

                # use sub_context as is if it is not .*
                else:
                    val_tmp = get_val_operated(val, param, node, year, tech, context_tmp, sub_context,
                                               val_operator, row_index, create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                       context=context_tmp, sub_context=sub_context)

        # use context as is if it is not .*
        else:
            if sub_context == '.*':
                try:
                    # search through all sub_contexts in node given tech
                    sub_contexts = list(model.get_param(param, node, year, tech=tech, context=context).keys())
                except:
                    return
                for sub_context_tmp in sub_contexts:
                    val_tmp = get_val_operated(val, param, node, year, tech, context, sub_context_tmp,
                                               val_operator, row_index, create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                       context=context, sub_context=sub_context_tmp)

            # use sub_context as is if it is not .*
            else:
                val_tmp = get_val_operated(val, param, node, year, tech, context, sub_context,
                                           val_operator, row_index, create_missing)
                if val_tmp:
                    model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                   context=context, sub_context=sub_context)


def create_param(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
                 param_source=None, branch=None, row_index=None):
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
        The year which you are interested in. `year` is not required for parameters specified at
        the node level and which by definition cannot change year to year. For example,
        'competition type' can be retrieved without specifying a year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology. If tech is `.*`, all possible tech keys will be searched at the
        specified node, param, year, context, and sub_context.
    context : str, optional
        Used when there is context available in the node. Analogous to the 'context' column in the model
        description. If context is `.*`, all possible context keys will be searched at the specified node, param,
        year, sub_context, and tech.
    sub_context : str, optional
        Must be used only if context is given. Analogous to the 'sub_context' column in the model description.
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
    val_dict = create_value_dict(val, param_source=param_source, branch=branch)

    # *********
    # If there is a tech specified, check if it exists and create context (tech, param, context,
    # sub_context) accordingly
    # *********
    if tech:
        # add technology if it does not exist
        if tech not in data['technologies']:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                context_dict = {context: sub_context_dict}
                param_dict = {param: context_dict}
            elif context:
                context_dict = {context: val_dict}
                param_dict = {param: context_dict}
            else:
                param_dict = {param: val_dict}
            data['technologies'][tech] = param_dict
        # add param if it does not exist
        elif param not in data['technologies'][tech]:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                context_dict = {context: sub_context_dict}
                data['technologies'][tech][param] = context_dict
            elif context:
                context_dict = {context: val_dict}
                data['technologies'][tech][param] = context_dict
            else:
                data['technologies'][tech][param] = val_dict
        # add context if it does not exist
        elif context not in data['technologies'][tech][param]:
            if sub_context:
                sub_context_dict = {sub_context: val_dict}
                data['technologies'][tech][param][context] = sub_context_dict
            else:
                data['technologies'][tech][param][context] = val_dict
        # add sub_context if it does not exist
        elif sub_context not in data['technologies'][tech][param][context]:
            data['technologies'][tech][param][context][sub_context] = val_dict

    # *********
    # Check if param exists and create context (param, context, sub_context) accordingly
    # *********
    elif param not in data:
        if sub_context:
            sub_context_dict = {sub_context: val_dict}
            context_dict = {context: sub_context_dict}
            data[param] = context_dict
        if context:
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

