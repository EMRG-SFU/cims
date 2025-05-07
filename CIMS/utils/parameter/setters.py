import pandas as pd
import operator
import re

from . import list as PARAM
from ..model_description import column_list as COL


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
        The name of the node (branch notation) whose parameter you are interested in matching.
    year : str, optional
        The year(s) which you are setting a value for. `year` is not required
        for parameters specified at the node level and which by definition
        cannot change year to year (e.g. competition type).
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
        Must be used only if context is given. Analogous to the `subcontext` column in the model description.
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
            techs = list(model.graph.nodes[node][year][PARAM.technologies].keys())
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
                            val_tmp = get_val_operated(model, val, param, node, year, tech_tmp,
                                                       context_tmp, sub_context_tmp, val_operator,
                                                       row_index, create_missing)
                            if val_tmp:
                                model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                               context=context_tmp, sub_context=sub_context_tmp)

                    # use sub_context as is if it is not .*
                    else:
                        val_tmp = get_val_operated(model, val, param, node, year, tech_tmp,
                                                   context_tmp, sub_context, val_operator,
                                                   row_index, create_missing)
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
                        val_tmp = get_val_operated(model, val, param, node, year, tech_tmp, context,
                                                   sub_context_tmp, val_operator, row_index,
                                                   create_missing)
                        if val_tmp:
                            model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                           context=context, sub_context=sub_context_tmp)

                # use sub_context as is if it is not .*
                else:
                    val_tmp = get_val_operated(model, val, param, node, year, tech_tmp, context,
                                               sub_context, val_operator, row_index, create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech_tmp,
                                       context=context, sub_context=sub_context)

    # use tech as is if it is not .*
    else:
        if context == '.*':
            try:
                # search through all contexts in node given tech
                if tech:
                    contexts = list(model.get_param(param, node, year, tech=tech, dict_expected=True).keys())
                else:
                    contexts = list(model.get_param(param, node, year, dict_expected=True).keys())
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
                        val_tmp = get_val_operated(model, val, param, node, year, tech, context_tmp,
                                                   sub_context_tmp, val_operator, row_index,
                                                   create_missing)
                        if val_tmp:
                            model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                           context=context_tmp, sub_context=sub_context_tmp)

                # use sub_context as is if it is not .*
                else:
                    val_tmp = get_val_operated(model, val, param, node, year, tech, context_tmp,
                                               sub_context, val_operator, row_index, create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                       context=context_tmp, sub_context=sub_context)

        # use context as is if it is not .*
        else:
            if sub_context == '.*':
                try:
                    # search through all sub_contexts in node given tech
                    sub_contexts = list(model.get_param(param, node, year, tech=tech, context=context).keys())
                except AttributeError:
                    print(f"Unable to access parameter at "
                          f"get_param({param}, {node}, {year}, {tech}, {context}, {sub_context}). \n"
                          f"Corresponding value was not set.")
                    return
                for sub_context_tmp in sub_contexts:
                    val_tmp = get_val_operated(model, val, param, node, year, tech, context,
                                               sub_context_tmp, val_operator, row_index,
                                               create_missing)
                    if val_tmp:
                        model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                       context=context, sub_context=sub_context_tmp)

            # use sub_context as is if it is not .*
            else:
                val_tmp = get_val_operated(model, val, param, node, year, tech, context,
                                           sub_context, val_operator, row_index, create_missing)
                if val_tmp:
                    model.set_param(val=val_tmp, param=param, node=node, year=year, tech=tech,
                                   context=context, sub_context=sub_context)


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
        param = row['param'].lower() if row['param'] != 'None' else None
        tech = row['tech'] if row['tech'] != 'None' else None
        context = row[PARAM.context] if row[PARAM.context] != 'None' else None
        sub_context = row[PARAM.sub_context] if row[PARAM.sub_context] != 'None' else None
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
        The name of the node (branch notation) whose parameter you are interested in set.
    year : str or list, optional
        The year(s) which you are setting a value for. `year` is not required
        for parameters specified at the node level and which by definition
        cannot change year to year (e.g. competition type).
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the `context` column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the `subcontext` column in the model description
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
        model : CIMS.Model
            The model containing the parameter value of interest.
        new_value : dict
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch notation) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        context : str, optional
                Used when there is context available in the node. Analogous to the `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the `subcontext` column in the model description
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
        model : CIMS.Model
            The model containing the parameter value of interest.
        new_value : dict
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch notation) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        tech : str
            The name of the technology you are interested in.
        context : str, optional
                Used when there is context available in the node. Analogous to the `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the `subcontext` column in the model description
        """
        # Set Parameter from Description
        # ******************************
        # If the parameter's value is in the model description for that node, year, & technology, use it
        data = model.graph.nodes[node][year][PARAM.technologies][tech]
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
            if tech in node_data[PARAM.technologies]:
                tech_data = model.graph.nodes[node][year[i]][PARAM.technologies][tech]
                if param in tech_data:
                    set_tech_param(model, val[i], param, node, year[i], tech, context, sub_context)
                else:
                    value = val[i][PARAM.year_value]
                    param_source = val[i][PARAM.param_source] if PARAM.param_source in val[i] else None
                    target = val[i][PARAM.target] if PARAM.target in val[i] else None
                    model.create_param(val=value, param=param, node=node, year=year[i], tech=tech,
                                       context=context, sub_context=sub_context, target=target,
                                       param_source=param_source)
            else:
                value = val[i][PARAM.year_value]
                param_source = val[i][PARAM.param_source] if PARAM.param_source in val[i] else None
                target = val[i][PARAM.target] if PARAM.target in val[i] else None
                model.create_param(val=value, param=param, node=node, year=year[i], tech=tech,
                                   context=context, sub_context=sub_context, target=target,
                                   param_source=param_source)
        else:
            if param in node_data:
                set_node_param(model, val[i], param, node, year[i], context, sub_context)
            else:
                value = val[i][PARAM.year_value]
                model.create_param(val=value, param=param, node=node, year=year[i],
                                   context=context, sub_context=sub_context, param_source=val[i][PARAM.param_source])


def set_param(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
              save=True):
    """
    Sets a parameter's value, given a specific context (node, year, tech, context, sub-context).
    This is intended for when you are using this function outside of model.run to make single changes
    to the model description.

    Parameters
    ----------
    model : CIMS.Model
        The model containing the parameter value of interest.
    val : any or list of any
        The new value(s) to be set at the specified `param` at `node`, given the context provided by
        `year`, `tech`, `context`, and `sub_context`.
    param : str
        The name of the parameter whose value is being set.
    node : str
        The name of the node (branch notation) whose parameter you are interested in set.
    year : str or list, optional
        The year(s) which you are interested in. `year` is not required for parameters specified at
        the node level and which by definition cannot change year to year. For example,
        competition type can be retrieved without specifying a year.
    tech : str, optional
        The name of the technology you are interested in. `tech` is not required for parameters
        that are specified at the node level. `tech` is required to get any parameter that is
        stored within a technology.
    context : str, optional
        Used when there is context available in the node. Analogous to the `context` column in the model description
    sub_context : str, optional
        Must be used only if context is given. Analogous to the `subcontext` column in the model description
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
        model : CIMS.Model
            The model containing the parameter value of interest.
        new_val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch notation) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        context : str, optional
            Used when there is context available in the node. Analogous to the `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the `subcontext` column in the model description
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
                        # If the value is a dictionary, check if `year_value` can be accessed.
                        if isinstance(val[context][sub_context], dict) and PARAM.year_value in val[context][sub_context]:
                            prev_val = val[context][sub_context][PARAM.year_value]
                            val[context][sub_context][PARAM.year_value] = new_val
                        else:
                            prev_val = val[context][sub_context]
                            val[context][sub_context] = new_val
                    else:
                        # If the value is a dictionary, check if `year_value` can be accessed.
                        if isinstance(val[context], dict) and PARAM.year_value in val[context]:
                            prev_val = val[context][PARAM.year_value]
                            val[context][PARAM.year_value] = new_val
                        else:
                            prev_val = val[context]
                            val[context] = new_val
                elif PARAM.year_value in val:
                    prev_val = val[PARAM.year_value]
                    val[PARAM.year_value] = new_val
                elif None in val:
                    # If the value is a dictionary, check if `year_value` can be accessed.
                    if isinstance(val[None], dict) and PARAM.year_value in val[None]:
                        prev_val = val[None][PARAM.year_value]
                        val[None][PARAM.year_value] = new_val
                    else:
                        prev_val = val[None]
                        val[None] = new_val
                elif len(val.keys()) == 1:
                    # If the value is a dictionary, check if `year_value` can be accessed.
                    if PARAM.year_value in val[list(val.keys())[0]]:
                        prev_val = val[list(val.keys())[0]][PARAM.year_value]
                        val[list(val.keys())[0]][PARAM.year_value] = new_val
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
                change_log = {
                    'base_model_description': [filename],
                    COL.parameter.lower(): [param],
                    COL.branch.lower(): [node],
                    'year': [year],
                    COL.technology.lower(): None,
                    COL.context.lower(): [context],
                    COL.context.lower(): [sub_context],
                    'old_value': [prev_val],
                    'new_value': [new_val]}
                model.change_history = pd.concat([model.change_history, pd.DataFrame(change_log)], ignore_index=True)
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
        model : CIMS.Model
            The model containing the parameter value of interest.
        new_val : any
            The new value to be set at the specified `param` at `node`, given the context provided by
            `year`, `tech`, `context`, and `sub_context`.
        param : str
            The name of the parameter whose value is being set.
        node : str
            The name of the node (branch notation) whose parameter you are interested in set.
        year : str
            The year which you are interested in. `year` must be provided for all parameters stored at
            the technology level, even if the parameter doesn't change year to year.
        tech : str
            The name of the technology you are interested in.
        context : str, optional
            Used when there is context available in the node. Analogous to the `context` column in the model description
        sub_context : str, optional
            Must be used only if context is given. Analogous to the `subcontext` column in the model description
        save : bool, optional
            This specifies whether the change should be saved in the change_log csv where True means
            the change will be saved and False means it will not be saved
        """

        # Set Parameter from Description
        # ******************************
        # If the parameter's value is in the model description for that node, year, & technology, use it
        data = model.graph.nodes[node][year][PARAM.technologies][tech]
        if param in data:
            val = data[param]
            # If the value is a dictionary, use its nested result
            if isinstance(val, dict):
                if context:
                    if sub_context:
                        # If the value is a dictionary, check if `year_value` can be accessed.
                        if isinstance(val[context][sub_context], dict) and (PARAM.year_value in val[context][sub_context]):
                            prev_val = val[context][sub_context][PARAM.year_value]
                            val[context][sub_context][PARAM.year_value] = new_val
                        else:
                            prev_val = val[context][sub_context]
                            val[context][sub_context] = new_val
                    else:
                        # If the value is a dictionary, check if `year_value` can be accessed.
                        if isinstance(val[context], dict) and (PARAM.year_value in val[context]):
                            prev_val = val[context][PARAM.year_value]
                            val[context][PARAM.year_value] = new_val
                        else:
                            prev_val = val[context]
                            val[context] = new_val
                elif None in val:
                    # If the value is a dictionary, check if `year_value` can be accessed.
                    if isinstance(val[None], dict) and (PARAM.year_value in val[None]):
                        prev_val = val[None][PARAM.year_value]
                        val[None][PARAM.year_value] = new_val
                    else:
                        prev_val = val[None]
                        val[None] = new_val
                else:
                    # If the value is a dictionary, check if `year_value` can be accessed.
                    if PARAM.year_value in val:
                        prev_val = data[param][PARAM.year_value]
                        data[param][PARAM.year_value] = new_val
            else:
                prev_val = data[param]
                data[param] = new_val

            # Save Change
            # ******************************
            # Append the change made to model.change_history DataFrame if save is set to True
            if save:
                filename = model.model_description_file.split('/')[-1].split('.')[0]
                change_log = {'base_model_description': [filename],
                              COL.parameter.lower(): [param],
                              COL.branch.lower(): [node],
                              'year': year,
                              COL.technology.lower(): [tech],
                              COL.context.lower(): [context],
                              COL.sub_context.lower(): [sub_context],
                              'old_value': [prev_val],
                              'new_value': [new_val]}
                changes_to_concat = [model.change_history, pd.DataFrame(change_log)]
                model.change_history = pd.concat([df for df in changes_to_concat if len(df) != 0], ignore_index=True)

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



