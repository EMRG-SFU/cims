import pandas as pd
import operator
import re

from CIMS.utils.parameter import list as PARAM
from CIMS.utils.model_description import column_list as COL

def set_param_calibration(model, val, param, node, year=None, tech=None, context=None, sub_context=None,
              save=True):
    """
    Hacked version of `set_param`, so that if the param isn't found it will be created. Some of these other
    `set` methods in this module do that, but not this one.

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
                  f"Corresponding value was not set to {val[i]}. \n"
                  f"Parameter created.")
            tmp = model.create_param(val=val[i], param=param, node=node, year=year[i], tech=tech,
                               context=context, sub_context=sub_context)
            if not tmp:
                raise

            continue
        if tech:
            set_tech_param_script(model, val[i], param, node, year[i], tech, context, sub_context, save)

        else:
            set_node_param_script(model, val[i], param, node, year[i], context, sub_context, save)



