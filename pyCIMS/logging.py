"""
Module providing all the logging (write Model parameters to CSV file) functionality for the pyCIMS
package.
"""
import warnings
from copy import deepcopy
from scipy.interpolate import interp1d

import pandas as pd

from pyCIMS.model import ProvidedQuantity, RequestedQuantity
from pyCIMS.emissions import Emissions, EmissionRates, EmissionsCost


class ValueLog:
    """Class used to store the information needed to log a single parameter value."""

    def __init__(self, context=None, sub_context=None, branch=None, unit=None, value=None):
        self.context = context
        self.sub_context = sub_context
        self.branch = branch
        self.unit = unit
        self.value = value

    def tuple(self):
        """
        Returns
        -------
        tuple :
            Returns a tuple containing the (context, sub-context, branch, unit, and value). Used to
            create the logging CSV.
        """
        return self.context, self.sub_context, self.branch, self.unit, self.value


def _has_techs(node_year_data):
    """Checks if a node has technologies."""
    return 'technologies' in node_year_data.keys()


def log_int(val):
    """Creates a logging-ready representation of an integer."""
    return [ValueLog(value=float(val))]


def log_float(val):
    """Creates a logging-ready representation of a float."""
    return [ValueLog(value=val)]


def log_str(val):
    """Creates a logging-ready representation of a string."""
    return [ValueLog(value=val)]


def log_bool(val):
    """Creates a logging-ready representation of a boolean."""
    return [ValueLog(value=val)]


def log_ProvidedQuantity(val):
    """Creates a logging-ready representation of a ProvidedQuantity object, using the total number
       of units provided by a node."""
    return [ValueLog(value=float(val.get_total_quantity()))]


def log_RequestedQuantity(val):
    """
    Examines the RequestedQuantity object and provides a list of tuples to be used in the logger.

    Parameters
    ----------
    val : pyCIMS.RequestedQuantity
        The RequestedQuantity object containing the record of all requested quantities which can be
        traced back to a node, from either it's own requested services, or those of it's successors.

    Returns
    -------
    list of tuples
        Returns a list of tuples, where each tuple contains context, unit, and value for a specific
        service being requested by the node.
    """
    rqs = []

    # Log quantities per tech
    for key, quant in val.get_total_quantities_requested().items():
        rqs.append(ValueLog(branch=key,
                            value=quant
                            ))

    # Log total quantities
    rqs.append(ValueLog(context='Total',
                        value=val.sum_requested_quantities()
                        ))

    return rqs


def log_Emissions(val):
    """ Provides a list of tuples to be used for logging, based on the provided Emissions object. A
    tuple is created for each GHG/Emission Type combination that exists in the node."""
    result = []

    emissions = val.summarize_emissions()
    for ghg in emissions:
        for emission_type in emissions[ghg]:
            val = emissions[ghg][emission_type]
            result.append(ValueLog(context=ghg,
                                   sub_context=emission_type,
                                   value=val))

    return result


def log_EmissionRates(val):
    """ Provides a list of tuples to be used for logging, based on the provided EmissionRates
    object. A tuple is created for each GHG/Emission Type combination that exists in the node."""

    result = []

    rates = val.summarize_rates()
    for ghg in rates:
        for emission_type in rates[ghg]:
            val = rates[ghg][emission_type]
            result.append(ValueLog(context=ghg,
                                   sub_context=emission_type,
                                   value=val))
    return result


def log_EmissionsCost(val):
    """ Provides a list of tuples to be used for logging, based on the provided EmissionsCost
    object. A tuple is created for each GHG/Emission Type combination that exists in the node."""

    result = []

    rates = val.summarize()
    for ghg in rates:
        for emission_type in rates[ghg]:
            val = rates[ghg][emission_type]
            result.append(ValueLog(context=ghg,
                                   sub_context=emission_type,
                                   value=val))
    return result


def log_list(val):
    """ List of dictionaries. For each item, extract the value and year_value"""
    val_pairs = []
    for entry in val:
        val_log = ValueLog()
        val_log.context = entry['context'] if 'context' in entry.keys() else None
        val_log.sub_context = entry['sub_context'] if 'sub_context' in entry.keys() else None
        val_log.branch = entry['branch'] if 'branch' in entry.keys() else None
        val_log.unit = entry['unit'] if 'unit' in entry.keys() else None
        val_log.value = entry['year_value']

        val_pairs.append(deepcopy(val_log))

    return val_pairs


def log_dict(val):
    """ Dictionary. May be base or be a dictionary containing base dictionaries"""
    # Check if base dictionary
    val_log = ValueLog()

    if 'year_value' in val.keys():
        val_log.context = val['context'] if 'context' in val.keys() else None
        val_log.sub_context = val['sub_context'] if 'sub_context' in val.keys() else None
        val_log.branch = val['branch'] if 'branch' in val.keys() else None
        val_log.unit = val['unit'] if 'unit' in val.keys() else None

        year_value = val['year_value']

        if year_value is None:
            return [val_log]
        if isinstance(year_value, ProvidedQuantity):
            return log_ProvidedQuantity(year_value)
        if isinstance(year_value, RequestedQuantity):
            return log_RequestedQuantity(year_value)
        if isinstance(year_value, Emissions):
            return log_Emissions(year_value)
        if isinstance(year_value, EmissionRates):
            return log_EmissionRates(year_value)
        if isinstance(year_value, EmissionsCost):
            return log_EmissionsCost(year_value)
        if isinstance(year_value, dict):
            return log_dict(year_value)
        val_log.value = float(year_value)
        return [val_log]
    else:
        val_pairs = []
        for key, inner_value in val.items():
            val_log.context = key

            if isinstance(inner_value, dict):
                if 'year_value' in inner_value:
                    val_log.sub_context = inner_value['sub_context'] \
                        if 'sub_context' in inner_value.keys() else None
                    val_log.branch = inner_value['branch'] \
                        if 'branch' in inner_value.keys() else None
                    val_log.unit = inner_value['unit'] if 'unit' in inner_value.keys() else None
                    val_log.value = inner_value['year_value']
                else:
                    for sub_context, base_val in inner_value.items():
                        val_log.sub_context = sub_context
                        val_log.branch = base_val['branch'] if 'branch' in base_val.keys() else None
                        val_log.unit = base_val['unit'] if 'unit' in base_val.keys() else None
                        val_log.value = base_val['year_value']
                        val_pairs.append(deepcopy(val_log))

            elif isinstance(inner_value, (int, float)):
                val_log.value = float(inner_value)
                val_pairs.append(deepcopy(val_log))

        return val_pairs


def do_not_log(_):
    """A placeholder function to take care of types we don't want to be logged"""
    return []


def _open_file(path):
    """Helper function for opening txt file"""
    with open(path) as file:
        p_list = file.readlines()
        p_list = [x.strip() for x in p_list]

    return p_list


def _slim_list(default_list):
    """Define slim list example, change the content in p_list if you want a different list"""

    if default_list == 'slim':
        p_list = ['new_market_share', 'life cycle cost', 'competition type',
                  'service requested', 'capital cost_overnight']

    # this is for validating if we have defined the default name
    else:
        raise ValueError("ValueError exception thrown: default_list name not exist")

    return p_list


def add_log_item(all_logs, log_tuple):
    """Process the log_tuple (according to its type) and add it to the list of all_logs"""
    log_func = {int: log_int,
                float: log_float,
                ProvidedQuantity: log_ProvidedQuantity,
                RequestedQuantity: log_RequestedQuantity,
                Emissions: log_Emissions,
                EmissionRates: log_EmissionRates,
                EmissionsCost: log_EmissionsCost,
                list: log_list,
                dict: log_dict,
                str: log_str,
                bool: log_bool,
                interp1d: do_not_log}

    node, year, tech, param, val = log_tuple
    # Process the value & year value
    try:
        prepped_val = log_func[type(val)](val)
    except KeyError:
        prepped_val = val

    # Log
    if isinstance(prepped_val, list):
        for val in prepped_val:
            log = node, year, tech, param, val
            all_logs.append(log)
    else:
        log = node, year, tech, param, prepped_val
        all_logs.append(log)
    return all_logs


def _full_parameter_list(model):
    """Helper function which returns all parameters in the model and store as a list"""
    model_list = []

    for node in model.graph.nodes:
        for param, _ in model.graph.nodes[node].items():
            if param not in model_list:
                model_list.append(param)

        for year in model.years:
            ny_data = model.graph.nodes[node][year]
            for param in ny_data:
                if param not in model_list:
                    model_list.append(param)

            for param in ny_data:
                if param == 'technologies':
                    for tech_data in ny_data['technologies'].values():
                        for tech_param in tech_data:
                            if tech_param not in model_list:
                                model_list.append(tech_param)
    return model_list


def search_parameter(model, search: [str] = None):
    """Function to search for model parameters that contain any strings present in the search list.
    """
    model_list = _full_parameter_list(model)

    print('You are searching if any parameter in the model contains ', search)
    search_list = []
    for parameter in search:
        matching = [x for x in model_list if parameter in x]
        search_list += matching

    if len(search_list) == 0:
        warnings.warn(
            "You search term doesn't match with any parameter in the model")
        return

    print('Here are all the parameters contain your search term : ')
    return search_list


def log_model(model, output_file, parameter_list: [str] = None, path: str = None,
              default_list: str = None):
    """
    Log a model's current state to an output CSV file.

    Parameters
    ----------
    model : pyCIMS.Model
        Model that is being logged to a CSV file
    output_file : str
        Path to the output CSV file location
    parameter_list : list of str, optional
        A list of strings
    path : str, optional
        Path to a text file containing the list of parameters to log
    default_list : str, optional
        The name of a default parameter list. Currently two default lists are defined:
        (1) 'all' will log all parameters and
        (2) 'slim' will return 5 pre-defined parameters ('new_market_share', 'life cycle cost',
            'competition type', 'service requested', 'capital cost_overnight'

    Returns
    -------
    pandas.DataFrame :
        The DataFrame containing the model's current parameter values. Additionally, a CSV file is
        written to output_path.
    """
    # if no argument chosen or defualt_list = 'all', return all parameters
    if parameter_list is None and path is None and (default_list is None or default_list == 'all'):
        all_logs = []
        for node in model.graph.nodes:
            # Log Year Agnostic Values
            for param, val in model.graph.nodes[node].items():
                if param not in model.years:
                    log = node, None, None, param, val
                    add_log_item(all_logs, log)

            # Log Year Specific Values
            for year in model.years:
                ny_data = model.graph.nodes[node][year]
                for param, val in ny_data.items():
                    if param == 'technologies':
                        for tech, tech_data in ny_data['technologies'].items():
                            for tech_param, tech_val in tech_data.items():
                                if tech_param not in ['aggregate_emissions_cost_rates']:
                                    log = node, year, tech, tech_param, tech_val
                                    add_log_item(all_logs, log)
                    else:
                        log = node, year, None, param, val
                        if param not in ['aggregate_emissions_cost_rates']:
                            add_log_item(all_logs, log)

    else:
        # path argument exist
        if path and (parameter_list is None and default_list is None):
            p_list = _open_file(path)

        # parameter_list argument exist
        elif parameter_list and (default_list is None and path is None):
            p_list = parameter_list

        # default_list argument exist
        elif default_list and (parameter_list is None and path is None):
            p_list = _slim_list(default_list)

        # Warning if there are more than 2 argument specified
        else:
            raise ValueError("ValueError exception thrown: multiple parameters specified")

        all_logs = []
        total_parameter_list = _full_parameter_list(model)

        for node in model.graph.nodes:
            # Log Year Agnostic Values
            for param_to_log in p_list:
                # check if the input parameter exists.
                if param_to_log not in total_parameter_list:
                    message = "parameter {parameter:} does not exist".format(parameter=param_to_log)
                    warnings.warn(message)

                for param, val in model.graph.nodes[node].items():
                    if param == param_to_log:
                        if param not in model.years:
                            log = node, None, None, param, val
                            add_log_item(all_logs, log)

                # Log Year Specific Values
                for year in model.years:
                    ny_data = model.graph.nodes[node][year]

                    for param, val in ny_data.items():
                        if param == 'technologies':
                            for tech, tech_data in ny_data['technologies'].items():
                                for tech_param, tech_val in tech_data.items():
                                    if tech_param == param_to_log:
                                        log = node, year, tech, tech_param, tech_val
                                        add_log_item(all_logs, log)
                        else:
                            if param == param_to_log:
                                log = node, year, None, param, val
                                add_log_item(all_logs, log)

    # data_tuples = [log.tuple() for log in all_logs]
    log_df = pd.DataFrame(all_logs)
    log_df.columns = ['node', 'year', 'technology', 'parameter', 'value']

    # Split Value Log values
    split_columns = ['context', 'sub_context', 'branch', 'unit', 'value']
    log_df[split_columns] = pd.DataFrame(log_df['value'].apply(lambda x: x.tuple()).to_list())

    # Select final columns
    columns = ['node', 'year', 'technology', 'parameter', 'context', 'sub_context', 'branch',
               'unit', 'value']
    log_df = log_df[columns]

    # Write to file
    log_df.to_csv(output_file, index=False)

    return log_df
