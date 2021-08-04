import pandas as pd
import warnings
from pyCIMS.model import ProvidedQuantity, RequestedQuantity


class ValueLog:
    def __init__(self, sub_param=None, context=None, unit=None, value=None):
        self.sub_param = sub_param
        self.context = context
        self.unit = unit
        self.value = value

    def tuple(self):
        return self.sub_param, self.context,  self.unit,  self.value


def has_techs(node_year_data):
    return 'technologies' in node_year_data.keys()


def log_int(val):
    return [ValueLog(value=float(val))]


def log_float(val):
    return [ValueLog(value=val)]


def log_str(val):
    return [ValueLog(value=val)]


def log_bool(val):
    return [ValueLog(value=val)]


def log_ProvidedQuantity(val):
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
    for k, v in val.get_total_quantities_requested().items():
        rqs.append(ValueLog(context=k,
                            value=v
                            ))

    # Log total quantities
    rqs.append(ValueLog(context='Total',
                        value=val.sum_requested_quantities()
                        ))

    return rqs


def log_list(val):
    """ List of dictionaries. For each item, extract the value and year_value"""
    val_pairs = []
    for entry in val:
        val_log = ValueLog()
        val_log.context = entry['value']
        val_log.sub_param = entry['sub_param'] if 'sub_param' in entry.keys() else None
        val_log.unit = entry['unit'] if 'unit' in entry.keys() else None
        val_log.value = entry['year_value']

        val_pairs.append(val_log)

    return val_pairs


def log_dict(val):
    """ Dictionary. May be base or be a dictionary containing base dictionaries"""
    # Check if base dictionary
    val_log = ValueLog()

    if 'year_value' in val.keys():
        val_log.context = val['value'] if 'value' in val.keys() else None
        val_log.sub_param = val['sub_param'] if 'sub_param' in val.keys() else None
        val_log.unit = val['unit'] if 'unit' in val.keys() else None

        year_value = val['year_value']

        if year_value is None:
            return [val_log]
        elif isinstance(year_value, ProvidedQuantity):
            return log_ProvidedQuantity(year_value)
        elif isinstance(year_value, RequestedQuantity):
            return log_RequestedQuantity(year_value)
        elif isinstance(year_value, dict):
            return log_dict(year_value)
        else:
            val_log.value = float(year_value)
            return [val_log]
    else:
        val_pairs = []
        for k, v in val.items():
            val_log.context = k

            if isinstance(v, dict):
                val_log.sub_param = v['sub_param'] if 'sub_param' in v.keys() else None
                val_log.unit = v['unit'] if 'unit' in v.keys() else None
                val_log.value = v['year_value']
                val_pairs.append(val_log)
            elif isinstance(v, int) or isinstance(v, float):
                val_log.value = float(v)
                val_pairs.append(val_log)

        return val_pairs


# helper function for opening txt file
def openfile(path):
    with open(path) as f:
        p_list = f.readlines()
        p_list = [x.strip() for x in p_list]

    return p_list


# define slim list example, change the content in parameter_list1 if you want a different list
def slimlist(default_list):
    # add more combinations to the list as we grow the default list

    if default_list == 'slim':
        p_list = ['new_market_share', 'Life Cycle Cost', 'competition type',
                    'Service requested', 'Capital cost_overnight']

    # this is for validating if we have defined the default name
    else:
        raise ValueError("ValueError exception thrown: default_list name not exist")

    return p_list


def add_log_item(all_logs, log_tuple):
    log_func = {int: log_int,
                float: log_float,
                ProvidedQuantity: log_ProvidedQuantity,
                RequestedQuantity: log_RequestedQuantity,
                list: log_list,
                dict: log_dict,
                str: log_str,
                bool: log_bool}

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


# Add model_parameter helper function which returns all parameters in the model and store as a list
def model_parameter(model):
    model_list = []

    for node in model.graph.nodes:
        for param, val in model.graph.nodes[node].items():
            # print(param)
            if param not in model_list:
                model_list.append(param)

        for year in model.years:
            ny_data = model.graph.nodes[node][year]
            for param, val in ny_data.items():
                if param not in model_list:
                    model_list.append(param)

            for param, val in ny_data.items():
                if param == 'technologies':
                    for tech, tech_data in ny_data['technologies'].items():
                        for tech_param, tech_val in tech_data.items():
                            if tech_param not in model_list:
                                model_list.append(tech_param)
    return model_list


def search_parameter(model, search: [str] = None):
    model_list = model_parameter(model)

    print('You are searching if any parameter in the model contains ', search)
    search_list = []
    for i in range(len(search)):
        m = search[i]
        matching = [x for x in model_list if m in x]

        search_list += matching

    if len(search_list) == 0:
        warnings.warn(
            "You search term doesn't match with any parameter in the model")
        return

    print('Here are all the parameters contain your search term : ')
    return search_list


def log_model(model, output_file, parameter_list: [str] = None, path: str = None, default_list: str = None):
    '''
    parameter_list: a list of string such as ['aa', 'bb','cc']
    path: str path of the txt file such as 'test.txt'
    default_list: str of default list, right now 'all' return all parameters and 'slim' return a pre-defined 5 parameters
    '''

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
                                log = node, year, tech, tech_param, tech_val
                                add_log_item(all_logs, log)
                    else:
                        log = node, year, None, param, val
                        add_log_item(all_logs, log)

    else:
        # path argument exist
        if path and (parameter_list is None and default_list is None):
            p_list = openfile(path)

        # parameter_list argument exist
        elif parameter_list and (default_list is None and path is None):
            p_list = parameter_list

        # default_list argument exist
        elif default_list and (parameter_list is None and path is None):
            p_list = slimlist(default_list)

        # Warning if there are more than 2 argument specified
        else:
            raise ValueError("ValueError exception thrown: multiple parameters specified")

            return

        l = len(p_list)
        all_logs = []
        total_parameter_list = model_parameter(model)

        for node in model.graph.nodes:
            # Log Year Agnostic Values
            for i in range(l):
                # check if the input parameter exists.
                if p_list[i] not in total_parameter_list:
                    message = "parameter {parameter:} does not exist".format(parameter=p_list[i])
                    warnings.warn(message)

                for param, val in model.graph.nodes[node].items():
                    if param == p_list[i]:
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
                                    if tech_param == p_list[i]:
                                        log = node, year, tech, tech_param, tech_val
                                        add_log_item(all_logs, log)
                        else:
                            if param == p_list[i]:
                                log = node, year, None, param, val
                                add_log_item(all_logs, log)

    # data_tuples = [log.tuple() for log in all_logs]
    log_df = pd.DataFrame(all_logs)
    log_df.columns = ['node', 'year', 'technology', 'parameter', 'value']

    # Split Value Log values
    log_df[['sub_parameter', 'context', 'unit', 'value']] = pd.DataFrame(log_df['value'].apply(lambda x: x.tuple()).to_list())
    log_df = log_df[['node', 'year', 'technology', 'parameter', 'sub_parameter', 'context', 'unit', 'value']]

    # Write to file
    log_df.to_csv(output_file, index=False)

    return log_df
