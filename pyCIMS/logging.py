import pandas as pd
from pyCIMS.model import NodeQuantity


def has_techs(node_year_data):
    return 'technologies' in node_year_data.keys()


def log_int(val):
    return [(None, None, float(val))]


def log_float(val):
    return [(None, None, val)]


def log_str(val):
    return [(None, None, val)]


def log_bool(val):
    return [(None, None, val)]


def log_NodeQuantity(val):
    return [(None, None, float(val.get_total_quantity()))]


def log_list(val):
    """ List of dictionaries. For each item, extract the value and year_value"""
    val_pairs = []
    for entry in val:
        value = entry['value']
        year_value = entry['year_value']
        unit = entry['unit']
        val_pairs.append((value, unit, year_value))

    return val_pairs


def log_dict(val):
    """ Dictionary. May be base or be a dictionary containing base dictionaries"""
    # Check if there is a value to use for context
    if 'value' in val.keys():
        context = val['value']
    else:
        context = None

    # Check if base dictionary
    if 'year_value' in val.keys():
        unit = val['unit']
        year_value = val['year_value']

        if year_value is None:
            return [(context, unit, None)]
        else:
            return [(context, unit, float(year_value))]
    else:
        val_pairs = []
        for k, v in val.items():
            year_value = v['year_value']
            unit = v['unit']
            val_pairs.append((k, unit, year_value))
        return val_pairs


# helper function for opening txt file
def openfile(path):
    with open(path) as f:
        p_list = f.readlines()
        p_list = [x.strip() for x in p_list]

    return p_list


# define slim list example, change the content in parameter_list1 if you want a different list
def slimlist(default_list):
    if default_list == 'slim':
        p_list = ['new_market_share', 'Life Cycle Cost', 'Competition type',
                  'Service requested', 'Capital cost_overnight']
    return p_list


def add_log_item(all_logs, log_tuple):
    log_func = {int: log_int,
                float: log_float,
                NodeQuantity: log_NodeQuantity,
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


def log_model(model, output_file, parameter_list: [str] = None, path: str = None, default_list: str = None):
    # parameter_list: a list of string such as ['aa', 'bb','cc']
    # path: str path of the txt file such as 'test.txt'
    # default_list: str of default list, right now 'all' return all parameters and 'slim' return a pre-defined 5 parameters

    # if no argument chosen or defualt_list = 'all', return all parameters
    if parameter_list is None and path is None and (default_list is None or default_list == 'all'):

        data_tuples = []
        for node in model.graph.nodes:
            # Log Year Agnostic Values
            for param, val in model.graph.nodes[node].items():
                if param not in model.years:
                    log = node, None, None, param, val
                    add_log_item(data_tuples, log)

            # Log Year Specific Values
            for year in model.years:
                ny_data = model.graph.nodes[node][year]
                for param, val in ny_data.items():
                    if param == 'technologies':
                        for tech, tech_data in ny_data['technologies'].items():
                            for tech_param, tech_val in tech_data.items():
                                log = node, year, tech, tech_param, tech_val
                                add_log_item(data_tuples, log)
                    else:
                        log = node, year, None, param, val
                        add_log_item(data_tuples, log)


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

        # print this if there are more than 2 argument specified
        else:
            print(
                'Error! You can only use parameter_list or path or default_list. You cannot specify them at the same time!')
            return

        l = len(p_list)
        data_tuples = []
        for node in model.graph.nodes:
            # Log Year Agnostic Values
            for i in range(l):

                for param, val in model.graph.nodes[node].items():
                    if param == p_list[i]:
                        if param not in model.years:
                            log = node, None, None, param, val
                            add_log_item(data_tuples, log)

                # Log Year Specific Values
                for year in model.years:
                    ny_data = model.graph.nodes[node][year]

                    for param, val in ny_data.items():
                        if param == 'technologies':
                            for tech, tech_data in ny_data['technologies'].items():
                                for tech_param, tech_val in tech_data.items():
                                    if tech_param == p_list[i]:
                                        log = node, year, tech, tech_param, tech_val
                                        add_log_item(data_tuples, log)
                        else:
                            if param == p_list[i]:
                                log = node, year, None, param, val
                                add_log_item(data_tuples, log)

    log_df = pd.DataFrame(data_tuples)
    log_df.columns = ['node', 'year', 'technology', 'parameter', 'value']

    # Split tupled values
    log_df[['context', 'unit', 'value']] = pd.DataFrame(log_df['value'].to_list())
    log_df = log_df[['node', 'year', 'technology', 'parameter', 'context', 'unit', 'value']]

    # Write to file
    log_df.to_csv(output_file, index=False)

    return log_df
