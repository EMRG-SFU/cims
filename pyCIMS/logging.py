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


def log_model(model, output_file):
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

    log_df = pd.DataFrame(data_tuples)
    log_df.columns = ['node', 'year', 'technology', 'parameter', 'value']

    # Split tupled values
    log_df[['context', 'unit', 'value']] = pd.DataFrame(log_df['value'].to_list())
    log_df = log_df[['node', 'year', 'technology', 'parameter', 'context', 'unit', 'value']]

    # Write to file
    log_df.to_csv(output_file, index=False)

    return log_df