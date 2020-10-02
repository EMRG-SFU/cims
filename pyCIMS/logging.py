# import pandas as pd
# import pyCIMS
# import pprint as pp
# from pyCIMS.model import NodeQuantity
# import numpy as np
#
# file = 'pycims_prototype/pyCIMS_model_description_Alberta_Test.xlsb'
# my_reader = pyCIMS.ModelReader(infile=file,
#                                sheet_map={'model': 'Model',
#                                           'incompatible': 'Incompatible',
#                                           'default_tech': 'Technologies'},
#                                node_col='Node')
# my_model = pyCIMS.Model(my_reader)
# my_model.run(show_warnings=False, max_iterations=5)
#


def has_techs(node_year_data):
    return 'technologies' in node_year_data.keys()


def log_int(val):
    return [(None, float(val))]


def log_float(val):
    return [(None, val)]


def log_NodeQuantity(val):
    return [(None, float(val.get_total_quantity()))]


def log_list(val):
    """ List of dictionaries. For each item, extract the value and year_value"""
    val_pairs = []
    for entry in val:
        value = entry['value']
        year_value = entry['year_value']
        val_pairs.append((value, year_value))

    return val_pairs


def log_dict(val):
    """ Dictionary. May be base or be a dictionary containing base dictionaries"""

    # Check if base dictionary
    if 'year_value' in val.keys():
        year_value = val['year_value']
        if year_value is None:
            return [(None, None)]
        else:
            return [(None, float(year_value))]
    else:
        val_pairs = []
        for k, v in val.items():
            year_value = v['year_value']
            val_pairs.append((k, year_value))
        return val_pairs


def log_model(model):
    def add_log_item(all_logs, log_tuple):
        log_func = {int: log_int,
                    float: log_float,
                    NodeQuantity: log_NodeQuantity,
                    list: log_list,
                    dict: log_dict}

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
    log_df[['value', 'year_value']] = pd.DataFrame(log_df['value'].to_list())

    return log_df


log_df = log_model(my_model)

log_df.head(10000).to_csv("TESTER_SAMPLE.csv", index=False)
# log_df['value_types'] = log_df['value'].apply(lambda x: type(x))
# log_df['value_types'].unique()