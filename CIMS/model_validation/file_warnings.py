import pandas as pd
import numpy as np

from .validation_utils import get_year_cols

from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM
from ..utils.parameter.parse import is_year


def missing_parameter_default(validator):
    """
    Identify parameters in the model file which are missing from the default
    parameter file.

    If no default parameters have been specified, then this check is ignored.
    """
    if len(validator.default_param_df) == 0:
        missing_parameter_default = []
        # Create Warning information
        concern_desc = "!!! No default parameters file was provided"

    else:
        # The model's DataFrame
        data = validator.model_df.dropna(how='all')

        # Find all parameter names
        params_no_defs = data[~data[COL.parameter].isin(
            validator.default_param_df[COL.parameter])][COL.parameter].value_counts()

        # Find Unique Node/Branch + Technology rows
        missing_parameter_default = []
        for parameter, occurences in params_no_defs.items():
            missing_parameter_default.append((parameter,
                                              f"{occurences} occurences"))

        # Create Warning information
        concern_desc = "parameters are in the model, but do not have default \
            values"

    return missing_parameter_default, concern_desc


def unrequested_nodes(validator, providers, requested):
    """
    Identify any non-root nodes which are specified in the model description but
    are never requested by other nodes.
    """
    unrequested_nodes = [(i, v) for i, v in providers.items() if
                         (v not in requested.values) and (v != validator.root_node)]

    concern_desc = "nodes are defined in the model, but are not requested by other nodes"

    return unrequested_nodes, concern_desc


def nodes_no_requested_service(validator):
    """
    Identify nodes or technologies which have been specified in the model description but don't
    request services from other nodes.
    """
    nodes_techs_no_serv_req = []

    # Nodes (without techs)
    nodes = validator.model_df.groupby(validator.node_col)
    for node, df in nodes:
        if pd.isna(df[COL.technology]).all():
            if PARAM.service_request not in df[COL.parameter].values:
                nodes_techs_no_serv_req.append(
                    (validator.branch2node_index_map[node], node, None))

    # Technologies
    node_techs = validator.model_df.groupby(
        [validator.node_col, COL.technology])
    for (node, tech), df in node_techs:
        if PARAM.service_request not in df[COL.parameter].values:
            nodes_techs_no_serv_req.append(
                (validator.branch2node_index_map[node], node, tech))

    nodes_techs_no_serv_req.sort(key=lambda x: x[0])

    concern_desc = "nodes or technologies request no services"
    return nodes_techs_no_serv_req, concern_desc


def duplicate_service_requests(validator):
    """
    Identify nodes and technologies which request the same service twice.
    """
    # The model's DataFrame
    data = validator.model_df

    serv_request = data[data[COL.parameter] == PARAM.service_request]
    duplicated = serv_request[serv_request.duplicated(
        subset=[validator.node_col, COL.technology, validator.target_col],
        keep=False)]

    if len(duplicated) > 0:
        # Group & list rows (index) where duplicates exist
        duplicated_with_idx = duplicated.reset_index()
        duplicated_groups = duplicated_with_idx.groupby(
            [validator.node_col, COL.technology, validator.target_col],
            dropna=False)['index'].apply(list)
        duplicated_groups = duplicated_groups.reset_index()

        # Create our Warning information
        duplicate_req = list(zip(duplicated_groups['index'],
                                 duplicated_groups[COL.branch],
                                 duplicated_groups[COL.technology]))
    else:
        duplicate_req = []

    concern_desc = "nodes/technologies request the same service more than once"

    return duplicate_req, concern_desc


def bad_service_req(validator):
    """
    Identify nodes/technologies that have a service requested line, but where the values in these
    lines are either blank or exogenously specified as 0.
    """
    # The model's DataFrame
    data = validator.model_df

    # Filter to Only Include Service Requested
    services_req = data[data[COL.parameter] == PARAM.service_request]

    # Select only the year columns
    year_cols = [c for c in services_req.columns if is_year(c)]
    year_values = services_req[year_cols]

    # Identify rows that have 0's or missing values
    row_has_bad_values = year_values.isin([0, np.nan]).all(axis=1)
    rows_with_bad_values = services_req[row_has_bad_values]

    # Create our Warning information
    bad_service_requests = list(
        zip(rows_with_bad_values.index, rows_with_bad_values[validator.node_col]))

    concern_desc = "nodes/technologies have Service requested values of only \
        0's or are missing all values"

    return bad_service_requests, concern_desc


def zero_requested_nodes(validator, providers):
    """
    Identify any non-root nodes which are specified in the model description
    but are only requested by node's via service request rows exogenously set to
    0.
    """
    data = validator.model_df
    request_lines = data[data[COL.parameter] == PARAM.service_request]
    all_requested = set(request_lines[validator.target_col])

    numeric_values = request_lines[get_year_cols(
        data)].replace("None", None).astype(float)
    zero_request_line = numeric_values.sum(axis=1) == 0
    non_zero_request_lines = request_lines[~zero_request_line]
    non_zero_requested = set(non_zero_request_lines[validator.target_col])

    zero_requested = [(i, v) for i, v in providers.items() if
                      (v in all_requested) and
                      (v not in non_zero_requested) and
                      (v != validator.root_node)]

    concern_desc = "nodes are defined in the model, but are only requested by nodes where all Service requested values are 0"

    return zero_requested, concern_desc
