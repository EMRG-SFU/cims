import pandas as pd
import numpy as np

from ..utils.model_description import column_list as COL
from ..utils.parameter import list as PARAM


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
            missing_parameter_default.append((parameter, int(occurences)))

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
    Identify nodes and technologies which request the same service more than once.

    Returns {(node, tech, target): {year: count}} so the caller can see exactly
    which (node, technology, target) combinations have duplicates and how many
    times each year appears. All years duplicated at the same count suggests the
    same rows exist in multiple files; only some years duplicated suggests a
    data entry error.
    """
    data = validator.model_df

    serv_request = data[data[COL.parameter] == PARAM.service_request]
    duplicated = serv_request[serv_request.duplicated(
        subset=[validator.node_col, COL.technology, validator.target_col, "Year"],
        keep=False)]

    result: dict = {}
    for _, row in duplicated.iterrows():
        key = (row[validator.node_col], row[COL.technology], row[validator.target_col])
        year = row["Year"]
        result.setdefault(key, {})
        result[key][year] = result[key].get(year, 0) + 1

    concern_desc = "(node, technology, target) combination(s) have duplicate service_request rows"

    return result, concern_desc


def bad_service_req(validator):
    """
    Identify nodes/technologies that have a service requested line, but where the values in these
    lines are either blank or exogenously specified as 0.
    """
    data = validator.model_df
    services_req = data[data[COL.parameter] == PARAM.service_request].copy()
    services_req["Value_num"] = pd.to_numeric(services_req["Value"], errors="coerce")
    group_keys = [validator.node_col, COL.technology, validator.target_col]

    # True for each row where ALL values in the (node, tech, target) group are 0 or NaN
    all_bad_mask = services_req.groupby(group_keys, dropna=False)["Value_num"] \
        .transform(lambda g: (g.isna() | (g == 0)).all())

    bad_rows = services_req[all_bad_mask].drop_duplicates(subset=group_keys)
    bad_service_requests = list(zip(
        bad_rows.index,
        bad_rows[validator.node_col],
        bad_rows[COL.technology],
        bad_rows[validator.target_col],
    ))

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
    request_lines = data[data[COL.parameter] == PARAM.service_request].copy()
    request_lines["Value_num"] = pd.to_numeric(request_lines["Value"], errors="coerce").fillna(0)
    all_requested = set(request_lines[validator.target_col])

    group_keys = [validator.node_col, COL.technology, validator.target_col]
    group_sums = request_lines.groupby(group_keys, dropna=False)["Value_num"].sum()
    non_zero_requested = set(group_sums[group_sums != 0].reset_index()[validator.target_col])

    zero_requested = [(i, v) for i, v in providers.items() if
                      (v in all_requested) and
                      (v not in non_zero_requested) and
                      (v != validator.root_node)]

    concern_desc = "nodes are defined in the model, but are only requested by nodes where all Service requested values are 0"

    return zero_requested, concern_desc
