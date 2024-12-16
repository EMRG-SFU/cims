import pandas as pd
import numpy as np

from ..old_utils import is_year
from .validation_utils import get_providers, get_year_cols, get_nodes

from ..utils import model_columns as COL
from ..utils import parameters as PARAM


def invalid_competition_type(df):
    """
    Find list of nodes with an invalid competition type.
    """
    valid_comp_types = [
        'Root',
        'Region',
        'Sector',
        'Tech Compete',
        'Node Tech Compete',
        'Fixed Ratio',
        'Supply - Fixed Price',
        'Supply - Cost Curve Annual',
        'Supply - Cost Curve Cumulative'
    ]

    invalid_rows = df[(df[COL.parameter] == PARAM.competition_type) &
                      (~df[COL.context].isin(valid_comp_types))]
    invalid_nodes = list(zip(invalid_rows.index, invalid_rows[COL.branch]))

    concern_desc = "nodes have an invalid 'Competition Type'"
    return invalid_nodes, concern_desc


def undefined_nodes(providers, requested):
    """
    Identify any nodes which are targets of other nodes, but have not been specified within
    the model description.
    """
    referenced_unspecified = [(i, v) for i, v in requested.items() if v not in providers.values]
    concern_desc = "nodes are requested by other nodes without being defined in the model"

    return referenced_unspecified, concern_desc


def unrequested_nodes(providers, requested, root_node):
    """
    Identify any non-root nodes which are specified in the model description but
    are never requested by other nodes.
    """
    unrequested_nodes = [(i, v) for i, v in providers.items() if
                            (v not in requested.values) and (v != root_node)]

    concern_desc = "nodes are defined in the model, but are not requested by other nodes"

    return unrequested_nodes, concern_desc


def nodes_no_provided_service(validator):
    """
    Identify any nodes which are specified but do not provide a service.
    """
    providers = get_providers(validator.model_df, node_col=validator.node_col)
    nodes = get_nodes(validator.model_df, validator.node_col)
    nodes_no_service = [(i, n) for i, n in nodes.items() if n not in providers.values]

    concern_desc = "nodes are specified but have no 'Service Provided'"
    return nodes_no_service, concern_desc


def nodes_requesting_self(validator):
    """
    Identifies any nodes which request services of themselves.
    """
    request_rows = validator.model_df[validator.model_df[COL.parameter] == PARAM.service_requested]
    self_requests = request_rows[
        request_rows[validator.node_col] == request_rows[validator.target_col]]
    self_requesting = [(i, node) for i, node in
                       zip(self_requests[validator.node_col], self_requests.index)]

    concern_desc = "nodes have a Service requested of themselves"
    return self_requesting, concern_desc


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
            if PARAM.service_requested not in df[COL.parameter].values:
                nodes_techs_no_serv_req.append((validator.branch2node_index_map[node], node, None))

    # Technologies
    node_techs = validator.model_df.groupby([validator.node_col, COL.technology])
    for (node, tech), df in node_techs:
        if PARAM.service_requested not in df[COL.parameter].values:
            nodes_techs_no_serv_req.append((validator.branch2node_index_map[node], node, tech))

    nodes_techs_no_serv_req.sort(key=lambda x: x[0])

    concern_desc = "nodes or technologies request no services"
    return nodes_techs_no_serv_req, concern_desc


def nodes_with_zero_output(validator):
    """
    Identify nodes or technologies where the "Output" line has been set to 0 for
    any of the year values in the model description.
    """
    year_cols = get_year_cols(validator.model_df)
    output = validator.model_df[validator.model_df[COL.parameter] == PARAM.output]
    zero_outputs = output[(output[year_cols] == 0).any(axis=1)]
    zero_output_nodes = list(zero_outputs[COL.branch].items())

    concern_desc = "nodes have an Output value of 0"
    return zero_output_nodes, concern_desc


def supply_without_lcc_or_price(validator):
    """
    Identify supply nodes (fixed price or cost curve) that have no price, 
    lcc financial, or cost curve price specified in the base year.
    """
    supply_nodes = validator.model_df[(validator.model_df[COL.parameter] == PARAM.competition_type) &
                                       (validator.model_df[COL.context].str.contains('supply', case=False, na=False))][COL.branch]

    cost_df = validator.model_df[validator.model_df[COL.parameter].isin([PARAM.lcc_financial, PARAM.price, PARAM.cost_curve_price])]
    has_base_year_cost = cost_df[~cost_df['2000'].isna()][COL.branch]
    no_prod_cost = [(validator.branch2node_index_map[f], f) for f in supply_nodes if
                    f not in has_base_year_cost.values]

    concern_desc = "supply nodes (fixed price or cost curve) are missing a price"
    return no_prod_cost, concern_desc


def techs_no_base_market_share(validator):
    """
    Identify technologies which have a market share line, but do not have a base year market share.
    """
    # The model's DataFrame
    data = validator.model_df

    base_year = [c for c in data.columns if is_year(c)][0]
    base_year_market_shares = data[data[COL.parameter] == PARAM.new_market_share]
    no_base_year_ms = base_year_market_shares[base_year_market_shares[base_year].isna()]

    techs_no_base_year_ms = []
    for idx in no_base_year_ms.index:
        techs_no_base_year_ms.append((idx,
                                      data.loc[idx, validator.node_col],
                                      data.loc[idx, COL.technology]))

    concern_desc = "technologies are missing a base year Market share"

    return techs_no_base_year_ms, concern_desc


def duplicate_service_requests(validator):
    """
    Identify nodes and technologies which request the same service twice.
    """
    # The model's DataFrame
    data = validator.model_df

    serv_request = data[data[COL.parameter] == PARAM.service_requested]
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
    services_req = data[data[COL.parameter] == PARAM.service_requested]

    # Select only the year columns
    year_cols = [c for c in services_req.columns if is_year(c)]
    year_values = services_req[year_cols]

    # Identify rows that have 0's or missing values
    row_has_bad_values = year_values.isin([0, np.nan]).all(axis=1)
    rows_with_bad_values = services_req[row_has_bad_values]

    # Create our Warning information
    bad_service_requests = list(zip(rows_with_bad_values.index, rows_with_bad_values[validator.node_col]))

    concern_desc = "nodes/technologies have Service requested values of only \
        0's or are missing all values"

    return bad_service_requests, concern_desc


def tech_compete_nodes_no_techs(validator):
    """
    Identify tech compete nodes that don't contain any technologies in the COL.technology column.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all Tech Compete Nodes
    tech_compete_nodes = data[(data[COL.parameter] == PARAM.competition_type) &
                              (data[COL.context] == 'Tech Compete')][validator.node_col]

    # Find all Technology Header Rows
    techs = data[data[COL.parameter] == COL.technology.lower()]

    # Determine which Tech Compete nodes don't have a Technology header
    tc_nodes_no_techs = []
    for idx, node in tech_compete_nodes.items():
        if node not in techs[validator.node_col].values:
            tc_nodes_no_techs.append((idx, node))

    concern_desc = "tech compete nodes contain no technologies"

    return tc_nodes_no_techs, concern_desc


def revenue_recycling_at_techs(validator):
    """
    Revenue recycling should only happen at nodes, never at techs
    """

    # The model's DataFrame
    data = validator.model_df

    # Find Recycled Revenues Rows
    rr_tech_df = data[(data[COL.parameter] == PARAM.recycled_revenues) &
                                  (~data[COL.technology].isna())]

    techs_recycling_revenues = []
    for idx in rr_tech_df.index:
        techs_recycling_revenues.append((
            idx,
            rr_tech_df.loc[idx, validator.node_col],
            rr_tech_df.loc[idx, COL.technology]
        ))

    concern_desc = "technologies have a Recycled revenues parameter (this \
        should only occur at nodes)"

    return techs_recycling_revenues, concern_desc


def both_cop_p2000_defined(validator):
    """
    No node should have both COP & P2000 exogenously defined
    """

    data = validator.model_df

    # Find all instances of COP & P2000 in the model description
    cop_p2000 = data[data[COL.parameter].isin([PARAM.cop, PARAM.p2000])]

    # Only keep the rows that aren't completely None
    cop_p2000 = cop_p2000.dropna(how='all',
                                 subset=[c for c in cop_p2000.columns if is_year(c)])
    duplicated = cop_p2000[cop_p2000.duplicated(
                 subset=[validator.node_col],
                 keep='first')]

    nodes_with_cop_and_p2000 = []
    for i, node in duplicated[validator.node_col].items():
        nodes_with_cop_and_p2000.append((i, node))

    concern_desc = "nodes have both COP & P2000 exogenously defined"
    return nodes_with_cop_and_p2000, concern_desc


def inconsistent_tech_refs(validator):
    """
    Identify nodes which include `Technology` column values and reference a
    technology which does not exist at that node.
    """
    # The model's DataFrame
    data = validator.model_df
    tech_data = data[~data[COL.technology].isna()]

    # Build a Node -> [Technologies] map
    tech_rows = tech_data[tech_data[COL.parameter]==COL.technology.lower()]
    node_tech_map = {}
    for node, tech_name in zip(tech_rows[validator.node_col], tech_rows[COL.technology]):
        if node not in node_tech_map:
            node_tech_map[node] = []
        node_tech_map[node].append(tech_name)

    # Find Unique Node/Branch + Technology rows
    inconsistent_tech_refs = []
    tech_other = tech_data[~tech_data.isin(tech_rows)].dropna(how='all')
    for idx, node, tech in zip(tech_other.index, tech_other[validator.node_col], tech_other[COL.technology]):
        try:
            if tech not in node_tech_map[node]:
                inconsistent_tech_refs.append((idx, node, tech))
        except KeyError:
            inconsistent_tech_refs.append((idx, node, tech))


    # Create Warning information
    concern_desc = "rows have inconsistent technology names"

    return inconsistent_tech_refs, concern_desc


def service_req_at_tech_node(validator):
    """
    Identify tech or node-tech compete nodes where a service request is
    specified at the node level.

    The implication of this is that values such as cumul_emissions_cost_rate
    will be incorrect.
    """
    # The model's DataFrame
    data = validator.model_df

    # Find all [Node-]Tech compete nodes
    tech_nodes = data[(data[COL.parameter]==PARAM.competition_type) &
                      (data[COL.context].str.lower().isin(['node tech compete', 'tech compete']))][validator.node_col].unique()


    # Find service request rows specified at the node level of a [node-]tech
    # compete node
    req_at_tech_node_rows = data[(data[COL.parameter] == PARAM.service_requested) &
                                 (data[COL.technology].isna()) &
                                 (data[validator.node_col].isin(tech_nodes))]

    # Find Unique Node/Branch + Technology rows
    service_req_at_tech_node = []
    for idx, node in zip(req_at_tech_node_rows.index,
                               req_at_tech_node_rows[validator.node_col]):
        service_req_at_tech_node.append((idx, node))

    # Create Warning information
    concern_desc = "Tech Compete nodes have node-level Service requests (these \
        should only occur at the tech-level)"

    return service_req_at_tech_node, concern_desc


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
        params_no_defs = data[~data[COL.parameter].isin(validator.default_param_df[COL.parameter])][COL.parameter].value_counts()

        # Find Unique Node/Branch + Technology rows
        missing_parameter_default = []
        for parameter, occurences in params_no_defs.items():
            missing_parameter_default.append((parameter,
                                              f"{occurences} occurences"))

        # Create Warning information
        concern_desc = "parameters are in the model, but do not have default \
            values"

    return missing_parameter_default, concern_desc


def min_max_conflicts(validator):
    """
    Identify technologies where the market share limits set conflict with one
    another. For example, max=0.5<min=0.7.
    """
    # The model's DataFrame
    data = validator.model_df

    # Min/Max Marketshare Limits
    ms_min_limits = data[data[COL.parameter] == PARAM.market_share_new_min]
    ms_max_limits = data[data[COL.parameter] == PARAM.market_share_new_max]

    # Build a Node -> [Technologies] map
    df = pd.merge(ms_min_limits, ms_max_limits,
                  how='inner', validate='many_to_many',
                  on=[COL.branch, COL.region, COL.sector, COL.technology],
                  suffixes=["_min", "_max"])

    issues = {}
    for y in get_year_cols(data):
        incongruent_nodes = df[df[f"{y}_min"] > df[f"{y}_max"]]
        for branch, tech in zip(incongruent_nodes[COL.branch], incongruent_nodes[COL.technology]):
            if (branch, tech) not in issues:
                issues[(branch, tech)] = []
            issues[(branch, tech)].append(y)


    # Find Unique Node/Branch + Technology rows
    min_max_conflicts = []
    for ((node, tech), years) in issues.items():
        min_max_conflicts.append((node, tech, years))

    # Create Warning information
    concern_desc = "technologies contain market share limits that  conflict \
        with one another (e.g., min > max)"

    return min_max_conflicts, concern_desc


def new_nodes_in_scenario(validator):
    """
    Identify new nodes included in the scenario models (i.e. were not in the
    base model) but which don't have a service provided parameter.
    """
    if validator.scenario_files:
        # The model dataframes
        base_data = validator._get_model_df(read_scenario_files=False)
        scenario_data = validator._get_model_df(read_base_file=False)

        # Find nodes from base and scenario files
        base_nodes = set(base_data[validator.node_col].dropna())
        scen_nodes = set(scenario_data[validator.node_col].dropna())
        declared_new_nodes = set(scenario_data[scenario_data[COL.parameter]==PARAM.service_provided]\
            [validator.node_col].dropna())

        # Find new nodes which haven't been declared without a service provided line
        new_nodes_in_scenario = list(scen_nodes\
                                    .difference(declared_new_nodes)\
                                    .difference(base_nodes))
    else:
        new_nodes_in_scenario = []

    # Create Warning information
    concern_desc = "nodes were included in scenario/model update files without \
        a Service provided parameter"

    return new_nodes_in_scenario, concern_desc


def new_techs_in_scenario(validator):
    """
    Identify new technologies included in the scenario models (i.e. were not in
    the base model) but which don't have a technology parameter.
    """
    if validator.scenario_files:
        # The model dataframes
        base_data = validator._get_model_df(read_scenario_files=False)
        scenario_data = validator._get_model_df(read_base_file=False)

        # Find nodes from base and scenario files
        base_techs = set([tuple(x) for x in
                        base_data[[validator.node_col, COL.technology]]\
                            .dropna().drop_duplicates().values])
        scen_techs = set([tuple(x) for x in
                        scenario_data[[validator.node_col, COL.technology]]\
                            .dropna().drop_duplicates().values])

        scen_declared_techs = scenario_data[scenario_data[COL.parameter]==COL.technology.lower()]
        declared_new_techs = set([tuple(x) for x in
                                scen_declared_techs[[validator.node_col, COL.technology]]\
                                    .dropna().drop_duplicates().values])

        # Find new nodes which haven't been declared without a service provided line
        new_techs_in_scenario = list(scen_techs\
                                    .difference(declared_new_techs)\
                                    .difference(base_techs))
    else:
        new_techs_in_scenario = []

    # Create Warning information
    concern_desc = "technologies were included in scenario/model update files \
        without a Technology parameter"

    return new_techs_in_scenario, concern_desc


def zero_requested_nodes(validator, providers, root_node):
    """
    Identify any non-root nodes which are specified in the model description
    but are only requested by node's via service request rows exogenously set to
    0.
    """
    data = validator.model_df
    request_lines = data[data[COL.parameter]==PARAM.service_requested]
    all_requested = set(request_lines[validator.target_col])

    zero_request_line = request_lines[get_year_cols(data)].sum(axis=1)==0
    non_zero_request_lines = request_lines[~zero_request_line]
    non_zero_requested = set(non_zero_request_lines[validator.target_col])


    zero_requested = [(i, v) for i, v in providers.items() if
                      (v in all_requested) and
                      (v not in non_zero_requested) and
                      (v != root_node)]

    concern_desc = "nodes are defined in the model, but are only requested by \
        nodes where all Service requested values are 0"

    return zero_requested, concern_desc


def lcc_at_tech_node(validator):
    """
    Identify any tech-compete or node-tech-compete nodes where an LCC value
    has been set exogenously.
    """
    tech_nodes = validator.model_df[COL.branch][(validator.model_df[COL.parameter] == PARAM.competition_type) & (validator.model_df[COL.context].str.lower().str.contains('tech compete'))]
    lcc_nodes = validator.model_df[COL.branch][
        validator.model_df[COL.technology].isna() &
        validator.model_df[COL.parameter].str.lower().str.contains('lcc')]
    
    lcc_at_tech_nodes = [(i, n) for i, n in lcc_nodes.items() if n in tech_nodes]

    concern_desc = "tech compete nodes have exogenously defined LCC values"

    return lcc_at_tech_nodes, concern_desc


def lcc_at_tech(validator):
    """
    Identify any technologies where an LCC value has been set exogenously.
    """
    techs = validator.model_df[[COL.branch, COL.technology]].drop_duplicates().dropna(how='any')
    
    lcc_techs = validator.model_df[COL.branch][
        ~validator.model_df[COL.technology].isna() &
        validator.model_df[COL.parameter].str.lower().str.contains('lcc')]
    
    lcc_at_techs = [(i, n) for i, n in lcc_techs.items() if n in techs]

    concern_desc = "technologies have exogenously defined LCC values"

    return lcc_at_techs, concern_desc


